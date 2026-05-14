"""Scene-level number-classifier refiner.

Per-cluster mbconv hits 95.94% on round3 but the remaining ~4% of errors
poison v9 because the encoder is hyper-sensitive to wrong anchors. This
model learns to REFINE mbconv's per-cluster predictions by attending to
neighbors in the same frame.

Architecture
────────────
For each frame:
  1. Spatial-CC cluster the v8 number mask (dilate-28). Same as v9.
  2. mbconv runs per-cluster on each number cluster's 64×64 binary crop
     → 9-D logits.
  3. cc_tokenizer also produces yardline tokens (no NGS_x class — pure
     positional anchors).
  4. Build per-frame token features:
        [9 mbconv logits | 1 has_logits | 1 is_yardline | 3 geometry]
                                                       (cx, cy, log_area)
     Total feature_dim = 14. Yard tokens get zeros for the 9 logit slots
     and 0 for has_logits, 1 for is_yardline.
  5. Tiny transformer (2 layers × 4 heads × d_model=32) lets every token
     attend to every other.
  6. Per-number-token output head → 9-class softmax (refined).

Loss: CE on each number token's refined output vs GT class (from H-projection).

Yard tokens are context only — no loss applied to them.

At v9 inference: replace the per-cluster mbconv argmax with this model's
refined argmax.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "pipeline"))

from cc_tokenizer_v2 import (    # noqa: E402
    SRC_W, SRC_H, MIN_CC_PX, DEFAULT_DILATE_PX,
    LOG_AREA_DIVISOR, _make_classifier_crop,
)
NUM_MASK_THRESH = 0.5    # cc_tokenizer_v2 hardcodes this; mirror locally.
from src.homography.field_model import (    # noqa: E402
    NUMBER_Y_NEAR, NUMBER_Y_FAR, ngs_x_to_field_number, TEN_YARD_POSITIONS,
)


# ────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────
N_CLASSES = 9
TOK_NUM = 0
TOK_YARD = 1
NGS_BUCKET_TOL = 5.0
NGS_Y_TOL = 6.0
NGS_X_FIELD_MIN = 10.0
NGS_X_FIELD_MAX = 110.0
V9_VAL_GAME = "2024090802"
CLASSES = ["10L", "10R", "20L", "20R", "30L", "30R", "40L", "40R", "50"]
_CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def class_for_ngs_x(ngs_x: float) -> str | None:
    """Map NGS_x ∈ {20, 30, ..., 100} to class label."""
    bucket = int(round(ngs_x / 10.0)) * 10
    if bucket not in TEN_YARD_POSITIONS:
        return None
    n = ngs_x_to_field_number(bucket)
    if n == 50:
        return "50"
    return f"{n}{'L' if bucket < 60 else 'R'}"


# ────────────────────────────────────────────────────────────────────────
# Per-frame token build
# ────────────────────────────────────────────────────────────────────────

def build_frame_tokens(masks: np.ndarray, H: np.ndarray, K: np.ndarray,
                          dist: np.ndarray, mbconv_fn,
                          dilate_px: int = DEFAULT_DILATE_PX) -> dict:
    """Returns dict with:
      tokens: (N, 14) float — N = num + yard tokens in this frame
      gt_class: (N,) int — 0..8 for number tokens, -1 for yard / unclassifiable
      is_number: (N,) bool
    """
    out_tokens = []
    out_gt = []
    out_is_number = []

    # ── Number-channel clusters (with mbconv logits) ──
    num_mask = masks[..., 3].astype(np.float32)
    bin_mask = (num_mask > NUM_MASK_THRESH).astype(np.uint8)
    if bin_mask.sum() > 0:
        ks = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        dilated = cv2.dilate(bin_mask, kernel, iterations=1)
        n_cl, lab, _, _ = cv2.connectedComponentsWithStats(
            dilated, connectivity=8)
        crops = []
        records = []
        for cid in range(1, n_cl):
            mm = (lab == cid) & (bin_mask > 0)
            if mm.sum() < MIN_CC_PX:
                continue
            ys, xs = np.where(mm)
            x_min = int(xs.min()); y_min = int(ys.min())
            x_max = int(xs.max()) + 1; y_max = int(ys.max()) + 1
            cx = float(xs.mean()); cy = float(ys.mean())
            area = int(mm.sum())
            crop = _make_classifier_crop(
                bin_mask, x_min, y_min, x_max, y_max,
                cluster_label_map=lab, cluster_id=cid)
            crops.append(crop)
            records.append((cx, cy, area))

        if crops:
            # Run mbconv batched
            logits = mbconv_fn(crops)    # (M, 9)
            # GT: project each cluster centroid through H to NGS, snap.
            cents = np.asarray([[r[0], r[1]] for r in records],
                                  dtype=np.float64)
            ngs = _dist_to_ngs(cents, H, K, dist)
            for i, (cx, cy, area) in enumerate(records):
                ngs_x, ngs_y = ngs[i]
                cls_idx = -1
                if NGS_X_FIELD_MIN <= ngs_x <= NGS_X_FIELD_MAX:
                    bucket = int(round(ngs_x / 10.0)) * 10
                    if bucket in TEN_YARD_POSITIONS \
                            and abs(ngs_x - bucket) <= NGS_BUCKET_TOL:
                        dy = min(abs(ngs_y - NUMBER_Y_NEAR),
                                  abs(ngs_y - NUMBER_Y_FAR))
                        if dy <= NGS_Y_TOL:
                            cls_name = class_for_ngs_x(float(bucket))
                            if cls_name in _CLASS_TO_IDX:
                                cls_idx = _CLASS_TO_IDX[cls_name]
                # Token features
                tok = _make_token_features(
                    logits[i], cx=cx, cy=cy, area=area,
                    is_number=True)
                out_tokens.append(tok)
                out_gt.append(cls_idx)
                out_is_number.append(True)

    # ── Yardline clusters (geometry only) ──
    yard_mask_prob = masks[..., 0].astype(np.float32)
    yard_bin = (yard_mask_prob > NUM_MASK_THRESH).astype(np.uint8)
    if yard_bin.sum() > 0:
        n_cl, lab, _, _ = cv2.connectedComponentsWithStats(
            yard_bin, connectivity=8)
        for cid in range(1, n_cl):
            mm = (lab == cid)
            if mm.sum() < MIN_CC_PX:
                continue
            ys, xs = np.where(mm)
            cx = float(xs.mean()); cy = float(ys.mean())
            area = int(mm.sum())
            tok = _make_token_features(
                np.zeros(9, dtype=np.float32),
                cx=cx, cy=cy, area=area, is_number=False)
            out_tokens.append(tok)
            out_gt.append(-1)
            out_is_number.append(False)

    if not out_tokens:
        return dict(
            tokens=np.zeros((0, 14), dtype=np.float32),
            gt_class=np.zeros(0, dtype=np.int64),
            is_number=np.zeros(0, dtype=bool),
        )
    return dict(
        tokens=np.stack(out_tokens, axis=0).astype(np.float32),
        gt_class=np.asarray(out_gt, dtype=np.int64),
        is_number=np.asarray(out_is_number, dtype=bool),
    )


def _make_token_features(logits: np.ndarray, cx: float, cy: float,
                            area: int, is_number: bool) -> np.ndarray:
    feat = np.zeros(14, dtype=np.float32)
    if is_number:
        feat[0:9] = logits
        feat[9] = 1.0    # has_logits
        feat[10] = 0.0   # is_yardline
    else:
        # logits all zeros
        feat[9] = 0.0
        feat[10] = 1.0
    feat[11] = cx / SRC_W
    feat[12] = cy / SRC_H
    feat[13] = float(np.log(max(1, area))) / LOG_AREA_DIVISOR
    return feat


def _dist_to_ngs(pts_dist, H, K, dist):
    if len(pts_dist) == 0:
        return pts_dist
    und = cv2.undistortPoints(
        pts_dist.reshape(-1, 1, 2).astype(np.float64),
        K, np.asarray(dist).reshape(-1), P=K).reshape(-1, 2)
    und_h = np.concatenate([und, np.ones((und.shape[0], 1))], axis=1)
    ngs_h = (H @ und_h.T).T
    return ngs_h[:, :2] / ngs_h[:, 2:3]


# ────────────────────────────────────────────────────────────────────────
# Model
# ────────────────────────────────────────────────────────────────────────

class SceneRefiner(nn.Module):
    """Tiny transformer over per-frame number + yard tokens.

    Input: (B, N, 14) tokens + (B, N) padding mask.
    Output: (B, N, 9) refined logits — only loss on tokens where
            is_number=True.
    """
    def __init__(self, d_model: int = 32, n_layers: int = 2,
                 n_heads: int = 4, ffn_dim: int = 64,
                 dropout: float = 0.1, n_classes: int = N_CLASSES,
                 input_dim: int = 14):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ffn_dim, dropout=dropout,
            activation="relu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, tokens, padding_mask):
        x = self.embed(tokens)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.head(x)


# ────────────────────────────────────────────────────────────────────────
# Dataset (per-frame)
# ────────────────────────────────────────────────────────────────────────

class SceneRefinerDataset(Dataset):
    def __init__(self, entries, cache_dir, intrinsics_by_clip, mbconv_fn,
                 cache_to_ram: bool = True):
        self.entries = entries
        self.cache_dir = cache_dir
        self.intrinsics_by_clip = intrinsics_by_clip
        self.mbconv_fn = mbconv_fn
        self.cache_to_ram = cache_to_ram
        self._ram_cache = None
        if cache_to_ram:
            self._build_cache()

    def _build_cache(self):
        t0 = time.time()
        self._ram_cache = []
        for i in range(len(self.entries)):
            self._ram_cache.append(self._compute(i))
        n_num = sum(int(item["is_number"].sum())
                    for item in self._ram_cache)
        n_yard = sum(int((~item["is_number"]).sum())
                     for item in self._ram_cache)
        print(f"  cached {len(self.entries)} frames "
              f"({n_num} num + {n_yard} yard tokens) "
              f"in {time.time() - t0:.1f}s", flush=True)

    def _compute(self, idx):
        e = self.entries[idx]
        cp = os.path.join(self.cache_dir, f"{e['id']}.npz")
        d = np.load(cp)
        masks = d["masks"].astype(np.float32)
        intr = self.intrinsics_by_clip.get(e["clip"], {})
        K = np.asarray(intr.get("K", np.eye(3)), dtype=np.float64)
        if K.shape == (9,):
            K = K.reshape(3, 3)
        dist = np.asarray(intr.get("dist", [0, 0, 0, 0, 0]),
                              dtype=np.float64)
        H = np.asarray(e["H"], dtype=np.float64)
        item = build_frame_tokens(masks, H, K, dist, self.mbconv_fn)
        return {
            "tokens": torch.from_numpy(item["tokens"]),
            "gt_class": torch.from_numpy(item["gt_class"]),
            "is_number": torch.from_numpy(item["is_number"]),
            "frame_id": e["id"],
        }

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        if self._ram_cache is not None:
            return self._ram_cache[idx]
        return self._compute(idx)


def collate(batch):
    """Pad variable-length token sets to per-batch max."""
    max_n = max(item["tokens"].shape[0] for item in batch)
    if max_n == 0:
        max_n = 1
    B = len(batch)
    F_dim = 14
    tokens = torch.zeros(B, max_n, F_dim, dtype=torch.float32)
    gt = torch.full((B, max_n), -1, dtype=torch.long)
    is_num = torch.zeros(B, max_n, dtype=torch.bool)
    pad = torch.ones(B, max_n, dtype=torch.bool)
    for i, item in enumerate(batch):
        n = item["tokens"].shape[0]
        if n > 0:
            tokens[i, :n] = item["tokens"]
            gt[i, :n] = item["gt_class"]
            is_num[i, :n] = item["is_number"]
            pad[i, :n] = False
    return {
        "tokens": tokens, "gt_class": gt,
        "is_number": is_num, "padding_mask": pad,
    }


# ────────────────────────────────────────────────────────────────────────
# mbconv loader
# ────────────────────────────────────────────────────────────────────────

def make_backbone_logits_fn(ckpt_path: str, arch: str,
                                device: torch.device):
    from train_compare_classifiers import build_model
    from train_number_classifier import PIXEL_MEAN, PIXEL_STD
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck.get("model_state_dict", ck)
    classes = ck.get("classes", CLASSES)
    if list(classes) != CLASSES:
        idx_map = [classes.index(c) for c in CLASSES if c in classes]
        if len(idx_map) != N_CLASSES:
            raise ValueError(f"ckpt classes {classes} don't match {CLASSES}")
    else:
        idx_map = None
    model = build_model(arch, dropout=0.0, num_classes=len(classes))
    model.load_state_dict(state)
    model.to(device).eval()

    @torch.no_grad()
    def _fn(crops):
        if not crops:
            return np.zeros((0, N_CLASSES), dtype=np.float32)
        arr = np.stack([c.astype(np.float32) for c in crops], axis=0)
        arr = (arr / 255.0 - PIXEL_MEAN) / PIXEL_STD
        x = torch.from_numpy(arr).unsqueeze(1).to(device)
        logits = model(x).cpu().numpy().astype(np.float32)
        if idx_map is not None:
            logits = logits[:, idx_map]
        return logits

    return _fn


# ────────────────────────────────────────────────────────────────────────
# Train / eval
# ────────────────────────────────────────────────────────────────────────

def split_by_game(entries, val_game=V9_VAL_GAME):
    train = [e for e in entries if e["clip"].split("/", 1)[0] != val_game]
    val = [e for e in entries if e["clip"].split("/", 1)[0] == val_game]
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone-ckpt", default=os.path.join(
        PROJECT_ROOT, "models/mbconv_round3/best.pth"))
    ap.add_argument("--backbone-arch", default="mbconv",
                     choices=["mbconv", "mbconv_mini", "dsresnet10",
                                "dsresnet10w", "tiny", "mininet"])
    ap.add_argument("--manifest-file", default=os.path.join(
        PROJECT_ROOT, "data/h_pool_and_intrinsics.json"))
    ap.add_argument("--cache-dir", default=os.path.join(
        PROJECT_ROOT, "data/dense_regression/cache"))
    ap.add_argument("--out-dir", default=os.path.join(
        PROJECT_ROOT, "models/scene_refiner"))
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--ffn-dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading manifest...")
    m = json.load(open(args.manifest_file))
    intr = m["intrinsics_by_clip"]
    train_e, val_e = split_by_game(m["entries"])
    print(f"Split: train={len(train_e)}  val={len(val_e)}")

    print(f"Loading {args.backbone_arch} from {args.backbone_ckpt}...")
    mbconv_fn = make_backbone_logits_fn(
        args.backbone_ckpt, args.backbone_arch, device)

    print(f"Building train cache...")
    train_ds = SceneRefinerDataset(
        train_e, args.cache_dir, intr, mbconv_fn, cache_to_ram=True)
    print(f"Building val cache...")
    val_ds = SceneRefinerDataset(
        val_e, args.cache_dir, intr, mbconv_fn, cache_to_ram=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate)

    model = SceneRefiner(
        d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, ffn_dim=args.ffn_dim,
        dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SceneRefiner: {n_params:,} params  ({n_params/1e3:.2f}K)")

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr_min)
    crit = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing, ignore_index=-1)

    log_path = os.path.join(args.out_dir, "training_log.jsonl")
    open(log_path, "w").close()
    best_val_acc = 0.0
    t0 = time.time()

    # Baseline: mbconv argmax accuracy on val (number tokens only).
    base_correct = 0; base_total = 0
    for batch in val_loader:
        toks = batch["tokens"]; is_num = batch["is_number"]; gt = batch["gt_class"]
        # mbconv logits live in tokens[..., :9]; argmax for number tokens
        pred = toks[..., :9].argmax(-1)
        m_ = is_num & (gt >= 0)
        if m_.any():
            base_correct += int((pred[m_] == gt[m_]).sum())
            base_total += int(m_.sum())
    base_acc = base_correct / max(base_total, 1)
    print(f"Baseline (mbconv argmax) val acc: {base_acc*100:.2f}% "
          f"({base_correct}/{base_total})")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0; train_n = 0
        train_correct = 0; train_total = 0
        for batch in train_loader:
            tokens = batch["tokens"].to(device)
            pad = batch["padding_mask"].to(device)
            gt = batch["gt_class"].to(device)
            is_num = batch["is_number"].to(device)
            logits = model(tokens, pad)    # (B, N, 9)
            # Flatten and compute loss only on number-valid tokens.
            flat_logits = logits.view(-1, N_CLASSES)
            flat_gt = gt.view(-1)
            valid_mask = is_num.view(-1) & (flat_gt >= 0)
            if not valid_mask.any():
                continue
            sel_logits = flat_logits[valid_mask]
            sel_gt = flat_gt[valid_mask]
            loss = crit(sel_logits, sel_gt)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            with torch.no_grad():
                pred = sel_logits.argmax(-1)
                train_correct += int((pred == sel_gt).sum())
                train_total += int(sel_gt.numel())
            train_loss += loss.item() * sel_gt.numel()
            train_n += sel_gt.numel()
        sched.step()
        train_acc = train_correct / max(train_total, 1)
        train_loss /= max(train_n, 1)

        # Val
        model.eval()
        val_correct = 0; val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                tokens = batch["tokens"].to(device)
                pad = batch["padding_mask"].to(device)
                gt = batch["gt_class"].to(device)
                is_num = batch["is_number"].to(device)
                logits = model(tokens, pad).view(-1, N_CLASSES)
                flat_gt = gt.view(-1)
                m_ = is_num.view(-1) & (flat_gt >= 0)
                if m_.any():
                    pred = logits[m_].argmax(-1)
                    val_correct += int((pred == flat_gt[m_]).sum())
                    val_total += int(m_.sum())
        val_acc = val_correct / max(val_total, 1)
        elapsed = time.time() - t0

        print(f"Ep {epoch+1:3d}/{args.epochs}  "
              f"loss={train_loss:.4f}  train_acc={train_acc*100:.2f}%  "
              f"val_acc={val_acc*100:.2f}%  baseline={base_acc*100:.2f}%  "
              f"({elapsed:.0f}s)", flush=True)

        with open(log_path, "a") as f:
            json.dump({
                "epoch": epoch+1, "lr": sched.get_last_lr()[0],
                "train_loss": train_loss, "train_acc": train_acc,
                "val_acc": val_acc, "baseline_val_acc": base_acc,
            }, f)
            f.write("\n")

        ckpt = {"model_state_dict": model.state_dict(),
                "epoch": epoch+1, "args": vars(args),
                "classes": CLASSES, "val_acc": val_acc,
                "baseline_val_acc": base_acc}
        torch.save(ckpt, os.path.join(args.out_dir, "last.pth"))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, os.path.join(args.out_dir, "best.pth"))
            print(f"   ↑ new best val_acc = {val_acc*100:.2f}%", flush=True)

    print(f"\nDone. Best val_acc: {best_val_acc*100:.2f}%  "
          f"(baseline: {base_acc*100:.2f}%, "
          f"Δ = {(best_val_acc - base_acc)*100:+.2f}pp)")


if __name__ == "__main__":
    main()
