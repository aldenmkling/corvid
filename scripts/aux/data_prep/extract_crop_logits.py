"""Extract per-frame number-crop classifier logits for every pseudo-labeled
frame. Phase 2/3 training needs these (RFB's input is (encoder_feat, crop_logits)).

For each pseudo_labels/<clip>.npz:
  1. Open source mp4
  2. Read frames at d["frame_idx"][k]
  3. Undistort + run UNet + tokenize (cc_tokenizer_v3 with return_aux=True)
  4. Run crop classifier on aux["num_crops"]
  5. Save sidecar to data/pseudo_labels_crops/<clip>.npz:
       frame_idx        (n_frames,)            (matches the source npz)
       num_token_idx    object array (n_frames,) → per-frame indices of
                                                 num tokens in the source
                                                 npz's token order
       crop_logits      object array (n_frames,) → (n_num_tok, 9) float32

Token-order alignment: cc_tokenizer_v3 is deterministic; re-running on the
same UNet output (which is also deterministic) yields the same tokens in
the same order. The script verifies this by checking that the count of
num tokens in the re-tokenized frame matches the source npz.

Resume-friendly: skips clips whose sidecar already exists.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import json
import numpy as np
import torch
import segmentation_models_pytorch as smp

cv2.setNumThreads(1)
try: torch.set_num_threads(1)
except Exception: pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "pipeline"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "data_prep"))

from cc_tokenizer_v2 import SRC_W, SRC_H, null_classifier  # noqa: E402
from cc_tokenizer_v3 import cc_tokens_from_frame_v3  # noqa: E402
from train_rf_a import make_painted_logits_fn  # noqa: E402

TYPE_NUM = 3
UH, UW = 512, 896
IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_unified(path, device):
    m = smp.Unet("mit_b0", encoder_weights=None, in_channels=3, classes=4)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck.get("model_state_dict", ck))
    return m.to(device).eval()


def pre(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (UW, UH))
    x = (rgb.astype(np.float32) / 255.0 - IM_MEAN) / IM_STD
    return torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0)


@torch.no_grad()
def pred(model, frame, device):
    x = pre(frame).to(device)
    p = torch.sigmoid(model(x))[0].cpu().numpy()
    h0, w0 = frame.shape[:2]
    out = np.zeros((h0, w0, 4), dtype=np.float32)
    for ci in range(4):
        out[..., ci] = cv2.resize(p[ci], (w0, h0), interpolation=cv2.INTER_LINEAR)
    return out


def npz_to_clip_path(clip_id):
    """2019092204_play_001_sideline → videos/clips/2019092204/play_001/sideline.mp4"""
    parts = clip_id.split("_")
    return os.path.join(PROJECT_ROOT, "videos", "clips",
                         parts[0], f"{parts[1]}_{parts[2]}",
                         f"{'_'.join(parts[3:])}.mp4")


def process_clip(npz_path, out_path, unet, crop_fn, intr_by_clip, device,
                  verbose=False):
    """Extract per-frame crop logits for one clip's npz. Returns (n_frames, n_with_nums)."""
    clip_id = os.path.basename(npz_path).replace(".npz", "")
    clip_path = npz_to_clip_path(clip_id)
    if not os.path.exists(clip_path):
        print(f"  [missing-mp4] {clip_path}", flush=True)
        return 0, 0

    d = np.load(npz_path, allow_pickle=True)
    frame_idx_arr = np.asarray(d["frame_idx"], dtype=np.int64)
    n_frames = len(frame_idx_arr)
    source_tokens = d["tokens"]
    source_type_idx = d["type_idx"]

    rel = os.path.relpath(clip_path, os.path.join(PROJECT_ROOT, "videos/clips"))
    intr = intr_by_clip.get(rel, {})
    K = np.asarray(intr.get("K", np.eye(3).tolist()), dtype=np.float64)
    if K.shape == (9,): K = K.reshape(3, 3)
    dist = np.asarray(intr.get("dist", [0]*5), dtype=np.float64)

    cap = cv2.VideoCapture(clip_path)
    out_logits = np.empty(n_frames, dtype=object)
    out_num_idx = np.empty(n_frames, dtype=object)
    # Initialize all slots to empty (safe defaults for any frame we
    # never reach, e.g., truncated mp4).
    for k in range(n_frames):
        out_logits[k] = np.zeros((0, 9), dtype=np.float32)
        out_num_idx[k] = np.zeros((0,), dtype=np.int64)
    n_with_nums = 0

    # Read frames SEQUENTIALLY and skip ones not in frame_idx_arr.
    # cap.set(POS_FRAMES) per-frame forces a keyframe-rewind in cv2's
    # FFmpeg backend, which can be 10-50x slower than sequential reads.
    wanted = set(int(x) for x in frame_idx_arr)
    max_wanted = max(wanted) if wanted else -1
    fi_to_k = {int(frame_idx_arr[k]): k for k in range(n_frames)}

    t0 = time.time()
    fi = -1
    while True:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        fi += 1
        if fi > max_wanted:
            break
        if fi not in wanted:
            continue
        k = fi_to_k[fi]
        if fr.shape[1] != SRC_W or fr.shape[0] != SRC_H:
            fr = cv2.resize(fr, (SRC_W, SRC_H))
        masks = cv2.undistort(pred(unet, fr, device).astype(np.float32), K, dist)
        toks, aux = cc_tokens_from_frame_v3(masks, null_classifier, return_aux=True)
        # Indices of num tokens (in toks's order — must match source npz).
        if toks.shape[0] == 0:
            out_logits[k] = np.zeros((0, 9), dtype=np.float32)
            out_num_idx[k] = np.zeros((0,), dtype=np.int64)
            continue
        type_idx_new = toks[..., :4].argmax(-1)
        num_idx_new = np.where(type_idx_new == TYPE_NUM)[0]
        # Cross-check vs source npz num count.
        type_idx_src = np.asarray(source_type_idx[k], dtype=np.int64)
        num_idx_src = np.where(type_idx_src == TYPE_NUM)[0]
        if len(num_idx_new) != len(num_idx_src):
            # Tokens drifted vs source; emit empty + warning.
            if verbose:
                print(f"  [tok-mismatch] {clip_id} f{fi}: "
                      f"new={len(num_idx_new)} src={len(num_idx_src)}",
                      flush=True)
            out_logits[k] = np.zeros((0, 9), dtype=np.float32)
            out_num_idx[k] = np.zeros((0,), dtype=np.int64)
            continue
        crops = aux["num_crops"]
        if not crops:
            out_logits[k] = np.zeros((0, 9), dtype=np.float32)
            out_num_idx[k] = num_idx_src.astype(np.int64)
            continue
        logits = crop_fn(crops)   # (n_num, 9) float32
        out_logits[k] = logits.astype(np.float32)
        out_num_idx[k] = num_idx_src.astype(np.int64)
        n_with_nums += 1
    cap.release()

    np.savez_compressed(
        out_path,
        frame_idx=frame_idx_arr.astype(np.int32),
        num_token_idx=out_num_idx,
        crop_logits=out_logits,
    )
    if verbose:
        dt = time.time() - t0
        print(f"  {clip_id}  {n_frames} fr / {n_with_nums} with nums  "
              f"{dt:.0f}s", flush=True)
    return n_frames, n_with_nums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="data/pseudo_labels")
    ap.add_argument("--out-dir", default="data/pseudo_labels_crops")
    ap.add_argument("--manifest",
                   default="data/h_pool_and_intrinsics.json")
    ap.add_argument("--unified-weights",
                   default="models/unet_unified_v8_yardside_recover/best.pth")
    ap.add_argument("--crop-ckpt",
                   default="models/dsresnet10ww_round3_128x32/best.pth")
    ap.add_argument("--crop-arch", default="dsresnet10ww")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--max-clips", type=int, default=None,
                   help="cap for quick local tests")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    npz_dir = os.path.join(PROJECT_ROOT, args.npz_dir)
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Loading UNet + crop classifier on {device} ...", flush=True)
    unet = load_unified(args.unified_weights, device)
    crop_fn = make_painted_logits_fn(args.crop_ckpt, args.crop_arch, device)
    intr_by_clip = json.load(open(args.manifest))["intrinsics_by_clip"]

    files = sorted(f for f in os.listdir(npz_dir) if f.endswith(".npz"))
    # Worker sharding (modulo).
    if args.num_workers > 1:
        files = files[args.worker_id::args.num_workers]
    # Skip already-done.
    files = [f for f in files
             if not os.path.exists(os.path.join(out_dir, f))]
    if args.max_clips:
        files = files[:args.max_clips]
    print(f"[worker {args.worker_id}/{args.num_workers}] "
          f"{len(files)} clips to process", flush=True)

    t0 = time.time()
    for i, f in enumerate(files, 1):
        npz_path = os.path.join(npz_dir, f)
        out_path = os.path.join(out_dir, f)
        try:
            process_clip(npz_path, out_path, unet, crop_fn, intr_by_clip,
                         device, verbose=args.verbose)
        except Exception as e:
            print(f"  [err] {f}: {e}", flush=True)
            continue
        if i % 20 == 0 or i == len(files):
            dt = time.time() - t0
            rate = i / dt
            eta = (len(files) - i) / max(rate, 1e-6)
            print(f"  [{i}/{len(files)}] {dt:.0f}s "
                  f"({rate:.2f} clip/s, ETA {eta:.0f}s)", flush=True)

    print(f"Done. Total: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
