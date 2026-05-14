"""Per-frame field-mapping pipeline.

One `FieldMappingPipeline` instance loads the 5 models once and exposes a
`__call__(frame_bgr, K, dist)` that runs the full per-frame chain:

  frame → UNet → 4-channel mask → tokenizer → tokens
                                            ↓
            encoder (phase 1) → encoded features
                                            ↓
                       crop classifier → number_refiner (phase 2)
                                            ↓
                          token_labeler (phase 3) → NGS-x label per token
                                            ↓
                  keypoints (tokens → image↔NGS correspondences)

Used by both the full pipeline (`src.pipeline`) and by aux scripts (viz,
NGS comparison, diagnostics) that need per-frame correspondences.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

from .tokenizer import (
    tokenize_frame, null_classifier,
    TYPE_NUM, TYPE_YARD, TYPE_SIDE, TYPE_HASH,
    SRC_W, SRC_H,
)
from .encoder import TokenEncoder, encoder_features
from .number_refiner import NumberRefiner
from .token_labeler import TokenLabeler, refine_number_tokens, PAINTED_TO_21
from .crop_classifier import make_painted_logits_fn
from .classes import N_NGS_X_CLASSES
from .keypoints import extract_keypoints


# UNet input dimensions + ImageNet preprocessing constants.
_UH, _UW = 512, 896
_IMM = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMS = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_unet(path: str, device) -> nn.Module:    # noqa: F821 (forward ref)
    m = smp.Unet("mit_b0", encoder_weights=None, in_channels=3, classes=4)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck.get("model_state_dict", ck))
    return m.to(device).eval()


def _preprocess_for_unet(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (_UW, _UH))
    x = (rgb.astype(np.float32) / 255.0 - _IMM) / _IMS
    return torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0)


@torch.no_grad()
def _predict_masks(unet, frame_bgr: np.ndarray, device) -> np.ndarray:
    """Returns (H, W, 4) probabilities in [0, 1] at the original frame size."""
    x = _preprocess_for_unet(frame_bgr).to(device)
    p = torch.sigmoid(unet(x))[0].cpu().numpy()
    h0, w0 = frame_bgr.shape[:2]
    out = np.zeros((h0, w0, 4), dtype=np.float32)
    for c in range(4):
        out[..., c] = cv2.resize(p[c], (w0, h0), interpolation=cv2.INTER_LINEAR)
    return out


# Default model paths (relative to PROJECT_ROOT).
_DEFAULT_UNET = "models/unet_unified_v8_yardside_recover/best.pth"
_DEFAULT_ENCODER = "models/token_only_v10_phase1_pseudo/best.pth"
_DEFAULT_NUMBER_REFINER = "models/rf_b_phase2_pseudo/best.pth"
_DEFAULT_TOKEN_LABELER = "models/v10c_phase3_pseudo/best.pth"
_DEFAULT_CROP_CLASSIFIER = "models/dsresnet10ww_round3_128x32/best.pth"


class FieldMappingPipeline:
    """Loads the 5 field-mapping models and runs the per-frame chain.

    Each call: frame → list of (image↔NGS) correspondences.
    Solving H is left to the caller (use src.field_mapping.homography).

    Args:
        device: torch device for model inference.
        project_root: base dir for the default model paths (defaults to
            two levels up from this file, i.e. <repo>/).
        unet_path, encoder_path, etc.: optional overrides for non-default
            model checkpoints.
    """
    def __init__(self, device,
                 project_root: str | None = None,
                 unet_path: str | None = None,
                 encoder_path: str | None = None,
                 number_refiner_path: str | None = None,
                 token_labeler_path: str | None = None,
                 crop_classifier_path: str | None = None):
        if project_root is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
        def _resolve(p, default):
            return p if (p and os.path.isabs(p)) else os.path.join(
                project_root, p or default)

        self.device = device

        # UNet (mask predictor).
        self.unet = _load_unet(_resolve(unet_path, _DEFAULT_UNET), device)

        # Encoder (phase 1).
        s1 = torch.load(_resolve(encoder_path, _DEFAULT_ENCODER),
                        map_location="cpu", weights_only=False)
        ea = s1["args"]
        self._encoder_args = ea
        self.encoder = TokenEncoder(
            n_layers=ea["n_layers"], n_heads=ea["n_heads"],
            d_model=ea["d_model"], ffn_dim=ea["ffn_dim"],
            dropout=0.0, token_dropout=0.0).to(device).eval()
        self.encoder.load_state_dict(s1["model_state_dict"])

        # Number refiner (phase 2).
        nr = torch.load(_resolve(number_refiner_path, _DEFAULT_NUMBER_REFINER),
                        map_location="cpu", weights_only=False)
        ra = nr["args"]
        self.number_refiner = NumberRefiner(
            d_enc=ea["d_model"], d_model=ra["d_model"],
            n_heads=ra["n_heads"], ffn_dim=ra["ffn_dim"],
            dropout=0.0, with_row=True).to(device).eval()
        self.number_refiner.load_state_dict(nr["model_state_dict"])

        # Token labeler (phase 3).
        tl = torch.load(_resolve(token_labeler_path, _DEFAULT_TOKEN_LABELER),
                        map_location="cpu", weights_only=False)
        ta = tl["args"]
        self.token_labeler = TokenLabeler(
            n_layers=ta["n_layers"], n_heads=ta["n_heads"],
            d_model=ta["d_model"], ffn_dim=ta["ffn_dim"],
            dropout=0.0, token_dropout=0.0).to(device).eval()
        self.token_labeler.load_state_dict(tl["model_state_dict"])

        # Number crop classifier.
        self.crop_classifier = make_painted_logits_fn(
            _resolve(crop_classifier_path, _DEFAULT_CROP_CLASSIFIER),
            "dsresnet10ww", device)

    def __call__(self, frame_bgr: np.ndarray, K: np.ndarray, dist: np.ndarray):
        """Run the full per-frame chain.

        Returns:
            dict with keys:
              corrs : list of {pixel_u, field, kind, source} dicts —
                      image↔NGS correspondences for the H solver
              fits  : dict — yardline/sideline/hash fit metadata
              tokens: (N, 16) token features (for debugging / viz)
              aux   : dict with num_crops, pixel_sets, etc. (from tokenizer)
            Returns None if the frame produced no tokens.
        """
        masks_d = _predict_masks(self.unet, frame_bgr, self.device)
        masks = cv2.undistort(masks_d.astype(np.float32), K, dist)
        tokens_np, aux = tokenize_frame(masks, null_classifier, return_aux=True)
        if tokens_np.shape[0] == 0:
            return None
        type_idx = tokens_np[..., :4].argmax(-1)
        is_num = (type_idx == TYPE_NUM)
        is_yard = (type_idx == TYPE_YARD)
        is_side = (type_idx == TYPE_SIDE)
        is_hash = (type_idx == TYPE_HASH)

        toks_t = torch.from_numpy(tokens_np).unsqueeze(0).to(self.device)
        pad = torch.zeros(1, tokens_np.shape[0], dtype=torch.bool,
                          device=self.device)
        with torch.no_grad():
            enc_feat = encoder_features(self.encoder, toks_t, pad)[0]

        nac = np.full(tokens_np.shape[0], -1, dtype=np.int64)
        nar = np.zeros(tokens_np.shape[0], dtype=np.float32)
        rfb_pre_full = torch.zeros(1, tokens_np.shape[0],
                                    self._encoder_args["d_model"],
                                    device=self.device)
        if is_num.any() and aux["num_crops"]:
            ni = np.where(is_num)[0]
            cl = torch.from_numpy(self.crop_classifier(aux["num_crops"])
                                   ).float().to(self.device)
            pad_n = torch.zeros(1, len(ni), dtype=torch.bool, device=self.device)
            with torch.no_grad():
                rl, rr, rpp = refine_number_tokens(
                    self.number_refiner, enc_feat[ni].unsqueeze(0),
                    cl.unsqueeze(0), pad_n)
            pp = rl[0].argmax(-1).cpu().numpy()
            p21 = PAINTED_TO_21.numpy()[pp]
            prw = torch.sigmoid(rr[0]).cpu().numpy()
            for j, ti in enumerate(ni):
                nac[ti] = int(p21[j])
                nar[ti] = float(prw[j])
                rfb_pre_full[0, ti] = rpp[0, j]

        nca_t = torch.from_numpy(nac).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.token_labeler(toks_t, pad,
                                      num_class_gt=nca_t,
                                      num_rfb_features=rfb_pre_full)
            p2 = out["logits_pass2"][0]
        ngs_cls = p2[:, :N_NGS_X_CLASSES].argmax(-1).cpu().numpy()
        rows = (p2[:, N_NGS_X_CLASSES] > 0).cpu().numpy().astype(int)

        corrs, fits = extract_keypoints(
            pixel_sets=aux["pixel_sets"],
            yard_classes=ngs_cls[is_yard].tolist(),
            side_rows=rows[is_side].tolist(),
            hash_classes=ngs_cls[is_hash].tolist(),
            hash_rows=rows[is_hash].tolist(),
            num_classes=nac[is_num].tolist(),
            num_rows=(nar[is_num] > 0.5).astype(int).tolist(),
            yard_mask=masks[..., 0],
        )
        return {
            "corrs": corrs,
            "fits": fits,
            "tokens": tokens_np,
            "aux": aux,
        }


# Fix the forward reference in _load_unet now that we've imported torch.nn.
import torch.nn as nn   # noqa: E402
_load_unet.__annotations__["return"] = nn.Module
