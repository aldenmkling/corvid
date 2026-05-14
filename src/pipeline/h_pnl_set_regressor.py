"""Phase 6 PnL set-transformer model + DLT-from-predictions wiring.

Architecture:
  Tokenize CC masks (cc_tokenizer.cc_tokens_from_frame) — same as Phase 5b.
  Token embed + positional encoding + transformer encoder — same as Phase 5b.
  PER-TOKEN output head emits 3 scalars:
      [ngs_x_logit, row_logit, conf_logit]
  These are interpreted differently per token type:
      • YARDLINE (line): ngs_x_pred → vertical NGS-line (1, 0, -X);
                         image-line built from token's centroid + orientation.
      • SIDELINE (line): row_logit → horizontal NGS-line (0, 1, -Y_row);
                         image-line same as above.
      • HASH    (point): ngs_x_pred + row_logit → NGS point (X, Y_hash_row);
                         image point = token centroid.
      • NUMBER  (point): KNOWN ngs_x label + row_logit → NGS point;
                         image point = (cluster) centroid.

The per-token (image, NGS) correspondences feed a confidence-weighted DLT
(see h_pnl_dlt.solve_h_dlt_weighted), which is differentiable end-to-end.

Coordinate convention:
  Both image and NGS coordinates are NORMALIZED to [0, 1] before going into
  the DLT solver — image by (SRC_W, SRC_H), NGS by (NGS_X_MAX, NGS_Y_MAX).
  The solved H is therefore a "normalized H" matching the project's
  existing h_pixel_to_norm convention. This is also our pre-conditioning
  for fp32 stability (the raw-pixel system is much worse-conditioned;
  see h_pnl_dlt._self_test).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "pipeline"))
sys.path.insert(0, PROJECT_ROOT)

from cc_tokenizer import (    # noqa: E402
    TOKEN_FEATURE_DIM, TYPE_YARD, TYPE_SIDE, TYPE_HASH, TYPE_NUM,
)
from h_pnl_dlt import (    # noqa: E402
    point_rows, line_rows, line_from_centroid_orientation,
    solve_h_dlt_weighted,
)
from src.homography.field_model import (    # noqa: E402
    FIELD_WIDTH, FIELD_LENGTH, HASH_Y_NEAR, HASH_Y_FAR,
    NUMBER_Y_NEAR, NUMBER_Y_FAR,
)


# Normalized field constants (NGS_y / NGS_Y_MAX, in [0, 1]).
HASH_Y_NEAR_NORM = HASH_Y_NEAR / FIELD_WIDTH        # 0.4423
HASH_Y_FAR_NORM = HASH_Y_FAR / FIELD_WIDTH          # 0.5579
NUM_Y_NEAR_NORM = NUMBER_Y_NEAR / FIELD_WIDTH       # 0.2438
NUM_Y_FAR_NORM = NUMBER_Y_FAR / FIELD_WIDTH         # 0.7562
SIDE_Y_NEAR_NORM = 0.0
SIDE_Y_FAR_NORM = 1.0


# ── Sinusoidal positional encoding from a 2D centroid ──
def make_2d_pos_enc(cx: torch.Tensor, cy: torch.Tensor,
                     dim: int = 128) -> torch.Tensor:
    """(B, N) cx/cy in [0, 1] → (B, N, dim) sinusoidal PE.

    Identical to Phase 5b — same convention.
    """
    assert dim % 4 == 0
    quarter = dim // 4
    div = torch.exp(
        torch.arange(0, quarter, device=cx.device, dtype=cx.dtype)
        * -(math.log(10000.0) / quarter))
    cx_t = cx.unsqueeze(-1) * 2.0 * math.pi * 100.0 * div
    cy_t = cy.unsqueeze(-1) * 2.0 * math.pi * 100.0 * div
    pe_x = torch.cat([torch.sin(cx_t), torch.cos(cx_t)], dim=-1)
    pe_y = torch.cat([torch.sin(cy_t), torch.cos(cy_t)], dim=-1)
    return torch.cat([pe_x, pe_y], dim=-1)


# ────────────────────────────────────────────────────────────────────────────
# Build per-token correspondences and solve the weighted DLT system.
# ────────────────────────────────────────────────────────────────────────────

def build_corrs_and_solve_h(
    tokens: torch.Tensor,            # (B, N, 16) normalized features
    pred_logits: torch.Tensor,       # (B, N, 3) [ngs_x_logit, row_logit, conf_logit]
    padding_mask: torch.Tensor,      # (B, N) True = pad
    return_intermediate: bool = False,
):
    """Run the full per-token DLT pipeline.

    Returns:
        H_norm: (B, 3, 3) homography in normalized image → normalized NGS coords.
        Optionally: dict of intermediates (predictions, per-token NGS targets,
        per-token confidence, etc.) for use by per-type losses.
    """
    B, N, _ = tokens.shape
    device, dtype = tokens.device, tokens.dtype

    # ── Token features (already normalized) ──
    type_oh = tokens[..., :4]                          # (B, N, 4)
    type_idx = type_oh.argmax(dim=-1)                  # (B, N)
    cx = tokens[..., 4]                                # in [0, 1]
    cy = tokens[..., 5]
    cos_t = tokens[..., 11]                            # major-axis dx
    sin_t = tokens[..., 12]                            # major-axis dy
    label_ngs_x = tokens[..., 13]                      # NGS_x_norm (only valid for nums)

    valid = (~padding_mask).to(dtype)                  # (B, N) 1.0 if real

    is_yard = (type_idx == TYPE_YARD).to(dtype) * valid
    is_side = (type_idx == TYPE_SIDE).to(dtype) * valid
    is_hash = (type_idx == TYPE_HASH).to(dtype) * valid
    is_num = (type_idx == TYPE_NUM).to(dtype) * valid

    # ── Predictions (squashed to interpretable ranges) ──
    ngs_x_pred = torch.sigmoid(pred_logits[..., 0])    # in [0, 1] = NGS_x_norm
    row_prob = torch.sigmoid(pred_logits[..., 1])      # 0=near, 1=far
    conf = torch.sigmoid(pred_logits[..., 2])          # in [0, 1]

    # ── Image-side centroid + line params (normalized) ──
    centroid = torch.stack([cx, cy], dim=-1)           # (B, N, 2)
    direction = torch.stack([cos_t, sin_t], dim=-1)    # (B, N, 2)
    l_img = line_from_centroid_orientation(centroid, direction)  # (B, N, 3)

    # ── Per-type NGS-side targets ──
    ones = torch.ones_like(cx)
    zeros = torch.zeros_like(cx)

    # Yardline NGS line: vertical at x = ngs_x_pred  →  (1, 0, -X)
    yard_l_ngs = torch.stack([ones, zeros, -ngs_x_pred], dim=-1)
    # Sideline NGS line: horizontal at y = row_prob  →  (0, 1, -Y_row)
    # (since SIDE_Y_NEAR_NORM=0, SIDE_Y_FAR_NORM=1, row_prob ∈ [0,1] is the y)
    side_l_ngs = torch.stack([zeros, ones, -row_prob], dim=-1)

    # Choose the right NGS line per token type (line-type tokens only)
    l_ngs = (is_yard.unsqueeze(-1) * yard_l_ngs
             + is_side.unsqueeze(-1) * side_l_ngs)

    # Hash NGS point: (ngs_x_pred, hash_y_near + row_prob*(hash_y_far-hash_y_near))
    hash_dy = (HASH_Y_FAR_NORM - HASH_Y_NEAR_NORM)
    hash_y = HASH_Y_NEAR_NORM + row_prob * hash_dy
    hash_p_ngs = torch.stack([ngs_x_pred, hash_y], dim=-1)

    # Number NGS point: (label_ngs_x, num_y_near + row_prob*(num_y_far-num_y_near))
    num_dy = (NUM_Y_FAR_NORM - NUM_Y_NEAR_NORM)
    num_y = NUM_Y_NEAR_NORM + row_prob * num_dy
    num_p_ngs = torch.stack([label_ngs_x, num_y], dim=-1)

    # Effective NGS point per token (point-type tokens only)
    p_ngs = (is_hash.unsqueeze(-1) * hash_p_ngs
             + is_num.unsqueeze(-1) * num_p_ngs)
    p_img = centroid                                    # (B, N, 2)

    # ── Per-token row blocks ──
    line_block = line_rows(l_img, l_ngs)               # (B, N, 2, 9)
    point_block = point_rows(p_img, p_ngs)             # (B, N, 2, 9)

    is_line = is_yard + is_side                         # (B, N)
    is_point = is_hash + is_num
    rows = (is_line.unsqueeze(-1).unsqueeze(-1) * line_block
            + is_point.unsqueeze(-1).unsqueeze(-1) * point_block)
    rows = rows.reshape(B, N * 2, 9)

    # ── Per-row weights = per-token confidence × valid (× 1.0) ──
    # Padded tokens or zero-typed tokens get effective weight 0.
    type_valid = (is_line + is_point)                  # (B, N)  1 if real type
    w_token = conf * type_valid                         # (B, N)
    weights = w_token.unsqueeze(-1).expand(-1, -1, 2).reshape(B, N * 2)

    H_norm = solve_h_dlt_weighted(rows, weights)        # (B, 3, 3)

    if not return_intermediate:
        return H_norm

    info = {
        "ngs_x_pred": ngs_x_pred,
        "row_prob": row_prob,
        "conf": conf,
        "is_yard": is_yard, "is_side": is_side,
        "is_hash": is_hash, "is_num": is_num,
        "type_valid": type_valid,
        "weights": weights,
    }
    return H_norm, info


# ────────────────────────────────────────────────────────────────────────────
# Model.
# ────────────────────────────────────────────────────────────────────────────

class HSetRegressorPnL(nn.Module):
    """PnL-DLT set-transformer for homography regression.

    forward(tokens, padding_mask) → H_pred: (B, 3, 3),
                                      info: dict for the trainer's losses.

    Same encoder shape as Phase 5b's HSetRegressor (4×4×128). The
    difference is the PER-TOKEN output head (3 logits) instead of a
    CLS-token H head.
    """

    def __init__(self, n_layers: int = 4, n_heads: int = 4,
                 d_model: int = 128, ffn_dim: int = 256,
                 dropout: float = 0.1, init_conf_bias: float = -2.0):
        super().__init__()
        self.d_model = d_model

        self.token_embed = nn.Sequential(
            nn.Linear(TOKEN_FEATURE_DIM, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ffn_dim, dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        # Per-token head: 3 logits (ngs_x, row, conf).
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )
        # Init: small weights + sensible biases.
        nn.init.normal_(self.head[-1].weight, std=0.001)
        with torch.no_grad():
            self.head[-1].bias.zero_()
            # Bias the conf logit low so the H consistency loss starts soft.
            self.head[-1].bias[2] = init_conf_bias

    def forward(self, tokens: torch.Tensor,
                  padding_mask: torch.Tensor):
        feat = self.token_embed(tokens)
        cx = tokens[..., 4]
        cy = tokens[..., 5]
        pe = make_2d_pos_enc(cx, cy, dim=self.d_model)
        pe = pe * (~padding_mask).unsqueeze(-1).to(pe.dtype)
        feat = feat + pe

        out = self.encoder(feat, src_key_padding_mask=padding_mask)
        pred = self.head(out)                          # (B, N, 3)

        H_pred, info = build_corrs_and_solve_h(
            tokens, pred, padding_mask, return_intermediate=True)
        info["pred_logits"] = pred
        return H_pred, info


# ────────────────────────────────────────────────────────────────────────────
# Smoke test.
# ────────────────────────────────────────────────────────────────────────────

def _smoke_test():
    """End-to-end smoke test: random tokens → model → DLT → finite H."""
    print("[smoke] HSetRegressorPnL")
    torch.manual_seed(0)
    B, N = 2, 24
    tokens = torch.zeros(B, N, TOKEN_FEATURE_DIM)

    # Build a few of each type.
    rng = np.random.default_rng(0)
    for b in range(B):
        for i in range(N):
            t = rng.integers(0, 4)
            tokens[b, i, t] = 1.0
            tokens[b, i, 4] = float(rng.random())          # cx
            tokens[b, i, 5] = float(rng.random())          # cy
            tokens[b, i, 6:10] = torch.tensor(             # bbox
                [tokens[b, i, 4] - 0.05, tokens[b, i, 5] - 0.05,
                 tokens[b, i, 4] + 0.05, tokens[b, i, 5] + 0.05])
            tokens[b, i, 10] = 0.5                          # log_area
            theta = float(rng.random()) * math.pi
            tokens[b, i, 11] = math.cos(theta)
            tokens[b, i, 12] = math.sin(theta)
            if t == TYPE_NUM:
                tokens[b, i, 13] = float(rng.choice(
                    [20, 30, 40, 50, 40, 30, 20])) / 120.0
                tokens[b, i, 14] = 1.0
            tokens[b, i, 15] = 1.0                          # confidence

    pad_mask = torch.zeros(B, N, dtype=torch.bool)
    pad_mask[1, -4:] = True                                 # last 4 padded

    model = HSetRegressorPnL()
    H_pred, info = model(tokens, pad_mask)
    print(f"H_pred shape: {tuple(H_pred.shape)}")
    print(f"H_pred[0] (random init):\n{H_pred[0].detach().numpy()}")
    print(f"info keys: {list(info.keys())}")
    print(f"conf range: [{info['conf'].min().item():.4f}, "
          f"{info['conf'].max().item():.4f}]")
    print(f"ngs_x_pred range: [{info['ngs_x_pred'].min().item():.4f}, "
          f"{info['ngs_x_pred'].max().item():.4f}]")
    print(f"row_prob range: [{info['row_prob'].min().item():.4f}, "
          f"{info['row_prob'].max().item():.4f}]")
    assert torch.isfinite(H_pred).all(), "H_pred has NaN/inf"

    # Backprop check
    loss = H_pred.pow(2).sum()
    loss.backward()
    g = next(model.parameters()).grad
    assert g is not None and torch.isfinite(g).all()
    print("backprop: OK")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"n_params: {n_params:.2f}M")
    print("[smoke] PASS")


if __name__ == "__main__":
    _smoke_test()
