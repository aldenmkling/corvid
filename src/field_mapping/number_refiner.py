"""Number refiner — phase 2 of the field-mapping pipeline.

Takes the encoder features for number tokens and the crop classifier's
per-crop logits, and produces refined NGS-x labels for each number token
(plus a binary near/far row prediction). The refinement is a 1-layer
transformer that attends across the number tokens in a single frame —
so a frame with three painted numbers can use their relative positions
to disambiguate ambiguous individual classifications.

Public API:
- NumberRefiner — model class (used to be called RFB)
- N_PAINTED_CLASSES = 9
"""
from __future__ import annotations

import torch
import torch.nn as nn


N_PAINTED_CLASSES = 9


class NumberRefiner(nn.Module):
    """1-layer transformer over number tokens.

    Input: encoder features (d_enc) concatenated with crop classifier
    logits (9). Internal transformer attends across the number tokens.
    Output: refined 9-class logits per number token (and optionally a
    near/far row prediction).
    """
    def __init__(self, d_enc: int, n_classes: int = N_PAINTED_CLASSES,
                 d_model: int = 64, n_heads: int = 4,
                 ffn_dim: int = 128, dropout: float = 0.1,
                 with_row: bool = False):
        super().__init__()
        self.embed = nn.Linear(d_enc + n_classes, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ffn_dim, dropout=dropout,
            activation="relu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(d_model, n_classes)
        self.with_row = with_row
        if with_row:
            self.head_row = nn.Linear(d_model, 1)

    def forward(self, enc_feat: torch.Tensor,
                crop_logits: torch.Tensor,
                padding_mask: torch.Tensor):
        """
        enc_feat:     (B, N, d_enc) — phase 1 encoder output for number tokens
        crop_logits:  (B, N, n_classes) — per-crop classifier logits
        padding_mask: (B, N) bool, True = padded
        Returns: (B, N, n_classes) refined logits.
                  If with_row=True, returns (cls_logits, row_logits) where
                  row_logits is (B, N) — sigmoid for near/far prediction.
        """
        x = torch.cat([enc_feat, crop_logits], dim=-1)
        x = self.embed(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        cls = self.head(x)
        if self.with_row:
            return cls, self.head_row(x).squeeze(-1)
        return cls
