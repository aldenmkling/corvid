"""TokenClassifyV10 — v9 with an inline number classifier.

v9 (TokenClassifyV6) takes number tokens with NGS_x already filled in
(from an external classifier — mbconv, SceneRefiner, v2 per-pixel mode-vote,
etc.) and uses that NGS_x as the cross-attention anchor signal.

v10 strips NGS_x from the input number tokens, lets the encoder learn
geometry-only scene structure, then adds an inline classifier head that
reads encoder output and predicts NGS_x for each number token. The
classifier's prediction (or ground-truth during teacher-forced training)
feeds the existing cross-attention pass 1 anchor.

Architecture:
  embed → PE → 4-layer encoder → encoder_out
                                    ├─→ classifier head (Linear→9-class)
                                    │      → aux CE loss (vs GT classes)
                                    │      → soft NGS_x via expected value
                                    │
                                    └─→ cross-attn pass 1 (uses GT or
                                          classifier-soft NGS_x as anchor)
                                          → pass 2 → readout

Stage 1 training: forward(..., num_class_gt=GT) — teacher forcing. The
cross-attention path always sees clean GT anchors. The classifier head
trains independently against the same GT (aux CE loss). At validation /
inference, num_class_gt=None and the classifier's soft NGS_x feeds the
anchor.

Stage 2 (later): freeze encoder + classifier, fine-tune cross-attention
on classifier-predicted anchors. Closes the train/inference distribution
gap.

Reuses TokenClassifyV6's encoder, anchor projection, cross-attention
modules, head, and forward logic — only the number-anchor source differs.
"""
from __future__ import annotations

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "pipeline"))

from cc_tokenizer import (    # noqa: E402
    TOKEN_FEATURE_DIM, TYPE_YARD, TYPE_HASH, TYPE_NUM,
)
from train_h_set_regressor import make_2d_pos_enc    # noqa: E402
from train_token_v6 import (    # noqa: E402
    TokenClassifyV6, N_NGS_X_CLASSES, make_class_to_ngs_x_norm,
)


class TokenClassifyV10(TokenClassifyV6):
    """TokenClassifyV6 + inline number classifier head + GT-injection path.

    Adds:
      - num_classifier: small head reading encoder output for number tokens.
        Output is N_NGS_X_CLASSES + 1 logits (matches existing self.head
        format, so we reuse make_class_to_ngs_x_norm for soft NGS_x).
      - forward signature accepts num_class_gt (B, N) int tensor with the
        21-class GT per number token (-1 elsewhere). When provided AND
        self.training, the cross-attn anchor uses the GT-derived NGS_x
        instead of the classifier's soft prediction (teacher forcing).

    Returns dict adds:
      - num_logits: (B, N, N_NGS_X_CLASSES + 1) — classifier output for
        every token (only the is_num positions are loss-meaningful).
    """
    def __init__(self, n_layers: int = 4, n_heads: int = 4,
                 d_model: int = 128, ffn_dim: int = 256,
                 dropout: float = 0.1,
                 token_dropout: float = 0.2,
                 min_keep_tokens: int = 4):
        super().__init__(n_layers=n_layers, n_heads=n_heads,
                          d_model=d_model, ffn_dim=ffn_dim,
                          dropout=dropout, token_dropout=token_dropout,
                          min_keep_tokens=min_keep_tokens)
        self.num_classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, N_NGS_X_CLASSES + 1),
        )
        nn.init.normal_(self.num_classifier[-1].weight, std=0.001)
        nn.init.zeros_(self.num_classifier[-1].bias)

    def forward(self, tokens: torch.Tensor,
                  padding_mask: torch.Tensor,
                  num_class_gt: torch.Tensor | None = None):
        """Returns dict with logits_pass1, logits_pass2, num_logits.

        Args:
            tokens: (B, N, 16) — number tokens should have feat[13]=0 and
                    feat[14]=0 (no NGS_x baked in; v10 derives it from the
                    encoder output).
            padding_mask: (B, N) bool — True = padded.
            num_class_gt: (B, N) int — 21-class GT for each number token,
                    -1 elsewhere. When provided AND self.training, the
                    cross-attn pass-1 anchor uses GT-derived NGS_x.
        """
        B, N, _ = tokens.shape
        eff_padding = self._apply_token_dropout(padding_mask)
        valid = ~eff_padding

        # v10: defensively zero feat[13] (NGS_x) and feat[14] (has_ngs)
        # for number tokens so the encoder doesn't see externally-derived
        # labels. The cross-attention path will receive label_ngs_x from
        # the classifier (or GT during teacher forcing) instead.
        type_idx_pre = tokens[..., :4].argmax(dim=-1)
        is_num_pre = (type_idx_pre == TYPE_NUM)
        tokens = tokens.clone()
        tokens[..., 13] = torch.where(
            is_num_pre, torch.zeros_like(tokens[..., 13]), tokens[..., 13])
        tokens[..., 14] = torch.where(
            is_num_pre, torch.zeros_like(tokens[..., 14]), tokens[..., 14])

        # ── Token features + PE + encoder ──
        feat = self.token_embed(tokens)
        cx = tokens[..., 4]; cy = tokens[..., 5]
        pe = make_2d_pos_enc(cx, cy, dim=self.d_model)
        pe_masked = pe * valid.unsqueeze(-1).to(pe.dtype)
        feat = feat + pe_masked
        encoded = self.encoder(feat, src_key_padding_mask=eff_padding)

        # ── NEW: classify number tokens from encoder output ──
        num_logits = self.num_classifier(encoded)    # (B, N, 22)

        # ── Choose label_ngs_x source for pass-1 anchor ──
        type_idx = tokens[..., :4].argmax(dim=-1)
        is_num = (type_idx == TYPE_NUM) & valid
        is_yard = (type_idx == TYPE_YARD) & valid
        is_hash = (type_idx == TYPE_HASH) & valid

        class_ngs_x_norm = make_class_to_ngs_x_norm(tokens.device)    # (21,)

        if num_class_gt is not None:
            # GT-injected anchor (Stage 1 — both train and eval). The
            # classifier head still trains via aux CE loss but its
            # output is not used in the cross-attention path.
            gt_clamped = num_class_gt.clamp(min=0)
            label_ngs_x = class_ngs_x_norm[gt_clamped]    # (B, N)
        else:
            # Use classifier's soft NGS_x (expected value over softmax).
            with torch.amp.autocast(
                enabled=False,
                device_type=tokens.device.type
                if tokens.device.type in ("cuda", "cpu") else "cpu"):
                num_class_logits = num_logits[..., :N_NGS_X_CLASSES].float()
                num_probs = F.softmax(num_class_logits, dim=-1)
                label_ngs_x = (num_probs * class_ngs_x_norm).sum(dim=-1)

        # Mask: only number tokens carry NGS_x.
        label_ngs_x = torch.where(is_num, label_ngs_x,
                                       torch.zeros_like(label_ngs_x))

        # ── Cross-attention pass 1 (anchors = numbers w/ derived NGS_x) ──
        anchor_input_1 = torch.stack([label_ngs_x, cx, cy], dim=-1)
        x_pass1 = self._cross_attend(
            encoded, anchor_input_1, anchor_valid_mask=is_num,
            attn_module=self.cross_attn_1, norm_module=self.cross_norm_1,
            raw_pe=pe_masked,
        )
        logits_pass1 = self.head(x_pass1)

        # ── Pass 2: same as V6 — yards/hashes use pass-1 soft preds ──
        with torch.amp.autocast(
            enabled=False,
            device_type=tokens.device.type
            if tokens.device.type in ("cuda", "cpu") else "cpu"):
            ngs_x_logits_pass1 = logits_pass1[..., :N_NGS_X_CLASSES].float()
            probs_pass1 = F.softmax(ngs_x_logits_pass1, dim=-1)
            soft_ngs_x = (probs_pass1 * class_ngs_x_norm).sum(dim=-1)

        ngs_x_for_anchor = torch.where(is_num, label_ngs_x, soft_ngs_x)
        anchor_input_2 = torch.stack([ngs_x_for_anchor, cx, cy], dim=-1)
        is_anchor_2 = is_num | is_yard | is_hash

        x_pass2 = self._cross_attend(
            x_pass1, anchor_input_2, anchor_valid_mask=is_anchor_2,
            attn_module=self.cross_attn_2, norm_module=self.cross_norm_2,
            raw_pe=pe_masked,
        )
        logits_pass2 = self.head(x_pass2)

        return {
            "logits_pass1": logits_pass1,
            "logits_pass2": logits_pass2,
            "num_logits": num_logits,
        }
