"""Token encoder — phase 1 of the field-mapping pipeline.

Takes per-token features (output of the tokenizer) and runs a transformer
encoder over them to produce contextual embeddings. Each token attends to
all other tokens in the same frame (yardlines, sidelines, hashes,
numbers), so the encoder builds a frame-level understanding of where each
detected region sits relative to the others.

The output feed phase 2 (`number_refiner`) for the number tokens, and
phase 3 (`token_labeler`) for the final NGS-x labels across all tokens.

Public API:
- TokenEncoder       — model class
- encoder_features() — run the encoder, return per-token features
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import TOKEN_FEATURE_DIM, TYPE_YARD, TYPE_HASH, TYPE_NUM
from .classes import N_NGS_X_CLASSES, make_class_to_ngs_x_norm


# ── 2D positional encoding ────────────────────────────────────────────────
def make_2d_pos_enc(cx: torch.Tensor, cy: torch.Tensor,
                     dim: int = 128) -> torch.Tensor:
    """Args:
        cx, cy: (B, N) tensors in [0, 1].
        dim:    output dimension; must be a multiple of 4.
    Returns:
        (B, N, dim) sinusoidal positional encoding.
    """
    assert dim % 4 == 0, "PE dim must be divisible by 4"
    quarter = dim // 4
    div = torch.exp(
        torch.arange(0, quarter, device=cx.device, dtype=cx.dtype)
        * -(math.log(10000.0) / quarter))
    # Scale cx, cy by 2π so they cycle within [0, 1] range
    cx_t = cx.unsqueeze(-1) * 2.0 * math.pi * 100.0 * div
    cy_t = cy.unsqueeze(-1) * 2.0 * math.pi * 100.0 * div
    pe_x = torch.cat([torch.sin(cx_t), torch.cos(cx_t)], dim=-1)
    pe_y = torch.cat([torch.sin(cy_t), torch.cos(cy_t)], dim=-1)
    return torch.cat([pe_x, pe_y], dim=-1)



# ── Internal: shared transformer base (the original TokenClassifyV6) ──────
class _TokenTransformerBase(nn.Module):
    def __init__(self, n_layers: int = 4, n_heads: int = 4,
                 d_model: int = 128, ffn_dim: int = 256,
                 dropout: float = 0.1,
                 token_dropout: float = 0.2,
                 min_keep_tokens: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.token_dropout = token_dropout
        self.min_keep_tokens = min_keep_tokens

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

        # Anchor projection: (ngs_x_norm, cx, cy) → d_model.
        # SHARED between pass 1 and pass 2 (same input format).
        self.anchor_proj = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, d_model),
        )

        # Learned global anchor — always-valid extra key for empty-anchor
        # frames. SHARED between passes.
        self.global_anchor = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.global_anchor, std=0.02)

        # Two cross-attention blocks (separate weights — they do different
        # jobs: pass-1 reads only numbers, pass-2 reads everything).
        self.cross_attn_1 = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_norm_1 = nn.LayerNorm(d_model)
        self.cross_attn_2 = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_norm_2 = nn.LayerNorm(d_model)

        # Head: shared between pass 1 and pass 2 outputs (same task)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, N_NGS_X_CLASSES + 1),
        )
        nn.init.normal_(self.head[-1].weight, std=0.001)
        nn.init.zeros_(self.head[-1].bias)

    def _apply_token_dropout(self, padding_mask: torch.Tensor) -> torch.Tensor:
        if not self.training or self.token_dropout <= 0:
            return padding_mask
        valid = ~padding_mask
        drop = torch.rand(valid.shape, device=padding_mask.device) < \
                self.token_dropout
        drop = drop & valid
        new_pad = padding_mask | drop
        n_kept = (~new_pad).sum(dim=-1)
        safe = n_kept >= self.min_keep_tokens
        return torch.where(safe.unsqueeze(-1), new_pad, padding_mask)

    def _cross_attend(self, query, anchor_input, anchor_valid_mask,
                        attn_module, norm_module, raw_pe):
        """One cross-attention pass.

        query:               (B, N, d) — token features (encoder out / pass-1 out)
        anchor_input:        (B, N, 3) — (ngs_x_norm, cx, cy) per token
        anchor_valid_mask:   (B, N) — True for tokens that should serve as keys
        attn_module:         the MHA module to use
        norm_module:         the LayerNorm to apply after residual
        raw_pe:              (B, N, d) — fresh positional encoding to add to query

        Returns refined token features (B, N, d).
        """
        B, N, d = query.shape
        # Build keys/values from anchor input
        anchor_keys = self.anchor_proj(anchor_input)            # (B, N, d)
        # Append always-valid global anchor
        global_k = self.global_anchor.expand(B, -1, -1)         # (B, 1, d)
        full_kv = torch.cat([anchor_keys, global_k], dim=1)      # (B, N+1, d)
        cross_pad = torch.cat([
            ~anchor_valid_mask,                                  # True = invalid
            torch.zeros(B, 1, dtype=torch.bool, device=query.device),
        ], dim=1)

        # Add raw positional encoding to query (so Q-K can match by image-x)
        query_pos = query + raw_pe

        out, _ = attn_module(
            query=query_pos, key=full_kv, value=full_kv,
            key_padding_mask=cross_pad)
        return norm_module(query + out)                          # residual

    def forward(self, tokens: torch.Tensor,
                  padding_mask: torch.Tensor):
        """Returns dict with two sets of logits (pass-1, pass-2)."""
        B, N, _ = tokens.shape
        eff_padding = self._apply_token_dropout(padding_mask)
        valid = ~eff_padding

        # ── Token features ──
        feat = self.token_embed(tokens)
        cx = tokens[..., 4]; cy = tokens[..., 5]
        pe = make_2d_pos_enc(cx, cy, dim=self.d_model)
        pe_masked = pe * valid.unsqueeze(-1).to(pe.dtype)        # zero-out padded
        feat = feat + pe_masked

        # ── Self-attention encoder ──
        encoded = self.encoder(feat, src_key_padding_mask=eff_padding)

        # ── Pass 1: anchors = numbers only (label_ngs_x as anchor signal) ──
        type_idx = tokens[..., :4].argmax(dim=-1)
        is_num = (type_idx == TYPE_NUM) & valid
        is_yard = (type_idx == TYPE_YARD) & valid
        is_hash = (type_idx == TYPE_HASH) & valid

        label_ngs_x = tokens[..., 13]                            # in [0, 1]
        anchor_input_1 = torch.stack([label_ngs_x, cx, cy], dim=-1)

        x_pass1 = self._cross_attend(
            encoded, anchor_input_1, anchor_valid_mask=is_num,
            attn_module=self.cross_attn_1, norm_module=self.cross_norm_1,
            raw_pe=pe_masked,
        )
        logits_pass1 = self.head(x_pass1)

        # ── Pass 2 setup: build new anchors from pass-1 predictions ──
        # For numbers: keep label_ngs_x (known, trustworthy).
        # For yards/hashes: use pass-1 SOFT predicted NGS_x.
        #
        # Soft NGS_x = expected value over softmax probabilities. Differentiable.
        with torch.amp.autocast(enabled=False, device_type=tokens.device.type
                                if tokens.device.type in ("cuda", "cpu") else "cpu"):
            ngs_x_logits_pass1 = logits_pass1[..., :N_NGS_X_CLASSES].float()
            probs_pass1 = F.softmax(ngs_x_logits_pass1, dim=-1)   # (B, N, 21)
            class_ngs_x_norm = make_class_to_ngs_x_norm(tokens.device)
            soft_ngs_x = (probs_pass1 * class_ngs_x_norm).sum(dim=-1)  # (B, N)

        # Choose ngs_x source per token
        ngs_x_for_anchor = torch.where(is_num, label_ngs_x, soft_ngs_x)
        anchor_input_2 = torch.stack([ngs_x_for_anchor, cx, cy], dim=-1)

        # In pass 2, valid anchors = number ∪ yard ∪ hash (NOT sidelines —
        # they don't have meaningful NGS_x).
        is_anchor_2 = is_num | is_yard | is_hash

        # ── Pass 2 cross-attention ──
        x_pass2 = self._cross_attend(
            x_pass1, anchor_input_2, anchor_valid_mask=is_anchor_2,
            attn_module=self.cross_attn_2, norm_module=self.cross_norm_2,
            raw_pe=pe_masked,
        )
        logits_pass2 = self.head(x_pass2)

        return {"logits_pass1": logits_pass1, "logits_pass2": logits_pass2}



# ── Phase 1 encoder (the original TokenClassifyV10) ───────────────────────
class TokenEncoder(_TokenTransformerBase):
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



# ── Helper: run just the encoder portion ──────────────────────────────────
def encoder_features(model: 'TokenEncoder', tokens: torch.Tensor,
                        padding_mask: torch.Tensor) -> torch.Tensor:
    """Run the encoder portion of TokenClassifyV10 to get h_4 (per-token feat)."""
    # Replicate model.forward up through the encoder.
    # Defensive zero-out of NGS_x for number tokens (matches v10's forward).
    type_idx = tokens[..., :4].argmax(dim=-1)
    is_num = (type_idx == TYPE_NUM)
    tokens = tokens.clone()
    tokens[..., 13] = torch.where(
        is_num, torch.zeros_like(tokens[..., 13]), tokens[..., 13])
    tokens[..., 14] = torch.where(
        is_num, torch.zeros_like(tokens[..., 14]), tokens[..., 14])

    feat = model.token_embed(tokens)
    cx = tokens[..., 4]; cy = tokens[..., 5]
    pe = make_2d_pos_enc(cx, cy, dim=model.d_model)
    feat = feat + pe * (~padding_mask).unsqueeze(-1).to(pe.dtype)
    return model.encoder(feat, src_key_padding_mask=padding_mask)
