"""TokenClassifyV10b — v10 with locked, RF-B-enriched number features.

Architectural change vs v10:
  - Number tokens get an external feature input from RF-B's post-attention
    representation (d=96, matches v10's d_model). It's residual-added to
    the encoder output for number tokens, enriching them with crop +
    cross-number scene reasoning.
  - Number tokens are LOCKED through cross-attention: their own query
    features stay at "encoder + RF-B residual" throughout pass 1 and
    pass 2. Yard/hash/side queries still attend to number anchors AND
    refine via cross-attn as before.

Why locking matters: number features after encoder + RF-B already
encode (geometry, image content via crop classifier, cross-number
attention via RF-B). Allowing cross-attention to update them risks
contamination from confused yards/hashes early in training. Locking
keeps them clean while letting other tokens pull info from them.

The locking is implemented as a residual mask: cross-attention runs
normally but the output for locked tokens reverts to their input query.

Usage:
    out = model(tokens, padding_mask,
                num_class_gt=...,        # for cross-attn label_ngs_x source
                num_rfb_features=...)    # (B, N, d_model) — RF-B post-attn features
                                          # for number tokens; can be zeros for non-num.

If num_rfb_features is None, behaves exactly like v10 (no enrichment, no
lock — purely the inline classifier path).
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
    TYPE_YARD, TYPE_HASH, TYPE_NUM,
)
from train_h_set_regressor import make_2d_pos_enc    # noqa: E402
from train_token_v6 import N_NGS_X_CLASSES, make_class_to_ngs_x_norm    # noqa: E402
from model_token_v10 import TokenClassifyV10    # noqa: E402


class TokenClassifyV10b(TokenClassifyV10):
    """v10 + (1) RF-B residual enrichment for number tokens, (2) lock numbers
    in cross-attention so they don't get refined."""

    def _cross_attend_skip_queries(self, query, anchor_input,
                                       anchor_valid_mask,
                                       attn_module, norm_module, raw_pe,
                                       skip_mask=None):
        """Cross-attention where queries flagged True in `skip_mask` are
        excluded from the attention compute entirely (genuine compute
        skip via gather/scatter, not a post-hoc torch.where).

        Skipped queries pass through unchanged (output = input). Their
        anchor_input STILL contributes to the K/V set so other queries
        can attend to them — only the query side is skipped.

        Gradient-equivalent to the prior `_cross_attend_locked` with
        `lock_mask=skip_mask`. The behavioural intent ("number tokens
        are not queries in cross-attention") is now visible in code.
        """
        B, N, d = query.shape

        # K/V built from per-token anchors (unchanged).
        anchor_keys = self.anchor_proj(anchor_input)
        global_k = self.global_anchor.expand(B, -1, -1)
        full_kv = torch.cat([anchor_keys, global_k], dim=1)
        cross_pad = torch.cat([
            ~anchor_valid_mask,
            torch.zeros(B, 1, dtype=torch.bool, device=query.device),
        ], dim=1)

        if skip_mask is None or not skip_mask.any():
            # Nothing to skip: compute over all queries.
            query_pos = query + raw_pe
            out, _ = attn_module(query=query_pos, key=full_kv, value=full_kv,
                                    key_padding_mask=cross_pad)
            return norm_module(query + out)

        # Gather non-skipped queries via stable argsort: False (=0, keep)
        # is sorted before True (=1, skip) in each batch row.
        sort_idx = torch.argsort(skip_mask.long(), dim=1,
                                       stable=True)               # (B, N)
        n_keep = (~skip_mask).sum(dim=1)                            # (B,)
        max_keep = int(n_keep.max().item())
        if max_keep == 0:
            return query  # All queries are skipped — nothing to do.

        # Pack the first max_keep positions of the sorted index into a
        # (B, max_keep) gather index. Positions >= n_keep[b] are
        # placeholders (will be masked out by `keep_valid`).
        packed_idx = sort_idx[:, :max_keep]                         # (B, M)
        expand_idx = packed_idx.unsqueeze(-1).expand(-1, -1, d)     # (B,M,d)
        packed_q = torch.gather(query, 1, expand_idx)
        packed_pe = torch.gather(raw_pe, 1, expand_idx)
        packed_q_pos = packed_q + packed_pe

        # Run attention only on the packed (non-skipped) queries.
        out, _ = attn_module(query=packed_q_pos, key=full_kv,
                                  value=full_kv,
                                  key_padding_mask=cross_pad)
        packed_result = norm_module(packed_q + out)

        # Scatter back: start with `query` (skipped queries pass through),
        # then overwrite the non-skip positions with the attended values.
        result = query.clone()
        positions = torch.arange(max_keep, device=query.device) \
            .expand(B, -1)                                          # (B, M)
        keep_valid = positions < n_keep.unsqueeze(1)                # (B, M)
        # For each (b, m) with keep_valid: write packed_result[b, m] to
        # result[b, packed_idx[b, m]]. Use index_put_ with the linearised
        # (b, packed_idx[b, m]) pairs from keep_valid==True.
        b_idx, m_idx = keep_valid.nonzero(as_tuple=True)
        tgt_idx = packed_idx[b_idx, m_idx]
        result[b_idx, tgt_idx] = packed_result[b_idx, m_idx]
        return result

    # Backwards-compat alias: legacy callers still pass `lock_mask`.
    def _cross_attend_locked(self, query, anchor_input, anchor_valid_mask,
                                attn_module, norm_module, raw_pe,
                                lock_mask=None):
        return self._cross_attend_skip_queries(
            query, anchor_input, anchor_valid_mask,
            attn_module, norm_module, raw_pe, skip_mask=lock_mask)

    def forward(self, tokens: torch.Tensor,
                  padding_mask: torch.Tensor,
                  num_class_gt: torch.Tensor | None = None,
                  num_rfb_features: torch.Tensor | None = None):
        B, N, _ = tokens.shape
        eff_padding = self._apply_token_dropout(padding_mask)
        valid = ~eff_padding

        # Defensive zero-out (matches v10).
        type_idx_pre = tokens[..., :4].argmax(dim=-1)
        is_num_pre = (type_idx_pre == TYPE_NUM)
        tokens = tokens.clone()
        tokens[..., 13] = torch.where(
            is_num_pre, torch.zeros_like(tokens[..., 13]), tokens[..., 13])
        tokens[..., 14] = torch.where(
            is_num_pre, torch.zeros_like(tokens[..., 14]), tokens[..., 14])

        # Token features + PE + encoder.
        feat = self.token_embed(tokens)
        cx = tokens[..., 4]; cy = tokens[..., 5]
        pe = make_2d_pos_enc(cx, cy, dim=self.d_model)
        pe_masked = pe * valid.unsqueeze(-1).to(pe.dtype)
        feat = feat + pe_masked
        encoded = self.encoder(feat, src_key_padding_mask=eff_padding)

        # ── v10b: enrich number tokens with RF-B post-attention features ──
        # Residual add. num_rfb_features expected (B, N, d_model) with
        # zeros for non-number tokens.
        if num_rfb_features is not None:
            type_idx = tokens[..., :4].argmax(dim=-1)
            is_num_for_enrich = (type_idx == TYPE_NUM) & valid
            # Apply only on number tokens.
            encoded = encoded + num_rfb_features * \
                is_num_for_enrich.unsqueeze(-1).to(encoded.dtype)

        # Inline classifier head (still aux-trained, but main path uses
        # external num_class_gt — which in v10b carries RF-B's argmax).
        num_logits = self.num_classifier(encoded)

        # Choose label_ngs_x source (anchor input).
        type_idx = tokens[..., :4].argmax(dim=-1)
        is_num = (type_idx == TYPE_NUM) & valid
        is_yard = (type_idx == TYPE_YARD) & valid
        is_hash = (type_idx == TYPE_HASH) & valid

        class_ngs_x_norm = make_class_to_ngs_x_norm(tokens.device)

        # Soft fallback from the inline num_classifier head — used for
        # (a) inference when no anchor is passed at all, and (b) tokens
        # whose anchor is missing (num_class_gt == -1, e.g. conf-filtered).
        with torch.amp.autocast(
            enabled=False,
            device_type=tokens.device.type
            if tokens.device.type in ("cuda", "cpu") else "cpu"):
            num_class_logits_soft = num_logits[..., :N_NGS_X_CLASSES].float()
            num_probs_soft = F.softmax(num_class_logits_soft, dim=-1)
            soft_label_ngs_x = (num_probs_soft * class_ngs_x_norm).sum(dim=-1)

        if num_class_gt is not None:
            has_anchor = (num_class_gt >= 0)
            gt_clamped = num_class_gt.clamp(min=0)
            anchor_label_ngs_x = class_ngs_x_norm[gt_clamped]
            label_ngs_x = torch.where(has_anchor,
                                          anchor_label_ngs_x,
                                          soft_label_ngs_x)
        else:
            has_anchor = torch.zeros_like(is_num)
            label_ngs_x = soft_label_ngs_x

        label_ngs_x = torch.where(is_num, label_ngs_x,
                                       torch.zeros_like(label_ngs_x))

        # Cross-attention pass 1.
        # - anchor_valid_mask = is_num: all number tokens contribute as
        #   K/V (their label_ngs_x is either the GT anchor or the soft
        #   fallback from num_classifier). Yard / hash / side queries
        #   attend to them.
        # - skip_mask = numbers WITH a valid anchor: those queries are
        #   excluded from the cross-attention compute entirely — their
        #   output equals their input (the encoder feature + RF-B
        #   residual). Numbers WITHOUT an anchor (-1) get the standard
        #   refinement so the model can predict their NGS_x from context.
        skip_num = is_num & has_anchor
        anchor_input_1 = torch.stack([label_ngs_x, cx, cy], dim=-1)
        x_pass1 = self._cross_attend_skip_queries(
            encoded, anchor_input_1, anchor_valid_mask=is_num,
            attn_module=self.cross_attn_1, norm_module=self.cross_norm_1,
            raw_pe=pe_masked,
            skip_mask=skip_num)
        logits_pass1 = self.head(x_pass1)

        # Pass 2 setup.
        with torch.amp.autocast(
            enabled=False,
            device_type=tokens.device.type
            if tokens.device.type in ("cuda", "cpu") else "cpu"):
            ngs_x_logits_pass1 = logits_pass1[..., :N_NGS_X_CLASSES].float()
            probs_pass1 = F.softmax(ngs_x_logits_pass1, dim=-1)
            soft_ngs_x = (probs_pass1 * class_ngs_x_norm).sum(dim=-1)

        # For numbers with a valid anchor, keep using the GT label.
        # For numbers without an anchor, use pass-1's refined soft prediction.
        ngs_x_for_num = torch.where(has_anchor, label_ngs_x, soft_ngs_x)
        ngs_x_for_anchor = torch.where(is_num, ngs_x_for_num, soft_ngs_x)
        anchor_input_2 = torch.stack([ngs_x_for_anchor, cx, cy], dim=-1)
        is_anchor_2 = is_num | is_yard | is_hash

        # Pass 2 cross-attention. Skip number queries with a valid anchor
        # (same convention as pass 1).
        x_pass2 = self._cross_attend_skip_queries(
            x_pass1, anchor_input_2, anchor_valid_mask=is_anchor_2,
            attn_module=self.cross_attn_2, norm_module=self.cross_norm_2,
            raw_pe=pe_masked,
            skip_mask=skip_num)
        logits_pass2 = self.head(x_pass2)

        return {
            "logits_pass1": logits_pass1,
            "logits_pass2": logits_pass2,
            "num_logits": num_logits,
        }
