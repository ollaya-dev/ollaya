"""Export-friendly forward passes for Valen-Decision-1.5 (a Qwen3.5 vision-language backbone, block-structured
full attention, the Mixer candidate head).

The vision tower is `decider_vision.graphs.VisionGraph` (the same Qwen3.5 tower). `DecoderGraph` runs one request:
token embeddings with the visual tokens written in, interleaved M-RoPE from explicit position ids, upstream's
`bbox_blocks` mask in the full-attention layers (the Gated DeltaNet layers stay causal, as upstream), then the
authors' Mixer head on four pooled roles per candidate.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..decider_vision.graphs import MropeTrunk, VisionGraph  # noqa: F401  (VisionGraph: re-exported for export.py)
from .layout import COMMON, PADDING, PREFIX


def block_bias(blocks, dtype):
    """Upstream `block_attention.segment_mask` as an additive bias [B, 1, T, T]: a prefix token sees the prefix; any
    other token sees the prefix, the shared glue and its own block. Padding is never seen.

    blocks [B, T]: -1 prefix, 0 glue, a positive block id, -2 padding. Upstream's prefix is the positions before
    the row's prefix length and its valid keys the positions before the row's length; here both are read off
    `blocks`, which says the same thing. A padding query sees the prefix, so its softmax stays finite (upstream
    gives it key 0 for the same reason): a NaN there would reach real tokens through the masked sums of the
    chunked DeltaNet layers and the span means."""
    q, k = blocks[:, :, None], blocks[:, None, :]
    key_prefix = k == PREFIX
    # Boolean algebra, not torch.where: ONNX Runtime has no Where kernel for bool branches.
    visible = (key_prefix | ((q != PREFIX) & ((k == COMMON) | (k == q)))) & (k != PADDING)
    visible = visible | ((q == PADDING) & key_prefix)
    bias = torch.zeros(visible.shape, dtype=dtype, device=blocks.device).masked_fill(~visible, float("-inf"))
    return bias[:, None]


class BlockTrunk(MropeTrunk):
    """MropeTrunk with the full-attention layers' mask as an argument."""

    def embeds(self, x, position_ids, bias):
        m = self.m
        f = position_ids[..., None].float() * self.inv_freq                      # [3, B, T, rd/2]
        f = torch.where(self.h_sel, f[1], torch.where(self.w_sel, f[2], f[0]))   # [B, T, rd/2]
        emb = torch.cat((f, f), dim=-1)[:, None]                                 # [B, 1, T, rd]
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
        for layer, kind in zip(m.layers, self.layer_types):
            h = layer.input_layernorm(x)
            if kind == "linear_attention":
                h = self._deltanet(layer.linear_attn, h)
            else:
                h = self._attention(layer.self_attn, h, cos, sin, bias)
            x = x + h
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return m.norm(x)


def span_mean(values, positions, spans):
    """values [n, d], spans [s, 2] -> the mean of values[start:end] per span, [s, d]."""
    inside = (positions[None, :] >= spans[:, :1]) & (positions[None, :] < spans[:, 1:])
    m = inside.to(values.dtype)
    return (m @ values) / m.sum(dim=-1, keepdim=True)


def role_features(hidden, blocks, instruction_spans, decision_pos, candidate_spans, candidate_question):
    """Upstream `heads.role_features` for every candidate of every question at once: [o, 4, H] holding the mean
    of the prefix, the mean of the question's instruction span, the mean of the candidate's span and the
    question's decision token. hidden [T, H]; blocks [T]."""
    positions = torch.arange(hidden.shape[0], device=hidden.device)
    prefix = (blocks == PREFIX).to(hidden.dtype)
    context = (prefix @ hidden) / prefix.sum()
    instruction = span_mean(hidden, positions, instruction_spans).index_select(0, candidate_question)
    candidate = span_mean(hidden, positions, candidate_spans)
    decision = hidden.index_select(0, decision_pos).index_select(0, candidate_question)
    return torch.stack((context[None].expand_as(candidate), instruction, candidate, decision), dim=1)


class DecoderGraph(nn.Module):
    """input_ids [1, T] (T a multiple of 64, right-padded), position_ids [3, 1, T], image_embeds [M, H] written at
    the flat positions image_pos [M], blocks [1, T], instruction_spans [Q, 2], decision_pos [Q],
    candidate_spans [O, 2], candidate_question [O] -> logits [O], every question's candidates in turn."""

    def __init__(self, text_model, head, mrope_section):
        super().__init__()
        self.trunk = BlockTrunk(text_model, mrope_section)
        self.head = head   # the authors' MixerHead, fp32

    def hidden(self, input_ids, position_ids, image_embeds, image_pos, blocks):
        R, T = input_ids.shape
        x = self.trunk.m.embed_tokens(input_ids)
        flat = x.reshape(R * T, x.shape[-1]).index_copy(0, image_pos, image_embeds.to(x.dtype))
        return self.trunk.embeds(flat.reshape(R, T, -1), position_ids, block_bias(blocks, x.dtype))

    def forward(self, input_ids, position_ids, image_embeds, image_pos, blocks, instruction_spans, decision_pos,
                candidate_spans, candidate_question):
        h = self.hidden(input_ids, position_ids, image_embeds, image_pos, blocks)[0].float()
        features = role_features(h, blocks[0], instruction_spans, decision_pos, candidate_spans, candidate_question)
        return self.head.score_features(features)
