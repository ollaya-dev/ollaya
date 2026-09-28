"""Export-friendly forward passes for decider-2b-vision (Qwen3.5-2B vision-language).

VisionGraph: one image's patches -> its merged visual tokens, with the grid-dependent tables (the
bilinear taps into the learned position table, the 2-D rotary positions) as inputs, so the graph has no
grid arithmetic. DecoderGraph: token rows with visual tokens written into the embeddings at given
positions, interleaved multimodal RoPE from explicit position ids, letter logits at each row's slot.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..llm_common.qwen35 import Qwen35Trunk


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class VisionGraph(nn.Module):
    """patches [N, C*T*P*P] (merge-block order), pos_idx/pos_w [N, 4] (bilinear taps into the 48x48
    position table), rot_ids [N, 2] (h, w patch coordinates) -> visual tokens [N / 4, out_hidden]."""

    def __init__(self, visual):
        super().__init__()
        self.v = visual
        self.register_buffer("inv_freq", visual.rotary_pos_emb.inv_freq.detach().float().clone(), persistent=False)

    def forward(self, patches, pos_idx, pos_w, rot_ids):
        v = self.v
        h = v.patch_embed(patches)
        h = h + (v.pos_embed(pos_idx) * pos_w[:, :, None]).sum(1).to(h.dtype)
        freq = rot_ids[..., None].float() * self.inv_freq            # [N, 2, D/4]
        freq = torch.cat([freq[:, 0], freq[:, 1]], dim=-1)            # [N, D/2]
        freq = torch.cat([freq, freq], dim=-1)                         # [N, D]
        cos, sin = freq.cos()[:, None, :], freq.sin()[:, None, :]     # [N, 1, D]
        for blk in v.blocks:
            a = blk.attn
            x = blk.norm1(h)
            n = x.shape[0]
            q, k, val = a.qkv(x).reshape(n, 3, a.num_heads, -1).permute(1, 0, 2, 3).unbind(0)   # [N, H, D]
            q = q.float() * cos + _rotate_half(q.float()) * sin
            k = k.float() * cos + _rotate_half(k.float()) * sin
            q, k, val = q.transpose(0, 1), k.transpose(0, 1), val.transpose(0, 1)              # [H, N, D]
            w = torch.softmax((q @ k.transpose(-1, -2)) * a.scaling, dim=-1)
            o = (w @ val).transpose(0, 1).reshape(n, -1)
            h = h + a.proj(o)
            h = h + blk.mlp(blk.norm2(h))
        return v.merger(h)


class MropeTrunk(Qwen35Trunk):
    """Qwen35Trunk from input embeddings, with interleaved multimodal RoPE from position_ids [3, B, T]
    (text-only rows, whose three position rows are equal, reduce to the plain 1-D RoPE)."""

    def __init__(self, text_model, mrope_section):
        super().__init__(text_model)
        rd = self.inv_freq.shape[0]
        idx = torch.arange(rd)
        self.register_buffer("h_sel", ((idx % 3 == 1) & (idx < mrope_section[1] * 3))[None, None, :], persistent=False)
        self.register_buffer("w_sel", ((idx % 3 == 2) & (idx < mrope_section[2] * 3))[None, None, :], persistent=False)

    def embeds(self, x, position_ids):
        m = self.m
        T = x.shape[1]
        f = position_ids[..., None].float() * self.inv_freq                      # [3, B, T, rd/2]
        f = torch.where(self.h_sel, f[1], torch.where(self.w_sel, f[2], f[0]))   # [B, T, rd/2]
        emb = torch.cat((f, f), dim=-1)[:, None]                                 # [B, 1, T, rd]
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
        bias = torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype).triu(1)
        for layer, kind in zip(m.layers, self.layer_types):
            h = layer.input_layernorm(x)
            if kind == "linear_attention":
                h = self._deltanet(layer.linear_attn, h)
            else:
                h = self._attention(layer.self_attn, h, cos, sin, bias)
            x = x + h
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return m.norm(x)


class DecoderGraph(nn.Module):
    """input_ids [R, T] (T a multiple of 64, right-padded), position_ids [3, R, T], image_embeds [M, H]
    written at flat positions image_pos [M] (row * T + column), slot_pos [R] -> letter logits [R, L]."""

    def __init__(self, text_model, lm_head, letter_ids, mrope_section):
        super().__init__()
        self.trunk = MropeTrunk(text_model, mrope_section)
        self.lm_head = lm_head
        self.register_buffer("letter_ids", torch.tensor(letter_ids, dtype=torch.long), persistent=False)

    def forward(self, input_ids, position_ids, image_embeds, image_pos, slot_pos):
        R, T = input_ids.shape
        x = self.trunk.m.embed_tokens(input_ids)
        flat = x.reshape(R * T, x.shape[-1]).index_copy(0, image_pos, image_embeds.to(x.dtype))
        h = self.trunk.embeds(flat.reshape(R, T, -1), position_ids)
        hs = h[torch.arange(R, device=h.device), slot_pos]
        return hs @ self.lm_head.weight[self.letter_ids].transpose(0, 1)
