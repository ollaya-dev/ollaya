"""An export-friendly forward pass for Qwen3 (dense, full attention) text backbones.

Same idea as qwen35.py: recompute the HF forward from the HF submodules (norms, projections, MLPs) with
an explicit causal attention and full RoPE, positions 0..T-1, no mask input (rows are right-padded and
every layer is causal). No sequence-length restriction.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class Qwen3Trunk(nn.Module):
    def __init__(self, text_model):
        super().__init__()
        self.m = text_model
        self.register_buffer("inv_freq", text_model.rotary_emb.inv_freq.detach().float().clone(), persistent=False)
        self.scaling = float(getattr(text_model.rotary_emb, "attention_scaling", 1.0))

    def _attention(self, mod, h, cos, sin, bias):
        B, T, _ = h.shape
        hd = mod.head_dim
        q = mod.q_norm(mod.q_proj(h).view(B, T, -1, hd)).transpose(1, 2)
        k = mod.k_norm(mod.k_proj(h).view(B, T, -1, hd)).transpose(1, 2)
        v = mod.v_proj(h).view(B, T, -1, hd).transpose(1, 2)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        rep = mod.num_key_value_groups
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        w = (q @ k.transpose(-1, -2)) * mod.scaling + bias
        w = torch.softmax(w.float(), dim=-1).to(q.dtype)
        o = (w @ v).transpose(1, 2).reshape(B, T, -1)
        return mod.o_proj(o)

    def forward(self, input_ids):
        m = self.m
        x = m.embed_tokens(input_ids)
        T = input_ids.shape[1]
        pos = torch.arange(T, device=input_ids.device, dtype=torch.float32)
        freqs = pos[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = (emb.cos() * self.scaling).to(x.dtype), (emb.sin() * self.scaling).to(x.dtype)
        bias = torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype).triu(1)
        for layer in m.layers:
            x = x + self._attention(layer.self_attn, layer.input_layernorm(x), cos, sin, bias)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return m.norm(x)
