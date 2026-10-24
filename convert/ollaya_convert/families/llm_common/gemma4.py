"""An export-friendly forward pass for Gemma 4 text decoders (`Gemma4TextModel`; the E2B and E4B "effective" models).

Same idea as gemma3.py: recompute the HF forward from the HF submodules (scaled embeddings, norms, projections,
MLPs) with explicit attention, positions 0..T-1 and no mask input (rows are right-padded and every layer is causal,
so padding never reaches an earlier position). Gemma 4 specifics, as `modeling_gemma4.py` (transformers 5.17):

  * per-layer embeddings: a second table gives every layer a `hidden_size_per_layer_input` vector per token,
    combined with a projection of the input embedding (`(norm(proj(x) / sqrt(hidden)) + table) / sqrt(2)`), which
    each layer adds through a gated projection after its MLP (`per_layer_input_gate`, `per_layer_projection`,
    `post_per_layer_input_norm`);
  * two layer types with their own head size and RoPE: sliding layers (`sliding_window`, local theta) and
    full-attention layers (`global_head_dim`, "proportional" partial RoPE, whose inverse frequencies are zero on the
    unrotated channels);
  * KV sharing: the last `num_kv_shared_layers` layers have no K/V projections and reuse the keys and values of the
    last non-shared layer of their type;
  * q_norm, k_norm and a scale-free v_norm per head, `scaling` 1.0, no attention soft-capping, a `layer_scalar` on
    every layer's output, and the double-wide MLP of the shared layers (the HF module's own shape).

Mixture-of-experts blocks (26B-A4B), `attention_k_eq_v` and bidirectional attention are not implemented.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class Gemma4Trunk(nn.Module):
    """input_ids [B, T] -> last hidden state [B, T, hidden] (after the final norm)."""

    def __init__(self, text_model):
        super().__init__()
        self.m = text_model
        cfg = text_model.config
        for flag in ("attn_logit_softcapping", "enable_moe_block", "attention_k_eq_v", "use_bidirectional_attention"):
            if getattr(cfg, flag, None):
                raise SystemExit("Gemma4Trunk does not implement %s" % flag)
        self.window = int(cfg.sliding_window)
        self.types = list(cfg.layer_types)
        self.ple = bool(cfg.hidden_size_per_layer_input)
        r = text_model.rotary_emb
        for t in sorted(set(self.types)):
            self.register_buffer("inv_freq_" + t, getattr(r, t + "_inv_freq").detach().float().clone(), persistent=False)
            setattr(self, "scale_" + t, float(getattr(r, t + "_attention_scaling")))

    def _rope(self, pos, inv_freq, scale, dtype):
        freqs = pos[:, None] * inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * scale).to(dtype)[None, :, None, :], (emb.sin() * scale).to(dtype)[None, :, None, :]

    def _attention(self, attn, h, cos, sin, bias, shared):
        B, T, _ = h.shape
        hd = attn.head_dim
        q = attn.q_norm(attn.q_proj(h).view(B, T, -1, hd))
        q = (q * cos + _rotate_half(q) * sin).transpose(1, 2)
        if attn.is_kv_shared_layer:
            k, v = shared[attn.layer_type]
        else:
            k = attn.k_norm(attn.k_proj(h).view(B, T, -1, hd))
            k = (k * cos + _rotate_half(k) * sin).transpose(1, 2)
            v = attn.v_norm(attn.v_proj(h).view(B, T, -1, hd)).transpose(1, 2)
        if attn.store_full_length_kv:
            shared[attn.layer_type] = (k, v)
        rep = attn.num_key_value_groups
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        w = (q @ k.transpose(-1, -2)) * attn.scaling + bias
        w = torch.softmax(w.float(), dim=-1).to(q.dtype)
        o = (w @ v).transpose(1, 2).reshape(B, T, -1)
        return attn.o_proj(o)

    def forward(self, input_ids):
        m = self.m
        x = m.embed_tokens(input_ids)   # Gemma4TextScaledWordEmbedding: times sqrt(hidden_size)
        B, T = input_ids.shape
        if self.ple:
            L, P = len(self.types), m.hidden_size_per_layer_input
            table = m.embed_tokens_per_layer(input_ids).reshape(B, T, L, P)
            proj = (m.per_layer_model_projection(x) * m.per_layer_model_projection_scale).reshape(B, T, L, P)
            per_layer = (m.per_layer_projection_norm(proj) + table) * m.per_layer_input_scale
        pos = torch.arange(T, device=input_ids.device, dtype=torch.float32)
        rope = {t: self._rope(pos, getattr(self, "inv_freq_" + t), getattr(self, "scale_" + t), x.dtype)
                for t in set(self.types)}
        i = torch.arange(T, device=input_ids.device)
        future = i[None, :] > i[:, None]
        too_far = i[None, :] <= i[:, None] - self.window
        zero = torch.zeros((T, T), device=x.device, dtype=x.dtype)
        bias = {"full_attention": zero.masked_fill(future, float("-inf")),
                "sliding_attention": zero.masked_fill(future | too_far, float("-inf"))}
        shared = {}
        for li, (layer, t) in enumerate(zip(m.layers, self.types)):
            cos, sin = rope[t]
            h = self._attention(layer.self_attn, layer.input_layernorm(x), cos, sin, bias[t], shared)
            x = x + layer.post_attention_layernorm(h)
            h = layer.mlp(layer.pre_feedforward_layernorm(x))
            x = x + layer.post_feedforward_layernorm(h)
            if self.ple:
                g = layer.act_fn(layer.per_layer_input_gate(x)) * per_layer[:, :, li, :]
                x = x + layer.post_per_layer_input_norm(layer.per_layer_projection(g))
            x = x * layer.layer_scalar
        return m.norm(x)
