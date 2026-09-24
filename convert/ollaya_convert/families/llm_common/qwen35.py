"""An export-friendly forward pass for Qwen3.5 text backbones (hybrid Gated DeltaNet + gated attention).

`transformers`' own forward cannot go through `torch.onnx.export(dynamo=True)` with a dynamic sequence
axis: the chunked gated-delta rule loops over `num_chunks` in Python, a data-dependent trip count. This
module recomputes the same function from the same HF submodules and weights, with two changes that
keep the numbers and make the graph exportable:

  * the inter-chunk recurrence runs as `torch._higher_order_ops.scan` over chunks, which the exporter
    lowers to an ONNX `Scan` whose trip count is the (dynamic) number of 64-token chunks;
  * the unit lower-triangular solve of the UT transform (`solve_triangular` in HF eager) is computed by
    recursive block doubling: log2(64) = 6 levels of masked 64x64 matmuls, numerically the standard
    recursive triangular inverse (no Neumann series, so no cancellation).

Everything else (embedding, RMSNorms, MLPs, projections, the depthwise causal conv, the gated RMSNorm)
calls the HF modules themselves. Positions are 0..T-1 (text-only mRoPE uses the same position for all
three sections, so it reduces to plain partial RoPE on the first `rotary_dim` channels).

Rows are right-padded. Every layer is causal, so tokens after a readout position never influence it:
no attention mask is needed, and pad ids may be anything.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._higher_order_ops.scan import scan

CHUNK = 64


def _doubling_masks(c: int) -> torch.Tensor:
    """[log2 c, c, c] float masks. Level l selects, inside every diagonal block of size 2s (s = 2**l),
    its lower-left s x s block."""
    i = torch.arange(c)
    out, s = [], 1
    while s < c:
        same = (i[:, None] // (2 * s)) == (i[None, :] // (2 * s))
        lower_left = ((i[:, None] % (2 * s)) >= s) & ((i[None, :] % (2 * s)) < s)
        out.append((same & lower_left).float())
        s *= 2
    return torch.stack(out)


def unit_lower_inverse(strict_lower: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """(I + A)^-1 for strictly lower-triangular A [..., c, c].

    X_1 = I; X_2s = X_s - X_s (A * M_s) X_s, where X_s holds the inverses of the s x s diagonal blocks
    and M_s selects the lower-left s-blocks of the 2s-blocks. Exact in real arithmetic."""
    c = strict_lower.shape[-1]
    x = torch.eye(c, dtype=strict_lower.dtype, device=strict_lower.device).expand_as(strict_lower)
    for level in range(masks.shape[0]):
        x = x - x @ (strict_lower * masks[level]) @ x
    return x


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _chunk_step(state, xs):
    new_v, k_cumdecay, q, intra, k, chunk_decay = xs
    v_new = new_v - k_cumdecay @ state
    out = q @ state + intra @ v_new
    state = state * chunk_decay + k.transpose(-1, -2) @ v_new
    return state, out


def gated_delta_rule(q, k, v, g, beta, masks, strict_upper):
    """Same contract as transformers' torch_chunk_gated_delta_rule(use_qk_l2norm_in_kernel=True,
    initial_state=None, output_final_state=False). q, k [B,T,H,dk]; v [B,T,H,dv]; g, beta [B,T,H].

    T must be a multiple of CHUNK. HF pads to that multiple internally; padding in-graph would put
    `T % 64` arithmetic on the dynamic axis, which torch.export cannot keep symbolic, so the caller pads
    (right padding is exact: the recurrence is causal)."""
    in_dtype = q.dtype
    q, k, v, beta, g = [x.transpose(1, 2).float() for x in (q, k, v, beta, g)]
    q = _l2norm(q)
    k = _l2norm(k)
    q = q * (q.shape[-1] ** -0.5)
    B, H, T, dk = k.shape
    dv = v.shape[-1]
    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)
    q, k, k_beta, v_beta = (x.reshape(B, H, -1, CHUNK, x.shape[-1]) for x in (q, k, k_beta, v_beta))
    g = g.reshape(B, H, -1, CHUNK)
    cum = g.cumsum(dim=-1)
    pair = (cum.unsqueeze(-1) - cum.unsqueeze(-2)).masked_fill(strict_upper, float("-inf")).exp()
    ut = (k_beta @ k.transpose(-1, -2)) * pair
    intra = (q @ k.transpose(-1, -2)) * pair
    decayed_k_beta = k_beta * cum.exp().unsqueeze(-1)
    inv = unit_lower_inverse(ut.masked_fill(~strict_upper.transpose(0, 1), 0.0), masks)
    new_v = inv @ v_beta
    k_cumdecay = inv @ decayed_k_beta
    q = q * cum.exp().unsqueeze(-1)
    k = k * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    chunk_decay = cum[..., -1].exp()[..., None, None]
    xs = [x.movedim(2, 0) for x in (new_v, k_cumdecay, q, intra, k, chunk_decay)]
    state0 = q.new_zeros((B, H, dk, dv))
    _, out = scan(_chunk_step, state0, xs)  # [N, B, H, CHUNK, dv]
    out = out.movedim(0, 2).reshape(B, H, T, dv)
    return out.transpose(1, 2).to(in_dtype)


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class Qwen35Trunk(nn.Module):
    """input_ids [B, T] -> last hidden state [B, T, hidden] (after the final norm), positions 0..T-1.

    T must be a multiple of CHUNK (64): right-pad every row. Pads never reach earlier positions."""

    def __init__(self, text_model):
        super().__init__()
        self.m = text_model
        cfg = text_model.config
        self.layer_types = list(cfg.layer_types)
        self.register_buffer("inv_freq", text_model.rotary_emb.inv_freq.detach().float().clone(), persistent=False)
        self.register_buffer("masks", _doubling_masks(CHUNK), persistent=False)
        self.register_buffer("strict_upper", torch.ones(CHUNK, CHUNK, dtype=torch.bool).triu(1), persistent=False)

    def _deltanet(self, mod, h):
        B, T, _ = h.shape
        mixed = mod.in_proj_qkv(h).transpose(1, 2)
        mixed = F.conv1d(mixed, mod.conv1d.weight, mod.conv1d.bias, padding=mod.conv_kernel_size - 1,
                         groups=mixed.shape[1])[:, :, :T]
        mixed = F.silu(mixed).transpose(1, 2)
        z = mod.in_proj_z(h).reshape(B, T, -1, mod.head_v_dim)
        b = mod.in_proj_b(h)
        a = mod.in_proj_a(h)
        q, k, v = torch.split(mixed, [mod.key_dim, mod.key_dim, mod.value_dim], dim=-1)
        q = q.reshape(B, T, -1, mod.head_k_dim)
        k = k.reshape(B, T, -1, mod.head_k_dim)
        v = v.reshape(B, T, -1, mod.head_v_dim)
        beta = b.sigmoid()
        g = -mod.A_log.float().exp() * F.softplus(a.float() + mod.dt_bias)
        rep = mod.num_v_heads // mod.num_k_heads
        if rep > 1:
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)
        core = gated_delta_rule(q, k, v, g, beta, self.masks, self.strict_upper)
        core = mod.norm(core.reshape(-1, mod.head_v_dim), z.reshape(-1, mod.head_v_dim))
        return mod.out_proj(core.reshape(B, T, -1))

    def _attention(self, mod, h, cos, sin, bias):
        B, T, _ = h.shape
        hd = mod.head_dim
        q, gate = torch.chunk(mod.q_proj(h).view(B, T, -1, hd * 2), 2, dim=-1)
        gate = gate.reshape(B, T, -1)
        q = mod.q_norm(q).transpose(1, 2)
        k = mod.k_norm(mod.k_proj(h).view(B, T, -1, hd)).transpose(1, 2)
        v = mod.v_proj(h).view(B, T, -1, hd).transpose(1, 2)
        rd = cos.shape[-1]
        q = torch.cat([q[..., :rd] * cos + _rotate_half(q[..., :rd]) * sin, q[..., rd:]], dim=-1)
        k = torch.cat([k[..., :rd] * cos + _rotate_half(k[..., :rd]) * sin, k[..., rd:]], dim=-1)
        rep = mod.num_key_value_groups
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        w = (q @ k.transpose(-1, -2)) * mod.scaling + bias
        w = torch.softmax(w.float(), dim=-1).to(q.dtype)
        o = (w @ v).transpose(1, 2).reshape(B, T, -1)
        o = o * torch.sigmoid(gate)
        return mod.o_proj(o)

    def forward(self, input_ids):
        m = self.m
        x = m.embed_tokens(input_ids)
        T = input_ids.shape[1]
        pos = torch.arange(T, device=input_ids.device, dtype=torch.float32)
        freqs = pos[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
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
