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

`Qwen35Trunk.fork` is the same function for rows that share a prefix: the prefix runs once and each branch
continues it from the DeltaNet state, the conv window and the attention keys at the prefix's end ("exact state
forking", docs/decisions/0010-state-forking.md). The prefix length is a multiple of CHUNK, so every chunk is the row's.
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


def gated_delta_rule(q, k, v, g, beta, masks, strict_upper, initial_state=None, final_state=False):
    """Same contract as transformers' torch_chunk_gated_delta_rule(use_qk_l2norm_in_kernel=True,
    initial_state, output_final_state). q, k [B,T,H,dk]; v [B,T,H,dv]; g, beta [B,T,H]; the state
    [B,H,dk,dv] is float32. With `final_state`, returns (out, state at the end).

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
    state0 = q.new_zeros((B, H, dk, dv)) if initial_state is None else initial_state
    last, out = scan(_chunk_step, state0, xs)  # [N, B, H, CHUNK, dv]
    out = out.movedim(0, 2).reshape(B, H, T, dv)
    out = out.transpose(1, 2).to(in_dtype)
    return (out, last) if final_state else out


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def half_trunk(trunk):
    """fp16 compute for an exported trunk: weights and activations in fp16, while the RMSNorms, the gated
    delta rule and the attention softmax stay fp32, as HF's modules and this file upcast them. The rotary
    frequencies and the doubling masks stay fp32 (fp16 positions are off by whole radians at 8,000 tokens)."""
    keep = dict(trunk.named_buffers(recurse=False))   # rotary tables and masks, in every trunk
    trunk.half()
    for k, v in keep.items():
        setattr(trunk, k, v)
    return trunk


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

    def _deltanet(self, mod, h, before=None, state=None, carry=False):
        """`before` [B, C, K-1]: the conv inputs of the K-1 tokens ahead of the row (zeros when None); `state`: the
        recurrent state at the row's start. With `carry`, also returns the row's own last K-1 conv inputs and its
        final state, for rows that continue it."""
        B, T, _ = h.shape
        mixed = mod.in_proj_qkv(h).transpose(1, 2)
        tail = mixed[:, :, 1 - mod.conv_kernel_size:] if carry else None
        if before is None:
            mixed = F.conv1d(mixed, mod.conv1d.weight, mod.conv1d.bias, padding=mod.conv_kernel_size - 1,
                             groups=mixed.shape[1])[:, :, :T]
        else:
            mixed = F.conv1d(torch.cat([before, mixed], dim=2), mod.conv1d.weight, mod.conv1d.bias,
                             groups=mixed.shape[1])
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
        core = gated_delta_rule(q, k, v, g, beta, self.masks, self.strict_upper, initial_state=state,
                                final_state=carry)
        if carry:
            core, last = core
        core = mod.norm(core.reshape(-1, mod.head_v_dim), z.reshape(-1, mod.head_v_dim))
        out = mod.out_proj(core.reshape(B, T, -1))
        return (out, tail, last) if carry else out

    def _attention(self, mod, h, cos, sin, bias, past=None, keep=False):
        """`past`: keys and values [B, kv heads, P, head_dim] ahead of the row, which `bias` [T, P + T] covers.
        With `keep`, also returns the row's own keys and values."""
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
        own = (k, v)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        rep = mod.num_key_value_groups
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        w = (q @ k.transpose(-1, -2)) * mod.scaling + bias
        w = torch.softmax(w.float(), dim=-1).to(q.dtype)
        o = (w @ v).transpose(1, 2).reshape(B, T, -1)
        o = o * torch.sigmoid(gate)
        out = mod.o_proj(o)
        return (out, own) if keep else out

    def _rope(self, T, dtype, device):
        pos = torch.arange(T, device=device, dtype=torch.float32)
        freqs = pos[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, input_ids, bias=None):
        """`bias`: the full-attention layers' additive mask, broadcast to [B, heads, T, T]; causal when None.
        The linear-attention layers are causal either way."""
        m = self.m
        x = m.embed_tokens(input_ids)
        T = input_ids.shape[1]
        cos, sin = self._rope(T, x.dtype, input_ids.device)
        if bias is None:
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

    def fork(self, prefix_ids, branch_ids, branch_job):
        """prefix_ids [J, P], branch_ids [B, S], branch_job [B] -> (prefix hidden [J, P, hidden], branch hidden
        [B, S, hidden]), after the final norm.

        Branch b is the row prefix_ids[branch_job[b]] ++ branch_ids[b] of `forward`, at positions P..P+S-1: it starts
        from its prefix's DeltaNet state and conv window and attends to its prefix's keys, so every prefix runs once.
        P and S must be multiples of CHUNK, so every chunk of the recurrence is the row's. Branches are right-padded."""
        m = self.m
        xp, xb = m.embed_tokens(prefix_ids), m.embed_tokens(branch_ids)
        P, S = prefix_ids.shape[1], branch_ids.shape[1]
        dev = prefix_ids.device
        cos, sin = self._rope(P + S, xp.dtype, dev)
        cos_p, sin_p, cos_b, sin_b = cos[:P], sin[:P], cos[P:], sin[P:]
        bias_p = torch.full((P, P), float("-inf"), device=dev, dtype=xp.dtype).triu(1)
        # query s sits at P + s; it sees every prefix key and the branch keys up to itself
        later = torch.arange(P + S, device=dev)[None, :] > (torch.arange(S, device=dev)[:, None] + P)
        bias_b = torch.zeros((S, P + S), device=dev, dtype=xp.dtype).masked_fill(later, float("-inf"))
        for layer, kind in zip(m.layers, self.layer_types):
            hp, hb = layer.input_layernorm(xp), layer.input_layernorm(xb)
            if kind == "linear_attention":
                mod = layer.linear_attn
                hp, tail, state = self._deltanet(mod, hp, carry=True)
                hb = self._deltanet(mod, hb, before=tail[branch_job], state=state[branch_job])
            else:
                mod = layer.self_attn
                hp, (k, v) = self._attention(mod, hp, cos_p, sin_p, bias_p, keep=True)
                hb = self._attention(mod, hb, cos_b, sin_b, bias_b, past=(k[branch_job], v[branch_job]))
            xp = xp + hp
            xp = xp + layer.mlp(layer.post_attention_layernorm(xp))
            xb = xb + hb
            xb = xb + layer.mlp(layer.post_attention_layernorm(xb))
        return m.norm(xp), m.norm(xb)


def fork_rows(rows, pad):
    """A request's rows (token lists) as `fork` inputs: (prefixes [J][P], branches [B][S], branch_job [B]), as the
    runner splits them (`crates/ollaya-runner/src/fork.rs`). Rows that share CHUNK tokens or more fork at their longest
    common prefix, rounded down to CHUNK and short of every row's end (J = 1). Otherwise each row is its own prefix,
    cut one chunk before the end of the longest row (a shorter row sits whole in its prefix, padded, and its branch is
    unread padding), which costs what the row form costs; rows of one chunk or less run in a one-chunk prefix."""
    shortest = min(len(r) for r in rows)
    common = shortest
    for i in range(shortest):
        if any(r[i] != rows[0][i] for r in rows):
            common = i
            break
    up = lambda n: -(-n // CHUNK) * CHUNK
    p = min(common, shortest - 1) // CHUNK * CHUNK
    if p:
        prefixes, branches, job = [rows[0][:p]], [r[p:] for r in rows], [0] * len(rows)
    else:
        p = max(up(max(len(r) for r in rows)) - CHUNK, CHUNK)
        prefixes = [(r[:p] + [pad] * p)[:p] for r in rows]
        branches, job = [r[p:] for r in rows], list(range(len(rows)))
    S = max(up(max(len(b) for b in branches)), CHUNK)
    return prefixes, [b + [pad] * (S - len(b)) for b in branches], job


def fork_trace(vocab):
    """Made-up fork inputs for an export (prefix_ids, branch_ids, branch_job): J 2, B 3, P 256, S 320, so no axis is 1
    (specialized) and no two are equal (tied). Read positions for them lie in [0, 576)."""
    g = torch.Generator().manual_seed(0)
    ids = lambda rows, chunks: torch.randint(100, vocab, (rows, chunks * CHUNK), generator=g)
    return ids(2, 4), ids(3, 5), torch.tensor([0, 1, 1])


def fork_shapes(max_rows=4096):
    """(dynamic shapes of the three fork inputs, the branch axis for the graph's read inputs)."""
    from torch.export import Dim

    B = Dim("branches", min=1, max=max_rows)
    J = Dim("jobs", min=1, max=max_rows)
    NP, NS = Dim("prefix_chunks", min=1, max=4096), Dim("branch_chunks", min=1, max=4096)
    return {"prefix_ids": {0: J, 1: CHUNK * NP}, "branch_ids": {0: B, 1: CHUNK * NS}, "branch_job": {0: B}}, B


def gather_hidden(hp, hb, branch_job, pos):
    """Hidden states of `fork`'s outputs at positions `pos` [B, ...] of each branch's whole row (its prefix, then
    the branch): [B, ..., hidden]. A position inside the prefix reads the prefix's row."""
    P = hp.shape[1]
    inside = pos < P
    shape = (-1,) + (1,) * (pos.dim() - 1)
    rows = torch.arange(pos.shape[0], device=pos.device).view(shape)
    a = hp[branch_job.view(shape), torch.where(inside, pos, 0)]
    b = hb[rows, torch.where(inside, 0, pos - P)]
    return torch.where(inside.unsqueeze(-1), a, b)
