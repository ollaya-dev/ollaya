"""Ternary weights (one scale per output row) and low-bit embeddings, trained (`train.py --quant`).

    python ternary.py      # self-test

Training (`train.py --quant`): every weight matrix of the encoder layers and of the decision head goes through
a parametrization, so the forward pass sees the quantized weight and the gradient reaches the latent one
(straight-through estimator). Two ternary quantizers:

- `absmean` (BitNet b1.58): g = mean(|w_row|), w_q = g * RoundClip(w / g, -1, 1).
- `learned` (ParetoQ's ternary form): w_q = a/1.5 * round(1.5 * clamp(w / a, -0.99, 0.99)), with a learnable
  per-row scale a (LSQ-style gradient, scaled by 1/sqrt(row length)). It is initialised at a = 1.5 * mean(|w_row|),
  where it equals absmean; ParetoQ starts at max(|w_row|).

Both give exactly {-g, 0, +g} per row once finalized. Token embeddings can be fake-quantized to 4 or 8 bits per
row (symmetric, s = max(|row|) / (2^(b-1) - 1)).

webgraph.py turns a finished checkpoint's graph into the browser's, with the weights at 2 and 4 bits.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parametrize


def absmean_ternary(w: torch.Tensor) -> torch.Tensor:
    g = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
    return (w / g).round().clamp(-1, 1) * g


def _ste_round(x):
    return x + (x.round() - x).detach()


class Ternary(nn.Module):
    """Parametrization: the forward sees the ternary weight, the gradient flows to the latent one.
    `mix` < 1 blends in the latent weight (quantization warm-up, `set_mix`)."""

    def __init__(self, w: torch.Tensor, learned: bool = False):
        super().__init__()
        self.learned, self.mix = learned, 1.0
        if learned:
            self.scale = nn.Parameter(1.5 * w.detach().abs().mean(dim=1, keepdim=True).float().clamp_min(1e-6))
            k = 1.0 / math.sqrt(w.shape[1])
            self.scale.register_hook(lambda g: g * k)

    def quantize(self, w):
        if not self.learned:
            return absmean_ternary(w)
        a = self.scale.abs().clamp_min(1e-6)
        return (1.5 * (w / a).clamp(-0.99, 0.99)).round() * (a / 1.5)

    def forward(self, w):
        if not self.learned:
            return w + (self.mix * (absmean_ternary(w) - w)).detach()
        a = self.scale.abs().clamp_min(1e-6)
        q = _ste_round(1.5 * (w / a).clamp(-0.99, 0.99)) * (a / 1.5)
        return q if self.mix >= 1 else w + self.mix * (q - w)


class IntRows(nn.Module):
    """Parametrization: symmetric per-row integer weights of `bits` bits (for the token embeddings)."""

    def __init__(self, bits: int):
        super().__init__()
        self.top, self.mix = 2 ** (bits - 1) - 1, 1.0

    def quantize(self, w):
        s = (w.abs().amax(dim=1, keepdim=True) / self.top).clamp_min(1e-12)
        return (w / s).round().clamp(-self.top, self.top) * s

    def forward(self, w):
        return w + (self.mix * (self.quantize(w) - w)).detach()


def ternary_targets(model):
    """(module, parameter name) of every weight matrix trained ternary: encoder layers and the decision head."""
    out = []
    for part in (model.encoder.layers, model.head):
        for m in part.modules():
            if isinstance(m, nn.Linear):
                out.append((m, "weight"))
            elif isinstance(m, nn.MultiheadAttention) and m.in_proj_weight is not None:
                out.append((m, "in_proj_weight"))
    return out


def quantize_(model, quant: str, emb_bits: int = 32):
    """Register the training parametrizations: `quant` in {"absmean", "learned"}, embeddings at `emb_bits`."""
    for m, name in ternary_targets(model):
        parametrize.register_parametrization(m, name, Ternary(getattr(m, name), learned=quant == "learned"))
    if emb_bits < 32:
        parametrize.register_parametrization(model.encoder.embeddings.tok_embeddings, "weight", IntRows(emb_bits))


def set_mix(model, mix: float):
    """Quantization warm-up: the forward sees w + mix * (q(w) - w); 1 is fully quantized."""
    for m in model.modules():
        if isinstance(m, (Ternary, IntRows)):
            m.mix = mix


@torch.no_grad()
def finalize_(model):
    """Replace every parametrized weight by its exact quantized value and drop the parametrizations."""
    for m in list(model.modules()):
        if not parametrize.is_parametrized(m):
            continue
        for name in list(m.parametrizations.keys()):
            p = m.parametrizations[name]
            # + 0.0 turns -0.0 into +0.0: the same weight, and one byte pattern for each value
            q = p[0].quantize(p.original.float()).to(p.original.dtype) + 0.0
            parametrize.remove_parametrizations(m, name, leave_parametrized=False)
            getattr(m, name).copy_(q)


@torch.no_grad()
def zero_ratio(model) -> float:
    """Share of zero weights in the ternary matrices (a runaway value means a collapsing scale)."""
    zeros = total = 0
    for m, name in ternary_targets(model):
        w = getattr(m, name)
        zeros += int((w == 0).sum())
        total += w.numel()
    return zeros / max(1, total)


def _row_scale(w):
    """The per-row g of a ternary matrix, or None when the matrix is not ternary."""
    g = np.abs(w).max(axis=1)
    ok = np.all((w == 0) | (w == g[:, None]) | (w == -g[:, None]))
    return g if ok else None


def _int_rows(w, bits):
    """(scale, codes) when every row is exactly q * s, else None. Training's s went through max|row| = top * s
    in float32, so max|row| / top can be an ulp or two off it: try the neighbours too."""
    top = np.float32(2 ** (bits - 1) - 1)
    s0 = np.maximum((np.abs(w).max(axis=1) / top).astype(np.float32), np.float32(1e-12))
    s, q, ok = s0.copy(), np.zeros(w.shape, np.float32), np.zeros(len(w), bool)
    for k in (0, 1, -1, 2, -2, 3, -3):
        sk = (s0.view(np.int32) + k).view(np.float32)
        qk = np.rint(w / sk[:, None])
        good = ~ok & (np.abs(qk).max(axis=1) <= top) & np.all((qk * sk[:, None]).astype(np.float32) == w, axis=1)
        s[good], q[good] = sk[good], qk[good]
        ok |= good
    return (s, q.astype(np.int8)) if ok.all() else None


def _self_test():
    torch.manual_seed(0)
    w = torch.randn(7, 13)
    for learned in (False, True):
        t = Ternary(w, learned)
        q = t.quantize(w).detach().numpy()
        assert _row_scale(q) is not None
        assert torch.allclose(t(w), t.quantize(w), atol=1e-6)
        t.mix = 0.0
        assert torch.allclose(t(w), w, atol=1e-6)
    assert np.allclose(_row_scale(absmean_ternary(w).numpy()), np.abs(w.numpy()).mean(1), rtol=1e-6)
    for bits in (4, 8):
        e = IntRows(bits).quantize(torch.randn(5, 9)).numpy()
        assert _int_rows(e, bits) is not None


if __name__ == "__main__":
    _self_test()
    print("ok")
