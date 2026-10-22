"""Ternary weights (BitNet b1.58 absmean, one scale per output row) and their 2-bit packing.

    python ternary.py pack <checkpoint> <out-dir>     # weights.bin + weights.json for the browser
    python ternary.py check <checkpoint> <out-dir>    # unpack and compare with model.safetensors, bit for bit

A ternary row holds only {-g, 0, +g}, g = mean(|w_row|). Packed: four 2-bit codes per byte (0 -> 0,
1 -> +g, 2 -> -g), row-major, plus g as float32 per row. Tensors that are not ternary (embeddings, norms,
the scorer) go as int8 per row with a float32 scale when `train.py --int8-embeddings` made them so,
otherwise as raw float32. Unpacking multiplies in float32, so the weights come back identical.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn


def absmean_ternary(w: torch.Tensor) -> torch.Tensor:
    g = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
    return (w / g).round().clamp(-1, 1) * g


class Ternary(nn.Module):
    """Parametrization: the forward sees the ternary weight, the gradient flows to the latent one (STE)."""

    def forward(self, w):
        return w + (absmean_ternary(w) - w).detach()


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


def int8_rows(w: torch.Tensor) -> torch.Tensor:
    s = (w.abs().amax(dim=1, keepdim=True) / 127).clamp_min(1e-12)
    return (w / s).round().clamp(-127, 127) * s


@torch.no_grad()
def ternarize_(model, int8_embeddings=False):
    """Replace the latent weights by their exact ternary values (and the embeddings by int8 ones)."""
    for m, name in ternary_targets(model):
        p = getattr(m, name)
        p.copy_(absmean_ternary(p.float()).to(p.dtype))
    if int8_embeddings:
        e = model.encoder.embeddings.tok_embeddings.weight
        e.copy_(int8_rows(e.float()).to(e.dtype))


def _row_scale(w):
    """The per-row g of a ternary matrix, or None when the matrix is not ternary."""
    g = np.abs(w).max(axis=1)
    ok = np.all((w == 0) | (w == g[:, None]) | (w == -g[:, None]))
    return g if ok else None


def _int8_scale(w):
    s = np.float32(np.abs(w).max(axis=1) / np.float32(127))
    s = np.maximum(s, np.float32(1e-12))
    q = np.rint(w / s[:, None])
    return (s, q.astype(np.int8)) if np.array_equal((q.astype(np.float32) * s[:, None]).astype(np.float32), w) else None


def pack(ckpt, out):
    from safetensors.numpy import load_file

    weights = load_file(os.path.join(ckpt, "model.safetensors"))
    os.makedirs(out, exist_ok=True)
    index, blob = {}, bytearray()
    for name, w in weights.items():
        w = w.astype(np.float32)
        entry = {"shape": list(w.shape), "offset": len(blob)}
        g = _row_scale(w) if w.ndim == 2 and w.shape[0] > 1 else None
        i8 = _int8_scale(w) if g is None and w.ndim == 2 and w.shape[0] > 1 else None
        if g is not None:
            codes = np.where(w > 0, 1, np.where(w < 0, 2, 0)).astype(np.uint8).reshape(-1)
            codes = np.concatenate([codes, np.zeros((-len(codes)) % 4, np.uint8)]).reshape(-1, 4)
            packed = codes[:, 0] | (codes[:, 1] << 2) | (codes[:, 2] << 4) | (codes[:, 3] << 6)
            entry["kind"] = "ternary"
            blob += g.astype("<f4").tobytes()
            blob += packed.astype(np.uint8).tobytes()
        elif i8 is not None:
            s, q = i8
            entry["kind"] = "int8"
            blob += s.astype("<f4").tobytes()
            blob += q.tobytes()
        else:
            entry["kind"] = "f32"
            blob += w.astype("<f4").tobytes()
        entry["bytes"] = len(blob) - entry["offset"]
        blob += b"\0" * ((-len(blob)) % 4)
        index[name] = entry
    with open(os.path.join(out, "weights.bin"), "wb") as f:
        f.write(blob)
    with open(os.path.join(out, "weights.json"), "w") as f:
        json.dump(index, f, indent=0)
    kinds = {}
    for e in index.values():
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + e["bytes"]
    print("packed %d tensors, %.1f MB: %s" % (len(index), len(blob) / 1e6, {k: round(v / 1e6, 1) for k, v in kinds.items()}))


def unpack(out):
    with open(os.path.join(out, "weights.json")) as f:
        index = json.load(f)
    blob = open(os.path.join(out, "weights.bin"), "rb").read()
    res = {}
    for name, e in index.items():
        shape, off = e["shape"], e["offset"]
        n = int(np.prod(shape))
        if e["kind"] == "f32":
            res[name] = np.frombuffer(blob, "<f4", n, off).reshape(shape)
            continue
        rows, cols = shape
        s = np.frombuffer(blob, "<f4", rows, off)
        if e["kind"] == "int8":
            q = np.frombuffer(blob, np.int8, n, off + 4 * rows).reshape(rows, cols).astype(np.float32)
            res[name] = (q * s[:, None]).astype(np.float32)
            continue
        packed = np.frombuffer(blob, np.uint8, (n + 3) // 4, off + 4 * rows)
        codes = np.stack([(packed >> sh) & 3 for sh in (0, 2, 4, 6)], 1).reshape(-1)[:n].reshape(rows, cols)
        res[name] = np.where(codes == 1, s[:, None], np.where(codes == 2, -s[:, None], np.float32(0))).astype(np.float32)
    return res


def check(ckpt, out):
    from safetensors.numpy import load_file

    ref = load_file(os.path.join(ckpt, "model.safetensors"))
    got = unpack(out)
    bad = [n for n in ref if not np.array_equal(ref[n].astype(np.float32), got[n])]
    assert not bad, "differ: %s" % bad[:5]
    print("all %d tensors identical" % len(ref))


def _self_test():
    torch.manual_seed(0)
    w = torch.randn(7, 13)
    q = absmean_ternary(w).numpy()
    g = _row_scale(q)
    assert g is not None and np.allclose(g, np.abs(w.numpy()).mean(1), rtol=1e-6)
    e = int8_rows(torch.randn(5, 9)).numpy()
    assert _int8_scale(e) is not None


if __name__ == "__main__":
    _self_test()
    if len(sys.argv) == 4 and sys.argv[1] in ("pack", "check"):
        {"pack": pack, "check": check}[sys.argv[1]](sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
