"""Ternary weights (one scale per output row) and low-bit embeddings, trained and packed.

    python ternary.py pack <checkpoint> <out-dir>     # model.packed, for the browser
    python ternary.py check <checkpoint> <out-dir>    # unpack and compare with model.safetensors, byte for byte

Training (`train.py --quant`): every weight matrix of the encoder layers and of the decision head goes through
a parametrization, so the forward pass sees the quantized weight and the gradient reaches the latent one
(straight-through estimator). Two ternary quantizers:

- `absmean` (BitNet b1.58): g = mean(|w_row|), w_q = g * RoundClip(w / g, -1, 1).
- `learned` (ParetoQ's ternary form): w_q = a/1.5 * round(1.5 * clamp(w / a, -0.99, 0.99)), with a learnable
  per-row scale a (LSQ-style gradient, scaled by 1/sqrt(row length)). It is initialised at a = 1.5 * mean(|w_row|),
  where it equals absmean; ParetoQ starts at max(|w_row|).

Both give exactly {-g, 0, +g} per row once finalized. Token embeddings can be fake-quantized to 4 or 8 bits per
row (symmetric, s = max(|row|) / (2^(b-1) - 1)).

Packed (`model.packed`): a little-endian u64 length, that many bytes of JSON, then the tensor data. The JSON holds
the checkpoint's safetensors header verbatim and, per tensor, its kind and place in the data: ternary rows as four
2-bit codes per byte (0 -> 0, 1 -> +g, 2 -> -g) plus g as float32 per row; int4 rows as two nibbles per byte and
int8 rows as bytes, each with a float32 scale per row; everything else as raw float32. Unpacking multiplies in
float32 and writes each tensor where the header puts it, so it rebuilds model.safetensors byte for byte: the
graph addresses that file by its sha256 (web/ollaya.js `unpackWeights` does the same in the browser).
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import struct
import sys

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
            # + 0.0 turns -0.0 into +0.0: both pack to one code, and the packed file must round-trip byte for byte
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


def _header(path):
    """The raw safetensors header (JSON text, with its alignment padding)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8")


def pack(ckpt, out):
    from safetensors.numpy import load_file, save_file

    path = os.path.join(ckpt, "model.safetensors")
    weights = load_file(path)
    if any(np.signbit(w[w == 0]).any() for w in weights.values()):
        # Checkpoints finalized before finalize_ dropped -0.0: rewrite with +0.0, the same model.
        meta = json.loads(_header(path)).get("__metadata__")
        save_file({k: w + np.float32(0) for k, w in weights.items()}, path, metadata=meta)
        weights = load_file(path)
        print("rewrote %s without -0.0" % path)
    header = _header(path)
    assert all(e["dtype"] == "F32" for k, e in json.loads(header).items() if k != "__metadata__"), "not float32"
    os.makedirs(out, exist_ok=True)
    index, blob = {}, bytearray()
    for name, w in weights.items():
        entry = {"shape": list(w.shape), "offset": len(blob)}
        matrix = w.ndim == 2 and w.shape[0] > 1
        g = _row_scale(w) if matrix else None
        i4 = _int_rows(w, 4) if matrix and g is None else None
        i8 = _int_rows(w, 8) if matrix and g is None and i4 is None else None
        if g is not None:
            codes = np.where(w > 0, 1, np.where(w < 0, 2, 0)).astype(np.uint8).reshape(-1)
            codes = np.concatenate([codes, np.zeros((-len(codes)) % 4, np.uint8)]).reshape(-1, 4)
            entry["kind"] = "ternary"
            blob += g.astype("<f4").tobytes()
            blob += (codes[:, 0] | (codes[:, 1] << 2) | (codes[:, 2] << 4) | (codes[:, 3] << 6)).astype(np.uint8).tobytes()
        elif i4 is not None:
            s, q = i4
            nib = (q.reshape(-1).astype(np.int16) & 0xF).astype(np.uint8)
            nib = np.concatenate([nib, np.zeros(len(nib) % 2, np.uint8)]).reshape(-1, 2)
            entry["kind"] = "int4"
            blob += s.astype("<f4").tobytes()
            blob += (nib[:, 0] | (nib[:, 1] << 4)).astype(np.uint8).tobytes()
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
    meta = json.dumps({"safetensors_header": header, "tensors": index}).encode("utf-8")
    meta += b" " * ((-len(meta)) % 8)
    with open(os.path.join(out, "model.packed"), "wb") as f:
        f.write(struct.pack("<Q", len(meta)) + meta + bytes(blob))
    kinds = {}
    for e in index.values():
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + e["bytes"]
    print("packed %d tensors, %.1f MB: %s" % (len(index), (8 + len(meta) + len(blob)) / 1e6,
                                             {k: round(v / 1e6, 1) for k, v in kinds.items()}))


def unpack(out):
    """(tensors, safetensors header) from `out`/model.packed."""
    with open(os.path.join(out, "model.packed"), "rb") as f:
        data = f.read()
    n = struct.unpack_from("<Q", data)[0]
    meta = json.loads(data[8:8 + n])
    blob = memoryview(data)[8 + n:]
    res = {}
    for name, e in meta["tensors"].items():
        shape, off = e["shape"], e["offset"]
        count = int(np.prod(shape))
        if e["kind"] == "f32":
            res[name] = np.frombuffer(blob, "<f4", count, off).reshape(shape)
            continue
        rows, cols = shape
        s = np.frombuffer(blob, "<f4", rows, off)
        if e["kind"] == "int8":
            q = np.frombuffer(blob, np.int8, count, off + 4 * rows).reshape(rows, cols).astype(np.float32)
        elif e["kind"] == "int4":
            b = np.frombuffer(blob, np.uint8, (count + 1) // 2, off + 4 * rows)
            nib = np.stack([b & 0xF, b >> 4], 1).reshape(-1)[:count].astype(np.int8)
            q = np.where(nib > 7, nib - 16, nib).reshape(rows, cols).astype(np.float32)
        else:
            packed = np.frombuffer(blob, np.uint8, (count + 3) // 4, off + 4 * rows)
            codes = np.stack([(packed >> sh) & 3 for sh in (0, 2, 4, 6)], 1).reshape(-1)[:count].reshape(rows, cols)
            q = np.where(codes == 1, np.float32(1), np.where(codes == 2, np.float32(-1), np.float32(0)))
        res[name] = (q * s[:, None]).astype(np.float32)
    return res, meta["safetensors_header"]


def unpack_safetensors(out) -> bytes:
    """The model.safetensors that `out`/model.packed was packed from, rebuilt byte for byte."""
    tensors, header = unpack(out)
    hdr = header.encode("utf-8")
    entries = {k: e for k, e in json.loads(header).items() if k != "__metadata__"}
    base = 8 + len(hdr)
    buf = bytearray(base + max(e["data_offsets"][1] for e in entries.values()))
    buf[:base] = struct.pack("<Q", len(hdr)) + hdr
    for name, w in tensors.items():
        b, e = entries[name]["data_offsets"]
        buf[base + b:base + e] = w.astype("<f4").tobytes()
    return bytes(buf)


def check(ckpt, out):
    with open(os.path.join(ckpt, "model.safetensors"), "rb") as f:
        want = hashlib.sha256(f.read()).hexdigest()
    got = hashlib.sha256(unpack_safetensors(out)).hexdigest()
    assert got == want, "unpacked sha256 %s, model.safetensors %s" % (got, want)
    print("unpacked model.safetensors identical: sha256-%s" % got)


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
    import tempfile
    from safetensors.numpy import save_file

    t = absmean_ternary(torch.randn(6, 11)).numpy()
    t[0, :2] = (0.0, -0.0)
    t = t * np.float32(1)
    with tempfile.TemporaryDirectory() as d:
        save_file({"t": t, "i4": IntRows(4).quantize(torch.randn(5, 9)).numpy(),
                   "i8": IntRows(8).quantize(torch.randn(3, 300)).numpy(), "f": np.random.randn(7).astype(np.float32)},
                  os.path.join(d, "model.safetensors"))
        with contextlib.redirect_stdout(io.StringIO()):
            pack(d, d)
            check(d, d)


if __name__ == "__main__":
    _self_test()
    if len(sys.argv) == 4 and sys.argv[1] in ("pack", "check"):
        {"pack": pack, "check": check}[sys.argv[1]](sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
