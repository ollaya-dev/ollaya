"""`weightless.py`, generalised: bf16 safetensors, PyTorch zip checkpoints, several source files.

    uv run python -m ollaya_convert.families.weightless_ext out/von --out out/von-wl \
        --checkpoint ~/.cache/.../option_marker.pt

Same contract as `ollaya_convert.weightless` (read its docstring): every graph initializer that comes
from a checkpoint tensor becomes an external reference to the author's original bytes plus the
Cast / Transpose nodes that rebuild it, after an exact value check. Additions:

  * BF16 safetensors. numpy has no bfloat16, so values are compared through the exact bf16 -> fp32
    widening (`uint32(bits) << 16`); the graph gets a Cast(BFLOAT16 -> FLOAT) node.
  * PyTorch zip checkpoints (`torch.save`, the format of `*.pt` / `pytorch_model.bin` since torch 1.6).
    The archive stores every tensor storage as an uncompressed zip member, so its raw little-endian
    bytes sit at a fixed file offset, exactly like safetensors. The pickle only carries metadata; it is
    read here with a restricted unpickler that resolves nothing but tensor-rebuild records, and nothing
    is ever unpickled at runtime. Only contiguous tensors are referenced.
  * `--checkpoint PATH[=LOCATION]` may repeat (e.g. shards); each external reference names its own file.

Does not modify `ollaya_convert/weightless.py`; reuses its safetensors index reader.
"""
import argparse
import collections
import os
import pickle
import shutil
import struct
import zipfile
from collections import Counter

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..weightless import read_safetensors_index

# dtype tag -> (numpy dtype used for comparison, ONNX dtype of the stored bytes, itemsize)
DTYPES = {
    "F32": (np.float32, TensorProto.FLOAT, 4),
    "F16": (np.float16, TensorProto.FLOAT16, 2),
    "BF16": (np.float32, TensorProto.BFLOAT16, 2),  # widened exactly for comparison
    "I64": (np.int64, TensorProto.INT64, 8),
}
TORCH_STORAGES = {"FloatStorage": "F32", "HalfStorage": "F16", "BFloat16Storage": "BF16", "LongStorage": "I64"}


# ------------------------------------------------------------------------------ torch zip indexing


class _Storage(collections.namedtuple("_Storage", "dtype key numel")):
    pass


class _Tensor(collections.namedtuple("_Tensor", "storage offset size stride")):
    pass


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
    return _Tensor(storage, storage_offset, tuple(size), tuple(stride))


class _MetadataUnpickler(pickle.Unpickler):
    """Resolves only what a state_dict pickle needs; any other global is refused."""

    def find_class(self, module, name):
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor_v2
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if module == "torch" and name in TORCH_STORAGES:
            return name
        raise pickle.UnpicklingError("refusing to resolve %s.%s" % (module, name))

    def persistent_load(self, pid):
        kind, storage_type, key, _location, numel = pid
        if kind != "storage" or storage_type not in TORCH_STORAGES:
            raise pickle.UnpicklingError("unsupported storage %r" % (pid,))
        return _Storage(TORCH_STORAGES[storage_type], key, numel)


def _contiguous(size, stride):
    expect, acc = [], 1
    for d in reversed(size):
        expect.append(acc)
        acc *= d
    return tuple(reversed(expect)) == tuple(stride)


def read_torch_zip_index(path):
    """{name: (dtype, shape, absolute_offset, length)} for every contiguous tensor of a torch.save zip."""
    zf = zipfile.ZipFile(path)
    names = zf.namelist()
    pkl = next(n for n in names if n.endswith("/data.pkl"))
    root = pkl[: -len("data.pkl")]
    obj = _MetadataUnpickler(zf.open(pkl)).load()
    starts = {}
    with open(path, "rb") as f:
        for info in zf.infolist():
            if not info.filename.startswith(root + "data/"):
                continue
            if info.compress_type != zipfile.ZIP_STORED:
                raise SystemExit("%s is compressed; cannot reference it" % info.filename)
            f.seek(info.header_offset)
            hdr = f.read(30)
            n_name, n_extra = struct.unpack("<HH", hdr[26:30])
            starts[info.filename[len(root) + 5:]] = info.header_offset + 30 + n_name + n_extra
    index = {}
    for name, t in obj.items():
        if not isinstance(t, _Tensor) or not _contiguous(t.size, t.stride):
            continue
        item = DTYPES[t.storage.dtype][2]
        numel = int(np.prod(t.size)) if t.size else 1
        index[name] = (t.storage.dtype, t.size, starts[t.storage.key] + t.offset * item, numel * item)
    return index


# --------------------------------------------------------------------------------------- loading


def load_tensor(path, entry):
    dtype, shape, off, length = entry
    with open(path, "rb") as f:
        f.seek(off)
        raw = f.read(length)
    if dtype == "BF16":
        return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
    return np.frombuffer(raw, dtype=DTYPES[dtype][0]).reshape(shape)


def read_index(path):
    return read_safetensors_index(path) if path.endswith(".safetensors") else read_torch_zip_index(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="directory with model.onnx (+ .onnx.data) and sidecar json files")
    ap.add_argument("--checkpoint", action="append", required=True, help="PATH[=LOCATION], repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="", help="prefix the exporter added to parameter names")
    a = ap.parse_args()

    sources = []  # (path, location, index)
    for spec in a.checkpoint:
        path, _, loc = spec.partition("=")
        sources.append((path, loc or os.path.basename(path), read_index(path)))

    model = onnx.load(os.path.join(a.src, "model.onnx"), load_external_data=True)
    graph = model.graph
    by_shape = {}
    for si, (_, _, index) in enumerate(sources):
        for name, entry in index.items():
            by_shape.setdefault(entry[1], []).append((si, name))

    kept, new_nodes, stats, used = [], [], Counter(), set()
    inline_bytes = 0
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        name = init.name[len(a.prefix):] if init.name.startswith(a.prefix) else init.name
        src = transpose = None
        for si, (_, _, index) in enumerate(sources):
            if name in index and index[name][1] == arr.shape and (si, name) not in used:
                src = (si, name)
                break
        if src is None and arr.ndim == 2:  # exact value check below makes any match safe
            for si, cand in by_shape.get(arr.shape[::-1], []):
                if (si, cand) in used:
                    continue
                path, _, index = sources[si]
                if np.array_equal(load_tensor(path, index[cand]).T.astype(arr.dtype), arr):
                    src, transpose = (si, cand), True
                    break
        if src is None:
            kept.append(init)
            stats["inline"] += 1
            inline_bytes += arr.nbytes
            continue
        path, loc, index = sources[src[0]]
        orig = load_tensor(path, index[src[1]])
        expect = orig.T if transpose else orig
        if not np.array_equal(expect.astype(arr.dtype), arr):
            raise SystemExit("value mismatch for %s <- %s:%s" % (init.name, loc, src[1]))
        used.add(src)

        dtype, shape, off, length = index[src[1]]
        ext = onnx.TensorProto(name="w:%s:%s" % (loc, src[1]), data_type=DTYPES[dtype][1], dims=list(shape))
        ext.data_location = TensorProto.EXTERNAL
        for k, v in (("location", loc), ("offset", str(off)), ("length", str(length))):
            ext.external_data.add(key=k, value=v)
        kept.append(ext)
        cur = ext.name
        if DTYPES[dtype][1] != helper.np_dtype_to_tensor_dtype(arr.dtype):
            new_nodes.append(helper.make_node("Cast", [cur], [cur + ":cast"], to=helper.np_dtype_to_tensor_dtype(arr.dtype)))
            cur += ":cast"
            stats["cast"] += 1
        if transpose:
            new_nodes.append(helper.make_node("Transpose", [cur], [init.name], perm=[1, 0]))
            stats["transpose"] += 1
        else:
            new_nodes.append(helper.make_node("Identity", [cur], [init.name]))
        stats["external"] += 1

    del graph.initializer[:]
    graph.initializer.extend(kept)
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(new_nodes + nodes)

    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "model.onnx")
    onnx.save(model, out)
    for f in ("tokenizer.json", "decision.json", "calibration.json"):
        if os.path.exists(os.path.join(a.src, f)):
            shutil.copy(os.path.join(a.src, f), os.path.join(a.out, f))
    for path, loc, _ in sources:  # ORT needs external data next to the graph; a hard link is free
        link = os.path.join(a.out, loc)
        if os.path.lexists(link):
            os.remove(link)
        try:
            os.link(os.path.realpath(path), link)
        except OSError:
            shutil.copy(os.path.realpath(path), link)
    big_inline = [(i.name, numpy_helper.to_array(i).nbytes) for i in kept
                  if i.data_location != TensorProto.EXTERNAL and numpy_helper.to_array(i).nbytes > 1 << 20]
    unused = {loc: sorted(n for n in index if (si, n) not in used) for si, (_, loc, index) in enumerate(sources)}
    print("graph %s: %.1f MB | %s | inline %.2f MB | inline tensors > 1 MB: %s"
          % (out, os.path.getsize(out) / 2**20, dict(stats), inline_bytes / 2**20, big_inline))
    for loc, names in unused.items():
        print("  %s: %d checkpoint tensors unused %s" % (loc, len(names), names[:12]))


if __name__ == "__main__":
    main()
