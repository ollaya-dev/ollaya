"""Weightless graphs over several checkpoint files: shards, base + LoRA adapter, BF16 sources, torch zips.

    uv run python -m ollaya_convert.weightless_sharded SRC --out OUT \
        --source model.safetensors=/path/model.safetensors \
        --map trunk.m.=model.language_model.

The single-file tool (`weightless.py`) covers one float32/float16 `model.safetensors`. Decoder
checkpoints add four cases it does not handle, all solved the same way (an initializer becomes an
external reference to the original bytes, plus the Cast/Transpose that reproduces the exported value):

  * several files per graph: every source keeps its own `location`, so a sharded checkpoint, or a base
    model plus a LoRA adapter from another repo, stays byte-for-byte the author's files;
  * BF16 sources (numpy has no bfloat16): values are compared through an exact bf16 -> f32 upcast and
    the graph gets `Cast(to=FLOAT)`;
  * torch zip checkpoints (`torch.save`, e.g. kev's `head.pt`): their tensors are stored uncompressed
    at fixed offsets inside the zip, so they are byte-addressable too;
  * name maps: exporter names (`trunk.m.layers.0...`) are mapped to checkpoint names by prefix rules,
    with a shape-and-value search as fallback (pre-transposed Linear weights keep no usable name).

Every mapped initializer is checked for exact equality with its source after the transform. Anything
that has no source (masks, rotary tables, small derived constants) stays inline and is reported.
ONNX Runtime folds the Cast/Transpose chains at session creation, unless the model keeps its weights in
their stored precision (docs/decisions/0001-decoder-weights-in-memory.md).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import struct
import zipfile
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SAFETENSORS_DTYPES = {
    "F32": (np.float32, TensorProto.FLOAT),
    "F16": (np.float16, TensorProto.FLOAT16),
    "BF16": (np.uint16, TensorProto.BFLOAT16),
    "I64": (np.int64, TensorProto.INT64),
    "I32": (np.int32, TensorProto.INT32),
}
TORCH_DTYPES = {"float32": (np.float32, TensorProto.FLOAT), "float16": (np.float16, TensorProto.FLOAT16),
                "bfloat16": (np.uint16, TensorProto.BFLOAT16), "int64": (np.int64, TensorProto.INT64)}


@dataclass
class Entry:
    key: str
    dtype: str            # safetensors dtype name (F32, BF16, ...)
    shape: tuple
    offset: int           # absolute byte offset in the file
    length: int


@dataclass
class Source:
    """One checkpoint file. `location` is the file name the graph uses (next to model.onnx)."""
    location: str
    path: str
    entries: dict = field(default_factory=dict)
    repo: str | None = None
    revision: str | None = None
    filename: str | None = None

    def raw(self, e: Entry) -> np.ndarray:
        with open(self.path, "rb") as f:
            f.seek(e.offset)
            buf = f.read(e.length)
        return np.frombuffer(buf, dtype=SAFETENSORS_DTYPES[e.dtype][0]).reshape(e.shape)

    def value(self, e: Entry) -> np.ndarray:
        """The tensor as the exporter saw it after an exact upcast (bf16 -> f32)."""
        a = self.raw(e)
        if e.dtype == "BF16":
            return (a.astype(np.uint32) << 16).view(np.float32)
        return a


def safetensors_source(location, path, **meta) -> Source:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    base = 8 + n
    s = Source(location, path, **meta)
    for k, v in header.items():
        if v["dtype"] not in SAFETENSORS_DTYPES:
            continue
        s.entries[k] = Entry(k, v["dtype"], tuple(v["shape"]), base + v["data_offsets"][0],
                             v["data_offsets"][1] - v["data_offsets"][0])
    return s


def torchzip_source(location, path, **meta) -> Source:
    """Tensors of a `torch.save` zip, keyed by their dotted path in the saved (nested) dict. A tensor is
    byte-addressable when it is contiguous and covers its whole storage record, which holds for a
    saved state_dict; anything else is skipped (and stays inline if the graph needs it)."""
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    flat = {}

    def walk(prefix, o):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(prefix + str(k) + ".", v)
        elif isinstance(o, torch.Tensor):
            flat[prefix[:-1]] = o

    walk("", obj)
    s = Source(location, path, **meta)
    z = zipfile.ZipFile(path)
    records = []
    with open(path, "rb") as f:
        for info in z.infolist():
            if info.compress_type != zipfile.ZIP_STORED or "/data/" not in info.filename:
                continue
            f.seek(info.header_offset)
            local = f.read(30)
            n_name, n_extra = struct.unpack("<HH", local[26:30])
            off = info.header_offset + 30 + n_name + n_extra
            f.seek(off)
            records.append((off, info.file_size, f.read(info.file_size)))
    for key, t in flat.items():
        dt = str(t.dtype).removeprefix("torch.")
        if dt not in TORCH_DTYPES or not t.is_contiguous() or t.storage_offset() != 0:
            continue
        data = t.contiguous().view(torch.uint8).numpy().tobytes() if dt != "bfloat16" else \
            t.view(torch.int16).numpy().tobytes()
        if len(data) != t.untyped_storage().nbytes():
            continue
        for off, size, buf in records:
            if size == len(data) and buf == data:
                name = {"float32": "F32", "float16": "F16", "bfloat16": "BF16", "int64": "I64"}[dt]
                s.entries[key] = Entry(key, name, tuple(t.shape), off, size)
                break
    return s


def _candidates(name, maps):
    """Checkpoint names to try for a graph initializer: `maps` is a list of (graph_prefix, checkpoint_prefix)
    pairs, or a callable name -> [candidate names]."""
    if callable(maps):
        return list(maps(name)) + [name]
    out = []
    for g, c in maps:
        if name.startswith(g):
            out.append(c + name[len(g):])
    out.append(name)
    return out


def _initializer_array(init: TensorProto, base_dir: str) -> np.ndarray:
    """An initializer's value, read from its external data file on demand (one tensor in memory at a time)."""
    if init.data_location != TensorProto.EXTERNAL:
        return numpy_helper.to_array(init)
    info = {kv.key: kv.value for kv in init.external_data}
    dtype = np.dtype(helper.tensor_dtype_to_np_dtype(init.data_type))
    count = int(np.prod(init.dims, dtype=np.int64))
    with open(os.path.join(base_dir, info["location"]), "rb") as f:
        f.seek(int(info.get("offset", 0)))
        buf = f.read(int(info.get("length", count * dtype.itemsize)))
    return np.frombuffer(buf, dtype=dtype, count=count).reshape(tuple(init.dims))


def sink_casts_below_gathers(graph) -> int:
    """Move each widening Cast of a checkpoint tensor below the Gathers that read rows of it (the token embedding, and a
    tied LM head restricted to some rows): `Gather(Cast(W), ids)` becomes `Cast(Gather(W, ids))`, and likewise for
    `GatherND` (batch_dims 0) and through Identity or order-keeping Transpose nodes. The values are identical (Cast is
    elementwise), but only the gathered rows are widened. ONNX Runtime does not fold a Cast whose output exceeds its
    folding limit (1 GiB by default; a Qwen3.5 embedding widened to fp32 is 1 to 4 GB), and it folds none when the
    weights stay in their stored precision (decision.json `weights_in_memory: bf16`), so without this every forward
    pass would widen the whole embedding. Returns the number of Gathers rewritten."""
    external = {t.name: t.data_type for t in graph.initializer if t.data_location == TensorProto.EXTERNAL}
    all_nodes = [copy.deepcopy(n) for n in graph.node]   # plain copies: stable identities for the rewrite below
    producer = {o: n for n in all_nodes for o in n.output}

    def passthrough(n):
        """Identity, or a Transpose that keeps the axis order."""
        if n.op_type == "Identity":
            return True
        perm = [list(a.ints) for a in n.attribute if a.name == "perm"]
        return n.op_type == "Transpose" and bool(perm) and perm[0] == list(range(len(perm[0])))

    def source(name):
        """The checkpoint tensor `name` is a widened copy of (through Cast and pass-through nodes only), else None."""
        n = producer.get(name)
        while n is not None and passthrough(n):
            name, n = n.input[0], producer.get(n.input[0])
        if n is None or n.op_type != "Cast" or external.get(n.input[0]) not in (TensorProto.BFLOAT16, TensorProto.FLOAT16):
            return None
        return n.input[0], next(a.i for a in n.attribute if a.name == "to")

    def gathers_rows(n):
        attrs = {a.name: a.i for a in n.attribute}
        return (n.op_type == "Gather" and attrs.get("axis", 0) == 0) or \
            (n.op_type == "GatherND" and attrs.get("batch_dims", 0) == 0)

    rewritten, nodes = 0, []
    for n in all_nodes:
        src = source(n.input[0]) if gathers_rows(n) else None
        if src is None:
            nodes.append(n)
            continue
        w, to = src
        narrow = n.output[0] + ":narrow"
        gather = helper.make_node(n.op_type, [w, n.input[1]], [narrow], name=(n.name or narrow) + ":narrow")
        gather.attribute.extend(n.attribute)
        nodes += [gather, helper.make_node("Cast", [narrow], [n.output[0]], to=to, name=(n.name or narrow) + ":widen")]
        rewritten += 1
    if not rewritten:
        return 0
    # drop the Cast and pass-through chains nothing reads any more
    outputs = {o.name for o in graph.output}
    while True:
        used = {i for n in nodes for i in n.input} | outputs
        keep = [not ((n.op_type == "Cast" or passthrough(n)) and not any(o in used for o in n.output)) for n in nodes]
        if all(keep):
            break
        nodes = [n for n, k in zip(nodes, keep) if k]
    del graph.node[:]
    graph.node.extend(nodes)
    return rewritten


def make_weightless(src_dir: str, out_dir: str, sources: list[Source], maps,
                    link: bool = True, sidecars=("tokenizer.json", "decision.json", "calibration.json"),
                    sink_casts: bool = False):
    """Rewrite `src_dir/model.onnx` into `out_dir/model.onnx` whose weights reference `sources`.
    Returns a report dict (counts, inline initializers, unused checkpoint tensors).

    The exported values are read one initializer at a time, so peak memory stays near the largest tensor
    (a 9B export has 36 GB of fp32 external data). `sink_casts` applies `sink_casts_below_gathers`."""
    model = onnx.load(os.path.join(src_dir, "model.onnx"), load_external_data=False)
    graph = model.graph
    by_shape = {}
    for si, s in enumerate(sources):
        for k, e in s.entries.items():
            by_shape.setdefault(e.shape, []).append((si, k))

    def lookup(key):
        for si, s in enumerate(sources):
            if key in s.entries:
                return si, s.entries[key]
        return None

    kept, new_nodes, stats, inline, used = [], [], Counter(), [], set()
    def val(si, e):
        return sources[si].value(e)   # no cache: a 2B model upcast to f32 would double peak memory

    for init in graph.initializer:
        arr = _initializer_array(init, src_dir)
        found = None
        for key in _candidates(init.name, maps):
            hit = lookup(key)
            if hit is None:
                continue
            si, e = hit
            if e.shape == arr.shape and np.array_equal(val(si, e).astype(arr.dtype), arr):
                found = (si, e, False)
            elif arr.ndim == 2 and e.shape == arr.shape[::-1] and np.array_equal(val(si, e).T.astype(arr.dtype), arr):
                found = (si, e, True)
            if found:
                break
        if found is None and arr.size > 4096:
            for transpose, shape in ((False, arr.shape), (True, arr.shape[::-1])):
                if transpose and arr.ndim != 2:
                    continue
                for si, k in by_shape.get(tuple(shape), []):
                    if (si, k) in used:
                        continue
                    e = sources[si].entries[k]
                    v = val(si, e)
                    if np.array_equal((v.T if transpose else v).astype(arr.dtype), arr):
                        found = (si, e, transpose)
                        break
                if found:
                    break
        if found is None:
            kept.append(numpy_helper.from_array(arr, init.name) if init.data_location == TensorProto.EXTERNAL else init)
            stats["inline"] += 1
            inline.append((init.name, list(arr.shape), int(arr.nbytes)))
            continue
        si, e, transpose = found
        used.add((si, e.key))
        src = sources[si]
        ext = TensorProto(name="w:%s:%s" % (src.location, e.key), data_type=SAFETENSORS_DTYPES[e.dtype][1],
                          dims=list(e.shape))
        ext.data_location = TensorProto.EXTERNAL
        for k, v in (("location", src.location), ("offset", str(e.offset)), ("length", str(e.length))):
            ext.external_data.add(key=k, value=v)
        if not any(x.name == ext.name for x in kept):
            kept.append(ext)
        cur = ext.name
        want = helper.np_dtype_to_tensor_dtype(arr.dtype)
        if SAFETENSORS_DTYPES[e.dtype][1] != want:
            new_nodes.append(helper.make_node("Cast", [cur], [init.name + ":cast"], to=want))
            cur = init.name + ":cast"
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
    if sink_casts:
        stats["gather_before_cast"] = sink_casts_below_gathers(graph)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "model.onnx")
    onnx.save(model, out)
    for f in sidecars:
        p = os.path.join(src_dir, f)
        if os.path.exists(p) and os.path.abspath(p) != os.path.abspath(os.path.join(out_dir, f)):
            shutil.copy(p, os.path.join(out_dir, f))
    if link:
        # ONNX Runtime only resolves external data inside the model directory; hard links cost nothing.
        for s in sources:
            dst = os.path.join(out_dir, s.location)
            if os.path.lexists(dst):
                os.remove(dst)
            try:
                os.link(os.path.realpath(s.path), dst)
            except OSError:
                os.symlink(os.path.realpath(s.path), dst)
    unused = {s.location: sorted(k for k in s.entries if (i, k) not in used) for i, s in enumerate(sources)}
    used_bytes = Counter()
    for si, k in used:
        e = sources[si].entries[k]
        used_bytes[e.dtype] += e.length
    report = {"graph_bytes": os.path.getsize(out), "stats": dict(stats), "inline": inline,
              "inline_bytes": sum(b for _, _, b in inline), "unused": unused,
              "used_bytes_by_dtype": dict(used_bytes),
              "files": [{"location": s.location, "repo": s.repo, "revision": s.revision, "filename": s.filename,
                         "bytes": os.path.getsize(s.path)} for s in sources]}
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", action="append", required=True,
                    help="LOCATION=PATH; a .pt/.pth path is read as a torch zip, anything else as safetensors")
    ap.add_argument("--map", action="append", default=[], help="GRAPH_PREFIX=CHECKPOINT_PREFIX")
    a = ap.parse_args()
    sources = []
    for spec in a.source:
        loc, path = spec.split("=", 1)
        sources.append(torchzip_source(loc, path) if path.endswith((".pt", ".pth")) else safetensors_source(loc, path))
    maps = [tuple(m.split("=", 1)) for m in a.map]
    report = make_weightless(a.src, a.out, sources, maps)
    print(json.dumps({k: v for k, v in report.items() if k != "unused"}, indent=1))
    print("unused:", {k: len(v) for k, v in report["unused"].items()})


if __name__ == "__main__":
    main()
