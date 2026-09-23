"""Point an exported graph's weights at the original checkpoint files instead of copying them.

    uv run python -m ollaya_convert.weightless out/laya-en ~/models/laya/model.safetensors --prefix model. \
        --out out/laya-en-wl

An ONNX graph and its weights are separable: an initializer can live in an external file at a
byte offset. Safetensors stores each tensor as raw little-endian bytes at a known offset, which is
exactly that. So the graph Ollaya ships can reference the author's own `model.safetensors`, and
`ollaya pull` fetches weights from the original Hugging Face repo, unmodified.

Exporters rewrite some weights (dtype upcast, pre-transposed Linear weights). For every graph
initializer this finds the checkpoint tensor it came from, verifies the values match exactly after
the same transform, and replaces it with an external reference to the original bytes plus the
Cast/Transpose nodes that reproduce it; ONNX Runtime folds those at session creation. Initializers
with no source tensor (small constants: shapes, rotary tables) stay inline in the graph.

An fp16 graph (from `fp16.py`) over an fp16 checkpoint needs no Cast: its weights are the
checkpoint's bytes as they are.
"""
import argparse
import json
import os
import shutil
import struct
from collections import Counter

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ST_DTYPES = {"F16": (np.float16, TensorProto.FLOAT16), "BF16": (None, TensorProto.BFLOAT16),
             "F32": (np.float32, TensorProto.FLOAT), "I64": (np.int64, TensorProto.INT64)}


def read_safetensors_index(path):
    """{name: (dtype, shape, absolute_offset, length)} without loading any tensor data."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    base = 8 + n
    return {k: (v["dtype"], tuple(v["shape"]), base + v["data_offsets"][0],
                v["data_offsets"][1] - v["data_offsets"][0]) for k, v in header.items()}


def load_tensor(path, entry):
    dtype, shape, off, length = entry
    np_dtype = ST_DTYPES[dtype][0]
    with open(path, "rb") as f:
        f.seek(off)
        return np.frombuffer(f.read(length), dtype=np_dtype).reshape(shape)


def rewrite(model, checkpoint, prefix="", location="model.safetensors"):
    """Rewrite `model` in place so its weights reference `checkpoint`; returns stats."""
    index = read_safetensors_index(checkpoint)
    graph = model.graph
    by_shape = {}
    for name, entry in index.items():
        by_shape.setdefault(entry[1], []).append(name)

    kept, new_nodes, stats = [], [], Counter()
    used = set()
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        name = init.name[len(prefix):] if init.name.startswith(prefix) else init.name
        source = transpose = None
        if name in index and index[name][1] == arr.shape:
            source = name
        elif arr.ndim == 2 and arr.size > 4096:
            for cand in by_shape.get(arr.shape[::-1], []):
                if cand in used:
                    continue
                if np.array_equal(load_tensor(checkpoint, index[cand]).T.astype(arr.dtype), arr):
                    source, transpose = cand, True
                    break
        if source is None or index[source][0] not in ST_DTYPES or ST_DTYPES[index[source][0]][0] is None:
            kept.append(init)
            stats["inline"] += 1
            continue
        orig = load_tensor(checkpoint, index[source])
        expect = orig.T if transpose else orig
        if not np.array_equal(expect.astype(arr.dtype), arr):
            raise SystemExit("value mismatch for %s <- %s" % (init.name, source))
        used.add(source)

        dtype, shape, off, length = index[source]
        ext = onnx.TensorProto(name="w:" + source, data_type=ST_DTYPES[dtype][1], dims=list(shape))
        ext.data_location = TensorProto.EXTERNAL
        for k, v in (("location", location), ("offset", str(off)), ("length", str(length))):
            ext.external_data.add(key=k, value=v)
        kept.append(ext)

        cur = ext.name
        if orig.dtype != arr.dtype:
            to = helper.np_dtype_to_tensor_dtype(arr.dtype)
            new_nodes.append(helper.make_node("Cast", [cur], [cur + ":cast"], to=to))
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
    stats["unused"] = len(set(index) - used)
    return dict(stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="directory with model.onnx (+ .onnx.data) and sidecar json files")
    ap.add_argument("checkpoint", help="original model.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="", help="prefix the exporter added to parameter names")
    ap.add_argument("--location", default="model.safetensors",
                    help="file name the graph uses for the checkpoint, relative to model.onnx")
    a = ap.parse_args()

    model = onnx.load(os.path.join(a.src, "model.onnx"), load_external_data=True)
    stats = rewrite(model, a.checkpoint, a.prefix, a.location)
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "model.onnx")
    onnx.save(model, out)  # external initializers stay references; nothing is copied
    for f in ("tokenizer.json", "decision.json", "calibration.json"):
        if os.path.exists(os.path.join(a.src, f)):
            shutil.copy(os.path.join(a.src, f), os.path.join(a.out, f))
    # ONNX Runtime rejects external data that resolves outside the model's directory, so the
    # checkpoint must physically sit next to the graph. A hard link costs nothing; in the Ollaya
    # blob store the graph instead names the checkpoint's own blob file (`--location sha256-...`).
    link = os.path.join(a.out, a.location)
    if os.path.lexists(link):
        os.remove(link)
    try:
        os.link(os.path.realpath(a.checkpoint), link)
    except OSError:
        shutil.copy(os.path.realpath(a.checkpoint), link)
    print("graph %s: %.1f MB | %s" % (out, os.path.getsize(out) / 2**20, stats))


if __name__ == "__main__":
    main()
