"""The browser's graph of a ternary model: its weights stay low-bit inside onnxruntime-web.

    python webgraph.py <export dir> <out.onnx>
    python webgraph.py                            # self-test

<export dir> is `ollaya_convert.export`'s output for a ternary checkpoint (model.onnx and its external data). Every
MatMul whose weight is ternary (one g per output row, ternary.py) becomes com.microsoft MatMulNBits with 2-bit codes
and g as every block's scale; token embeddings that are int4 per row become GatherBlockQuantized. Both dequantize
to exactly the float32 weights, so the graph computes what the float32 one does up to the order of the sums; the
rest stays float32. One self-contained file, about the size of model.packed: unlike the float32 graph, whose
weights onnxruntime-web holds at 32 bits (several copies while it loads), it fits in a phone's memory.
"""
from __future__ import annotations

import collections
import os
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ternary import _int_rows, _row_scale


def _block(k: int) -> int:
    return next(b for b in (256, 128, 64, 32, 16) if k % b == 0)


def _pack(u: np.ndarray, bits: int) -> np.ndarray:
    """Unsigned codes [..., n] to bytes, the first code in the lowest bits, as MatMulNBits and GatherBlockQuantized read them."""
    per = 8 // bits
    u = u.reshape(*u.shape[:-1], u.shape[-1] // per, per)
    out = np.zeros(u.shape[:-1], np.uint8)
    for j in range(per):
        out |= u[..., j].astype(np.uint8) << (bits * j)
    return out


def matmul_nbits(node, w: np.ndarray, g: np.ndarray, name: str):
    """MatMul(x, w.T) with ternary w [N, K] as MatMulNBits, 2 bits per weight (zero point 2: codes 1, 2, 3)."""
    n, k = w.shape
    b = _block(k)
    g = np.where(g > 0, g, 1).astype(np.float32)
    codes = np.rint(w / g[:, None])
    assert np.array_equal((codes * g[:, None]).astype(np.float32), w), name
    q = _pack((codes + 2).astype(np.uint8).reshape(n, k // b, b), 2)
    inits = [numpy_helper.from_array(q, name + ".q2"), numpy_helper.from_array(np.repeat(g, k // b), name + ".scale")]
    new = helper.make_node("MatMulNBits", [node.input[0], name + ".q2", name + ".scale"], list(node.output),
                           name=node.name, domain="com.microsoft", K=k, N=n, bits=2, block_size=b)
    return new, inits


def gather_q4(node, w: np.ndarray, s: np.ndarray, q: np.ndarray, name: str):
    """Gather from int4 rows (q * s) as GatherBlockQuantized (zero point 8)."""
    v, h = w.shape
    b = _block(h)
    inits = [numpy_helper.from_array(_pack((q + 8).astype(np.uint8), 4), name + ".q4"),
             numpy_helper.from_array(np.repeat(s[:, None], h // b, axis=1), name + ".scale")]
    new = helper.make_node("GatherBlockQuantized", [name + ".q4", node.input[1], name + ".scale"], list(node.output),
                           name=node.name, domain="com.microsoft", gather_axis=0, quantize_axis=1, block_size=b, bits=4)
    return new, inits


def convert(src: str, out: str):
    m = onnx.load(os.path.join(src, "model.onnx"))
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    users = collections.Counter(i for n in g.node for i in n.input)
    nodes, gone, added, counts = [], set(), [], collections.Counter()
    for node in g.node:
        new = None
        if node.op_type == "MatMul" and node.input[1] in inits and users[node.input[1]] == 1:
            w = numpy_helper.to_array(inits[node.input[1]])
            if w.ndim == 2 and w.dtype == np.float32 and (scale := _row_scale(w.T)) is not None:
                new, more = matmul_nbits(node, np.ascontiguousarray(w.T), scale, node.input[1])
                counts["MatMulNBits"] += 1
        elif node.op_type == "Gather" and node.input[0] in inits and users[node.input[0]] == 1:
            w = numpy_helper.to_array(inits[node.input[0]])
            axis = next((a.i for a in node.attribute if a.name == "axis"), 0)
            if w.ndim == 2 and w.dtype == np.float32 and axis == 0 and (sq := _int_rows(w, 4)) is not None:
                new, more = gather_q4(node, w, *sq, node.input[0])
                counts["GatherBlockQuantized"] += 1
        if new is None:
            nodes.append(node)
            continue
        nodes.append(new)
        gone.add(node.input[1] if node.op_type == "MatMul" else node.input[0])
        added += more
    kept = [t for t in g.initializer if t.name not in gone]
    del g.node[:]
    g.node.extend(nodes)
    del g.initializer[:]
    g.initializer.extend(kept + added)
    if not any(o.domain == "com.microsoft" for o in m.opset_import):
        m.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    onnx.save(m, out)
    print(dict(counts), "kept float32:", sum(1 for t in kept if t.data_type == onnx.TensorProto.FLOAT),
          "%.1f MB" % (os.path.getsize(out) / 2**20))


def _self_test():
    """A ternary MatMul and an int4 Gather, converted, give the float32 graph's outputs exactly."""
    import tempfile

    import onnxruntime as ort
    from onnx import TensorProto as T

    from ternary import IntRows, absmean_ternary
    import torch

    w = absmean_ternary(torch.randn(48, 512)).numpy()                # [N, K]
    e = IntRows(4).quantize(torch.randn(30, 256)).numpy()
    nodes = [helper.make_node("Gather", ["E", "ids"], ["h"]), helper.make_node("MatMul", ["h", "W"], ["mid"]),
             helper.make_node("MatMul", ["x", "W2"], ["y2"])]
    graph = helper.make_graph(nodes, "t", [helper.make_tensor_value_info("ids", T.INT64, [2, 3]),
                                           helper.make_tensor_value_info("x", T.FLOAT, [2, 512])],
                              [helper.make_tensor_value_info(o, T.FLOAT, None) for o in ("mid", "y2")],
                              [numpy_helper.from_array(e, "E"), numpy_helper.from_array(np.ascontiguousarray(w[:, :256].T), "W"),
                               numpy_helper.from_array(np.ascontiguousarray(w.T), "W2")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    feeds = {"ids": np.array([[0, 5, 29], [3, 3, 7]], np.int64), "x": np.random.randn(2, 512).astype(np.float32)}
    with tempfile.TemporaryDirectory() as d:
        onnx.save(model, os.path.join(d, "model.onnx"))
        convert(d, os.path.join(d, "web.onnx"))
        run = lambda f: ort.InferenceSession(os.path.join(d, f), providers=["CPUExecutionProvider"]).run(None, feeds)
        for a, b in zip(run("model.onnx"), run("web.onnx")):
            assert np.allclose(a, b, rtol=1e-5, atol=1e-4), np.abs(a - b).max()   # the same sums, in another order
        ops = {n.op_type for n in onnx.load(os.path.join(d, "web.onnx")).graph.node}
        assert ops == {"GatherBlockQuantized", "MatMulNBits"}, ops


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _self_test()
        print("ok")
    elif len(sys.argv) == 3:
        convert(*sys.argv[1:])
    else:
        sys.exit(__doc__)
