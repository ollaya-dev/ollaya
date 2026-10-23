"""Derive the fp16 variant of an exported fp32 graph (the default GPU precision).

    uv run python -m ollaya_convert.fp16 out/laya-en out/laya-en-fp16

Inputs and outputs keep their fp32/int types, so runtimes feed both variants identically.
Constants beyond fp16 range (e.g. attention masks filled with float32's minimum) are clamped to
±1e4 by the converter, which keeps masked positions masked without producing -inf/NaN.
"""
import argparse
import os
import shutil

import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.transformers.onnx_model import OnnxModel


def weight_bytes(model):
    """The initializers' size; `ByteSize()` itself fails on a model past protobuf's 2 GB."""
    return sum(len(t.raw_data) for t in model.graph.initializer)


def to_fp16(model, keep_fp32=()):
    kw = {"op_block_list": list(keep_fp32)} if keep_fp32 else {}
    # The converter moves values below fp16's smallest subnormal to ±5.96e-8, which keeps a small constant such as an
    # epsilon non-zero. A weight gets a plain cast instead: the registry ships the fp32 checkpoint plus a Cast node
    # (weightless.py), which rounds such values to 0, and checks that it reproduces this graph exactly.
    plain = {t.name: numpy_helper.to_array(t).astype(np.float16) for t in model.graph.initializer
             if t.data_type == onnx.TensorProto.FLOAT and np.prod(t.dims) > 4096}
    # The converter's shape inference serializes the model, which protobuf caps at 2 GB (fp32 weights of
    # a 1B-parameter encoder); such a graph is converted without it, and its fp16 parity gate checks the result.
    big = weight_bytes(model) > 2**31 - 2**24
    model = convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=big, **kw)
    for i, t in enumerate(model.graph.initializer):
        if t.name in plain and t.data_type == onnx.TensorProto.FLOAT16:
            model.graph.initializer[i].CopyFrom(numpy_helper.from_array(plain[t.name], t.name))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--keep-fp32", default="", help="comma-separated op types left in fp32")
    a = ap.parse_args()

    model = onnx.load(os.path.join(a.src, "model.onnx"), load_external_data=True)
    model = to_fp16(model, [o for o in a.keep_fp32.split(",") if o])
    # The converter appends its Cast nodes out of order; ONNX requires a topologically sorted graph.
    wrapped = OnnxModel(model)
    wrapped.topological_sort()
    model = wrapped.model
    # With an op block list the converter leaves stale fp16 types in value_info for the outputs
    # of ops it kept in fp32. value_info is optional; drop it and let shape inference recompute.
    del model.graph.value_info[:]
    os.makedirs(a.dst, exist_ok=True)
    out = os.path.join(a.dst, "model.onnx")
    if weight_bytes(model) < 2**31 - 2**24:
        onnx.save(model, out)
    else:
        onnx.save(model, out, save_as_external_data=True, location="model.onnx.data")
    onnx.checker.check_model(out, full_check=True)
    for name in ("tokenizer.json", "decision.json", "calibration.json"):
        shutil.copy(os.path.join(a.src, name), os.path.join(a.dst, name))
    size = sum(os.path.getsize(os.path.join(a.dst, f)) for f in os.listdir(a.dst) if f.startswith("model.onnx"))
    print("wrote %s (%.0f MB)" % (out, size / 2**20))


if __name__ == "__main__":
    main()
