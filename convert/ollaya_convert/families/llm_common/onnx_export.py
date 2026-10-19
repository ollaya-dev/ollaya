"""Export helpers shared by the decoder families: dynamo export with a 64-multiple sequence axis,
weightless rewrite against the upstream checkpoint files, and file manifests."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from ...weightless_sharded import make_weightless

OPSET = 20
# decision.json "precision" of the -fp16 graphs (qwen35.half_trunk).
FP16_PRECISION = ("fp16 compute, with the RMSNorms, the gated delta rule and the attention softmax in fp32; weights "
                  "cast from the checkpoint's (at load, or per forward pass with weights_in_memory bf16)")
SEQ_MULTIPLE = 64


# 0 = no optimizer pass: the exact graphs parity was measured on (every weight byte-referenced).
# ONNX Runtime 1.30's CUDA EP miscomputes this raw export's shape arithmetic when memory reuse is on
# (Reshape receives shape values from reused buffers); run it with SessionOptions.enable_mem_reuse =
# False, or export with a small fold limit (e.g. OLLAYA_FOLD_LIMIT=64), which fixes the CUDA EP but
# folds a few tiny weight-derived tensors (-exp(A_log), 16 floats per layer) into the graph.
FOLD_LIMIT = int(os.environ.get("OLLAYA_FOLD_LIMIT", "0"))


def export_graph(module, args, input_names, output_names, dynamic_shapes, path, fold_limit=FOLD_LIMIT):
    """torch.onnx.export(dynamo=True), then the onnxscript optimizer with constant folding capped at
    `fold_limit` elements: every parameter keeps its name and value (so its checkpoint bytes stay
    referenceable) while the symbolic-shape plumbing the exporter emits is simplified.

    The cleanup is not cosmetic: ONNX Runtime 1.30's CUDA EP miscomputes the raw (unoptimized) export's
    shape arithmetic with memory reuse on (Reshape receives shape values from reused buffers)."""
    t0 = time.time()
    with torch.no_grad():
        program = torch.onnx.export(module, args, dynamo=True, opset_version=OPSET, input_names=input_names,
                                    output_names=output_names, dynamic_shapes=dynamic_shapes, optimize=False)
    if fold_limit:
        from onnxscript.optimizer import optimize_ir

        optimize_ir(program.model, input_size_limit=fold_limit, output_size_limit=fold_limit)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    program.save(path, external_data=True)
    model = onnx.load(path, load_external_data=False)
    if sequence_splits_to_split(model):
        onnx.save(model, path)
    onnx.checker.check_model(path, full_check=False)
    return time.time() - t0


def sequence_splits_to_split(model):
    """Rewrite each `SplitToSequence` read only by `SequenceAt` at constant positions (how torch.split and
    torch.chunk export) into one `Split`. ONNX Runtime has no CUDA kernel for SplitToSequence, so every one
    ran on the CPU with copies to the host and back: a quarter of a kev-4b request on an RTX 4090. The values
    are identical. Returns the number of nodes rewritten."""
    g = model.graph
    consts = {t.name: numpy_helper.to_array(t) for t in g.initializer if t.data_location != TensorProto.EXTERNAL}
    for n in g.node:
        if n.op_type == "Constant" and len(n.attribute) == 1 and n.attribute[0].name == "value":
            consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    readers = defaultdict(list)
    for n in g.node:
        for i in n.input:
            readers[i].append(n)
    graph_outputs = {o.name for o in g.output}
    dims = None
    replaced, nodes, done = {}, [], 0
    for n in g.node:
        if id(n) in replaced:
            nodes.append(replaced.pop(id(n)))
            continue
        users = readers[n.output[0]] if n.op_type == "SplitToSequence" else []
        if (len(n.input) < 2 or n.input[1] not in consts or not users or n.output[0] in graph_outputs
                or any(u.op_type != "SequenceAt" or u.input[1] not in consts for u in users)):
            nodes.append(n)
            continue
        axis = next((a.i for a in n.attribute if a.name == "axis"), 0)
        sizes = consts[n.input[1]].astype(np.int64)
        split_input = n.input[1]
        if sizes.ndim == 0:   # torch.chunk: equal parts of a dimension the graph must know
            if dims is None:
                inferred = onnx.shape_inference.infer_shapes(model).graph
                dims = {v.name: [d.dim_value if d.HasField("dim_value") else None for d in v.type.tensor_type.shape.dim]
                        for v in list(inferred.value_info) + list(inferred.input)}
            shape = dims.get(n.input[0])
            dim = shape[axis] if shape and -len(shape) <= axis < len(shape) else None
            if not dim:
                nodes.append(n)
                continue
            s = int(sizes)
            sizes = np.array([s] * (dim // s) + ([dim % s] if dim % s else []), dtype=np.int64)
            split_input = n.output[0] + "_sizes"
            g.initializer.append(numpy_helper.from_array(sizes, split_input))
        parts = ["%s_part%d" % (n.output[0], i) for i in range(len(sizes))]
        nodes.append(helper.make_node("Split", [n.input[0], split_input], parts, axis=axis, name=n.name))
        for u in users:
            i = int(consts[u.input[1]])
            replaced[id(u)] = helper.make_node("Identity", [parts[i + len(sizes) if i < 0 else i]], [u.output[0]],
                                               name=u.name)
        done += 1
    if done:
        del g.node[:]
        g.node.extend(nodes)
        kept = [v for v in g.value_info if not v.type.HasField("sequence_type")]
        del g.value_info[:]
        g.value_info.extend(kept)
    return done


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def hub_file_info(repo, revision, filename):
    """LFS sha256 + size as the Hub reports them (None when offline)."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        for s in info.siblings:
            if s.rfilename == filename:
                lfs = getattr(s, "lfs", None)
                return {"sha256": getattr(lfs, "sha256", None) if lfs else None, "size": s.size}
    except Exception:
        return None
    return None


def file_entry(role, repo, revision, filename, local_path, location=None, verify=True):
    e = {"role": role, "repo": repo, "revision": revision, "path": filename,
         "url": "https://huggingface.co/%s/resolve/%s/%s" % (repo, revision, filename),
         "bytes": os.path.getsize(local_path)}
    if location:
        e["location"] = location
    if verify:
        e["sha256"] = sha256_file(local_path)
        hub = hub_file_info(repo, revision, filename)
        if hub and hub.get("sha256"):
            e["hub_lfs_sha256"] = hub["sha256"]
            e["sha256_matches_hub"] = hub["sha256"] == e["sha256"]
    return e


def weightless(tmp_dir, out_dir, sources, maps, verify=True, sink_casts=True):
    report = make_weightless(tmp_dir, out_dir, sources, maps, link=True, sidecars=(), sink_casts=sink_casts,
                             verify=verify)
    return report


# Above this many bytes of BF16 checkpoint tensors (the 4B and 9B Qwen3.5 bases; decider-2b has 3.8 GB), the runtime
# keeps the weights BF16 in memory and widens them at each forward pass instead of once at load:
# docs/decisions/0002-decoder-weights-in-memory.md.
BF16_IN_MEMORY_ABOVE = 6 * 2**30


def weights_in_memory(report):
    """decision.json "weights_in_memory" for a weightless export: "bf16" for large BF16 checkpoints, else "fp32"."""
    return "bf16" if report["used_bytes_by_dtype"].get("BF16", 0) > BF16_IN_MEMORY_ABOVE else "fp32"


def scratch_dir(prefix):
    base = os.environ.get("OLLAYA_SCRATCH") or tempfile.gettempdir()
    return tempfile.mkdtemp(prefix=prefix, dir=base)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def cleanup(path):
    shutil.rmtree(path, ignore_errors=True)


def main():
    """Apply `sequence_splits_to_split` to exported model directories in place (graphs exported before it):

        python -m ollaya_convert.families.llm_common.onnx_export out/kev-4b out/nimble-9b-v2 ...
    """
    for d in sys.argv[1:]:
        path = os.path.join(d, "model.onnx")
        model = onnx.load(path, load_external_data=False)
        n = sequence_splits_to_split(model)
        if n:
            # No onnx.checker here: it refuses weights that are hard or symbolic links. Parity checks the graph.
            onnx.save(model, path + ".tmp")
            os.replace(path + ".tmp", path)   # a new file: never write through a hard link to a registry blob
            files = os.path.join(d, "files.json")
            if os.path.exists(files):
                f = json.load(open(files, encoding="utf-8"))
                for layer in f.get("layers", []):
                    if layer.get("path") == "model.onnx":
                        layer.update(bytes=os.path.getsize(path), sha256=sha256_file(path))
                write_json(files, f)
        print("%s: %d SplitToSequence rewritten" % (d, n))


if __name__ == "__main__":
    main()
