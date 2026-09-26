"""Export helpers shared by the decoder families: dynamo export with a 64-multiple sequence axis,
weightless rewrite against the upstream checkpoint files, and file manifests."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time

import onnx
import torch

from ...weightless_sharded import make_weightless

OPSET = 20
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
    onnx.checker.check_model(path, full_check=False)
    return time.time() - t0


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


def weightless(tmp_dir, out_dir, sources, maps):
    report = make_weightless(tmp_dir, out_dir, sources, maps, link=True, sidecars=(), sink_casts=True)
    return report


# Above this many bytes of BF16 checkpoint tensors (the 4B and 9B Qwen3.5 bases; decider-2b has 3.8 GB), the runtime
# keeps the weights BF16 in memory and widens them at each forward pass instead of once at load:
# docs/decisions/0001-decoder-weights-in-memory.md.
BF16_IN_MEMORY_ABOVE = 6 * 2**30


def weights_in_memory(report):
    """decision.json "weights_in_memory" for a weightless export: "bf16" for large BF16 checkpoints, else "fp32"."""
    return "bf16" if report["used_bytes_by_dtype"].get("BF16", 0) > BF16_IN_MEMORY_ABOVE else "fp32"


def scratch_dir(prefix):
    base = os.environ.get("OLLAYA_SCRATCH") or tempfile.gettempdir()
    return tempfile.mkdtemp(prefix=prefix, dir=base)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def cleanup(path):
    shutil.rmtree(path, ignore_errors=True)
