"""Export vllm-sr/Decision-2.0-* (a Qwen3.5 or, for Kai, Qwen3 text backbone + endpoint head) to a weightless ONNX graph.

    uv run python -m ollaya_convert.families.decision2.export decision2-eos --out out/decision2-eos [--root SNAPSHOT]

The graph is Decision 1.0's (`decision.export.DecisionGraph`, contract in docs/families/decision.md): `input_ids`
[rows, seq], `query_pos` [rows], `cand_pos` [rows, k] -> the head's raw `logits` [rows, k]. Kai's backbone is a dense
Qwen3 (`llm_common/qwen3.py`), so its rows pad to a multiple of 8 instead of the DeltaNet chunk (64).

Weights are not copied: every initializer references the author's files at their byte offsets. The packages store the
backbone's Linear weights in BF16 (as BF16 autocast rounds them; one Cast to float32 each) and every other tensor
(embeddings, norms, the DeltaNet's A_log, dt_bias and conv filters, the head) in F32.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch

from ..decision.export import DecisionGraph, rename
from ..llm_common import onnx_export as ox
from ..llm_common.qwen3 import Qwen3Trunk
from ..llm_common.qwen35 import CHUNK, Qwen35Trunk
from ...weightless_sharded import safetensors_source
from . import ref
from .layout import MAX_OPTIONS, MAX_SCORE_LEVELS, MIN_OPTIONS, NOUL_DEFAULTS, PROMPT_VERSION
from ..decision.layout import SUFFIX

INPUT_NAMES = ["input_ids", "query_pos", "cand_pos"]
OUTPUT_NAMES = ["logits"]
LAYOUT = "decision2-endpoint-v1"
QWEN3_MULTIPLE = 8  # upstream's collate pads to a multiple of 8; a dense causal graph needs no more

SAMPLE_STATE = "The customer was charged twice for order A-104 and wants the duplicate refunded today. " * 6
SAMPLE_QUESTIONS = {
    "a": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "charges", "tech": "bugs", "other": None}},
    "b": {"type": "score", "instructions": "How upset?", "criteria": ["calm", "annoyed", "furious"]},
}


def empty_model(d, root, qwen3):
    """The author's backbone and head classes with uninitialized fp32 storage (never written, so it costs no memory):
    the graph of a checkpoint too large to hold in fp32. Only the rotary frequencies, the one computed buffer, are real."""
    from transformers import AutoConfig

    if qwen3:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Model as Model, Qwen3RotaryEmbedding as Rotary
    else:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel as Model, \
            Qwen3_5TextRotaryEmbedding as Rotary

    config = AutoConfig.from_pretrained(os.path.join(root, "backbone"), local_files_only=True)
    config.use_cache = False
    with torch.device("meta"):
        backbone = Model(config)
        head = d.dm.CandidateHead(config.hidden_size, 256)
    buffers = [n for n, _ in backbone.named_buffers()]
    assert all(n.startswith("rotary_emb.") for n in buffers), buffers
    backbone = backbone.to_empty(device="cpu")
    backbone.rotary_emb = Rotary(config=config)
    return backbone.eval(), head.to_empty(device="cpu").eval()


def export(slug, out_dir, root, empty=False):
    """`empty`: export without the weights' values (see empty_model); the weightless step then maps initializers by
    name and shape, and the graph is checked by parity afterwards."""
    meta = ref.MODELS[slug]
    pkg = ref.package(root)
    cfg, pointer, manifest = pkg["decision"], pkg["pointer"], pkg["manifest"]
    assert cfg["prompt_version"] == PROMPT_VERSION and cfg["head_dim"] == 256, cfg
    assert cfg.get("head_variant", "shared") == "shared" and cfg["architecture"].endswith(
        "-text-endpoints-global-query-shared-bilinear-mlp"), cfg["architecture"]
    qwen3 = cfg["architecture"].startswith("qwen3-")
    multiple = QWEN3_MULTIPLE if qwen3 else CHUNK
    d = ref.load(root, slug, device="cpu", load_model=not empty)
    backbone, head = empty_model(d, root, qwen3) if empty else (d.model.backbone, d.model.head)
    graph = DecisionGraph(backbone, head, Qwen3Trunk if qwen3 else Qwen35Trunk).eval()

    # Two rows padded to a whole number of steps, three candidates each: every dynamic axis > 1. The rows come from the
    # author's encode, and the eager graph is checked against the author's forward on them.
    rows, items = d.encode(SAMPLE_STATE, SAMPLE_QUESTIONS)
    T = -(-max(len(it["ids"]) for it in items) // multiple) * multiple
    ids = torch.full((len(items), T), d.pad, dtype=torch.long)
    for i, it in enumerate(items):
        ids[i, :len(it["ids"])] = torch.tensor(it["ids"])
    args = (ids, torch.tensor([it["query_position"] for it in items]),
            torch.tensor([it["candidate_positions"] for it in items]))
    if not empty:
        with torch.no_grad():
            want = d.forward(items)
            got = graph(*args)
        print("eager graph vs the author's forward: %.2e" % max(float((g - w).abs().max()) for g, w in zip(got, want)))

    R = torch.export.Dim("rows", min=1, max=4096)
    N = torch.export.Dim("steps", min=1, max=4096)
    K = torch.export.Dim("options", min=1, max=MAX_OPTIONS)
    dyn = {"input_ids": {0: R, 1: multiple * N}, "query_pos": {0: R}, "cand_pos": {0: R, 1: K}}
    tmp = ox.scratch_dir("decision2-export-")
    secs = ox.export_graph(graph, args, INPUT_NAMES, OUTPUT_NAMES, dyn, os.path.join(tmp, "model.onnx"))
    print("exported in %.0fs" % secs)
    params = {"trunk.m." + n for n, _ in backbone.named_parameters()} | {"head." + n for n, _ in head.named_parameters()}
    del graph, backbone, head

    weights = pointer["backbone"]["weights"] + [pointer["decision_weights"]["decision_head"]]
    sources = [safetensors_source(os.path.basename(f), os.path.join(root, f), repo=meta["repo"],
                                  revision=meta["revision"], filename=f) for f in weights]
    report = ox.weightless(tmp, out_dir, sources, rename, verify=not empty, sink_casts=False)
    ox.cleanup(tmp)
    # Every parameter references the checkpoint; only buffers and constants stay inline.
    inline_params = sorted(params & {name for name, _, _ in report["inline"]})
    assert not inline_params and report["stats"]["external"] == len(params), (inline_params, report["stats"])
    tok_out = os.path.join(out_dir, "tokenizer.json")
    if os.path.exists(tok_out):  # a previous export's copy keeps the snapshot's read-only mode
        os.remove(tok_out)
    shutil.copyfile(os.path.join(root, "tokenizer.json"), tok_out)

    origin = manifest["origin"]
    decision = {
        "engine": "onnx",
        "family": "decision2",
        "layout": LAYOUT,
        "upstream": {"repo": meta["repo"], "revision": meta["revision"], "model_name": pointer["model_name"],
                     "origin": {"repo": origin["repo_id"], "revision": origin["revision"]},
                     "base_revision": cfg["base_revision"], "architecture": cfg["architecture"],
                     "runtime": {"entry": pointer["runtime"]["entry"], "profile": manifest["profile"],
                                 "files_sha256": {f: manifest["files_sha256"][f] for f in sorted(manifest["files_sha256"])
                                                  if f.startswith("decision2/")}}},
        "contract": {
            "inputs": {
                "input_ids": {"dtype": "int64", "shape": ["rows", "seq"],
                              "note": "one row per question; seq a multiple of seq_multiple; right-pad with any id (pad)"},
                "query_pos": {"dtype": "int64", "shape": ["rows"], "note": "position of the row's last token"},
                "cand_pos": {"dtype": "int64", "shape": ["rows", "k"],
                             "note": "position of each option segment's last token; rows with fewer options pad with 0"},
            },
            "outputs": {"logits": {"dtype": "float32", "shape": ["rows", "k"],
                                   "note": "raw candidate logits in option order; the first k_row entries are the row's"}},
            "seq_multiple": multiple,
            "positions": "0..seq-1, implicit",
            "attention": "causal; no mask input (right padding cannot reach earlier positions)",
        },
        "weights_in_memory": ox.weights_in_memory(report),
        "prompt_version": PROMPT_VERSION,
        "max_row_tokens": manifest["max_input_tokens"],
        "min_options": MIN_OPTIONS,
        "max_options": MAX_OPTIONS,
        "max_score_levels": MAX_SCORE_LEVELS,
        "choice_null_description": "null",
        "pad": d.pad,
        "templates": {
            "prefix": "Context:\n{payload(state)}\n\nTask type: {type}\nQuestion:\n{payload(instructions)}\nOptions:",
            "option": "\n<option>\n{canonical({\"key\": key, \"description\": description})}\n</option>",
            "suffix": SUFFIX,
            "payload": "a str verbatim, anything else canonical()",
            "canonical": "json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(',', ':'))",
            "tokenize": "each segment on its own, add_special_tokens=False; cand_pos = last token of each option, "
                        "query_pos = last token of the row",
            "noul_descriptions": NOUL_DEFAULTS,
        },
        "option_logits": {"choice": "logits[row, :k] in the caller's order",
                          "noul": "the false and true options' logits, as [false, true]",
                          "score": "logits[row, :levels], plus score_bias[levels] when listed"},
        "opset": ox.OPSET,
        "precision": "fp32 compute; backbone Linear weights BF16 (Cast), every other tensor F32",
    }
    if d.offsets:
        decision["score_bias"] = {str(k): v for k, v in sorted(d.offsets.items())}
    calibration = {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {},
                   "source": "none: the package has no calibration file and serves raw probabilities (temperature 1)"}
    layers = [{"role": "graph", "path": "model.onnx", "hosted_by": "ollaya",
               "bytes": os.path.getsize(os.path.join(out_dir, "model.onnx")),
               "sha256": ox.sha256_file(os.path.join(out_dir, "model.onnx"))}]
    for f in weights:
        layers.append(ox.file_entry("weights", meta["repo"], meta["revision"], f, os.path.join(root, f),
                                    location=os.path.basename(f)))
    layers += [
        ox.file_entry("tokenizer", meta["repo"], meta["revision"], "tokenizer.json", os.path.join(root, "tokenizer.json")),
        {"role": "decision", "path": "decision.json", "hosted_by": "ollaya"},
        {"role": "calibration", "path": "calibration.json", "hosted_by": "ollaya"},
        ox.file_entry("license", meta["repo"], meta["revision"], "LICENSE", os.path.join(root, "LICENSE"), verify=False)
        | {"note": "Apache-2.0 (the repo's LICENSE); the Qwen bases are Apache-2.0"},
    ]
    files = {"model": slug, "layers": layers, "weightless": {k: v for k, v in report.items() if k != "unused"},
             "unused_checkpoint_tensors": {k: len(v) for k, v in report["unused"].items()}}
    ox.write_json(os.path.join(out_dir, "decision.json"), decision)
    ox.write_json(os.path.join(out_dir, "calibration.json"), calibration)
    ox.write_json(os.path.join(out_dir, "files.json"), files)
    print(json.dumps(files["weightless"]["stats"]), "inline bytes", files["weightless"]["inline_bytes"],
          "graph MB %.1f" % (layers[0]["bytes"] / 2**20), "unused", files["unused_checkpoint_tensors"],
          "weights_in_memory", decision["weights_in_memory"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(ref.MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None, help="local snapshot of the model repo at the pinned revision")
    ap.add_argument("--empty-weights", action="store_true",
                    help="export without loading the weights (checkpoints too large for fp32 in memory); "
                         "initializers map by name and shape")
    a = ap.parse_args()
    export(a.model, a.out, a.root or ref.snapshot(a.model), empty=a.empty_weights)


if __name__ == "__main__":
    main()
