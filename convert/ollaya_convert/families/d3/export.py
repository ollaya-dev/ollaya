"""Export vllm-sr's d3 (a fine-tuned Qwen3.5 + a 255-code readout) to a weightless ONNX graph.

    uv run python -m ollaya_convert.families.d3.export d3-edge --out out/d3-edge

The graph is pplx's (`families/pplx/export.py`, contract in docs/families/pplx.md): `input_ids` [rows, seq] right-padded
to a multiple of 64, `lengths` [rows], `markers` [rows, q] -> `logits` [rows, q, 255]. d3 puts one question in a row
and reads its last token, so `markers` is [rows, 1] holding `lengths - 1`. The full-attention layers attend to the
whole row (upstream's `noncausal_full_attention`), the linear-attention layers stay causal. The graph references the
repository's BF16 shards and its fp32 `readout.safetensors` byte for byte.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil

import tokenizers
import torch

from ..llm_common import onnx_export as ox
from ..llm_common.qwen35 import CHUNK, half_trunk
from ..pplx.export import INPUT_NAMES, OUTPUT_NAMES, TOKENIZER_PROBES, PplxGraph, rename
from ...weightless_sharded import safetensors_source
from . import ref
from .layout import D3Layout

SENTINEL = "OLLAYA-USER-TEXT"
TOKENIZER = ("bespokelabs/Bespoke-Nimble-9B-v2", "4b8c04d1ac2cea3e41e5e3c4d2130bcead2c0abe")
SAMPLE_STATE = "The customer was charged twice for order A-104 and wants the duplicate refunded. " * 6
SAMPLE_QUESTIONS = {"team": {"type": "choice", "instructions": "Which team?",
                             "criteria": {"billing": "charges", "tech": "bugs", "other": None}},
                    "upset": {"type": "score", "instructions": "How upset?", "criteria": ["calm", "annoyed", "furious"]},
                    "refund": {"type": "noul", "instructions": "Is a refund requested?"}}


def templates(tokenizer, fmt):
    text = fmt._chat_text(tokenizer, [{"role": "system", "content": fmt.SYSTEM_PROMPT},
                                      {"role": "user", "content": SENTINEL}])
    pre, post = text.split(SENTINEL)
    return pre, post


def export(slug, out_dir, run_dir, fp16=False):
    meta = ref.MODELS[slug]
    d3 = ref.load(run_dir, device="cpu")
    rt, fmt = ref.runtime(run_dir)
    cfg = d3.config
    if cfg.get("attention_mode") != "noncausal_full_attention" or d3.backbone_type != "qwen3_5":
        raise SystemExit("%s: the pplx graph needs a qwen3_5 backbone with noncausal_full_attention" % slug)
    # Upstream tokenizes with AutoTokenizer, which transformers 5 rebuilds from the vocabulary with its own
    # pre-tokenizer; the repository's tokenizer.json splits combining marks differently. The rebuild equals
    # Bespoke-Nimble-9B-v2's tokenizer.json (as for nimble:9b-v3), so Ollaya references that file.
    from huggingface_hub import hf_hub_download

    tokenizer_file = hf_hub_download(TOKENIZER[0], "tokenizer.json", revision=TOKENIZER[1])
    if json.loads(d3.tokenizer.backend_tokenizer.to_str()) != json.load(open(tokenizer_file, encoding="utf-8")):
        raise SystemExit("upstream's tokenizer differs from %s@%s:tokenizer.json" % TOKENIZER)
    file_tok = tokenizers.Tokenizer.from_file(tokenizer_file)
    for s in TOKENIZER_PROBES + [SAMPLE_STATE]:
        if file_tok.encode(s, add_special_tokens=False).ids != d3.tokenizer(s, add_special_tokens=False)["input_ids"]:
            raise SystemExit("tokenizer.json differs from upstream's tokenizer on %r" % s)

    hidden = d3.readout.shape[1]
    readout = torch.nn.Linear(hidden, rt.MAX_OPTIONS, bias=False)
    readout.weight.data = d3.readout.float().cpu()
    graph = PplxGraph(d3.backbone.language_model, readout).eval()
    # every question in its own row, plus the first question again, so every dynamic axis is > 1
    prepared = ref.encode(d3, SAMPLE_STATE, SAMPLE_QUESTIONS)
    rows = ref.rows(prepared)
    rows.append(rows[0])
    T = -(-max(len(r["ids"]) for r in rows) // CHUNK) * CHUNK
    ids = torch.full((len(rows), T), d3.pad_id, dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r["ids"])] = torch.tensor(r["ids"])
    markers = torch.tensor([r["markers"] for r in rows], dtype=torch.long)
    args = (ids, torch.tensor([len(r["ids"]) for r in rows]), markers)
    wants = ref.forward(d3, prepared)
    wants = [wants[r["keys"][0]] for r in rows]
    if fp16:
        half_trunk(graph.trunk)   # after the fp32 reference: the trunk shares upstream's modules; the readout stays fp32
    with torch.no_grad():
        got = graph(*args)
    diff = max(float((got[i, 0, :len(w)] - torch.tensor(w)).abs().max()) for i, w in enumerate(wants))
    print("eager graph (padded rows) vs upstream logits: %.2e" % diff)

    R = torch.export.Dim("rows", min=1, max=4096)
    N = torch.export.Dim("chunks", min=1, max=4096)
    K = torch.export.Dim("questions", min=1, max=4096)
    dyn = {"input_ids": {0: R, 1: CHUNK * N}, "lengths": {0: R}, "markers": {0: R, 1: K}}
    tmp = ox.scratch_dir("d3-export-")
    secs = ox.export_graph(graph, args, INPUT_NAMES, OUTPUT_NAMES, dyn, os.path.join(tmp, "model.onnx"))
    print("exported in %.0fs" % secs)
    pre, post = templates(d3.tokenizer, fmt)
    pad = d3.pad_id
    temperature = d3.temperature
    del graph, d3, got
    gc.collect()

    shards = [os.path.join(run_dir, f) for f in meta["shards"]]
    head = os.path.join(run_dir, "readout.safetensors")
    sources = [safetensors_source(f, p, repo=meta["repo"], revision=meta["revision"], filename=f)
               for f, p in zip(meta["shards"], shards)]
    sources.append(safetensors_source("readout.safetensors", head, repo=meta["repo"], revision=meta["revision"],
                                      filename="readout.safetensors"))
    report = ox.weightless(tmp, out_dir, sources, rename)
    ox.cleanup(tmp)
    shutil.copy(tokenizer_file, os.path.join(out_dir, "tokenizer.json"))

    decision = {
        "engine": "onnx",
        "family": "d3",
        "layout": "d3-code-readout-v1",
        "upstream": {"repo": meta["repo"], "revision": meta["revision"], "base": cfg["base_model"],
                     "base_revision": cfg["revision"], "code": "d3_runtime.py, d3_format.py (in the model repository)",
                     "format_id": cfg["format_id"]},
        "contract": {
            "inputs": {
                "input_ids": {"dtype": "int64", "shape": ["rows", "seq"],
                              "note": "one question per row; seq a multiple of 64; right-pad with any id (pad)"},
                "lengths": {"dtype": "int64", "shape": ["rows"], "note": "each row's real tokens"},
                "markers": {"dtype": "int64", "shape": ["rows", 1], "note": "each row's last real token (lengths - 1)"},
            },
            "outputs": {"logits": {"dtype": "float32", "shape": ["rows", 1, 255],
                                   "note": "raw readout logits of the answer codes; a question reads its first k"}},
            "seq_multiple": CHUNK,
            "positions": "0..seq-1, implicit",
            "attention": "full-attention layers read the row's first `lengths` keys (not causal); "
                         "linear-attention layers causal",
        },
        "max_length": ref.MAX_LENGTH,
        "max_options": rt.MAX_OPTIONS,
        "score_levels": list(rt.SCORE_LEVELS),
        "codes": cfg["codes"],
        "special_tokens": {"pad": pad, "add_special_tokens": False},
        "templates": {"pre": pre, "post": post, "default_instructions": fmt._DEFAULT_INSTRUCTIONS,
                      "noul_defaults": dict(zip(("false", "true"), fmt.NOUL_DESCRIPTIONS)),
                      "empty_state": fmt._EMPTY_STATE, "closing_line": fmt._CLOSING_LINE,
                      "describe": "a string as is, else json.dumps(ensure_ascii=False)"},
        "option_logits": {"choice": "logits[row, 0, :k]", "noul": "logits[row, 0, :2] (0 = false, 1 = true)",
                          "score": "logits[row, 0, :levels]"},
        "opset": ox.OPSET,
        "precision": ox.FP16_PRECISION if fp16 else "fp32 compute; weights BF16, widened by Cast (at load, or per forward pass with "
                     "weights_in_memory bf16); readout fp32",
        "weights_in_memory": ox.weights_in_memory(report),
    }
    calibration = {"temperature": [float(temperature)] * 3, "temperature_by_options": {},
                   "source": "decision_config.json temperature"}
    files = {
        "model": slug,
        "layers": [
            {"role": "graph", "path": "model.onnx", "hosted_by": "ollaya",
             "bytes": os.path.getsize(os.path.join(out_dir, "model.onnx")),
             "sha256": ox.sha256_file(os.path.join(out_dir, "model.onnx"))},
            *[ox.file_entry("weights", meta["repo"], meta["revision"], f, p, location=f)
              for f, p in zip(meta["shards"], shards)],
            ox.file_entry("weights/head", meta["repo"], meta["revision"], "readout.safetensors", head,
                          location="readout.safetensors"),
            ox.file_entry("tokenizer", TOKENIZER[0], TOKENIZER[1], "tokenizer.json", tokenizer_file)
            | {"note": "what transformers 5's AutoTokenizer rebuilds from the repository's vocabulary (upstream's tokenizer)"},
            {"role": "decision", "path": "decision.json", "hosted_by": "ollaya"},
            {"role": "calibration", "path": "calibration.json", "hosted_by": "ollaya"},
            ox.file_entry("license", meta["repo"], meta["revision"], "LICENSE", os.path.join(run_dir, "LICENSE"))
            | {"note": "Apache-2.0 (see NOTICE); the base %s is Apache-2.0" % cfg["base_model"]},
        ],
        "weightless": {k: v for k, v in report.items() if k != "unused"},
        "unused_checkpoint_tensors": {k: len(v) for k, v in report["unused"].items()},
    }
    ox.write_json(os.path.join(out_dir, "decision.json"), decision)
    ox.write_json(os.path.join(out_dir, "calibration.json"), calibration)
    ox.write_json(os.path.join(out_dir, "files.json"), files)
    lay = D3Layout(tokenizers.Tokenizer.from_file(os.path.join(out_dir, "tokenizer.json")), decision)
    assert lay.encode(SAMPLE_STATE, SAMPLE_QUESTIONS) == rows[:-1], "layout port differs from upstream"
    print(json.dumps(files["weightless"]["stats"]), "inline bytes", files["weightless"]["inline_bytes"],
          "graph MB %.1f" % (files["layers"][0]["bytes"] / 2**20), "unused", files["unused_checkpoint_tensors"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(ref.MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default=None, help="local snapshot of the model repo")
    ap.add_argument("--fp16", action="store_true", help="fp16 compute in the trunk (the -fp16 GPU tags)")
    a = ap.parse_args()
    export(a.model, a.out, a.run or ref.snapshot(a.model), fp16=a.fp16)


if __name__ == "__main__":
    main()
