"""Export perplexity-ai/pplx-decider-flash-v1.2 (a post-trained Qwen3.5-4B + a 255-code readout) to a weightless ONNX graph.

    uv run python -m ollaya_convert.families.pplx.export pplx-decider-flash --out out/pplx-decider-flash

Graph (layout `pplx-joint-v1`, see layout.py and docs/families/pplx.md):
    inputs   input_ids  int64   [rows, seq]  one sequence per row (a request, or part of its questions);
                                             seq a multiple of 64, right-padded
             lengths    int64   [rows]       each row's real tokens: full attention reads only these keys
             markers    int64   [rows, q]    each question's `<|fim_suffix|>` position (pad with 0)
    outputs  logits     float32 [rows, q, 255]  readout(h[marker]): raw logits of the 255 answer codes

The full-attention layers attend to the whole row (upstream's `noncausal_full_attention`), the linear-attention
layers stay causal. The graph references the repository's two BF16 shards and `readout.safetensors` byte for byte.
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
from ..llm_common.qwen35 import CHUNK, Qwen35Trunk
from ...weightless_sharded import safetensors_source
from . import ref
from .layout import PplxLayout

INPUT_NAMES = ["input_ids", "lengths", "markers"]
OUTPUT_NAMES = ["logits"]
SENTINEL = "OLLAYA-USER-TEXT"
SAMPLE_STATE = "The customer was charged twice for order A-104 and wants the duplicate refunded. " * 6
SAMPLE_QUESTIONS = {"team": {"type": "choice", "instructions": "Which team?",
                             "criteria": {"billing": "charges", "tech": "bugs", "other": None}},
                    "upset": {"type": "score", "instructions": "How upset?", "criteria": ["calm", "annoyed", "furious"]},
                    "refund": {"type": "noul", "instructions": "Is a refund requested?"}}
# combining marks, where transformers' rebuilt Qwen tokenizer and some tokenizer.json files disagree
TOKENIZER_PROBES = ["नमस्ते दुनिया, यह एक परीक्षण है", "สวัสดีครับ ทดสอบ", "été café", "日本語のテキスト <|im_start|>"]


class PplxGraph(torch.nn.Module):
    def __init__(self, text_model, readout):
        super().__init__()
        self.trunk = Qwen35Trunk(text_model)
        self.readout = readout

    def forward(self, input_ids, lengths, markers):
        T = input_ids.shape[1]
        keys = torch.arange(T, device=input_ids.device)[None, :] < lengths[:, None]       # [R, T]
        bias = torch.zeros(keys.shape, dtype=torch.float32, device=input_ids.device).masked_fill(~keys, float("-inf"))
        h = self.trunk(input_ids, bias[:, None, None, :]).float()
        rows = torch.arange(h.shape[0], device=h.device).unsqueeze(1)
        return self.readout(h[rows, markers])                                             # [R, Q, 255]


def rename(name):
    """Graph initializer name -> checkpoint tensor names (backbone shards or readout)."""
    if name.startswith("trunk.m."):
        return ["language_model." + name[len("trunk.m."):]]
    if name == "readout.weight":
        return ["weight"]
    return [name]


def templates(processor, system_prompt):
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": SENTINEL}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pre, post = text.split(SENTINEL)
    return pre, post


def export(slug, out_dir, run_dir):
    meta = ref.MODELS[slug]
    decider = ref.load(run_dir, device="cpu")
    _, protocol = ref.package(run_dir)
    cfg = json.load(open(os.path.join(run_dir, "decision_config.json"), encoding="utf-8"))
    tok = decider.processor.tokenizer
    file_tok = tokenizers.Tokenizer.from_file(os.path.join(run_dir, "tokenizer.json"))
    for s in TOKENIZER_PROBES + [SAMPLE_STATE]:
        if file_tok.encode(s, add_special_tokens=False).ids != tok(s, add_special_tokens=True)["input_ids"]:
            raise SystemExit("tokenizer.json differs from upstream's tokenizer on %r" % s)

    graph = PplxGraph(decider.backbone.language_model, decider.readout).eval()
    # two rows (the sample and its first question alone), so every dynamic axis is > 1
    chunks = ref.encode(decider, SAMPLE_STATE, SAMPLE_QUESTIONS) + ref.encode(decider, SAMPLE_STATE, {"team": SAMPLE_QUESTIONS["team"]})
    rows = ref.rows(chunks)
    T = -(-max(len(r["ids"]) for r in rows) // CHUNK) * CHUNK
    Q = max(len(r["markers"]) for r in rows)
    ids = torch.full((len(rows), T), decider.encoder.pad_id, dtype=torch.long)
    markers = torch.zeros((len(rows), Q), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r["ids"])] = torch.tensor(r["ids"])
        markers[i, :len(r["markers"])] = torch.tensor(r["markers"])
    args = (ids, torch.tensor([len(r["ids"]) for r in rows]), markers)
    with torch.no_grad():
        got = graph(*args)
        diff = 0.0
        for i, (c, r) in enumerate(zip(chunks, rows)):
            for j, (want, n) in enumerate(zip(decider.reference_logits([c])[0], r["counts"])):
                diff = max(diff, float((got[i, j, :n] - torch.tensor(want)).abs().max()))
    print("eager graph (padded rows) vs upstream reference_logits: %.2e" % diff)

    R = torch.export.Dim("rows", min=1, max=4096)
    N = torch.export.Dim("chunks", min=1, max=4096)
    K = torch.export.Dim("questions", min=1, max=4096)
    dyn = {"input_ids": {0: R, 1: CHUNK * N}, "lengths": {0: R}, "markers": {0: R, 1: K}}
    tmp = ox.scratch_dir("pplx-export-")
    secs = ox.export_graph(graph, args, INPUT_NAMES, OUTPUT_NAMES, dyn, os.path.join(tmp, "model.onnx"))
    print("exported in %.0fs" % secs)
    pre, post = templates(decider.processor, protocol.SYSTEM_PROMPT)
    escapes = list(decider.encoder._escapes)
    marker = {"token": protocol.MARKER, "id": decider.encoder.marker_id}
    pad = decider.encoder.pad_id
    temperature = decider.temperature
    del graph, decider, got
    gc.collect()

    shards = [os.path.join(run_dir, f) for f in meta["shards"]]
    readout = os.path.join(run_dir, "readout.safetensors")
    sources = [safetensors_source(f, p, repo=meta["repo"], revision=meta["revision"], filename=f)
               for f, p in zip(meta["shards"], shards)]
    sources.append(safetensors_source("readout.safetensors", readout, repo=meta["repo"], revision=meta["revision"],
                                      filename="readout.safetensors"))
    report = ox.weightless(tmp, out_dir, sources, rename)
    ox.cleanup(tmp)
    shutil.copy(os.path.join(run_dir, "tokenizer.json"), os.path.join(out_dir, "tokenizer.json"))

    decision = {
        "engine": "onnx",
        "family": "pplx",
        "layout": "pplx-joint-v1",
        "upstream": {"repo": meta["repo"], "revision": meta["revision"], "base": cfg["base_model"],
                     "base_revision": cfg["revision"], "code": "source/src/pplx_decider (in the model repository)",
                     "protocol": json.load(open(os.path.join(run_dir, "joint_config.json"), encoding="utf-8"))},
        "contract": {
            "inputs": {
                "input_ids": {"dtype": "int64", "shape": ["rows", "seq"],
                              "note": "one sequence per row; seq a multiple of 64; right-pad with any id (pad)"},
                "lengths": {"dtype": "int64", "shape": ["rows"], "note": "each row's real tokens"},
                "markers": {"dtype": "int64", "shape": ["rows", "q"],
                            "note": "each question's marker position; rows with fewer questions pad with 0"},
            },
            "outputs": {"logits": {"dtype": "float32", "shape": ["rows", "q", 255],
                                   "note": "raw readout logits of the answer codes; a question reads its first k"}},
            "seq_multiple": CHUNK,
            "positions": "0..seq-1, implicit",
            "attention": "full-attention layers read the row's first `lengths` keys (not causal); "
                         "linear-attention layers causal",
        },
        "max_length": ref.MAX_LENGTH,
        "max_options": protocol.MAX_OPTIONS,
        "codes": cfg["codes"],
        "marker": marker,
        "escapes": escapes,
        "special_tokens": {"pad": pad, "add_special_tokens": False},
        "templates": {"pre": pre, "post": post, "default_instructions": protocol.DEFAULT_INSTRUCTIONS,
                      "noul_defaults": {"false": "No / false", "true": "Yes / true"},
                      "describe": "a string as is, else json.dumps(ensure_ascii=False)"},
        "option_logits": {"choice": "logits[row, q, :k]", "noul": "logits[row, q, :2] (0 = false, 1 = true)",
                          "score": "logits[row, q, :levels]"},
        "opset": ox.OPSET,
        "precision": "fp32 compute; weights BF16, widened by Cast (at load, or per forward pass with "
                     "weights_in_memory bf16)",
        "weights_in_memory": ox.weights_in_memory(report),
    }
    calibration = {"temperature": [float(temperature)] * 3, "temperature_by_options": {},
                   "source": "decision_config.json temperature (fitted upstream on held-out joint requests)"}
    files = {
        "model": slug,
        "layers": [
            {"role": "graph", "path": "model.onnx", "hosted_by": "ollaya",
             "bytes": os.path.getsize(os.path.join(out_dir, "model.onnx")),
             "sha256": ox.sha256_file(os.path.join(out_dir, "model.onnx"))},
            *[ox.file_entry("weights", meta["repo"], meta["revision"], f, p, location=f)
              for f, p in zip(meta["shards"], shards)],
            ox.file_entry("weights/head", meta["repo"], meta["revision"], "readout.safetensors", readout,
                          location="readout.safetensors"),
            ox.file_entry("tokenizer", meta["repo"], meta["revision"], "tokenizer.json", os.path.join(run_dir, "tokenizer.json")),
            {"role": "decision", "path": "decision.json", "hosted_by": "ollaya"},
            {"role": "calibration", "path": "calibration.json", "hosted_by": "ollaya"},
            ox.file_entry("license", meta["repo"], meta["revision"], "LICENSE", os.path.join(run_dir, "LICENSE"))
            | {"note": "Apache-2.0 (see NOTICE); the base Qwen3.5-4B is Apache-2.0"},
        ],
        "weightless": {k: v for k, v in report.items() if k != "unused"},
        "unused_checkpoint_tensors": {k: len(v) for k, v in report["unused"].items()},
    }
    ox.write_json(os.path.join(out_dir, "decision.json"), decision)
    ox.write_json(os.path.join(out_dir, "calibration.json"), calibration)
    ox.write_json(os.path.join(out_dir, "files.json"), files)
    lay = PplxLayout(tokenizers.Tokenizer.from_file(os.path.join(out_dir, "tokenizer.json")), decision)
    assert lay.encode(SAMPLE_STATE, SAMPLE_QUESTIONS) == rows[:1], "layout port differs from upstream"
    print(json.dumps(files["weightless"]["stats"]), "inline bytes", files["weightless"]["inline_bytes"],
          "graph MB %.1f" % (files["layers"][0]["bytes"] / 2**20), "unused", files["unused_checkpoint_tensors"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(ref.MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default=None, help="local snapshot of the model repo")
    a = ap.parse_args()
    export(a.model, a.out, a.run or ref.snapshot(a.model))


if __name__ == "__main__":
    main()
