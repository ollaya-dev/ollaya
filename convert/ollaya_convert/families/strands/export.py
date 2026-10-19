"""Export StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 (Qwen3.5-2B-Base + LoRA + pointer head) to a weightless ONNX graph.

    STRANDS_SRC=/path/to/strands-decider/src uv run --with peft==0.21.0 --with pydantic==2.12.5 \
        python -m ollaya_convert.families.strands.export strands-decider-2b --out out/strands-decider-2b

Graph (layout `strands-pointer-v1`, see layout.py and docs/families/strands.md):
    inputs   input_ids   int64   [rows, seq]  one row per question; seq a multiple of 64, right-padded
             decide_pos  int64   [rows]       position of the row's last token (`<answer>`)
             opt_pos     int64   [rows, k]    positions of the last token of each option line (pad with 0)
    outputs  scores      float32 [rows, k]    raw pointer scores (k(norm(h_opt)) . q(norm(h_decide))) / sqrt(768)

The LoRA is NOT merged: every adapted Linear runs as `x W^T + 2.0 * (x A^T) B^T`, so the graph references
the upstream files byte for byte: the base `model.safetensors-00001-of-00001.safetensors` (BF16,
Qwen/Qwen3.5-2B-Base), the adapter `lora/adapter_model.safetensors` (F32) and `head.safetensors` (F32).
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
from ..llm_common.qwen35 import CHUNK, Qwen35Trunk, half_trunk
from ...weightless_sharded import safetensors_source
from . import ref
from .layout import KINDS, StrandsLayout

INPUT_NAMES = ["input_ids", "decide_pos", "opt_pos"]
OUTPUT_NAMES = ["scores"]
SAMPLE_STATE = "The customer was charged twice for order A-104 and wants the duplicate refunded. " * 8
SAMPLE_QUESTIONS = {"team": {"type": "choice", "instructions": "Which team?",
                             "criteria": {"billing": "charges", "tech": "bugs", "other": None}},
                    "upset": {"type": "score", "instructions": "How upset?", "criteria": ["calm", "annoyed", "furious"]}}


class StrandsGraph(torch.nn.Module):
    def __init__(self, text_model, head):
        super().__init__()
        self.trunk = Qwen35Trunk(text_model)
        self.head = head

    def forward(self, input_ids, decide_pos, opt_pos):
        h = self.trunk(input_ids).float()
        rows = torch.arange(h.shape[0], device=h.device)
        q = self.head.q(self.head.norm(h[rows, decide_pos]))                  # [R, P]
        k = self.head.k(self.head.norm(h[rows.unsqueeze(1), opt_pos]))        # [R, K, P]
        return (k @ q.unsqueeze(-1)).squeeze(-1) * self.head.scale


def rename(name):
    """Graph initializer name -> checkpoint tensor names (base, adapter or head)."""
    if name.startswith("trunk.m."):
        x = name[len("trunk.m."):]
        if ".lora_" in x:
            return ["base_model.model." + x.replace(".default", "")]
        return ["model.language_model." + x.replace(".base_layer", "")]
    if name.startswith("head."):
        return [name[len("head."):]]
    return [name]


def export(slug, out_dir, run_dir, base_dir, fp16=False):
    meta = ref.MODELS[slug]
    model, engine = ref.load(run_dir, device="cpu")
    cfg = model.config
    if (cfg.base_model, cfg.base_revision) != (meta["base"], meta["base_revision"]):
        raise SystemExit("%s names the base %s@%s, not the pinned %s@%s"
                         % (slug, cfg.base_model, cfg.base_revision, meta["base"], meta["base_revision"]))
    if cfg.head_type != "pointer":
        raise SystemExit("%s has a %s head; this layout reads a pointer head" % (slug, cfg.head_type))
    model.eval()
    lora_model = model.torso.base_model.model   # Qwen3_5TextModel with peft LoRA Linear wrappers
    graph = StrandsGraph(lora_model, model.head).eval()

    # two rows, three options each: every dynamic axis > 1
    rows, _, _ = ref.encode(engine, SAMPLE_STATE, SAMPLE_QUESTIONS)
    T = -(-max(len(r["ids"]) for r in rows) // CHUNK) * CHUNK
    ids = torch.full((len(rows), T), engine.tok.pad_token_id, dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r["ids"])] = torch.tensor(r["ids"])
    args = (ids, torch.tensor([r["decide"] for r in rows]), torch.tensor([r["opts"] for r in rows]))
    with torch.no_grad():
        want = [torch.tensor(z) for z in ref.forward(model, rows)]
    if fp16:
        half_trunk(graph.trunk)   # after the fp32 reference: the trunk shares upstream's modules; the head stays fp32
    with torch.no_grad():
        got = graph(*args)
    print("eager graph vs upstream: %.2e" % max(float((g - w).abs().max()) for g, w in zip(got, want)))

    R = torch.export.Dim("rows", min=1, max=4096)
    N = torch.export.Dim("chunks", min=1, max=4096)
    K = torch.export.Dim("options", min=1, max=255)
    dyn = {"input_ids": {0: R, 1: CHUNK * N}, "decide_pos": {0: R}, "opt_pos": {0: R, 1: K}}
    tmp = ox.scratch_dir("strands-export-")
    secs = ox.export_graph(graph, args, INPUT_NAMES, OUTPUT_NAMES, dyn, os.path.join(tmp, "model.onnx"))
    print("exported in %.0fs" % secs)
    tok = engine.tok
    sample = "<state>\nx\n</state>\n"
    if tok(sample, add_special_tokens=True)["input_ids"] != tok(sample, add_special_tokens=False)["input_ids"]:
        raise SystemExit("the tokenizer adds special tokens to the state; the layout assumes it adds none")
    del graph, lora_model, model, engine, want, got
    gc.collect()

    base_ckpts = [os.path.join(base_dir, f) for f in meta["base_files"]]
    adapter = os.path.join(run_dir, "lora", "adapter_model.safetensors")
    head = os.path.join(run_dir, "head.safetensors")
    sources = [
        safetensors_source(f, p, repo=meta["base"], revision=meta["base_revision"], filename=f)
        for f, p in zip(meta["base_files"], base_ckpts)
    ] + [
        safetensors_source("adapter_model.safetensors", adapter, repo=meta["repo"], revision=meta["revision"],
                           filename="lora/adapter_model.safetensors"),
        safetensors_source("head.safetensors", head, repo=meta["repo"], revision=meta["revision"],
                           filename="head.safetensors"),
    ]
    report = ox.weightless(tmp, out_dir, sources, rename)
    ox.cleanup(tmp)
    shutil.copy(os.path.join(run_dir, "tokenizer.json"), os.path.join(out_dir, "tokenizer.json"))

    _, _, prompting, schema = ref._import()
    lora = json.load(open(os.path.join(run_dir, "lora", "adapter_config.json"), encoding="utf-8"))
    by_kind = cfg.temperature_by_kind or {}
    decision = {
        "engine": "onnx",
        "family": "strands",
        "layout": "strands-pointer-v1",
        "upstream": {"repo": meta["repo"], "revision": meta["revision"], "base": meta["base"],
                     "base_revision": meta["base_revision"], "code": ref.STRANDS_GIT,
                     "lora": {"r": lora["r"], "alpha": lora["lora_alpha"], "scaling": lora["lora_alpha"] / lora["r"],
                              "merged": False}},
        "contract": {
            "inputs": {
                "input_ids": {"dtype": "int64", "shape": ["rows", "seq"],
                              "note": "one row per question; seq a multiple of 64; right-pad with any id (pad)"},
                "decide_pos": {"dtype": "int64", "shape": ["rows"], "note": "position of the row's last token"},
                "opt_pos": {"dtype": "int64", "shape": ["rows", "k"],
                            "note": "position of the last token of each option line; rows with fewer options pad with 0"},
            },
            "outputs": {"scores": {"dtype": "float32", "shape": ["rows", "k"],
                                   "note": "raw pointer scores; the first k_row entries are the row's option logits"}},
            "seq_multiple": CHUNK,
            "positions": "0..seq-1, implicit",
            "attention": "causal; no mask input (right padding cannot reach earlier positions)",
        },
        # upstream EngineConfig defaults (max_batch, max_question_fraction) and the checkpoint's window
        "max_length": cfg.max_length,
        "max_question_fraction": 0.75,
        "max_batch": 32,
        "choice_options": [2, schema.MAX_CHOICE_OPTIONS],
        "score_levels": [schema.MIN_SCORE_LEVELS, schema.MAX_SCORE_LEVELS],
        "special_tokens": {"pad": tok.pad_token_id, "add_special_tokens": False},
        "templates": {
            "state": "<state>\\n{render(state)}\\n</state>\\n",
            "question": "<question type=\"{type}\">\\n{header}\\n{render(instructions)}\\n<options>\\n{lines}\\n"
                        "</options>\\n</question>\\n<answer>",
            "line": "{n}. {name} | {n}. {name} — {collapse(text)}",
            "headers": {"noul": "Decide whether the statement is true of the state.",
                        "choice": "Select exactly one option.",
                        "score": "Rate the state against the ordered levels below (lowest first)."},
            "noul_defaults": dict(prompting.NOUL_DEFAULT_CRITERIA),
            "render": "strands_decider.prompting.render_content: None->'', str->strip(), "
                      "else json.dumps(indent=2, ensure_ascii=False)",
        },
        "option_logits": {"choice": "scores[row, :k]", "noul": "scores[row, :2] (0 = false, 1 = true)",
                          "score": "scores[row, :levels]"},
        "opset": ox.OPSET,
        "precision": ox.FP16_PRECISION if fp16 else "fp32 compute; base weights BF16, widened by Cast (at load, or per forward pass with "
                     "weights_in_memory bf16); adapter and head F32",
        "weights_in_memory": ox.weights_in_memory(report),
    }
    calibration = {"temperature": [float(by_kind.get(k, cfg.temperature)) for k in KINDS], "temperature_by_options": {},
                   "source": "strands_decider_config.json temperature_by_kind (fitted upstream per question type; "
                             "order choice, score, noul)"}
    files = {
        "model": slug,
        "layers": [
            {"role": "graph", "path": "model.onnx", "hosted_by": "ollaya", "bytes": os.path.getsize(os.path.join(out_dir, "model.onnx")),
             "sha256": ox.sha256_file(os.path.join(out_dir, "model.onnx"))},
            *[ox.file_entry("weights/base", meta["base"], meta["base_revision"], f, p, location=f)
              for f, p in zip(meta["base_files"], base_ckpts)],
            ox.file_entry("weights/adapter", meta["repo"], meta["revision"], "lora/adapter_model.safetensors", adapter,
                          location="adapter_model.safetensors"),
            ox.file_entry("weights/head", meta["repo"], meta["revision"], "head.safetensors", head, location="head.safetensors"),
            ox.file_entry("tokenizer", meta["repo"], meta["revision"], "tokenizer.json", os.path.join(run_dir, "tokenizer.json")),
            {"role": "decision", "path": "decision.json", "hosted_by": "ollaya"},
            {"role": "calibration", "path": "calibration.json", "hosted_by": "ollaya"},
            ox.file_entry("license", meta["repo"], meta["revision"], "LICENSE.md", os.path.join(run_dir, "LICENSE.md"), verify=False)
            | {"note": "Apache-2.0 (adapter and head); the base %s is Apache-2.0" % meta["base"].split("/")[-1]},
        ],
        "weightless": {k: v for k, v in report.items() if k != "unused"},
        "unused_checkpoint_tensors": {k: len(v) for k, v in report["unused"].items()},
    }
    ox.write_json(os.path.join(out_dir, "decision.json"), decision)
    ox.write_json(os.path.join(out_dir, "calibration.json"), calibration)
    ox.write_json(os.path.join(out_dir, "files.json"), files)
    # the layout port must reproduce upstream's rows for the sample request
    lay = StrandsLayout(tokenizers.Tokenizer.from_file(os.path.join(out_dir, "tokenizer.json")), decision)
    assert lay.encode(SAMPLE_STATE, SAMPLE_QUESTIONS)[0] == rows, "layout port differs from upstream"
    print(json.dumps(files["weightless"]["stats"]), "inline bytes", files["weightless"]["inline_bytes"],
          "graph MB %.1f" % (files["layers"][0]["bytes"] / 2**20), "unused", files["unused_checkpoint_tensors"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(ref.MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default=None, help="local snapshot of the model repo")
    ap.add_argument("--base", default=None, help="local snapshot of the base repo at the pinned revision")
    ap.add_argument("--fp16", action="store_true", help="fp16 compute in the trunk (the -fp16 GPU tags)")
    a = ap.parse_args()
    run, base = (a.run, a.base) if a.run and a.base else ref.snapshot(a.model)
    export(a.model, a.out, run, base, fp16=a.fp16)


if __name__ == "__main__":
    main()
