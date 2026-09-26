"""Export jaredpalmer/kev-{0.8b,4b,9b} (Qwen3.5-Base + LoRA + pointer head) to a weightless ONNX graph.

    KEV_SRC=/path/to/kev uv run --with peft==0.21.0 --with pydantic==2.12.5 \
        python -m ollaya_convert.families.kev.export kev-0.8b --out out/kev-0.8b --run RUN --base BASE

Graph (layout `kev-pointer-v1`, see layout.py and docs/families/kev.md):
    inputs   input_ids   int64   [rows, seq]  one row per question; seq a multiple of 64, right-padded
             decide_pos  int64   [rows]       position of the row's <|fim_suffix|> (decide) token
             opt_pos     int64   [rows, k]    positions of the option-closing <|box_end|> tokens (pad with 0)
    outputs  scores      float32 [rows, k]    raw pointer-head scores (k(h_opt) . q(h_decide)) / 16

The LoRA is NOT merged: every adapted Linear runs as `x W^T + 2.0 * (x A^T) B^T`, so the graph keeps
the upstream files byte-referenced: the base shards `model.safetensors-0000i-of-0000n.safetensors` (BF16,
Qwen/Qwen3.5-*-Base), the adapter `adapter_model.safetensors` (F32) and the pointer head, which lives
in `head.pt` (a torch zip whose tensors are stored uncompressed, so they have byte offsets too).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil

import torch

from ..llm_common import onnx_export as ox
from ..llm_common.qwen35 import CHUNK, Qwen35Trunk
from ...weightless_sharded import safetensors_source, torchzip_source
from . import ref
from .layout import MAX_OPTIONS, SPECIAL

INPUT_NAMES = ["input_ids", "decide_pos", "opt_pos"]
OUTPUT_NAMES = ["scores"]


class KevGraph(torch.nn.Module):
    def __init__(self, text_model, head):
        super().__init__()
        self.trunk = Qwen35Trunk(text_model)
        self.head = head

    def forward(self, input_ids, decide_pos, opt_pos):
        h = self.trunk(input_ids).float()
        rows = torch.arange(h.shape[0], device=h.device)
        hd = h[rows, decide_pos]                      # [R, H]
        ho = h[rows.unsqueeze(1), opt_pos]            # [R, K, H]
        q = self.head.q(hd)                           # [R, P]
        k = self.head.k(ho)                           # [R, K, P]
        return (k @ q.unsqueeze(-1)).squeeze(-1) * self.head.scale


def rename(name):
    """Graph initializer name -> checkpoint tensor names (base, adapter or head.pt)."""
    if name.startswith("trunk.m."):
        x = name[len("trunk.m."):]
        if ".lora_" in x:
            return ["base_model.model." + x.replace(".default", "")]
        return ["model.language_model." + x.replace(".base_layer", "")]
    return [name]


def export(slug, out_dir, run_dir, base_dir):
    meta = ref.MODELS[slug]
    ck, tok, m = ref.load(run_dir, base_dir, device="cpu", merge=False)
    if ck.upstream_base != (meta["base"], meta["base_revision"]):
        raise SystemExit("%s head.pt names the base %s@%s, not the pinned %s@%s"
                         % (slug, *ck.upstream_base, meta["base"], meta["base_revision"]))
    m.eval()
    temperature = float(ck.meta.temperature)
    lora_model = m.lm.base_model.model  # Qwen3_5TextModel with peft LoRA Linear wrappers
    graph = KevGraph(lora_model, m.head).eval()

    # two rows (128 / 192 tokens), three options each: every dynamic axis > 1
    enc, _ = ref.encode(tok, m, "The customer was charged twice for order A-104 and wants the duplicate refunded. " * 8,
                        {"a": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "charges", "tech": "bugs", "other": None}},
                         "b": {"type": "score", "instructions": "How upset?", "criteria": ["calm", "annoyed", "furious"]}})
    rws = ref.rows(enc)
    T = 256
    ids = torch.full((2, T), tok.pad_token_id, dtype=torch.long)
    for i, r in enumerate(rws):
        ids[i, :len(r["ids"])] = torch.tensor(r["ids"])
    args = (ids, torch.tensor([r["decide"] for r in rws]), torch.tensor([r["opts"] for r in rws]))
    with torch.no_grad():
        want = [torch.tensor(z) for z in ref.forward(m, enc)]
        got = graph(*args)
    print("eager graph vs upstream (unmerged vs merged LoRA): %.2e" % max(float((g - w).abs().max()) for g, w in zip(got, want)))

    R = torch.export.Dim("rows", min=1, max=4096)
    N = torch.export.Dim("chunks", min=1, max=4096)
    K = torch.export.Dim("options", min=1, max=MAX_OPTIONS)
    dyn = {"input_ids": {0: R, 1: CHUNK * N}, "decide_pos": {0: R}, "opt_pos": {0: R, 1: K}}
    tmp = ox.scratch_dir("kev-export-")
    secs = ox.export_graph(graph, args, INPUT_NAMES, OUTPUT_NAMES, dyn, os.path.join(tmp, "model.onnx"))
    print("exported in %.0fs" % secs)
    del graph, lora_model, m, want, got   # the fp32 model (36 GB for a 9B) is not needed for the rewrite
    gc.collect()

    base_ckpts = [os.path.join(base_dir, f) for f in meta["base_files"]]
    sources = [
        safetensors_source(f, p, repo=meta["base"], revision=meta["base_revision"], filename=f)
        for f, p in zip(meta["base_files"], base_ckpts)
    ] + [
        safetensors_source("adapter_model.safetensors", os.path.join(run_dir, "adapter_model.safetensors"),
                           repo=meta["repo"], revision=meta["revision"], filename="adapter_model.safetensors"),
        torchzip_source("head.pt", os.path.join(run_dir, "head.pt"), repo=meta["repo"], revision=meta["revision"], filename="head.pt"),
    ]
    report = ox.weightless(tmp, out_dir, sources, rename)
    ox.cleanup(tmp)
    shutil.copy(os.path.join(run_dir, "tokenizer.json"), os.path.join(out_dir, "tokenizer.json"))

    sp_ids = [tok.convert_tokens_to_ids(t) for t in SPECIAL]
    cfg = json.load(open(os.path.join(run_dir, "adapter_config.json")))
    decision = {
        "engine": "onnx",
        "family": "kev",
        "layout": "kev-pointer-v1",
        "upstream": {"repo": meta["repo"], "revision": meta["revision"], "base": meta["base"],
                     "base_revision": meta["base_revision"], "code": ref.KEV_GIT,
                     "lora": {"r": cfg["r"], "alpha": cfg["lora_alpha"], "scaling": cfg["lora_alpha"] / cfg["r"],
                              "merged": False}},
        "contract": {
            "inputs": {
                "input_ids": {"dtype": "int64", "shape": ["rows", "seq"],
                              "note": "one row per question; seq a multiple of 64; right-pad with any id (pad)"},
                "decide_pos": {"dtype": "int64", "shape": ["rows"], "note": "position of the decide token"},
                "opt_pos": {"dtype": "int64", "shape": ["rows", "k"],
                            "note": "positions of each option's closing token; rows with fewer options pad with 0"},
            },
            "outputs": {"scores": {"dtype": "float32", "shape": ["rows", "k"],
                                   "note": "raw pointer scores; the first k_row entries are the row's option logits"}},
            "seq_multiple": CHUNK,
            "positions": "0..seq-1, implicit",
            "attention": "causal; no mask input (right padding cannot reach earlier positions)",
        },
        "max_state_tokens": 8192,
        "max_row_tokens": 8192,
        "min_options": 1,
        "max_options": MAX_OPTIONS,
        "special_tokens": {"state": sp_ids[0], "question": sp_ids[1], "option_open": sp_ids[2],
                           "option_close": sp_ids[3], "decide": sp_ids[4], "pad": tok.pad_token_id,
                           "texts": SPECIAL, "add_special_tokens": False},
        "user_text_escape": {"pattern": "<\\|([A-Za-z0-9_]+)\\|>", "replace": "<¦$1¦>"},
        "templates": {
            "row": "[state] user(render(state))[:max_state-1] [question] user(render(instructions)) "
                   "([option_open] user(option) [option_close])* [decide]",
            "choice_option": "{name} | {name}: {render(description)}",
            "noul_options": ["no | no: {render(false)}", "yes | yes: {render(true)}"],
            "score_option": "{render(level)}",
            "render": "kev.api.render: None->'', scalars->Python str(), list->'- item' lines, dict->'key: value' lines, 2-space nesting",
        },
        "option_logits": {"choice": "scores[row, :k]", "noul": "scores[row, :2] (0 = no = false, 1 = yes = true)",
                          "score": "scores[row, :levels]"},
        "opset": ox.OPSET,
        "precision": "fp32 compute; base weights BF16, widened by Cast (at load, or per forward pass with "
                     "weights_in_memory bf16); adapter and head F32",
        "weights_in_memory": ox.weights_in_memory(report),
    }
    calibration = {"temperature": [temperature] * 3, "temperature_by_options": {},
                   "source": "head.pt['temperature'] (fitted upstream on in-distribution development rows, scripts/calibrate_checkpoint.py)"}
    files = {
        "model": slug,
        "layers": [
            {"role": "graph", "path": "model.onnx", "hosted_by": "ollaya", "bytes": os.path.getsize(os.path.join(out_dir, "model.onnx")),
             "sha256": ox.sha256_file(os.path.join(out_dir, "model.onnx"))},
            *[ox.file_entry("weights/base", meta["base"], meta["base_revision"], f, p, location=f)
              for f, p in zip(meta["base_files"], base_ckpts)],
            ox.file_entry("weights/adapter", meta["repo"], meta["revision"], "adapter_model.safetensors",
                          os.path.join(run_dir, "adapter_model.safetensors"), location="adapter_model.safetensors"),
            ox.file_entry("weights/head", meta["repo"], meta["revision"], "head.pt", os.path.join(run_dir, "head.pt"), location="head.pt")
            | {"note": "torch zip; the pointer-head tensors are stored uncompressed and referenced by byte offset"},
            ox.file_entry("tokenizer", meta["repo"], meta["revision"], "tokenizer.json", os.path.join(run_dir, "tokenizer.json")),
            {"role": "decision", "path": "decision.json", "hosted_by": "ollaya"},
            {"role": "calibration", "path": "calibration.json", "hosted_by": "ollaya"},
            ox.file_entry("license", meta["repo"], meta["revision"], "README.md", os.path.join(run_dir, "README.md"), verify=False)
            | {"note": "Apache-2.0 per the model card (adapter + head); base %s is Apache-2.0 with a LICENSE file"
                       % meta["base"].split("/")[-1]},
        ],
        "weightless": {k: v for k, v in report.items() if k != "unused"},
        "unused_checkpoint_tensors": {k: len(v) for k, v in report["unused"].items()},
    }
    ox.write_json(os.path.join(out_dir, "decision.json"), decision)
    ox.write_json(os.path.join(out_dir, "calibration.json"), calibration)
    ox.write_json(os.path.join(out_dir, "files.json"), files)
    print(json.dumps(files["weightless"]["stats"]), "inline bytes", files["weightless"]["inline_bytes"],
          "graph MB %.1f" % (files["layers"][0]["bytes"] / 2**20), "unused", files["unused_checkpoint_tensors"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(ref.MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default=None, help="local snapshot of the kev repo")
    ap.add_argument("--base", default=None, help="local snapshot of the base repo at the pinned revision")
    a = ap.parse_args()
    run, base = (a.run, a.base) if a.run and a.base else ref.snapshot(a.model)
    export(a.model, a.out, run, base)


if __name__ == "__main__":
    main()
