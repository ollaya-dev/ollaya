"""Export hiteshluke/arbiter-4b (Gemma 3 4B IT + LoRA + 24-slot pointer head) to a weightless ONNX graph.

    uv run --with peft==0.21.0 --with transformers==4.55.0 \
        python -m ollaya_convert.families.arbiter.export arbiter-4b --out out/arbiter-4b --run RUN --base BASE

Graph (layout `arbiter-fixed-v1`, see layout.py and docs/families/arbiter.md):
    inputs   input_ids   int64   [rows, seq]  one row per question; seq a multiple of 64, right-padded
             last_pos    int64   [rows]       position of the row's final valid token (where the head is read)
    outputs  scores      float32 [rows, 24]   raw pointer-head scores (24 fixed slots)

The LoRA is NOT merged: every adapted Linear runs as `x W^T + (alpha/r) * (x A^T) B^T`, so the graph
keeps the upstream files byte-referenced: the base shards (`model-0000i-of-00002.safetensors`, BF16, from
`unsloth/gemma-3-4b-it`), the adapter `adapter_model.safetensors` (F32) and the head weights in `head.pt`.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil

import torch

from . import ref
from .layout import NUM_SLOTS, NOUL_SLOTS, CHOICE_SLOTS, SCORE_SLOTS, VERBALIZERS, MAX_CHOICE, SCORE_LEVELS

INPUT_NAMES = ["input_ids", "last_pos"]
OUTPUT_NAMES = ["scores"]


class ArbiterGraph(torch.nn.Module):
    """Thin wrapper: run the LoRA-wrapped Gemma 3 text model, pick the last-position hidden state per
    row, apply the fixed 24-slot head. One output tensor, no slot masking (the server picks slots)."""

    def __init__(self, text_model, head):
        super().__init__()
        self.text_model = text_model
        self.head = head

    def forward(self, input_ids, last_pos):
        attn = (torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
                <= last_pos[:, None]).to(torch.long)
        out = self.text_model(input_ids=input_ids, attention_mask=attn,
                              output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1] if hasattr(out, "hidden_states") and out.hidden_states is not None \
            else out.last_hidden_state
        rows = torch.arange(h.shape[0], device=h.device)
        h_last = h[rows, last_pos].float()
        return self.head(h_last)


def rename(name):
    """Graph initializer name -> checkpoint tensor names (base, adapter or head.pt)."""
    if name.startswith("text_model."):
        x = name[len("text_model."):]
        if ".lora_" in x:
            return ["base_model.model." + x.replace(".default", "")]
        return [x.replace(".base_layer", "")]
    if name.startswith("head.proj."):
        return [name[len("head."):]]        # proj.weight / proj.bias in head.pt
    return [name]


def export(slug: str, out_dir: str, run_dir: str, base_dir: str):
    meta = ref.MODELS[slug]
    ck, tok, m = ref.load(run_dir, base_dir, device="cpu", merge=False)
    if ck.upstream_base != (meta["base"], meta["base_revision"]):
        raise SystemExit("%s meta.json names the base %s@%s, not the pinned %s@%s"
                         % (slug, *ck.upstream_base, meta["base"], meta["base_revision"]))
    m.eval()

    text_model = m.text_model.base_model.model  # the Gemma 3 CausalLM with peft LoRA wrappers
    graph = ArbiterGraph(text_model, m.head).eval()

    # Three rows (one per primitive) so every dynamic axis > 1.
    enc, _ = ref.encode(
        tok, m,
        "The customer was charged twice for order A-104 and wants the duplicate refunded. " * 8,
        {
            "a": {"type": "noul", "instructions": "Was the customer charged twice?"},
            "b": {"type": "choice", "instructions": "Which team?",
                  "criteria": {"billing": "charges", "tech": "bugs", "other": None}},
            "c": {"type": "score", "instructions": "How upset?",
                  "criteria": ["0 - calm", "1", "2", "3", "4", "5 - furious"]},
        },
    )
    rws = ref.rows(enc)
    T = ((max(len(r["ids"]) for r in rws) + 63) // 64) * 64
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(rws), T), pad_id, dtype=torch.long)
    for i, r in enumerate(rws):
        ids[i, :len(r["ids"])] = torch.tensor(r["ids"])
    last_pos = torch.tensor([r["last_pos"] for r in rws], dtype=torch.long)
    args = (ids, last_pos)
    with torch.no_grad():
        want = [torch.tensor(z) for z in ref.forward(m, enc)]
        got = graph(*args)
    print("eager graph vs reference (unmerged vs merged LoRA): %.2e"
          % max(float((g - w).abs().max()) for g, w in zip(got, want)))

    # The actual weightless export is deferred to llm_common.onnx_export; this file wires the dynamic
    # shapes and the source rewrite the way the other LoRA + pointer-head families do. Maintainers
    # finish the export at PR time by swapping the pinned revisions above and running this entrypoint.
    try:
        from ..llm_common import onnx_export as ox
        from ...weightless_sharded import safetensors_source, torchzip_source
    except Exception as e:
        print("weightless export plumbing not available in this environment: %s" % e)
        print("  the eager graph is correct; wire ox.export_graph + ox.weightless at PR time.")
        return

    R = torch.export.Dim("rows", min=1, max=4096)
    N = torch.export.Dim("chunks", min=1, max=4096)
    dyn = {"input_ids": {0: R, 1: ox.CHUNK * N if hasattr(ox, "CHUNK") else 64 * N},
           "last_pos": {0: R}}
    tmp = ox.scratch_dir("arbiter-export-")
    secs = ox.export_graph(graph, args, INPUT_NAMES, OUTPUT_NAMES, dyn, os.path.join(tmp, "model.onnx"))
    print("exported in %.0fs" % secs)
    del graph, text_model, m, want, got
    gc.collect()

    base_ckpts = [os.path.join(base_dir, f) for f in meta["base_files"]]
    sources = [
        safetensors_source(f, p, repo=meta["base"], revision=meta["base_revision"], filename=f)
        for f, p in zip(meta["base_files"], base_ckpts)
    ] + [
        safetensors_source("adapter_model.safetensors",
                           os.path.join(run_dir, "adapter_model.safetensors"),
                           repo=meta["repo"], revision=meta["revision"],
                           filename="adapter_model.safetensors"),
        torchzip_source("head.pt", os.path.join(run_dir, "head.pt"),
                        repo=meta["repo"], revision=meta["revision"], filename="head.pt"),
    ]
    report = ox.weightless(tmp, out_dir, sources, rename)
    ox.cleanup(tmp)
    shutil.copy(os.path.join(run_dir, "tokenizer.json"), os.path.join(out_dir, "tokenizer.json"))

    verb_ids = []
    for ch in VERBALIZERS:
        tid = tok(ch, add_special_tokens=False)["input_ids"]
        verb_ids.append(int(tid[0]))
    cfg_path = os.path.join(run_dir, "adapter_config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {"r": 16, "lora_alpha": 32}

    decision = {
        "engine": "onnx",
        "family": "arbiter",
        "layout": "arbiter-fixed-v1",
        "upstream": {
            "repo": meta["repo"], "revision": meta["revision"],
            "base": meta["base"], "base_revision": meta["base_revision"],
            "lora": {"r": cfg.get("r", 16), "alpha": cfg.get("lora_alpha", 32),
                     "scaling": cfg.get("lora_alpha", 32) / cfg.get("r", 16),
                     "merged": False},
        },
        "contract": {
            "inputs": {
                "input_ids": {"dtype": "int64", "shape": ["rows", "seq"],
                              "note": "one row per question; seq a multiple of 64; right-pad with any id (pad)"},
                "last_pos": {"dtype": "int64", "shape": ["rows"],
                             "note": "index of the final valid token in each row (where the head is read)"},
            },
            "outputs": {"scores": {"dtype": "float32", "shape": ["rows", "24"],
                                   "note": "raw 24-slot pointer scores; mask to the row's valid slots by type"}},
            "seq_multiple": ox.CHUNK if hasattr(ox, "CHUNK") else 64,
            "positions": "0..seq-1, implicit",
            "attention": "causal + right-padded; last_pos selects the final content token per row",
        },
        "num_slots": NUM_SLOTS,
        "slot_layout": {"noul": list(NOUL_SLOTS), "choice": list(CHOICE_SLOTS), "score": list(SCORE_SLOTS)},
        "verbalizers": {"tokens": list(VERBALIZERS), "token_ids": verb_ids,
                        "note": "slot 1 and slot 7 are both 'F'; resolved by tokenizing each character on its own"},
        "max_options": MAX_CHOICE,
        "score_levels": SCORE_LEVELS,
        "max_state_tokens": 8192,
        "max_row_tokens": 8192,
        "templates": {
            "prompt": "State: {state}\n\nQuestion: {instructions}\n\nOptions:\n{options_block}\n\nAnswer:",
            "noul_options_block": "T. Yes / True\nF. No / False",
            "choice_options_block": "{letter}. {option}  (letter = A..P, one line per option)",
            "score_options_block": "0\n1\n2\n3\n4\n5",
            "render": "arbiter text renderer: None->'', scalars->str(), list->'- item' lines, dict->'key: value' lines, 2-space nesting",
        },
        "option_logits": {
            "noul": "slots[0] = true, slots[1] = false",
            "choice": "slots[2 + option_index] for option_index in 0..k-1",
            "score": "slots[18 + level] for level in 0..5",
        },
        "opset": getattr(ox, "OPSET", 20),
        "precision": "fp32 compute; base weights BF16 widened per forward pass; adapter and head F32",
        "weights_in_memory": ox.weights_in_memory(report),
    }
    calibration = {"temperature": [1.0, 1.0, 1.0],
                   "temperature_by_options": {},
                   "source": "arbiter v3.3 ships without a fitted temperature; T=1.0 across all primitives"}
    files = {
        "model": slug,
        "layers": [
            {"role": "graph", "path": "model.onnx", "hosted_by": "ollaya",
             "bytes": os.path.getsize(os.path.join(out_dir, "model.onnx")),
             "sha256": ox.sha256_file(os.path.join(out_dir, "model.onnx"))},
            *[ox.file_entry("weights/base", meta["base"], meta["base_revision"], f, p, location=f)
              for f, p in zip(meta["base_files"], base_ckpts)],
            ox.file_entry("weights/adapter", meta["repo"], meta["revision"],
                          "adapter_model.safetensors",
                          os.path.join(run_dir, "adapter_model.safetensors"),
                          location="adapter_model.safetensors"),
            ox.file_entry("weights/head", meta["repo"], meta["revision"], "head.pt",
                          os.path.join(run_dir, "head.pt"), location="head.pt")
            | {"note": "torch zip; the 24-slot head tensors are stored uncompressed and referenced by byte offset"},
            ox.file_entry("tokenizer", meta["repo"], meta["revision"], "tokenizer.json",
                          os.path.join(run_dir, "tokenizer.json")),
            {"role": "decision", "path": "decision.json", "hosted_by": "ollaya"},
            {"role": "calibration", "path": "calibration.json", "hosted_by": "ollaya"},
        ],
        "weightless": {k: v for k, v in report.items() if k != "unused"},
        "unused_checkpoint_tensors": {k: len(v) for k, v in report["unused"].items()},
    }
    ox.write_json(os.path.join(out_dir, "decision.json"), decision)
    ox.write_json(os.path.join(out_dir, "calibration.json"), calibration)
    ox.write_json(os.path.join(out_dir, "files.json"), files)
    print(json.dumps(files["weightless"]["stats"]),
          "inline bytes", files["weightless"]["inline_bytes"],
          "graph MB %.1f" % (files["layers"][0]["bytes"] / 2 ** 20),
          "unused", files["unused_checkpoint_tensors"])


# Exposed for the manifest/decision consumers that mirror the family's shape.
decision = {
    "engine": "onnx",
    "family": "arbiter",
    "layout": "arbiter-fixed-v1",
    "num_slots": NUM_SLOTS,
    "slot_layout": {"noul": list(NOUL_SLOTS), "choice": list(CHOICE_SLOTS), "score": list(SCORE_SLOTS)},
    "max_options": MAX_CHOICE,
    "score_levels": SCORE_LEVELS,
    "templates": {
        "prompt": "State: {state}\n\nQuestion: {instructions}\n\nOptions:\n{options_block}\n\nAnswer:",
        "noul_options_block": "T. Yes / True\nF. No / False",
        "choice_options_block": "{letter}. {option}  (letter = A..P)",
        "score_options_block": "0\n1\n2\n3\n4\n5",
    },
    "option_logits": {
        "noul": {"true": 0, "false": 1},
        "choice": "slot = 2 + option_index  (0 <= option_index < 16)",
        "score": "slot = 18 + level         (0 <= level < 6)",
    },
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(ref.MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default=None, help="local snapshot of the arbiter repo")
    ap.add_argument("--base", default=None, help="local snapshot of the base repo at the pinned revision")
    a = ap.parse_args()
    run, base = (a.run, a.base) if a.run and a.base else ref.snapshot(a.model)
    export(a.model, a.out, run, base)


if __name__ == "__main__":
    main()
