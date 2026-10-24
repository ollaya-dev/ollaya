"""Export d1-omni's text path to a weightless ONNX graph + tokenizer + decision/calibration config.

    uv run python -m ollaya_convert.families.d1_omni.export --out out/d1-omni-600m

Graph (contract M, as von and laya; layout `d1-omni-markers-v1`, see layout.py and docs/families/d1-omni.md):
    inputs   input_ids      int64 [rows, seq]      right-padded with the pad id
             attention_mask int64 [rows, seq]      1 on real tokens
             marker_pos     int64 [rows, markers]  position of each option's <|mask|>
             marker_mask    bool  [rows, markers]
             qtype          int64 [rows]           0 choice, 1 score, 2 noul (the head's type embedding)
    outputs  logits         float32 [rows, markers]  the scorer at each marker; masked slots -1e4

The network is upstream's own modules (`encoder.Trunk` with no media prefix, `encoder.DecisionHead`), fp32.
Weights are not copied: every initializer references the author's `model.safetensors` at its byte offset; the
vision and audio tensors stay unused. `model.safetensors` in the output directory is a hard link for local runs.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch

from ..llm_common import onnx_export as ox
from ...weightless_sharded import safetensors_source
from . import ref
from .layout import NOUL_DEFAULTS

LAYOUT = "d1-omni-markers-v1"
INPUT_NAMES = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
OUTPUT_NAMES = ["logits"]
MIN_MARKERS = 1


class Graph(torch.nn.Module):
    """`D1OmniModel._forward`'s text branch: embeddings, the trunk with no prefix, the head at the markers."""

    def __init__(self, m):
        super().__init__()
        self.encoder = m.encoder
        self.head = m.head

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        pad = attention_mask != 0
        h = self.encoder.embed_tokens(input_ids)
        h = self.encoder(h, pad, torch.zeros_like(qtype))
        return self.head(h, pad, marker_pos, marker_mask, qtype)


def sample_inputs(m, tok):
    """Three rows of different lengths and marker counts, so no axis is specialised."""
    state = {"subject": "Invoice #4411", "body": "We were billed twice for March, please refund the duplicate. " * 3}
    qs = {"dept": {"type": "choice", "instructions": "Which team handles this?",
                   "criteria": {"billing": "payments and invoices", "technical": "bugs", "sales": "", "other": ""}},
          "refund": {"type": "noul", "instructions": "The user asks for a refund."},
          "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["low", "medium", "high"]}}
    rows = ref.encode(m, state, qs)
    n, s, k = len(rows), max(len(r["ids"]) for r in rows), max(len(r["markers"]) for r in rows)
    ids = torch.full((n, s), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((n, s), dtype=torch.long)
    pos = torch.zeros((n, k), dtype=torch.long)
    mask = torch.zeros((n, k), dtype=torch.bool)
    for i, r in enumerate(rows):
        ids[i, : len(r["ids"])] = torch.tensor(r["ids"])
        att[i, : len(r["ids"])] = 1
        pos[i, : len(r["markers"])] = torch.tensor(r["markers"])
        mask[i, : len(r["markers"])] = True
    return ids, att, pos, mask, torch.tensor([ref.QTYPES[r["qtype"]] for r in rows])


def export(out_dir: str, root: str):
    m = ref.load(root, device="cpu")
    tok = m.tokenizer
    cfg = m.config
    # nn.TransformerEncoderLayer's fused fast path (`_transformer_encoder_layer_fwd`) is not exportable; the slow
    # path is the same arithmetic in standard ops.
    torch.backends.mha.set_fastpath_enabled(False)
    graph = Graph(m).eval()

    n = torch.export.Dim("rows", min=1, max=1024)
    s = torch.export.Dim("seq", min=8, max=cfg.max_length)
    k = torch.export.Dim("markers", min=1, max=255)
    dyn = {"input_ids": {0: n, 1: s}, "attention_mask": {0: n, 1: s}, "marker_pos": {0: n, 1: k},
           "marker_mask": {0: n, 1: k}, "qtype": {0: n}}
    tmp = ox.scratch_dir("d1-omni-export-")
    full = os.path.join(tmp, "model.onnx")
    secs = ox.export_graph(graph, sample_inputs(m, tok), INPUT_NAMES, OUTPUT_NAMES, dyn, full)
    print("exported in %.0fs" % secs)
    del graph, m

    ckpt = os.path.join(root, "model.safetensors")
    src = safetensors_source("model.safetensors", ckpt, repo=ref.REPO, revision=ref.REVISION, filename="model.safetensors")
    report = ox.weightless(tmp, out_dir, [src], [])
    ox.cleanup(tmp)

    shutil.copy(os.path.join(root, "tokenizer.json"), os.path.join(out_dir, "tokenizer.json"))
    ids_of = tok.convert_tokens_to_ids
    temps = {k_: float(v) for k_, v in cfg.temperatures.items()}
    decision = {
        "engine": "onnx",
        "family": "d1-omni",
        "layout": LAYOUT,
        "contract": "markers",
        "upstream": {"repo": ref.REPO, "revision": ref.REVISION, "base": ref.BASE,
                     "code": "modeling_d1.py, encoder.py and prompt.py in the model repo (text path only)"},
        "max_len": int(cfg.max_length),
        "special_tokens": {"bos": tok.bos_token_id, "pad": tok.pad_token_id, "mask": tok.mask_token_id,
                           "state": ids_of("<|reserved_7|>"), "question": ids_of("<|reserved_8|>"),
                           "option": ids_of("<|reserved_9|>"), "option_end": ids_of("<|reserved_10|>"),
                           "decide": ids_of("<|reserved_11|>"), "escape": "<|name|> -> <¦name¦>"},
        # prompt.encode: budget = max(min_budget, min(per_option * k + extra, max_len // 2)); per = max(min_per,
        # (budget - 3k) // k); the instructions keep at most max(min_question, budget) tokens (the Q marker included).
        "budget": {"per_option": 24, "extra": 32, "min_budget": 96, "min_question": 16, "min_per": 2},
        "noul": {"row_order": ["false", "true"], "default_false": NOUL_DEFAULTS[0], "default_true": NOUL_DEFAULTS[1]},
        "inputs": INPUT_NAMES,
        "outputs": OUTPUT_NAMES,
        "min_markers": MIN_MARKERS,
        "opset": ox.OPSET,
        "precision": "fp32 (the checkpoint's dtype)",
        "weights_in_memory": ox.weights_in_memory(report),
    }
    calibration = {
        # config.temperatures: Laya's buckets ("choice:3-5", …) with the type as the fallback (all 1.0 upstream).
        "temperature": [temps.get("choice", 1.0), temps.get("score", 1.0), temps.get("noul", 1.0)],
        "temperature_by_options": {k_: v for k_, v in temps.items() if ":" in k_},
        "source": "config.json temperatures, fitted upstream (text requests only; image and audio run uncalibrated)",
    }
    files = {
        "model": "d1-omni-600m",
        "layers": [
            {"role": "graph", "path": "model.onnx", "hosted_by": "ollaya",
             "bytes": os.path.getsize(os.path.join(out_dir, "model.onnx")),
             "sha256": ox.sha256_file(os.path.join(out_dir, "model.onnx"))},
            ox.file_entry("weights", ref.REPO, ref.REVISION, "model.safetensors", ckpt, location="model.safetensors"),
            ox.file_entry("tokenizer", ref.REPO, ref.REVISION, "tokenizer.json", os.path.join(root, "tokenizer.json")),
            {"role": "decision", "path": "decision.json", "hosted_by": "ollaya"},
            {"role": "calibration", "path": "calibration.json", "hosted_by": "ollaya"},
            ox.file_entry("license", ref.REPO, ref.REVISION, "LICENSE", os.path.join(root, "LICENSE"), verify=False)
            | {"note": "LFM Open License v1.0"},
        ],
        "weightless": {k_: v for k_, v in report.items() if k_ != "unused"},
        "unused_checkpoint_tensors": {k_: len(v) for k_, v in report["unused"].items()},
    }
    ox.write_json(os.path.join(out_dir, "decision.json"), decision)
    ox.write_json(os.path.join(out_dir, "calibration.json"), calibration)
    ox.write_json(os.path.join(out_dir, "files.json"), files)
    print(json.dumps(files["weightless"]["stats"]), "inline bytes", files["weightless"]["inline_bytes"],
          "graph MB %.1f" % (files["layers"][0]["bytes"] / 2**20), "unused", files["unused_checkpoint_tensors"])
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/d1-omni-600m")
    ap.add_argument("--root", default=None, help="local snapshot of the model repo (default: download pinned)")
    a = ap.parse_args()
    export(a.out, a.root or ref.snapshot())


if __name__ == "__main__":
    main()
