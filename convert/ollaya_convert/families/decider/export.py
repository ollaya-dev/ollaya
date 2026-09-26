"""Export Mapika/decider-{0.8b,2b,4b} to a weightless ONNX graph + tokenizer + decision/calibration config.

    uv run python -m ollaya_convert.families.decider.export decider-0.8b --out out/decider-0.8b

Graph (layout `decider-slots-v1`, see layout.py and docs/families/decider.md):
    inputs   input_ids    int64   [rows, seq]   seq a multiple of 64, right-padded (pad id is free)
             slot_pos     int64   [rows]        index of the row's "(" answer-slot token
    outputs  label_logits float32 [rows, 255]   hidden state at slot_pos . embed_tokens[label_ids]^T
                                                (the tied LM head restricted to the 255 option labels)

The backbone is the model's own Qwen3.5 weights, recomputed by llm_common.qwen35 (scan over 64-token
chunks for the Gated DeltaNet layers). Weights are not copied: every initializer references the
author's `model.safetensors` (BF16) at its byte offset, with a Cast to float32 folded by ONNX Runtime
at load. `model.safetensors` in the output directory is a hard link for local runs.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch

from ..llm_common import onnx_export as ox
from ..llm_common.qwen35 import CHUNK, Qwen35Trunk
from ...weightless_sharded import safetensors_source
from . import ref
from .layout import ISOLATED

INPUT_NAMES = ["input_ids", "slot_pos"]
OUTPUT_NAMES = ["label_logits"]


class DeciderGraph(torch.nn.Module):
    def __init__(self, lm, label_ids):
        super().__init__()
        self.trunk = Qwen35Trunk(lm.model)
        self.register_buffer("label_ids", torch.tensor(label_ids, dtype=torch.long), persistent=False)

    def forward(self, input_ids, slot_pos):
        h = self.trunk(input_ids)
        rows = torch.arange(h.shape[0], device=h.device)
        hs = h[rows, slot_pos]
        w = self.trunk.m.embed_tokens.weight[self.label_ids]      # tied LM head rows of the labels
        return hs @ w.transpose(0, 1)


def sample_inputs(tok):
    """Two rows of 128/192 tokens: every dynamic axis > 1 so nothing is specialised."""
    a = tok.encode("Context:\n" + "The customer was charged twice and wants a refund. " * 12, add_special_tokens=False)[:150]
    b = tok.encode("Context:\n" + "Checkout fails after the deploy. " * 20, add_special_tokens=False)[:100]
    T = 192
    ids = torch.full((2, T), tok.pad_token_id, dtype=torch.long)
    ids[0, :len(a)] = torch.tensor(a)
    ids[1, :len(b)] = torch.tensor(b)
    return ids, torch.tensor([len(a) - 1, len(b) - 1])


def export(slug: str, out_dir: str, root: str):
    from transformers import AutoModelForCausalLM

    meta = ref.MODELS[slug]
    sys.path.insert(0, root)
    from decider.prompt import NARROW, label_table

    cfg = json.load(open(os.path.join(root, "decider_config.json")))
    lm = AutoModelForCausalLM.from_pretrained(root, dtype=torch.float32).eval()
    assert lm.lm_head.weight.data_ptr() == lm.model.embed_tokens.weight.data_ptr(), "expected tied embeddings"
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(root)
    names, label_ids, open_ids = label_table(tok)
    graph = DeciderGraph(lm, label_ids).eval()

    R = torch.export.Dim("rows", min=1, max=4096)
    N = torch.export.Dim("chunks", min=1, max=4096)
    dyn = {"input_ids": {0: R, 1: CHUNK * N}, "slot_pos": {0: R}}
    tmp = ox.scratch_dir("decider-export-")
    full = os.path.join(tmp, "model.onnx")
    secs = ox.export_graph(graph, sample_inputs(tok), INPUT_NAMES, OUTPUT_NAMES, dyn, full)
    print("exported in %.0fs" % secs)
    del graph, lm

    ckpt = os.path.join(root, "model.safetensors")
    src = safetensors_source("model.safetensors", ckpt, repo=meta["repo"], revision=meta["revision"],
                             filename="model.safetensors")
    report = ox.weightless(tmp, out_dir, [src], [("trunk.m.", "model.language_model.")])
    ox.cleanup(tmp)

    shutil.copy(os.path.join(root, "tokenizer.json"), os.path.join(out_dir, "tokenizer.json"))
    # decider_config.json "temperature", and from decider-ai 1.4.0 an optional "temperature_by_type" map (choice, noul,
    # score; a missing type uses "temperature"). The score temperature divides each isolated level row's yes/no logits.
    T = float(cfg["temperature"])
    by_type = {k: float(v) for k, v in (cfg.get("temperature_by_type") or {}).items()}
    t_choice, t_noul, t_score = (by_type.get(k, T) for k in ("choice", "noul", "score"))
    decision = {
        "engine": "onnx",
        "family": "decider",
        "layout": "decider-slots-v1",
        "upstream": {"repo": meta["repo"], "revision": meta["revision"], "version": cfg.get("version"),
                     "base": meta["base"], "code": "decider/ in the model repo (inference subset of github.com/Mapika/decider)"},
        "contract": {
            "inputs": {
                "input_ids": {"dtype": "int64", "shape": ["rows", "seq"],
                              "note": "one row per scoring row; seq a multiple of 64; right-pad with any id (pad_id)"},
                "slot_pos": {"dtype": "int64", "shape": ["rows"], "note": "position of the row's answer-slot token"},
            },
            "outputs": {
                "label_logits": {"dtype": "float32", "shape": ["rows", 255],
                                 "note": "logits of labels[j] as the next token after the slot; use the first k"},
            },
            "seq_multiple": CHUNK,
            "positions": "0..seq-1, implicit",
            "attention": "causal; no mask input (right padding cannot reach earlier positions)",
        },
        "max_ctx_tokens": int(cfg.get("max_state_tokens", 32768)),
        "max_options": int(cfg.get("max_options", 255)),
        "min_options": 2,
        "max_levels": 10,
        "isolated_levels": bool(cfg.get("isolated_levels", False)),
        "isolated_row_temperature": t_score,
        "neutralize_none": bool(cfg.get("neutralize_none", True)),
        "independent_rows": True,
        "index_arrays_min_len": 8,
        "special_tokens": {"pad": tok.pad_token_id, "bos": None, "add_special_tokens": False},
        "labels": {"narrow": NARROW, "strings": names, "ids": label_ids, "open_ids": open_ids},
        "templates": {
            "context": "Context:\n{state}",
            "head": "\n\nQuestion: {question}\nOptions:",
            "narrow_option": "\n({label}) {option}",
            "wide_option": "enc(\"\\n(\") + [label_id] + enc(\") {option}\")",
            "tail": "\nAnswer: (",
            "isolated_question": ISOLATED,
            "choice_option": "{name} | {name}: {description}",
            "noul_options": ["no | no: {false}", "yes | yes: {true}"],
            "score_level_strip": "^\\s*-?\\d+\\s*:\\s*",
        },
        "option_logits": {
            "choice": "label_logits[row, :k]",
            "noul": "label_logits[row, :2]  (0 = no = false, 1 = yes = true)",
            "score": "log_sigmoid((label_logits[row_j, 1] - label_logits[row_j, 0]) / isolated_row_temperature), one row per level",
        },
        "opset": ox.OPSET,
        "precision": "fp32 compute; weights BF16 from the checkpoint, widened by Cast (at load, or per forward pass "
                     "with weights_in_memory bf16)",
        "weights_in_memory": ox.weights_in_memory(report),
    }
    calibration = {"temperature": [t_choice, 1.0, t_noul], "temperature_by_options": {},
                   "source": ("decider_config.json temperature_by_type (choice, noul; fitted upstream by NLL per answer type)"
                              if by_type else "decider_config.json temperature (fitted upstream on in-task data)")
                             + "; score is 1.0 because the isolated-row temperature is applied inside the score option logits"}
    files = {
        "model": slug,
        "layers": [
            {"role": "graph", "path": "model.onnx", "hosted_by": "ollaya", "bytes": os.path.getsize(os.path.join(out_dir, "model.onnx")),
             "sha256": ox.sha256_file(os.path.join(out_dir, "model.onnx"))},
            ox.file_entry("weights", meta["repo"], meta["revision"], "model.safetensors", ckpt, location="model.safetensors"),
            ox.file_entry("tokenizer", meta["repo"], meta["revision"], "tokenizer.json", os.path.join(root, "tokenizer.json")),
            {"role": "decision", "path": "decision.json", "hosted_by": "ollaya"},
            {"role": "calibration", "path": "calibration.json", "hosted_by": "ollaya"},
            ox.file_entry("license", meta["repo"], meta["revision"], "README.md", os.path.join(root, "README.md"), verify=False)
            | {"note": "Apache-2.0 per the model card; no LICENSE file in the repo"},
        ],
        "weightless": {k: v for k, v in report.items() if k != "unused"},
        "unused_checkpoint_tensors": {k: len(v) for k, v in report["unused"].items()},
    }
    ox.write_json(os.path.join(out_dir, "decision.json"), decision)
    ox.write_json(os.path.join(out_dir, "calibration.json"), calibration)
    ox.write_json(os.path.join(out_dir, "files.json"), files)
    print(json.dumps(files["weightless"]["stats"]), "inline bytes", files["weightless"]["inline_bytes"],
          "graph MB %.1f" % (files["layers"][0]["bytes"] / 2**20), "unused", files["unused_checkpoint_tensors"])
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(ref.MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None, help="local snapshot of the model repo (default: download pinned)")
    a = ap.parse_args()
    export(a.model, a.out, a.root or ref.snapshot(a.model))


if __name__ == "__main__":
    main()
