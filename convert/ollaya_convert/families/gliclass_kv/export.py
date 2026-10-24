"""Export GLiClass v4 (decoder-KV: Qwen3.5 decoder + DeBERTa scorer + recurrent cell + calibrator) to a weightless ONNX graph.

    uv run --with gliclass==0.1.21 python -m ollaya_convert.families.gliclass_kv.export gliclass-v4-2b --out out/gliclass-v4-2b

Graph (layout `gliclass-kv-v1`, see layout.py and docs/families/gliclass-kv.md):
    inputs   input_ids    int64 [rows, seq]      one row per question; seq a multiple of 64, right-padded (pad id is free)
             section_pos  int64 [rows, len]      positions of the label section's tokens (after the <<SEP>> that follows
                                                the context, through the final <<SEP>>); shorter rows repeat their last
             section_mask bool  [rows, len]
             label_pos    int64 [rows, k]        each <<LABEL>>'s index within the section
             label_mask   bool  [rows, k]
             sep_pos      int64 [rows]           the final <<SEP>>'s index within the section (its last token)
    outputs  logits       float32 [rows, k]      calibrated label logits (beta * z + bias); masked slots -1e4
             steps        int64   [rows]         recurrent steps the halting rule ran (1..recurrent_steps)

The decoder is `llm_common.qwen35` (the scan-based Gated-DeltaNet trunk) over the whole row, causal. The label
section's hidden states go through upstream's `DecoderKVScorer`: the DeBERTa encoder, the text representation at the
final <<SEP>>, each label's at its <<LABEL>>, the MLP; then `recurrent_steps - 1` gated steps of the same encoder,
each row frozen once no label's sigmoid probability moves by `recurrent_halt_threshold` (upstream's halting, as
`torch.where`); then the calibrator. Weights are not copied: every initializer references the repo's
`model.safetensors` (decoder and scorer BF16, calibrator F32) at its byte offset; the vision tower stays unused.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch

from ..llm_common import onnx_export as ox
from ..llm_common.qwen35 import CHUNK, Qwen35Trunk
from ...weightless_sharded import safetensors_source
from . import ref
from .layout import GliclassKvLayout

LAYOUT = "gliclass-kv-v1"
INPUT_NAMES = ["input_ids", "section_pos", "section_mask", "label_pos", "label_mask", "sep_pos"]
OUTPUT_NAMES = ["logits", "steps"]


class Graph(torch.nn.Module):
    def __init__(self, gl):
        super().__init__()
        # `model` first, so the shared decoder parameters keep the checkpoint's names (`model.decoder_model.…`).
        self.model = gl.model
        self.calibrator = gl.calibrator
        self.trunk = Qwen35Trunk(gl.model.decoder_model.language_model)
        cfg = gl.config
        self.steps = int(cfg.recurrent_steps)
        self.halt = float(cfg.recurrent_halt_threshold)

    def score(self, state, rows, label_pos, sep_pos):
        sc = self.model.scorer
        text = sc.text_projector(state[rows, sep_pos])                       # [R, D]
        labels = sc.label_projector(state[rows[:, None], label_pos])          # [R, K, D]
        combined = torch.cat([text[:, None, :].expand_as(labels), labels], dim=-1)
        return sc.mlp(combined).squeeze(-1), text, labels

    def forward(self, input_ids, section_pos, section_mask, label_pos, label_mask, sep_pos):
        sc = self.model.scorer
        h = self.trunk(input_ids).float()
        rows = torch.arange(h.shape[0], device=h.device)
        sec = h[rows[:, None], section_pos]                                   # [R, L, D]
        mask = section_mask.to(torch.int64)
        state = sc._encode(sec, mask)
        logits, text, labels = self.score(state, rows, label_pos, sep_pos)
        inputs = sc.reasoning_cell.normalize(sec)
        active = torch.ones_like(sep_pos, dtype=torch.bool)
        steps = torch.ones_like(sep_pos)
        for _ in range(1, self.steps):
            new_state = sc.reasoning_cell(state, inputs, mask, sc._encode)
            new_logits, new_text, new_labels = self.score(new_state, rows, label_pos, sep_pos)
            change = ((torch.sigmoid(new_logits) - torch.sigmoid(logits)).abs()
                      .masked_fill(~label_mask, 0.0).amax(dim=-1))
            state = torch.where(active[:, None, None], new_state, state)
            logits = torch.where(active[:, None], new_logits, logits)
            text = torch.where(active[:, None], new_text, text)
            labels = torch.where(active[:, None, None], new_labels, labels)
            steps = steps + active.to(torch.int64)
            active = active & change.ge(self.halt)
        beta, bias = self.calibrator(text, labels, logits, label_mask.to(torch.int64))
        out = beta * logits + bias
        return out.masked_fill(~label_mask, -1e4), steps


def collate(rows, pad_id, seq_multiple=CHUNK):
    """The graph's inputs for `rows` (ref.encode items), right-padded."""
    n = len(rows)
    s = -(-max(len(r["ids"]) for r in rows) // seq_multiple) * seq_multiple
    lens = [r["sep"] - r["section_start"] + 1 for r in rows]
    L, k = max(lens), max(len(r["markers"]) for r in rows)
    input_ids = torch.full((n, s), pad_id, dtype=torch.long)
    section_pos = torch.zeros((n, L), dtype=torch.long)
    section_mask = torch.zeros((n, L), dtype=torch.bool)
    label_pos = torch.zeros((n, k), dtype=torch.long)
    label_mask = torch.zeros((n, k), dtype=torch.bool)
    sep_pos = torch.zeros((n,), dtype=torch.long)
    for i, r in enumerate(rows):
        input_ids[i, : len(r["ids"])] = torch.tensor(r["ids"])
        st = r["section_start"]
        section_pos[i] = torch.tensor([min(st + j, r["sep"]) for j in range(L)])
        section_mask[i, : lens[i]] = True
        label_pos[i, : len(r["markers"])] = torch.tensor([m - st for m in r["markers"]])
        label_mask[i, : len(r["markers"])] = True
        sep_pos[i] = r["sep"] - st
    return input_ids, section_pos, section_mask, label_pos, label_mask, sep_pos


SAMPLE_STATE = {"subject": "Invoice #4411", "body": "We were billed twice for March, please refund the duplicate. " * 3}
SAMPLE_QUESTIONS = {
    "dept": {"type": "choice", "instructions": "Which team handles this?",
             "criteria": {"billing": "payments and invoices", "technical": "bugs", "sales": "", "other": ""}},
    "refund": {"type": "noul", "instructions": "The user asks for a refund."},
    "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["low", "medium", "high"]},
}


def export(slug: str, out_dir: str, root: str | None):
    m = ref.load(slug, "cpu", root)
    gl = m.model
    graph = Graph(gl).eval()
    enc = ref.encode(m, SAMPLE_STATE, SAMPLE_QUESTIONS)
    args = collate(enc["items"], m.tok.pad_token_id)
    with torch.no_grad():
        want = ref.forward(m, enc)
        got, steps = graph(*args)
    diff = max(float(abs(g[: len(w["logits"])] - torch.tensor(w["logits"])).max()) for g, w in zip(got, want))
    print("eager graph vs upstream: logits %.2e, steps %s vs %s" % (diff, steps.tolist(), [w["steps"] for w in want]))

    R = torch.export.Dim("rows", min=1, max=1024)
    N = torch.export.Dim("chunks", min=1, max=4096)
    L = torch.export.Dim("section", min=2, max=8192)
    K = torch.export.Dim("labels", min=1, max=255)
    dyn = {"input_ids": {0: R, 1: CHUNK * N}, "section_pos": {0: R, 1: L}, "section_mask": {0: R, 1: L},
           "label_pos": {0: R, 1: K}, "label_mask": {0: R, 1: K}, "sep_pos": {0: R}}
    tmp = ox.scratch_dir("gliclass-kv-export-")
    secs = ox.export_graph(graph, args, INPUT_NAMES, OUTPUT_NAMES, dyn, os.path.join(tmp, "model.onnx"))
    print("exported in %.0fs" % secs)
    cfg = gl.config
    tok = m.tok
    del graph, gl, m

    meta = ref.MODELS[slug]
    ckpt = os.path.join(root or ref.snapshot(slug), "model.safetensors")
    src = safetensors_source("model.safetensors", ckpt, repo=meta["repo"], revision=meta["revision"],
                             filename="model.safetensors")
    report = ox.weightless(tmp, out_dir, [src], [])
    ox.cleanup(tmp)
    snap = root or ref.snapshot(slug)
    shutil.copy(os.path.join(snap, "tokenizer.json"), os.path.join(out_dir, "tokenizer.json"))

    decision = {
        "engine": "onnx",
        "family": "gliclass-kv",
        "layout": LAYOUT,
        "upstream": {"repo": meta["repo"], "revision": meta["revision"], "base": meta["base"],
                     "code": ref.UPSTREAM, "git": ref.GLICLASS_GIT},
        "contract": {
            "inputs": {
                "input_ids": {"dtype": "int64", "shape": ["rows", "seq"],
                              "note": "one row per question; seq a multiple of 64; right-pad with any id (pad)"},
                "section_pos": {"dtype": "int64", "shape": ["rows", "section"],
                                "note": "positions of the label section's tokens; shorter rows repeat the final <<SEP>>"},
                "section_mask": {"dtype": "bool", "shape": ["rows", "section"]},
                "label_pos": {"dtype": "int64", "shape": ["rows", "k"], "note": "each <<LABEL>>'s index in the section"},
                "label_mask": {"dtype": "bool", "shape": ["rows", "k"]},
                "sep_pos": {"dtype": "int64", "shape": ["rows"], "note": "the final <<SEP>>'s index in the section"},
            },
            "outputs": {
                "logits": {"dtype": "float32", "shape": ["rows", "k"],
                           "note": "calibrated label logits (beta * z + bias); a row's first k_row are its labels'"},
                "steps": {"dtype": "int64", "shape": ["rows"], "note": "recurrent steps the halting rule ran"},
            },
            "seq_multiple": CHUNK,
            "positions": "0..seq-1, implicit",
            "attention": "causal in the decoder; the scorer attends over the label section with section_mask",
        },
        "max_len": ref.MAX_LEN,
        "special_tokens": {"pad": tok.pad_token_id, "label": cfg.class_token_index, "label_text": "<<LABEL>>",
                           "sep": cfg.sep_token_index, "sep_text": "<<SEP>>",
                           "example": cfg.example_token_index, "example_text": "<<EXAMPLE>>",
                           "add_special_tokens": False},
        "sanitize": list(ref.MARKERS),
        "noul": {"pair_when": "both true and false descriptions are non-empty",
                 "pair_labels": ["false", "true"], "single_label": "instructions", "single_prompt": ""},
        "recurrent": {"steps": int(cfg.recurrent_steps), "halt_threshold": float(cfg.recurrent_halt_threshold),
                      "probabilities": "sigmoid (problem_type multi_label_classification)"},
        "calibrator": {"in_graph": True, "beta_max": float(cfg.calibrator_beta_max),
                       "use_bias": bool(cfg.calibrator_use_bias)},
        "opset": ox.OPSET,
        "precision": "fp32 compute; decoder and scorer weights BF16 from the checkpoint, widened by Cast (at load, or "
                     "per forward pass with weights_in_memory bf16); calibrator F32",
        "weights_in_memory": ox.weights_in_memory(report),
    }
    calibration = {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {},
                   "source": "none: the model's own calibrator runs inside the graph (beta * logit + bias)"}
    files = {
        "model": slug,
        "layers": [
            {"role": "graph", "path": "model.onnx", "hosted_by": "ollaya",
             "bytes": os.path.getsize(os.path.join(out_dir, "model.onnx")),
             "sha256": ox.sha256_file(os.path.join(out_dir, "model.onnx"))},
            ox.file_entry("weights", meta["repo"], meta["revision"], "model.safetensors", ckpt, location="model.safetensors"),
            ox.file_entry("tokenizer", meta["repo"], meta["revision"], "tokenizer.json", os.path.join(snap, "tokenizer.json")),
            {"role": "decision", "path": "decision.json", "hosted_by": "ollaya"},
            {"role": "calibration", "path": "calibration.json", "hosted_by": "ollaya"},
            ox.file_entry("license", meta["repo"], meta["revision"], "README.md", os.path.join(snap, "README.md"), verify=False)
            | {"note": "Apache-2.0 per the model card; no LICENSE file in the repo"},
        ],
        "weightless": {k: v for k, v in report.items() if k != "unused"},
        "unused_checkpoint_tensors": {k: len(v) for k, v in report["unused"].items()},
    }
    ox.write_json(os.path.join(out_dir, "decision.json"), decision)
    ox.write_json(os.path.join(out_dir, "calibration.json"), calibration)
    ox.write_json(os.path.join(out_dir, "files.json"), files)
    # the layout port must reproduce upstream's rows for the sample request
    import tokenizers

    lay = GliclassKvLayout(tokenizers.Tokenizer.from_file(os.path.join(out_dir, "tokenizer.json")), decision)
    port = lay.encode(SAMPLE_STATE, SAMPLE_QUESTIONS)
    assert [(r["ids"], r["markers"], r["section_start"]) for r in port] == \
        [(r["ids"], r["markers"], r["section_start"]) for r in enc["items"]], "layout port differs from upstream"
    print(json.dumps(files["weightless"]["stats"]), "inline bytes", files["weightless"]["inline_bytes"],
          "graph MB %.1f" % (files["layers"][0]["bytes"] / 2**20), "unused", files["unused_checkpoint_tensors"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(ref.MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None, help="local snapshot of the model repo (default: download pinned)")
    a = ap.parse_args()
    export(a.model, a.out, a.root)


if __name__ == "__main__":
    main()
