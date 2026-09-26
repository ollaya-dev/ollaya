"""Export llm-semantic-router/Decision-1.0-* (Qwen3.5 text backbone + endpoint head) to a weightless ONNX graph.

    uv run python -m ollaya_convert.families.decision.export decision-eos --out out/decision-eos-0.8b [--root SNAPSHOT]

Graph (layout `decision-endpoint-v1`, see layout.py and docs/families/decision.md):
    inputs   input_ids  int64   [rows, seq]  one row per question; seq a multiple of 64, right-padded
             query_pos  int64   [rows]       position of the row's last token (the end of "Decision:")
             cand_pos   int64   [rows, k]    position of each option segment's last token (pad with 0)
    outputs  logits     float32 [rows, k]    raw candidate logits of the head (temperature not applied)

The head is the author's CandidateHead, recomputed from its own modules (the eager check below compares
the graph with the author's forward):
    c = candidate_norm(h[cand_pos]);  q = query_norm(h[query_pos])                      (LayerNorm, fp32)
    logits = (key(c) * query(q)).sum(-1) / sqrt(256) + scalar(gelu(candidate_mlp(c) + query_mlp(q)))

Weights are not copied: every initializer references the author's files at their byte offsets, the
backbone safetensors (BF16, one Cast to float32 each, folded by ONNX Runtime at load) and
`decision_head.safetensors` (F32).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil

import torch
import torch.nn.functional as F

from ..llm_common import onnx_export as ox
from ..llm_common.qwen35 import CHUNK, Qwen35Trunk
from ...weightless_sharded import safetensors_source
from . import ref
from .layout import MAX_OPTIONS, MAX_ROW_TOKENS, MAX_SCORE_LEVELS, MIN_OPTIONS, NOUL_DEFAULTS, PROMPT_VERSION, \
    SUFFIX

INPUT_NAMES = ["input_ids", "query_pos", "cand_pos"]
OUTPUT_NAMES = ["logits"]
LAYOUT = "decision-endpoint-v1"


class DecisionGraph(torch.nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.trunk = Qwen35Trunk(backbone)
        self.head = head
        self.scale = 1.0 / math.sqrt(head.head_dim)

    def forward(self, input_ids, query_pos, cand_pos):
        h = self.trunk(input_ids).float()
        rows = torch.arange(h.shape[0], device=h.device)
        c = self.head.candidate_norm(h[rows.unsqueeze(1), cand_pos])     # [R, K, H]
        q = self.head.query_norm(h[rows, query_pos])                      # [R, H]
        bilinear = (self.head.key(c) * self.head.query(q).unsqueeze(1)).sum(-1) * self.scale
        mlp = F.gelu(self.head.candidate_mlp(c) + self.head.query_mlp(q).unsqueeze(1))
        return bilinear + self.head.scalar(mlp).squeeze(-1)


def rename(name):
    """Graph initializer name -> checkpoint tensor names (backbone shards, then the head file)."""
    if name.startswith("trunk.m."):
        return [name[len("trunk.m."):]]
    if name.startswith("head."):
        return [name[len("head."):]]
    return [name]


SAMPLE_STATE = "The customer was charged twice for order A-104 and wants the duplicate refunded today. " * 6
SAMPLE_QUESTIONS = {
    "a": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "charges", "tech": "bugs", "other": None}},
    "b": {"type": "score", "instructions": "How upset?", "criteria": ["calm", "annoyed", "furious"]},
}


def empty_model(d, root):
    """The author's backbone and head classes with uninitialized fp32 storage (never written, so it
    costs no memory): the graph of a checkpoint too large to hold in fp32. Only the rotary
    frequencies, the one computed buffer, are real."""
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel, Qwen3_5TextRotaryEmbedding

    config = AutoConfig.from_pretrained(os.path.join(root, "backbone"), local_files_only=True)
    config.use_cache = False
    with torch.device("meta"):
        backbone = Qwen3_5TextModel(config)
        head = d.module.CandidateHead(config.hidden_size, 256)
    buffers = [n for n, _ in backbone.named_buffers()]
    assert all(n.startswith("rotary_emb.") for n in buffers), buffers
    backbone = backbone.to_empty(device="cpu")
    backbone.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
    return backbone.eval(), head.to_empty(device="cpu").eval()


def export(slug, out_dir, root, empty=False):
    """`empty`: export without the weights' values (see empty_model); the weightless step then maps
    initializers by name and shape, and the graph is checked by parity afterwards."""
    meta = ref.MODELS[slug]
    d = ref.load(root, slug, device="cpu", load_model=not empty)
    if empty:
        backbone, head = empty_model(d, root)
    else:
        backbone, head = d.model.backbone, d.model.head
    graph = DecisionGraph(backbone, head).eval()
    cfg = json.load(open(os.path.join(root, "decision_config.json")))
    assert cfg["prompt_version"] == PROMPT_VERSION and cfg["head_dim"] == 256, cfg

    # Two rows padded to 256 tokens, three candidates each: every dynamic axis > 1. The rows come from
    # the author's encode, and the eager graph is checked against the author's forward on them.
    rows, items = d.encode(SAMPLE_STATE, SAMPLE_QUESTIONS)
    T = -(-max(len(it["ids"]) for it in items) // CHUNK) * CHUNK
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
    N = torch.export.Dim("chunks", min=1, max=4096)
    K = torch.export.Dim("options", min=1, max=MAX_OPTIONS)
    dyn = {"input_ids": {0: R, 1: CHUNK * N}, "query_pos": {0: R}, "cand_pos": {0: R, 1: K}}
    tmp = ox.scratch_dir("decision-export-")
    secs = ox.export_graph(graph, args, INPUT_NAMES, OUTPUT_NAMES, dyn, os.path.join(tmp, "model.onnx"))
    print("exported in %.0fs" % secs)
    params = {"trunk.m." + n for n, _ in backbone.named_parameters()} | {"head." + n for n, _ in head.named_parameters()}
    del graph, backbone, head, d

    sources = [safetensors_source(os.path.basename(f), os.path.join(root, f), repo=meta["repo"],
                                  revision=meta["revision"], filename=f) for f in meta["backbone"]]
    sources.append(safetensors_source("decision_head.safetensors", os.path.join(root, "decision_head.safetensors"),
                                      repo=meta["repo"], revision=meta["revision"], filename="decision_head.safetensors"))
    report = ox.weightless(tmp, out_dir, sources, rename, verify=not empty, sink_casts=False)
    ox.cleanup(tmp)
    # Every parameter references the checkpoint; only buffers and constants stay inline.
    inline_params = sorted(params & {name for name, _, _ in report["inline"]})
    assert not inline_params and report["stats"]["external"] == len(params), (inline_params, report["stats"])
    tok_out = os.path.join(out_dir, "tokenizer.json")
    if os.path.exists(tok_out):  # a previous export's copy keeps the snapshot's read-only mode
        os.remove(tok_out)
    shutil.copyfile(os.path.join(root, "tokenizer.json"), tok_out)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(root, local_files_only=True)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    temps = ref.temperatures(root, slug)
    null_rule = ref.choice_null_description(root, slug)
    decision = {
        "engine": "onnx",
        "family": "decision",
        "layout": LAYOUT,
        "upstream": {"repo": meta["repo"], "revision": meta["revision"], "base": cfg["base_model"],
                     "base_revision": cfg["base_revision"], "model_name": cfg.get("model_name"),
                     "code": {"model": meta["model_code"], "model_sha256": ox.sha256_file(os.path.join(root, meta["model_code"])),
                              "api": meta["api"], "api_sha256": ox.sha256_file(os.path.join(root, meta["api"]))},
                     "architecture": cfg["architecture"]},
        "contract": {
            "inputs": {
                "input_ids": {"dtype": "int64", "shape": ["rows", "seq"],
                              "note": "one row per question; seq a multiple of 64; right-pad with any id (pad)"},
                "query_pos": {"dtype": "int64", "shape": ["rows"], "note": "position of the row's last token"},
                "cand_pos": {"dtype": "int64", "shape": ["rows", "k"],
                             "note": "position of each option segment's last token; rows with fewer options pad with 0"},
            },
            "outputs": {"logits": {"dtype": "float32", "shape": ["rows", "k"],
                                   "note": "raw candidate logits; the first k_row entries are the row's option logits"}},
            "seq_multiple": CHUNK,
            "positions": "0..seq-1, implicit",
            "attention": "causal; no mask input (right padding cannot reach earlier positions)",
        },
        "prompt_version": PROMPT_VERSION,
        "max_row_tokens": MAX_ROW_TOKENS,
        "min_options": MIN_OPTIONS,
        "max_options": MAX_OPTIONS,
        "max_score_levels": MAX_SCORE_LEVELS,
        "choice_null_description": null_rule,
        "pad": pad,
        "templates": {
            "prefix": "Context:\n{payload(state)}\n\nTask type: {type}\nQuestion:\n{payload(instructions)}\nOptions:",
            "option": "\n<option>\n{canonical({\"key\": key, \"description\": description})}\n</option>",
            "suffix": SUFFIX,
            "payload": "a str verbatim, anything else canonical()",
            "canonical": "json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(',', ':'))",
            "tokenize": "each segment on its own, add_special_tokens=False; cand_pos = last token of each option, "
                        "query_pos = last token of the row",
            "noul_descriptions": NOUL_DEFAULTS,
            "choice_null_description": "key: a null description becomes the key; null: it stays JSON null",
        },
        "option_logits": {"choice": "logits[row, :k] in key order", "noul": "logits[row, :2] (0 = false, 1 = true)",
                          "score": "logits[row, :levels]"},
        "opset": ox.OPSET,
        "precision": "fp32 compute; backbone weights BF16 (Cast at load), head F32",
    }
    calibration = {"temperature": [temps["choice"], temps["score"], temps["noul"]], "temperature_by_options": {},
                   "source": "%s (the released temperature, applied as softmax(logits / T))"
                             % meta["temperature"][0]}
    layers = [{"role": "graph", "path": "model.onnx", "hosted_by": "ollaya",
               "bytes": os.path.getsize(os.path.join(out_dir, "model.onnx")),
               "sha256": ox.sha256_file(os.path.join(out_dir, "model.onnx"))}]
    for f in meta["backbone"] + ["decision_head.safetensors"]:
        layers.append(ox.file_entry("weights", meta["repo"], meta["revision"], f, os.path.join(root, f),
                                    location=os.path.basename(f)))
    layers += [
        ox.file_entry("tokenizer", meta["repo"], meta["revision"], "tokenizer.json", os.path.join(root, "tokenizer.json")),
        {"role": "decision", "path": "decision.json", "hosted_by": "ollaya"},
        {"role": "calibration", "path": "calibration.json", "hosted_by": "ollaya"},
        ox.file_entry("license", meta["repo"], meta["revision"], "LICENSE", os.path.join(root, "LICENSE"), verify=False)
        | {"note": "Apache-2.0 (the repo's LICENSE); base Qwen3.5 is Apache-2.0"},
    ]
    files = {"model": slug, "layers": layers, "weightless": {k: v for k, v in report.items() if k != "unused"},
             "unused_checkpoint_tensors": {k: len(v) for k, v in report["unused"].items()}}
    ox.write_json(os.path.join(out_dir, "decision.json"), decision)
    ox.write_json(os.path.join(out_dir, "calibration.json"), calibration)
    ox.write_json(os.path.join(out_dir, "files.json"), files)
    print(json.dumps(files["weightless"]["stats"]), "inline bytes", files["weightless"]["inline_bytes"],
          "graph MB %.1f" % (layers[0]["bytes"] / 2**20), "unused", files["unused_checkpoint_tensors"])


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
