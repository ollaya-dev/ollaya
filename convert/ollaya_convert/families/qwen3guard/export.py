"""Export Qwen/Qwen3Guard-Gen-0.6B as a fixed-preset decision model (`qwen3guard-gen-v1`).

    uv run python -m ollaya_convert.families.qwen3guard.export --out out/qwen3guard-gen-0.6b [--root ROOT]

Graph:
    inputs   input_ids   int64   [rows, seq]   right-padded (any pad id), no sequence restriction
             last_pos    int64   [rows]        index of the row's last real token
    outputs  cand_logits float32 [rows, 13]    next-token logits of the 13 candidate first tokens:
                                               [ Safe, Cont, Unsafe] + the 10 category first tokens
Weights reference the author's `model.safetensors` (BF16) byte for byte (tied LM head = embedding rows).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch

from ..llm_common import onnx_export as ox
from ..llm_common.qwen3 import Qwen3Trunk
from ...weightless_sharded import safetensors_source
from . import ref

INPUT_NAMES = ["input_ids", "last_pos"]
OUTPUT_NAMES = ["cand_logits"]


class GuardGraph(torch.nn.Module):
    def __init__(self, lm, cand_ids):
        super().__init__()
        self.trunk = Qwen3Trunk(lm.model)
        self.register_buffer("cand_ids", torch.tensor(cand_ids, dtype=torch.long), persistent=False)

    def forward(self, input_ids, last_pos):
        h = self.trunk(input_ids)
        hs = h[torch.arange(h.shape[0], device=h.device), last_pos]
        w = self.trunk.m.embed_tokens.weight[self.cand_ids]
        return hs @ w.transpose(0, 1)


def export(out_dir, root):
    tok, lm = ref.load(root, device="cpu")
    assert lm.lm_head.weight.data_ptr() == lm.model.embed_tokens.weight.data_ptr()
    s_ids, c_ids = ref.first_tokens(tok, ref.SAFETY), ref.first_tokens(tok, ref.CATEGORIES)
    graph = GuardGraph(lm, s_ids + c_ids).eval()
    r = ref.rows(tok, "How can I make a bomb?")
    T = max(len(x) for x in r) + 3
    ids = torch.full((2, T), tok.pad_token_id, dtype=torch.long)
    for i, x in enumerate(r):
        ids[i, :len(x)] = torch.tensor(x)
    last = torch.tensor([len(x) - 1 for x in r])
    with torch.no_grad():
        want = lm(input_ids=torch.tensor([r[0]])).logits[0, -1, s_ids]
        got = graph(ids, last)[0, :3]
    print("eager trunk vs HF: %.2e" % float((want - got).abs().max()))
    R = torch.export.Dim("rows", min=1, max=4096)
    S = torch.export.Dim("seq", min=2, max=32768)
    tmp = ox.scratch_dir("qwen3guard-export-")
    secs = ox.export_graph(graph, (ids, last), INPUT_NAMES, OUTPUT_NAMES,
                           {"input_ids": {0: R, 1: S}, "last_pos": {0: R}}, os.path.join(tmp, "model.onnx"))
    print("exported in %.0fs" % secs)
    src = safetensors_source("model.safetensors", os.path.join(root, "model.safetensors"), repo=ref.REPO,
                             revision=ref.REVISION, filename="model.safetensors")
    report = ox.weightless(tmp, out_dir, [src], [("trunk.m.", "model.")])
    ox.cleanup(tmp)
    shutil.copy(os.path.join(root, "tokenizer.json"), os.path.join(out_dir, "tokenizer.json"))
    tail = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prompt = tok.apply_chat_template([{"role": "user", "content": "STATE"}], tokenize=False)
    head, rest = prompt.split("STATE")
    assert rest.endswith(tail)
    decision = {
        "engine": "onnx",
        "family": "qwen3guard",
        "layout": "qwen3guard-gen-v1",
        "upstream": {"repo": ref.REPO, "revision": ref.REVISION},
        "preset_only": True,
        "questions": ref.PRESET,
        "contract": {
            "inputs": {"input_ids": {"dtype": "int64", "shape": ["rows", "seq"], "note": "right-padded, any pad id"},
                       "last_pos": {"dtype": "int64", "shape": ["rows"], "note": "index of the last real token"}},
            "outputs": {"cand_logits": {"dtype": "float32", "shape": ["rows", len(s_ids) + len(c_ids)],
                                        "note": "[0:3] safety (safe, controversial, unsafe); [3:13] categories"}},
            "attention": "causal; no mask input", "positions": "0..seq-1, implicit"},
        "templates": {"prompt_head": head, "prompt_tail": rest, "state": "strings verbatim, else json.dumps(ensure_ascii=False)",
                      "row_safety": "{prompt_head}{state}{prompt_tail}Safety:",
                      "row_category": "{prompt_head}{state}{prompt_tail}Safety: Unsafe\nCategories:",
                      "tokenize": "whole row string, add_special_tokens=False, special tokens parsed"},
        "candidates": {"safety": {"labels": [k for k, _ in ref.SAFETY], "texts": [t for _, t in ref.SAFETY], "ids": s_ids},
                       "category": {"labels": [k for k, _ in ref.CATEGORIES], "texts": [t for _, t in ref.CATEGORIES], "ids": c_ids}},
        "option_logits": {"safety": "cand[r0, 0:3]", "unsafe": "[logsumexp(cand[r0,0], cand[r0,1]), cand[r0,2]]",
                          "unsafe_strict": "[cand[r0,0], logsumexp(cand[r0,1], cand[r0,2])]", "category": "cand[r1, 3:13]"},
        "max_len": 32768,
        "special_tokens": {"pad": tok.pad_token_id, "bos": None, "add_special_tokens": False},
        "opset": ox.OPSET,
    }
    calibration = {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}, "source": "raw (no fitted temperature upstream)"}
    files = {"model": "qwen3guard-gen-0.6b", "layers": [
        {"role": "graph", "path": "model.onnx", "hosted_by": "ollaya", "bytes": os.path.getsize(os.path.join(out_dir, "model.onnx")),
         "sha256": ox.sha256_file(os.path.join(out_dir, "model.onnx"))},
        ox.file_entry("weights", ref.REPO, ref.REVISION, "model.safetensors", os.path.join(root, "model.safetensors"), location="model.safetensors"),
        ox.file_entry("tokenizer", ref.REPO, ref.REVISION, "tokenizer.json", os.path.join(root, "tokenizer.json")),
        ox.file_entry("license", ref.REPO, ref.REVISION, "LICENSE", os.path.join(root, "LICENSE")),
        {"role": "decision", "path": "decision.json", "hosted_by": "ollaya"},
        {"role": "calibration", "path": "calibration.json", "hosted_by": "ollaya"}],
        "weightless": {k: v for k, v in report.items() if k != "unused"},
        "unused_checkpoint_tensors": {k: len(v) for k, v in report["unused"].items()}}
    for name, obj in (("decision.json", decision), ("calibration.json", calibration), ("files.json", files)):
        ox.write_json(os.path.join(out_dir, name), obj)
    print(json.dumps(files["weightless"]["stats"]), "graph MB %.1f" % (files["layers"][0]["bytes"] / 2**20),
          "unused", files["unused_checkpoint_tensors"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    if a.root is None:
        from huggingface_hub import snapshot_download

        a.root = snapshot_download(ref.REPO, revision=ref.REVISION)
    export(a.out, a.root)


if __name__ == "__main__":
    main()
