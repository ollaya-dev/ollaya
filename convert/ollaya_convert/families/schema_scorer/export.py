"""Export the schema-conditioned candidate scorer to one ONNX graph (Contract P, "pairs", one class).

    uv run python -m ollaya_convert.families.schema_scorer.export --out out/jev-schema-scorer-deberta-v3-large

Inputs (dynamic along rows `r` and sequence `s`), one row per (state, question+candidate) pair:
    input_ids      int64 [r, s]   [CLS] state [SEP] schema-json [SEP], padded with [PAD]
    attention_mask int64 [r, s]
Output:
    scores         float32 [r, 1] the candidate's logit (softmax across a question's candidates)
"""
import argparse
import json
import os
import shutil

import onnx
import torch

from . import ref

INPUT_NAMES = ["input_ids", "attention_mask"]
OUTPUT_NAMES = ["scores"]
OPSET = 20


class Graph(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # parameter names = "model." + checkpoint names

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits.float()


def export(out_dir: str) -> str:
    m = ref.load("cpu")
    graph = Graph(m.model).eval()
    state = {"subject": "Invoice #4411", "body": "We were billed twice for March, please refund the duplicate."}
    qs = {"dept": {"type": "choice", "instructions": "Which team handles this?",
                   "criteria": {"billing": "payments and invoices", "technical": "bugs", "other": None}},
          "refund": {"type": "noul", "instructions": "The user asks for a refund."}}
    b = ref.collate(ref.encode(m, state, qs), m.tok.pad_token_id)
    args = tuple(torch.from_numpy(b[n]) for n in INPUT_NAMES)
    r = torch.export.Dim("r", min=1, max=4096)
    s = torch.export.Dim("s", min=8, max=ref.MAX_LEN)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "model.onnx")
    with torch.no_grad():
        program = torch.onnx.export(graph, args, dynamo=True, opset_version=OPSET, input_names=INPUT_NAMES,
                                    output_names=OUTPUT_NAMES, optimize=True,
                                    dynamic_shapes={"input_ids": {0: r, 1: s}, "attention_mask": {0: r, 1: s}})
    program.save(path, external_data=True)
    onnx.checker.check_model(path, full_check=True)
    shutil.copy(os.path.join(m.dir, "tokenizer.json"), os.path.join(out_dir, "tokenizer.json"))
    tok = m.tok
    decision = {
        "engine": "onnx",
        "family": "schema-scorer",
        "layout": "schema-pairs-v1",
        "contract": "pairs",
        "source": {"repo": ref.REPO, "revision": ref.REVISION, "reference": "schema_scorer.py in the model repo",
                   "license": "MIT", "author": "mobarmg"},
        "encoder": "microsoft/deberta-v3-large",
        "max_len": ref.MAX_LEN,
        "tokenize": {"pair": True, "add_special_tokens": True, "template": "[CLS] $A [SEP] $B [SEP]",
                     "truncation": "none; reject rows longer than max_len"},
        "special_tokens": {"cls": tok.cls_token_id, "sep": tok.sep_token_id, "pad": tok.pad_token_id},
        "classes": {"score": 0},
        "noul_defaults": {"false": "No. The proposition is false for this state.",
                          "true": "Yes. The proposition is true for this state."},
        "inputs": INPUT_NAMES,
        "outputs": OUTPUT_NAMES,
        "opset": OPSET,
    }
    with open(os.path.join(out_dir, "decision.json"), "w") as f:
        json.dump(decision, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "calibration.json"), "w") as f:
        json.dump({"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}}, f, indent=2)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/jev-schema-scorer-deberta-v3-large")
    a = ap.parse_args()
    path = export(a.out)
    size = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out) if f.startswith("model.onnx"))
    print("wrote %s (%.0f MB with external data)" % (path, size / 2**20))


if __name__ == "__main__":
    main()
