"""Export a zero-shot NLI cross-encoder to one ONNX graph (Contract P, "pairs").

    uv run python -m ollaya_convert.families.nli.export deberta-v3-large --out out/deberta-v3-large-zeroshot-v2.0

Inputs (dynamic along rows `r` and sequence `s`), one row per (premise, hypothesis) pair:
    input_ids      int64 [r, s]   [CLS] premise [SEP] hypothesis [SEP], padded with [PAD]
    attention_mask int64 [r, s]
Output:
    scores         float32 [r, 2] raw logits, class 0 = entailment, 1 = not_entailment

No token_type_ids: DeBERTa-v3 here has type_vocab_size 0 and ModernBERT has none, so both ignore them.
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


def sample_inputs(m):
    state = {"subject": "Invoice #4411", "body": "We were billed twice for March, please refund the duplicate."}
    qs = {"dept": {"type": "choice", "instructions": "Which team handles this?",
                   "criteria": {"billing": "payments and invoices", "technical": "bugs", "other": ""}},
          "refund": {"type": "noul", "instructions": "The user asks for a refund."}}
    b = ref.collate(ref.encode(m, state, qs), m.tok.pad_token_id)
    return tuple(torch.from_numpy(b[n]) for n in INPUT_NAMES)


def export(name: str, out_dir: str) -> str:
    m = ref.load("cpu", name)
    graph = Graph(m.model).eval()
    args = sample_inputs(m)
    r = torch.export.Dim("r", min=1, max=4096)
    s = torch.export.Dim("s", min=8, max=ref.MAX_LEN)
    dynamic_shapes = {"input_ids": {0: r, 1: s}, "attention_mask": {0: r, 1: s}}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "model.onnx")
    with torch.no_grad():
        program = torch.onnx.export(graph, args, dynamo=True, opset_version=OPSET, input_names=INPUT_NAMES,
                                    output_names=OUTPUT_NAMES, dynamic_shapes=dynamic_shapes, optimize=True)
    program.save(path, external_data=True)
    onnx.checker.check_model(path, full_check=True)

    from huggingface_hub import hf_hub_download
    shutil.copy(hf_hub_download(m.repo, "tokenizer.json", revision=m.revision), os.path.join(out_dir, "tokenizer.json"))
    tok = m.tok
    decision = {
        "engine": "onnx",
        "family": "nli",
        "layout": "nli-pairs-v1",
        "contract": "pairs",
        "source": {"repo": m.repo, "revision": m.revision, "reference": "transformers==5.17 zero-shot-classification pipeline",
                   "license": "MIT" if "deberta" in name else "Apache-2.0", "author": "Moritz Laurer"},
        "encoder": m.model.config._name_or_path or m.model.config.model_type,
        "max_len": ref.MAX_LEN,
        "tokenize": {"pair": True, "add_special_tokens": True, "template": "[CLS] $A [SEP] $B [SEP]",
                     "truncation": "only_first (premise), from the right"},
        "special_tokens": {"cls": tok.cls_token_id, "sep": tok.sep_token_id, "pad": tok.pad_token_id},
        "classes": {"entailment": ref.ENTAILMENT, "not_entailment": ref.NOT_ENTAILMENT},
        "templates": ref.TEMPLATES[name],
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
    ap.add_argument("model", choices=sorted(ref.MODELS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    path = export(a.model, a.out)
    size = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out) if f.startswith("model.onnx"))
    print("wrote %s (%.0f MB with external data)" % (path, size / 2**20))


if __name__ == "__main__":
    main()
