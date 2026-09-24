"""Export Von (ModernBERT-large encoder + option-marker gather + OptionMarkerScorer) to one ONNX graph.

    uv run --with von-sdk==1.1.1 python -m ollaya_convert.families.von.export --out out/von

Contract M ("markers"), one row per encoder sequence (dynamic along rows `n`, sequence `s`, markers `k`):
    input_ids      int64 [n, s]
    attention_mask int64 [n, s]
    marker_pos     int64 [n, k]
    marker_mask    bool  [n, k]
    qtype          int64 [n]      accepted and ignored: Von has no type embedding
Output:
    logits         float32 [n, k] raw OptionMarkerScorer scores at each marker; masked slots are -1e4

The zero-shot noul debias (a second row with an empty state), the input-conditioned temperature map and
the softmax are runtime work; see docs/families/von.md.
"""
import argparse
import json
import os
import shutil

import onnx
import torch

from . import ref

INPUT_NAMES = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
OUTPUT_NAMES = ["logits"]
OPSET = 20
MIN_MARKERS = 1


class Graph(torch.nn.Module):
    """OptionMarkerModel.forward, batched: the per-sample `last_hidden[b, pos_list]` becomes a gather."""

    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.scorer = model.scorer

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        logits = self.scorer(torch.gather(h, 1, idx)).float()
        # qtype is part of the contract but Von has no use for it; keep it a live graph input.
        logits = logits + (qtype[:, None] * 0).to(logits.dtype)
        return logits.masked_fill(~marker_mask, -1e4)


def sample_inputs(tok):
    """A real request with rows of different lengths and marker counts (incl. a zero-shot noul pair)."""
    state = {"subject": "Invoice #4411", "body": "We were billed twice for March, please refund the duplicate."}
    qs = {
        "dept": {"type": "choice", "instructions": "Which team handles this?",
                 "criteria": {"billing": "payments and invoices", "technical": "bugs", "sales": "", "other": ""}},
        "refund": {"type": "noul", "instructions": "The user asks for a refund."},
        "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["low", "medium", "high"]},
    }
    b = ref.collate(ref.encode(tok, state, qs), tok.pad_token_id, MIN_MARKERS)
    return tuple(torch.from_numpy(b[n]) for n in INPUT_NAMES)


def export(out_dir: str) -> str:
    backend = ref.load("cpu")
    model = backend._get_model()
    tok = model.tokenizer
    graph = Graph(model).eval()
    args = sample_inputs(tok)

    n = torch.export.Dim("n", min=1, max=1024)
    s = torch.export.Dim("s", min=8, max=ref.MAX_LEN)
    k = torch.export.Dim("k", min=1, max=255)
    dynamic_shapes = {
        "input_ids": {0: n, 1: s},
        "attention_mask": {0: n, 1: s},
        "marker_pos": {0: n, 1: k},
        "marker_mask": {0: n, 1: k},
        "qtype": {0: n},
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "model.onnx")
    with torch.no_grad():
        program = torch.onnx.export(
            graph, args, dynamo=True, opset_version=OPSET,
            input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
            dynamic_shapes=dynamic_shapes, optimize=True,
        )
    program.save(path, external_data=True)
    onnx.checker.check_model(path, full_check=True)
    got = [i.name for i in onnx.load(path, load_external_data=False).graph.input]
    if got != INPUT_NAMES:
        raise RuntimeError("graph inputs %s != %s" % (got, INPUT_NAMES))

    # The tokenizer ships unchanged from the checkpoint; parity.py proves the Rust `tokenizers` core
    # (Tokenizer.from_file) reproduces the Python tokenizer on every row.
    from huggingface_hub import hf_hub_download
    shutil.copy(hf_hub_download(ref.REPO, "tokenizer.json", revision=ref.REVISION),
                os.path.join(out_dir, "tokenizer.json"))
    with open(hf_hub_download(ref.REPO, "marker_calibration.json", revision=ref.REVISION)) as f:
        mcal = json.load(f)

    decision = {
        "engine": "onnx",
        "family": "von",
        "layout": "von-option-marker-v1",
        "contract": "markers",
        "source": {"repo": ref.REPO, "revision": ref.REVISION, "reference": ref.UPSTREAM,
                   "license": "Apache-2.0", "author": "Victor Hugo Panisa"},
        "encoder": "answerdotai/ModernBERT-large",
        "max_len": ref.MAX_LEN,
        "tokenize": {"add_special_tokens": True, "template": "[CLS] $A [SEP]"},
        "special_tokens": {
            "cls": tok.cls_token_id, "sep": tok.sep_token_id, "mask": tok.mask_token_id,
            "pad": tok.pad_token_id, "mask_text": tok.mask_token, "sep_text": tok.sep_token,
        },
        "noul": {
            "row_order": ["true", "false"],
            "default_true": ref.NOUL_DEFAULT_TRUE,
            "default_false": ref.NOUL_DEFAULT_FALSE,
            # correction = a * (null_true - null_false) + b, applied only without noul criteria
            "zero_shot_prior": mcal.get("noul_zero_shot_prior") or {"a": ref.NOUL_PRIOR_COEF, "b": 0.0},
        },
        "inputs": INPUT_NAMES,
        "outputs": OUTPUT_NAMES,
        "min_markers": MIN_MARKERS,
        "opset": OPSET,
    }
    calibration = {
        # Upstream's fallback when the map is unusable: one scalar temperature for every question.
        "temperature": [mcal["temperature"]] * 3,
        "temperature_by_options": {},
        # Upstream's input-conditioned temperature (OptionMarkerBackend._effective_temperature).
        "temperature_map": {
            "kind": "von-entropy-length-v1",
            "formula": "T = clamp(bias + entropy*H_norm + log_tokens*log10(max(state_tokens,1))/4 + n_options*K/8, lo, hi)",
            **mcal["calibration_map"],
        },
    }
    with open(os.path.join(out_dir, "decision.json"), "w") as f:
        json.dump(decision, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "calibration.json"), "w") as f:
        json.dump(calibration, f, indent=2)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/von")
    a = ap.parse_args()
    path = export(a.out)
    size = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out) if f.startswith("model.onnx"))
    print("wrote %s (%.0f MB with external data)" % (path, size / 2**20))


if __name__ == "__main__":
    main()
