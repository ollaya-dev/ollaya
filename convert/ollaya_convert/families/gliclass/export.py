"""Export a GLiClass uni-encoder (encoder + label-span pooling + projectors + MLP scorer) to one ONNX graph.

    uv run --with gliclass==0.1.20 python -m ollaya_convert.families.gliclass.export large --out out/gliclass-instruct-large

Contract M ("markers"), one row per question (dynamic along rows `n`, sequence `s`, labels `k`):
    input_ids      int64 [n, s]
    attention_mask int64 [n, s]
    marker_pos     int64 [n, k]   positions of the <<LABEL>> tokens, ascending
    marker_mask    bool  [n, k]
    qtype          int64 [n]      accepted and ignored
Output:
    logits         float32 [n, k] GLiClass label logits; masked slots are -1e4

The graph is GLiClassUniEncoder.forward for this config, vectorised:
  * segment ids: 1 from the first <<SEP>> on (argmax semantics: all 1 if there is none), else 0;
    added to the word embeddings (`_create_segment_ids`; <<EXAMPLE>> never occurs after sanitising)
  * label k's embedding = mean of hidden states from marker k up to (excluding) marker k+1; the last
    label's span runs to the end of the sequence (`_extract_class_features_averaged` with
    extract_text_features=False), padding excluded
  * text embedding = hidden state 0 -> text_projector; labels -> classes_projector; MLPScorer
Sigmoid / softmax stay in the runtime.
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
    def __init__(self, gliclass_model):
        super().__init__()
        self.model = gliclass_model.model  # keeps parameter names == checkpoint names ("model.…")
        self.sep_id = int(gliclass_model.config.text_token_index)

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        u = self.model
        s = input_ids.shape[1]
        pos = torch.arange(s, device=input_ids.device)
        first_sep = torch.argmax((input_ids == self.sep_id).to(torch.int64), dim=1)
        segment = (pos[None, :] >= first_sep[:, None]).to(torch.int64)
        x = u.encoder_model.get_input_embeddings()(input_ids) + u.segment_embeddings(segment)
        h = u.encoder_model(inputs_embeds=x, attention_mask=attention_mask)[0]

        started = (marker_pos[:, :, None] <= pos[None, None, :]) & marker_mask[:, :, None]
        cum = started.to(torch.int64).sum(1)  # [n, s]: labels opened at or before each position
        k_range = torch.arange(marker_pos.shape[1], device=input_ids.device)[None, :, None]
        span = (cum[:, None, :] == k_range + 1) & attention_mask[:, None, :].bool()
        span = span.to(h.dtype)
        classes = torch.einsum("bks,bsd->bkd", span, h) / span.sum(-1, keepdim=True).clamp(min=1)

        pooled = u.text_projector(u.pooler(h))
        logits = u.scorer(pooled, u.classes_projector(classes)).float()
        logits = logits + (qtype[:, None] * 0).to(logits.dtype)
        return logits.masked_fill(~marker_mask, -1e4)


def sample_inputs(m):
    state = {"subject": "Invoice #4411", "body": "We were billed twice for March, please refund the duplicate."}
    qs = {
        "dept": {"type": "choice", "instructions": "Which team handles this?",
                 "criteria": {"billing": "payments and invoices", "technical": "bugs", "sales": "", "other": ""}},
        "refund": {"type": "noul", "instructions": "The user asks for a refund."},
        "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["low", "medium", "high"]},
    }
    b = ref.collate(ref.encode(m, state, qs), m.tok.pad_token_id, MIN_MARKERS)
    return tuple(torch.from_numpy(b[n]) for n in INPUT_NAMES)


def export(variant: str, out_dir: str) -> str:
    m = ref.load("cpu", variant)
    graph = Graph(m.model).eval()
    args = sample_inputs(m)
    n = torch.export.Dim("n", min=1, max=1024)
    s = torch.export.Dim("s", min=8, max=ref.MAX_LEN)
    k = torch.export.Dim("k", min=1, max=255)
    dynamic_shapes = {"input_ids": {0: n, 1: s}, "attention_mask": {0: n, 1: s},
                      "marker_pos": {0: n, 1: k}, "marker_mask": {0: n, 1: k}, "qtype": {0: n}}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "model.onnx")
    with torch.no_grad():
        program = torch.onnx.export(graph, args, dynamo=True, opset_version=OPSET, input_names=INPUT_NAMES,
                                    output_names=OUTPUT_NAMES, dynamic_shapes=dynamic_shapes, optimize=True)
    program.save(path, external_data=True)
    onnx.checker.check_model(path, full_check=True)
    got = [i.name for i in onnx.load(path, load_external_data=False).graph.input]
    if got != INPUT_NAMES:
        raise RuntimeError("graph inputs %s != %s" % (got, INPUT_NAMES))

    from huggingface_hub import hf_hub_download
    shutil.copy(hf_hub_download(m.repo, "tokenizer.json", revision=m.revision), os.path.join(out_dir, "tokenizer.json"))
    cfg = m.model.config
    tok = m.tok
    decision = {
        "engine": "onnx",
        "family": "gliclass",
        "layout": "gliclass-uni-v1",
        "contract": "markers",
        "source": {"repo": m.repo, "revision": m.revision, "reference": ref.UPSTREAM, "license": "Apache-2.0",
                   "author": "Knowledgator"},
        "encoder": cfg.encoder_model_name,
        "max_len": ref.MAX_LEN,
        "tokenize": {"add_special_tokens": True, "template": "[CLS] $A [SEP]", "truncation": "right, to max_len"},
        "special_tokens": {
            "cls": tok.cls_token_id, "sep": tok.sep_token_id, "pad": tok.pad_token_id,
            "label": cfg.class_token_index, "label_text": "<<LABEL>>",
            "text_sep": cfg.text_token_index, "text_sep_text": "<<SEP>>",
            "example": cfg.example_token_index, "example_text": "<<EXAMPLE>>",
        },
        "sanitize": ["<<LABEL>>", "<<SEP>>", "<<EXAMPLE>>"],
        "noul": {"pair_when": "both true and false descriptions are non-empty",
                 "pair_labels": ["false", "true"], "single_label": "instructions", "single_prompt": ""},
        "inputs": INPUT_NAMES,
        "outputs": OUTPUT_NAMES,
        "min_markers": MIN_MARKERS,
        "opset": OPSET,
    }
    with open(os.path.join(out_dir, "decision.json"), "w") as f:
        json.dump(decision, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "calibration.json"), "w") as f:
        json.dump({"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}}, f, indent=2)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variant", choices=sorted(ref.VARIANTS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    path = export(a.variant, a.out)
    size = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out) if f.startswith("model.onnx"))
    print("wrote %s (%.0f MB with external data)" % (path, size / 2**20))


if __name__ == "__main__":
    main()
