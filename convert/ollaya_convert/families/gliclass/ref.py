"""The PyTorch reference for GLiClass uni-encoder models, driven through the `gliclass` package.

    uv run --with gliclass==0.1.20 python -m ollaya_convert.families.gliclass.ref [large|edge]

Upstream (Apache-2.0): the `gliclass` package 0.1.20 (PyPI; github.com/Knowledgator/GLiClass @ 40baa67),
checkpoints `knowledgator/gliclass-instruct-large-v1.0` (DeBERTa-v3-large) and
`knowledgator/gliclass-instruct-edge-v1.0` (Ettin-32M, ModernBERT architecture).

GLiClass is a classifier, not a typed-decision model, so the question mapping below is Ollaya's. The
model call itself is upstream's `ZeroShotClassificationPipeline(texts=state, labels, prompt=...)`:
  * input text     UniEncoderZeroShotClassificationPipeline.prepare_input (labels, <<SEP>>, prompt, text)
  * tokenization   `tokenizer(inputs, truncation=True, max_length=1024, padding="longest")`, with the
                   tokenizer the checkpoint was trained with: the repo's tokenizer.json verbatim
                   (transformers 5 rebuilds DeBERTa's normalizer; see families/repo_tokenizer.py)
  * network        GLiClassModel.forward(**inputs, max_num_classes=len(labels))
  * scoring        single-label = softmax over the label logits, multi-label = per-label sigmoid
                   (BaseZeroShotClassificationPipeline._postprocess_logits)

Ollaya mapping (chosen on typed-decisions; see docs/families/gliclass.md):
  * text   = the state as Ollaya renders it everywhere (`laya.common.serialize_state`)
  * choice = labels `render_options` ("label" / "label: description"), prompt = instructions, softmax
  * score  = labels `render_options` ("level i: description"),          prompt = instructions, softmax
  * noul, both descriptions given = labels [false description, true description],
           prompt = instructions, softmax over the two
  * noul otherwise = one label, the instructions (a statement), no prompt, sigmoid
The special markers "<<LABEL>>", "<<SEP>>", "<<EXAMPLE>>" are replaced by " " in every user string:
upstream would turn them into extra classes / segments.
"""
import json
import os
from typing import Any, Dict, List

import numpy as np
import torch
from laya.common import render_criterion, render_options, serialize_state

from .. import repo_tokenizer

VARIANTS = {
    "large": ("knowledgator/gliclass-instruct-large-v1.0", "825e5478c1bf4bffbf297690517097ccbdb2e006"),
    "edge": ("knowledgator/gliclass-instruct-edge-v1.0", "727be8a417f6a7718e591b025e07054c146d8139"),
}
UPSTREAM = "gliclass==0.1.20"
MAX_LEN = 1024  # ZeroShotClassificationPipeline default max_length
MARKERS = ("<<LABEL>>", "<<SEP>>", "<<EXAMPLE>>")
QTYPES = {"choice": 0, "score": 1, "noul": 2}


class Model:
    def __init__(self, variant: str, device: str):
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline

        self.variant = variant
        self.repo, self.revision = VARIANTS[variant]
        self.model = GLiClassModel.from_pretrained(self.repo, revision=self.revision).to(device).eval()
        # The repo's tokenizer.json verbatim (training-time behaviour; see families/repo_tokenizer.py).
        self.tok, _ = repo_tokenizer.load(self.repo, self.revision)
        self.device = device
        pipe = ZeroShotClassificationPipeline(self.model, self.tok, max_length=MAX_LEN,
                                              classification_type="single-label", device=device, progress_bar=False)
        self.pipe = pipe.pipe  # UniEncoderZeroShotClassificationPipeline
        cfg = self.model.config
        for tok_text, idx in (("<<LABEL>>", cfg.class_token_index), ("<<SEP>>", cfg.text_token_index),
                              ("<<EXAMPLE>>", cfg.example_token_index)):
            if self.tok.convert_tokens_to_ids(tok_text) != idx:
                raise RuntimeError("%s is not token %d in %s" % (tok_text, idx, self.repo))
        assert cfg.architecture_type == "uni-encoder" and cfg.prompt_first
        assert cfg.class_token_pooling == "average" and cfg.embed_class_token and not cfg.extract_text_features
        assert cfg.pooling_strategy == "first" and cfg.scorer_type == "mlp" and cfg.use_segment_embeddings
        assert not (cfg.normalize_features or cfg.use_lstm or cfg.squeeze_layers or cfg.layer_wise)
        assert cfg.encoder_layer_id == -1


def load(device: str = "cpu", variant: str = "large") -> Model:
    return Model(variant, device)


def sanitize(text: str) -> str:
    for m in MARKERS:
        text = text.replace(m, " ")
    return text


def to_internal(qdef: Dict[str, Any]) -> Dict[str, Any]:
    """laya.Agent._to_internal: list choice -> {label: None}; noul keys lower-cased; json instructions."""
    t, crit = qdef["type"], qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    elif t == "noul" and isinstance(crit, dict):
        crit = {str(k).lower(): v for k, v in crit.items()}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins)
    return {"t": t, "ins": ins, "crit": crit}


def question_call(qdef: Dict[str, Any]) -> Dict[str, Any]:
    """The upstream pipeline call a question maps to: labels, prompt, and how logits become options."""
    q = to_internal(qdef)
    if q["t"] in ("choice", "score"):
        return {"labels": render_options(q), "prompt": q["ins"], "mode": "softmax"}
    crit = q["crit"] or {}
    t, f = crit.get("true"), crit.get("false")
    if t not in (None, "") and f not in (None, ""):
        return {"labels": [render_criterion(f), render_criterion(t)], "prompt": q["ins"], "mode": "pair"}
    return {"labels": [q["ins"]], "prompt": "", "mode": "sigmoid"}


def encode(m: Model, state: Any, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """One encoder row per question: the upstream input text, its ids and the <<LABEL>> positions.

    A question whose labels do not all fit in MAX_LEN is listed in `rejected` (the runtime answers 400)."""
    text = sanitize(serialize_state(state))
    items, rejected = [], {}
    for qid, qdef in questions.items():
        call = question_call(qdef)
        labels = [sanitize(l) for l in call["labels"]]
        prompt = sanitize(call["prompt"])
        row_text = m.pipe.prepare_input(text, labels, prompt=prompt)
        ids = m.tok([row_text], truncation=True, max_length=MAX_LEN)["input_ids"][0]
        markers = [i for i, t in enumerate(ids) if t == m.model.config.class_token_index]
        if len(markers) != len(labels):
            rejected[qid] = "labels do not fit in %d tokens" % MAX_LEN
            continue
        items.append({"qid": qid, "qtype": qdef["type"], "mode": call["mode"], "labels": labels,
                      "prompt": prompt, "text": row_text, "ids": ids, "markers": markers})
    return {"state_text": text, "items": items, "rejected": rejected}


def collate(enc: Dict[str, Any], pad_id: int, min_markers: int = 1):
    n = len(enc["items"])
    s = max(len(it["ids"]) for it in enc["items"])
    k = max(min_markers, max(len(it["markers"]) for it in enc["items"]))
    b = {"input_ids": np.full((n, s), pad_id, dtype=np.int64), "attention_mask": np.zeros((n, s), dtype=np.int64),
         "marker_pos": np.zeros((n, k), dtype=np.int64), "marker_mask": np.zeros((n, k), dtype=bool),
         "qtype": np.zeros((n,), dtype=np.int64)}
    for i, it in enumerate(enc["items"]):
        b["input_ids"][i, : len(it["ids"])] = it["ids"]
        b["attention_mask"][i, : len(it["ids"])] = 1
        b["marker_pos"][i, : len(it["markers"])] = it["markers"]
        b["marker_mask"][i, : len(it["markers"])] = True
        b["qtype"][i] = QTYPES[it["qtype"]]
    return b


@torch.no_grad()
def forward(m: Model, enc: Dict[str, Any]) -> List[np.ndarray]:
    """Upstream forward, one unpadded row per question (as a single pipeline call runs it)."""
    out = []
    for it in enc["items"]:
        ids = torch.tensor([it["ids"]], device=m.device)
        o = m.model(input_ids=ids, attention_mask=torch.ones_like(ids), max_num_classes=len(it["labels"]))
        out.append(o.logits[0, : len(it["labels"])].float().cpu().numpy())
    return out


def option_logits(item: Dict[str, Any], logits: np.ndarray) -> np.ndarray:
    """Ollaya option order: K for choice/score, [false, true] for noul."""
    z = np.asarray(logits, dtype=np.float64)
    if item["mode"] == "sigmoid":
        return np.array([0.0, z[0]])  # softmax([0, z]) == sigmoid(z), upstream's multi-label score
    return z


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def decide(m: Model, state: Any, questions: Dict[str, Dict[str, Any]]) -> Dict[str, np.ndarray]:
    enc = encode(m, state, questions)
    return {it["qid"]: softmax(option_logits(it, lg)) for it, lg in zip(enc["items"], forward(m, enc))}


def upstream_scores(m: Model, item: Dict[str, Any], state_text: str) -> np.ndarray:
    """The same question through the public pipeline call, in Ollaya option order."""
    cls = "multi-label" if item["mode"] == "sigmoid" else "single-label"
    res = m.pipe(state_text, item["labels"], prompt=item["prompt"] or None, classification_type=cls,
                 threshold=0.0, return_hierarchical=True)[0]
    p = np.array([res[l] for l in item["labels"]])
    return np.array([1 - p[0], p[0]]) if item["mode"] == "sigmoid" else p


def main():
    import sys

    os.environ.setdefault("USE_TF", "0")
    m = load("cpu", sys.argv[1] if len(sys.argv) > 1 else "large")
    state = {"subject": "Invoice #4411", "body": "We were billed twice for March, please refund."}
    qs = {"dept": {"type": "choice", "instructions": "Which team handles this?",
                   "criteria": {"billing": "payments", "technical": "bugs", "other": ""}},
          "refund": {"type": "noul", "instructions": "The user asks for a refund."},
          "angry": {"type": "noul", "instructions": "The customer is angry.",
                    "criteria": {"true": "strong negative emotion", "false": "calm"}}}
    enc = encode(m, state, qs)
    print(enc["items"][0]["text"])
    for it, lg in zip(enc["items"], forward(m, enc)):
        print(it["qid"], softmax(option_logits(it, lg)), upstream_scores(m, it, enc["state_text"]))


if __name__ == "__main__":
    main()
