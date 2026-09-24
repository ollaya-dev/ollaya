"""The PyTorch reference for zero-shot NLI models, run the way `transformers`' zero-shot pipeline runs them.

    uv run python -m ollaya_convert.families.nli.ref [deberta-v3-large|modernbert-large]

Upstream: `transformers.ZeroShotClassificationPipeline` (transformers 5.17), the documented way to use
MoritzLaurer's zeroshot-v2.0 models:
  * each (premise, hypothesis) pair is tokenized alone, with the repo's tokenizer.json verbatim
    (transformers 5 would rebuild DeBERTa's normalizer; see families/repo_tokenizer.py):
        tokenizer([premise, hypothesis], add_special_tokens=True, truncation=ONLY_FIRST)  (max 512)
  * the model scores it: logits [entailment, not_entailment] (config id2label {0: entailment, 1: not_entailment})
  * multi_label=False: softmax of the entailment logits across the candidates
  * one candidate / multi_label=True: softmax over [not_entailment, entailment] per candidate

Ollaya's typed-question mapping ("hypotheses"), with per-model templates (TEMPLATES) chosen on
typed-decisions (see docs/families/nli.md):
  * premise      the state as Ollaya renders it everywhere (`laya.common.serialize_state`)
  * choice/score one pair per option; option logit = entailment logit (softmax across options)
  * noul         both descriptions non-empty: two pairs, hypotheses = [false description, true
                 description]; option logits = their entailment logits
                 otherwise: one pair, hypothesis = the instructions; option logits
                 = [not_entailment, entailment] of that pair
Placeholders are substituted in a single left-to-right pass, so values containing "{...}" are inert.
"""
import json
import os
import re
from typing import Any, Dict, List

import numpy as np
import torch
from laya.common import render_criterion, serialize_state

from .. import repo_tokenizer

MODELS = {
    "deberta-v3-large": ("MoritzLaurer/deberta-v3-large-zeroshot-v2.0", "cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2"),
    "modernbert-large": ("MoritzLaurer/ModernBERT-large-zeroshot-v2.0", "a51e07b524299e309dd2b88d48b0cfa2bd9ec598"),
}
MAX_LEN = 512  # both tokenizers' model_max_length, which the pipeline truncates to
ENTAILMENT, NOT_ENTAILMENT = 0, 1

# Hypothesis templates. choice: `with_description` when the option has a (non-empty) description,
# else `without_description`. Fields: {instructions} {label} {description} {option}, where option is
# "label" or "label: description" (Laya's rendering). score: {instructions} {level} {description}.
TEMPLATES = {
    "deberta-v3-large": {
        "choice": {"with_description": 'The answer to "{instructions}" is: {option}',
                   "without_description": 'The answer to "{instructions}" is: {option}'},
        "score": 'The answer to "{instructions}" is: {description}',
        "noul_pair": "{description}",
        "noul_single": "{instructions}",
    },
    "modernbert-large": {
        "choice": {"with_description": "{description}",
                   "without_description": "This text is about {label}."},
        "score": "{description}",
        "noul_pair": "{description}",
        "noul_single": "{instructions}",
    },
}


def fill(template: str, **fields: str) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: fields[m.group(1)], template)


class Model:
    def __init__(self, name: str, device: str):
        from transformers import AutoModelForSequenceClassification, pipeline

        self.name = name
        self.repo, self.revision = MODELS[name]
        # The repo's tokenizer.json verbatim: the tokenizer these checkpoints were trained with under
        # transformers 4.x. transformers 5 rebuilds DeBERTa's normalizer (families/repo_tokenizer.py).
        self.tok, _ = repo_tokenizer.load(self.repo, self.revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.repo, revision=self.revision, dtype=torch.float32).to(device).eval()
        if getattr(self.model.config, "reference_compile", None):
            self.model.config.reference_compile = False
        assert self.model.config.id2label == {0: "entailment", 1: "not_entailment"}, self.model.config.id2label
        assert self.tok.model_max_length == MAX_LEN
        self.device = device
        self.templates = TEMPLATES[name]
        # Upstream call path, on the very same fp32 module.
        self.pipe = pipeline("zero-shot-classification", model=self.model, tokenizer=self.tok, device=device)


def load(device: str = "cpu", name: str = "deberta-v3-large") -> Model:
    return Model(name, device)


def question_rows(tpl: Dict[str, Any], qdef: Dict[str, Any]) -> Dict[str, Any]:
    """Hypotheses for one question and how their logits become option logits."""
    t = qdef["type"]
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins)  # Laya's rule, as the shared Rust parser does
    crit = qdef.get("criteria")
    if t == "choice":
        if isinstance(crit, list):
            crit = {c: None for c in crit}
        hyps = []
        for label, v in crit.items():
            desc = "" if v is None or v == "" else render_criterion(v)
            option = label if not desc else "%s: %s" % (label, desc)
            key = "with_description" if desc else "without_description"
            hyps.append(fill(tpl["choice"][key], instructions=ins, label=label, description=desc, option=option))
        return {"mode": "entail", "hypotheses": hyps}
    if t == "score":
        hyps = [fill(tpl["score"], instructions=ins, level=str(i), description=render_criterion(c))
                for i, c in enumerate(crit)]
        return {"mode": "entail", "hypotheses": hyps}
    crit = {str(k).lower(): v for k, v in (crit or {}).items()}
    tv, fv = crit.get("true"), crit.get("false")
    if tv not in (None, "") and fv not in (None, ""):
        hyps = [fill(tpl["noul_pair"], instructions=ins, description=render_criterion(fv)),
                fill(tpl["noul_pair"], instructions=ins, description=render_criterion(tv))]
        return {"mode": "entail", "hypotheses": hyps}
    return {"mode": "single", "hypotheses": [fill(tpl["noul_single"], instructions=ins)]}


def encode_pair(tok, premise: str, hypothesis: str) -> List[int]:
    # ZeroShotClassificationPipeline._parse_and_tokenize for one pair (its "too short" fallback to no
    # truncation is not taken: Ollaya rejects a hypothesis that leaves no room for the premise).
    try:
        return tok([[premise, hypothesis]], add_special_tokens=True, truncation="only_first",
                   max_length=MAX_LEN)["input_ids"][0]
    except Exception as e:  # tokenizers raises when only_first cannot make the pair fit
        raise ValueError("hypothesis too long for max_len=%d: %s" % (MAX_LEN, e))


def encode(m: Model, state: Any, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Rows per question; a question with a hypothesis too long for MAX_LEN goes to `rejected` (400)."""
    premise = serialize_state(state)
    items, rejected = [], {}
    for qid, qdef in questions.items():
        q = question_rows(m.templates, qdef)
        try:
            rows = [{"hypothesis": h, "ids": encode_pair(m.tok, premise, h)} for h in q["hypotheses"]]
        except ValueError as e:
            rejected[qid] = str(e)
            continue
        if any(len(r["ids"]) > MAX_LEN for r in rows):
            rejected[qid] = "does not fit in %d tokens" % MAX_LEN
            continue
        items.append({"qid": qid, "qtype": qdef["type"], "mode": q["mode"], "rows": rows})
    return {"premise": premise, "items": items, "rejected": rejected}


def collate(enc: Dict[str, Any], pad_id: int):
    rows = [r for it in enc["items"] for r in it["rows"]]
    s = max(len(r["ids"]) for r in rows)
    ids = np.full((len(rows), s), pad_id, dtype=np.int64)
    att = np.zeros((len(rows), s), dtype=np.int64)
    for i, r in enumerate(rows):
        ids[i, : len(r["ids"])] = r["ids"]
        att[i, : len(r["ids"])] = 1
    return {"input_ids": ids, "attention_mask": att}


@torch.no_grad()
def forward(m: Model, enc: Dict[str, Any]) -> np.ndarray:
    """Raw [entailment, not_entailment] logits per row, one unpadded row at a time (as the pipeline runs)."""
    out = []
    for it in enc["items"]:
        for r in it["rows"]:
            ids = torch.tensor([r["ids"]], device=m.device)
            out.append(m.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0].float().cpu().numpy())
    return np.stack(out)


def option_logits(item: Dict[str, Any], scores: np.ndarray) -> np.ndarray:
    """scores: this question's rows [r, 2]. Ollaya order: K for choice/score, [false, true] for noul."""
    scores = np.asarray(scores, dtype=np.float64)
    if item["mode"] == "single":
        return np.array([scores[0, NOT_ENTAILMENT], scores[0, ENTAILMENT]])
    return scores[:, ENTAILMENT]


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def split(enc, scores):
    i = 0
    for it in enc["items"]:
        yield it, scores[i:i + len(it["rows"])]
        i += len(it["rows"])


def decide(m: Model, state: Any, questions: Dict[str, Dict[str, Any]]) -> Dict[str, np.ndarray]:
    enc = encode(m, state, questions)
    return {it["qid"]: softmax(option_logits(it, s)) for it, s in split(enc, forward(m, enc))}


def upstream_probabilities(m: Model, premise: str, item: Dict[str, Any]):
    """The same question through `pipeline("zero-shot-classification")`, Ollaya order (None if ambiguous)."""
    hyps = [r["hypothesis"] for r in item["rows"]]
    if len(set(hyps)) != len(hyps) or not premise:
        return None  # the pipeline cannot map duplicate labels back, and refuses an empty sequence
    if item["mode"] == "entail" and len(hyps) == 1:
        # A one-option choice: the pipeline switches to multi-label (entail vs not) for a lone
        # candidate; Ollaya answers such a question with probability 1.0, as every layout does.
        return None
    res = m.pipe(premise, candidate_labels=hyps, hypothesis_template="{}", multi_label=False)
    p = dict(zip(res["labels"], res["scores"]))
    if item["mode"] == "single":
        return np.array([1 - p[hyps[0]], p[hyps[0]]])
    return np.array([p[h] for h in hyps])


def main():
    import sys

    os.environ.setdefault("USE_TF", "0")
    m = load("cpu", sys.argv[1] if len(sys.argv) > 1 else "deberta-v3-large")
    state = {"subject": "Invoice #4411", "body": "We were billed twice for March, please refund."}
    qs = {"dept": {"type": "choice", "instructions": "Which team handles this?",
                   "criteria": {"billing": "payments", "technical": "bugs", "other": ""}},
          "refund": {"type": "noul", "instructions": "The user asks for a refund."},
          "angry": {"type": "noul", "instructions": "The customer is angry.",
                    "criteria": {"true": "The customer is angry.", "false": "The customer is calm."}}}
    enc = encode(m, state, qs)
    for it, s in split(enc, forward(m, enc)):
        print(it["qid"], [r["hypothesis"] for r in it["rows"]])
        print("   ", softmax(option_logits(it, s)), upstream_probabilities(m, enc["premise"], it))


if __name__ == "__main__":
    main()
