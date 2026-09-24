"""The PyTorch reference for `mobarmg/jev-schema-scorer-deberta-v3-large`, run through its own code.

    uv run python -m ollaya_convert.families.schema_scorer.ref

Upstream (MIT): `schema_scorer.py`, shipped in the model repo @ ee092c35 and imported from there, so the
request compiler cannot drift:
  * compile_question(state, question) -> candidates [(id, description)], text pairs
        sequence_a = state if str else json.dumps(state, ensure_ascii=False)
        sequence_b = json.dumps({"candidate": {"id", "description"}, "type", "instructions"[, "criteria"]},
                                ensure_ascii=False)
  * LocalSystemOne.score_pairs: tokenizer(a, b, padding=True, truncation=False), batches of 32, a pair
    longer than max_position_embeddings (512) is an error; DebertaV2ForSequenceClassification with one
    logit per pair
  * decode_answer: softmax over a question's candidates; noul = p(candidate "true"), order [false, true]
The TypeSafe request is consumed as is: this model was trained on that schema, so there is no Ollaya
mapping layer at all.
"""
import importlib.util
import json
import os
import sys
from typing import Any, Dict

import numpy as np
import torch

REPO = "mobarmg/jev-schema-scorer-deberta-v3-large"
REVISION = "ee092c35cc4ba0bd81be06a351b8788a47668d8c"
MAX_LEN = 512


class Model:
    def __init__(self, device: str):
        from huggingface_hub import snapshot_download

        self.dir = snapshot_download(REPO, revision=REVISION)
        spec = importlib.util.spec_from_file_location("schema_scorer", os.path.join(self.dir, "schema_scorer.py"))
        self.upstream = importlib.util.module_from_spec(spec)
        sys.modules["schema_scorer"] = self.upstream  # its @dataclass needs the module registered
        spec.loader.exec_module(self.upstream)
        assert self.upstream.CANDIDATE_FIRST, "SCORER_CANDIDATE_FIRST must be unset (the trained layout)"
        self.scorer = self.upstream.LocalSystemOne(self.dir, device=torch.device(device))
        assert self.scorer.max_length == MAX_LEN
        self.tok = self.scorer.tokenizer
        self.model = self.scorer.model
        self.device = device


def load(device: str = "cpu") -> Model:
    return Model(device)


def to_upstream(qdef: Dict[str, Any]) -> Dict[str, Any]:
    """The one Ollaya extension: TypeSafe's list-of-labels choice becomes {label: None} (upstream
    requires a dict), exactly what `von`/Laya/TypeSafe mean by it. Everything else passes unchanged."""
    if qdef.get("type") == "choice" and isinstance(qdef.get("criteria"), list):
        qdef = dict(qdef, criteria={c: None for c in qdef["criteria"]})
    return qdef


def encode(m: Model, state: Any, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Rows per question. Questions upstream refuses (fewer than two candidates, malformed criteria)
    are listed in `rejected` with the upstream error; the runtime answers them with HTTP 400."""
    items, rejected = [], {}
    for qid, qdef in questions.items():
        try:
            candidates, pairs = m.upstream.compile_question(state, to_upstream(qdef))
        except ValueError as e:
            rejected[qid] = str(e)
            continue
        rows = []
        for (cid, _), (a, b) in zip(candidates, pairs):
            ids = m.tok(a, b)["input_ids"]  # special tokens on, no truncation, as score_pairs
            rows.append({"candidate": cid, "text_a": a, "text_b": b, "ids": ids})
        longest = max(len(r["ids"]) for r in rows)
        if longest > MAX_LEN:  # score_pairs raises for the whole batch; Ollaya rejects the question
            rejected[qid] = "State + question + criteria exceed the input limit (%d > %d tokens)." % (longest, MAX_LEN)
            continue
        items.append({"qid": qid, "qtype": qdef["type"], "rows": rows})
    return {"items": items, "rejected": rejected}


def collate(enc: Dict[str, Any], pad_id: int):
    rows = [r for it in enc["items"] for r in it["rows"]]
    s = max(len(r["ids"]) for r in rows)
    ids = np.full((len(rows), s), pad_id, dtype=np.int64)
    att = np.zeros((len(rows), s), dtype=np.int64)
    for i, r in enumerate(rows):
        ids[i, : len(r["ids"])] = r["ids"]
        att[i, : len(r["ids"])] = 1
    return {"input_ids": ids, "attention_mask": att}


def forward(m: Model, enc: Dict[str, Any]) -> np.ndarray:
    """Upstream LocalSystemOne.score_pairs over all rows of the request: [r, 1] fp32 logits."""
    pairs = [(r["text_a"], r["text_b"]) for it in enc["items"] for r in it["rows"]]
    return m.scorer.score_pairs(pairs).numpy()[:, None]


def split(enc, scores):
    i = 0
    for it in enc["items"]:
        yield it, scores[i:i + len(it["rows"])]
        i += len(it["rows"])


def option_logits(item, scores) -> np.ndarray:
    """Candidate order is Ollaya order already (noul candidates are [false, true])."""
    return np.asarray(scores, dtype=np.float64)[:, 0]


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def decide(m: Model, state: Any, questions: Dict[str, Dict[str, Any]]) -> Dict[str, np.ndarray]:
    enc = encode(m, state, questions)
    return {it["qid"]: softmax(option_logits(it, s)) for it, s in split(enc, forward(m, enc))}


def upstream_answers(m: Model, state, questions):
    return m.scorer.system_one(state, questions)["answers"]


def main():
    os.environ.setdefault("USE_TF", "0")
    m = load("cpu")
    state = {"subject": "Invoice #4411", "body": "We were billed twice for March, please refund."}
    qs = {"dept": {"type": "choice", "instructions": "Which team handles this?",
                   "criteria": {"billing": "payments", "technical": "bugs", "other": None}},
          "refund": {"type": "noul", "instructions": "The user asks for a refund."}}
    enc = encode(m, state, qs)
    print(enc["items"][0]["rows"][2]["text_b"])
    for it, s in split(enc, forward(m, enc)):
        print(it["qid"], softmax(option_logits(it, s)))
    print(json.dumps(upstream_answers(m, state, qs)))


if __name__ == "__main__":
    main()
