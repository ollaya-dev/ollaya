"""Golden fixtures for the Rust port of `arbiter-fixed-v1`, straight from the inline reference in fp32.

    uv run --with peft==0.21.0 --with transformers==4.55.0 \
        python -m ollaya_convert.families.arbiter.goldens arbiter-4b out/arbiter-4b --run RUN --base BASE

Writes `out/goldens-<model>.jsonl`, one JSON line per request:
    {"id", "state", "questions",
     "error": null | "<reference exception class>",
     "rows":   [{"ids": [...], "last_pos": int, "slots": [...]}],
     "plan":   [{"qid", "type", "k", "slots": [...],
                 "option_logits": [...],              # scores at the row's valid slots, fp32
                 "probabilities": [...]}],            # softmax(option_logits / temperature)
     "answers": {qid: {"type", "argmax_slot", "argmax_option", "probabilities"}}}
A rejected request is followed by a line "<id>#valid" with the questions the reference accepts on their own.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import ref
from .layout import ArbiterLayout, CHOICE_LETTERS, NOUL_SLOTS, SCORE_LEVELS


# A tiny, self-contained set of goldens; one per primitive, plus two more that exercise choice width
# and the two independent F slots. The real golden suite is built on top of llm_common/cases.py once
# this family runs through the shared harness.
GOLDENS = [
    ("noul-basic",
     "The customer was charged twice for order A-104.",
     {"q1": {"type": "noul", "instructions": "Was the customer charged twice?"}}),
    ("choice-3-way",
     "A refund request about a double charge.",
     {"q1": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "charges and refunds", "tech": "bugs and outages", "other": None}}}),
    ("choice-wide",
     "Classify this intent.",
     {"q1": {"type": "choice", "instructions": "Pick the best label.",
             "criteria": {letter: None for letter in CHOICE_LETTERS[:10]}}}),
    ("score-upset",
     "The customer is frustrated after three failed calls.",
     {"q1": {"type": "score", "instructions": "How upset is the customer?",
             "criteria": ["calm", "mild", "annoyed", "upset", "angry", "furious"]}}),
    ("multi-question",
     "The customer wants a refund for order A-104.",
     {"q1": {"type": "noul", "instructions": "Is this a refund request?"},
      "q2": {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": "charges", "tech": "bugs"}},
      "q3": {"type": "score", "instructions": "Urgency",
             "criteria": ["none", "low", "medium", "high", "critical", "severe"]}}),
]


def softmax(z, t):
    z = np.asarray(z, dtype=np.float64) / t
    e = np.exp(z - z.max())
    return (e / e.sum()).tolist()


def _slot_to_option(qtype: str, slot: int, k: int) -> str:
    if qtype == "noul":
        return "true" if slot == NOUL_SLOTS[0] else "false"
    if qtype == "choice":
        idx = slot - 2
        return CHOICE_LETTERS[idx] if 0 <= idx < k else "?"
    if qtype == "score":
        return str(slot - 18)
    return "?"


def record(tok, m, lay, T, cid, state, questions):
    enc, meta = ref.encode(tok, m, state, questions)
    rows = ref.rows(enc)
    port_rows, pmeta = lay.encode(state, questions)
    for a, b in zip(rows, port_rows):
        assert a["ids"] == b["ids"] and a["last_pos"] == b["last_pos"] and a["slots"] == b["slots"], cid
    scores = ref.forward(m, enc)
    plan = []
    answers = {}
    for q, s in zip(pmeta, scores):
        slots = q["slots"]
        opt_logits = [float(s[i]) for i in slots]
        probs = softmax(opt_logits, T)
        plan.append({"qid": q["qid"], "type": q["type"], "k": q["k"], "slots": slots,
                     "option_logits": opt_logits, "probabilities": probs})
        argmax_i = int(np.argmax(probs))
        answers[q["qid"]] = {
            "type": q["type"],
            "argmax_slot": slots[argmax_i],
            "argmax_option": _slot_to_option(q["type"], slots[argmax_i], q["k"]),
            "probabilities": probs,
        }
    return {"id": cid, "state": state, "questions": questions, "error": None,
            "rows": port_rows, "plan": plan, "answers": answers}


def main():
    import tokenizers
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(ref.MODELS))
    ap.add_argument("model_dir")
    ap.add_argument("--run", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    run, base = (a.run, a.base) if a.run and a.base else ref.snapshot(a.model)
    ck, tok, m = ref.load(run, base, device=a.device, merge=True)
    decision = json.load(open(os.path.join(a.model_dir, "decision.json")))
    T = json.load(open(os.path.join(a.model_dir, "calibration.json")))["temperature"][0]
    lay = ArbiterLayout(tokenizers.Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json")),
                        decision)
    path = a.out or os.path.join(os.path.dirname(os.path.abspath(a.model_dir)),
                                  "goldens-%s.jsonl" % a.model)
    n = 0
    with open(path, "w") as f:
        for cid, state, questions in GOLDENS:
            try:
                rec = record(tok, m, lay, T, cid, state, questions)
            except Exception as e:
                if isinstance(e, AssertionError):
                    raise
                f.write(json.dumps({"id": cid, "state": state, "questions": questions,
                                    "error": type(e).__name__}, ensure_ascii=False) + "\n")
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print("wrote %d records to %s (%.1f KB)" % (n, path, os.path.getsize(path) / 1024))


if __name__ == "__main__":
    main()
