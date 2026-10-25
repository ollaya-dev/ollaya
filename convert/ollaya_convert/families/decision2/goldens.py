"""Golden fixtures for the Rust port of `decision2-endpoint-v1`, straight from the author's code in fp32.

    uv run python -m ollaya_convert.families.decision2.goldens decision2-eos out/decision2-eos [--td-limit 20]

Writes out/goldens-<model>.jsonl in the format of `decision.goldens`, so `parity_decision` reads it:
    {"id", "state", "questions",
     "error": null | "<the author's exception class>",   # a question upstream refuses rejects the request
     "rows": [{"ids": [...], "query": int, "cands": [...]}],   # one row per question, request order
     "plan": [{"qid", "type", "k",
               "option_logits": [...],   # as the engine returns them: noul [false, true], score plus Kai's offsets
               "probabilities": [...]}], # softmax(option_logits), float64 (temperature 1)
     "answers": {...}}                   # the author's product_answer
A rejected request is followed by a line "<id>#valid" with the questions the author accepts on their own.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import tokenizers

from ..llm_common import cases
from . import ref
from .layout import Decision2Layout, LayoutError

LONG = "Read the whole policy below before you answer. " * 2000  # about 18,000 tokens: over every limit (16,384)
EXTRA_CASES = [
    ("decision2/noul_order", "The parcel arrived on time, but the box was crushed.",
     {"swapped": {"type": "noul", "instructions": "Was the box damaged?",
                  "criteria": {"true": "The box was damaged", "false": "The box was intact"}},
      "half": {"type": "noul", "instructions": "Was it late?", "criteria": {"true": "It was late"}},
      "bare": {"type": "noul", "instructions": "Did it arrive?", "criteria": {}}}),
    ("decision2/five_levels", "The support agent answered within a minute and fixed the issue.",
     {"s5": {"type": "score", "instructions": "How satisfied is the customer?",
             "criteria": ["Very unhappy", "Unhappy", "Neutral", "Happy", "Very happy"]},
      "s4": {"type": "score", "instructions": "How fast was the reply?", "criteria": ["Slow", "OK", "Fast", "Instant"]},
      "obj": {"type": "score", "instructions": {"rate": "effort"}, "criteria": [{"level": "low"}, ["mid"], "high"]}}),
    ("decision2/falsy", {"order": "A-104", "items": [1, 2.5, None, True], "note": "çok güzel 日本"},
     {"empty": {"type": "choice", "instructions": ["route", "it"], "criteria": {"a": "", "b": None, "c": {"k": []}}},
      "n": {"type": "noul", "instructions": {"q": "ok?"}, "criteria": {"false": "", "true": []}}}),
    ("decision2/state_null", None, {"q": {"type": "noul", "instructions": "Is there a state?"}}),
    ("decision2/state_number", 7, {"q": {"type": "noul", "instructions": "Is the state seven?"}}),
    ("decision2/state_empty", "", {"q": {"type": "noul", "instructions": "Is there a state?"}}),
    ("decision2/instructions_empty", "x", {"q": {"type": "noul", "instructions": ""},
                                           "r": {"type": "noul", "instructions": "ok?"}}),
    ("decision2/instructions_null", "x", {"q": {"type": "noul", "instructions": None},
                                          "r": {"type": "noul", "instructions": 3}}),
    ("decision2/noul_null_side", "x", {"q": {"type": "noul", "instructions": "ok?", "criteria": {"false": None}},
                                       "r": {"type": "noul", "instructions": "ok?", "criteria": {"true": 1}}}),
    ("decision2/score_numbers", "x", {"s": {"type": "score", "instructions": "Rate.", "criteria": [1, 2, 3]},
                                      "t": {"type": "score", "instructions": "Rate.", "criteria": ["low", None]}}),
    ("decision2/choice_number", "x", {"c": {"type": "choice", "instructions": "Pick.", "criteria": {"a": 1, "b": "two"}},
                                      "d": {"type": "choice", "instructions": "Pick.", "criteria": {"": "x", "b": "y"}}}),
    ("decision2/one_too_long", "A short state.",
     {"q": {"type": "noul", "instructions": LONG + "Is the state short?"},
      "r": {"type": "choice", "instructions": "Which word?", "criteria": {"short": None, "long": None}}}),
    ("decision2/long_state", "The order arrived late and damaged; the customer wants a refund. " * 400,
     {"refund": {"type": "noul", "instructions": "Does the customer want a refund?"},
      "lvl": {"type": "score", "instructions": "How upset?", "criteria": ["calm", "annoyed", "furious"]}}),
]


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return (e / e.sum()).tolist()


def record(d, lay, cid, state, questions):
    rows_up, items = d.encode(state, questions)  # the author's rejection propagates
    try:
        rows, pmeta = lay.encode(state, questions)
    except LayoutError as e:
        raise AssertionError("%s: the port rejects a request the author accepts: %s" % (cid, e)) from e
    up = [{"ids": it["ids"], "query": it["query_position"], "cands": it["candidate_positions"]} for it in items]
    assert up == rows, cid
    assert [it["keys"] for it in items] == [[o["key"] for o in r["options"]] for r in rows_up], cid
    logits = d.forward(items)
    answers = d.answers(rows_up, items, logits)
    plan = []
    for (qid, q), m, row, z in zip(questions.items(), pmeta, rows_up, logits):
        biased = d.biased(row, z)
        option_logits = lay.option_logits(q, [float(x) for x in z])
        # The port's offsets and order are the author's.
        assert sorted(option_logits) == sorted(biased) and len(option_logits) == m["k"], (cid, qid)
        plan.append({"qid": qid, "type": m["type"], "k": m["k"], "option_logits": option_logits,
                     "probabilities": softmax(option_logits)})
    return {"id": cid, "state": state, "questions": questions, "error": None, "rows": rows, "plan": plan,
            "answers": answers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(ref.MODELS))
    ap.add_argument("model_dir")
    ap.add_argument("--root", default=None)
    ap.add_argument("--td-limit", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = ref.load(a.root or ref.snapshot(a.model), a.model, device=a.device)
    decision = json.load(open(os.path.join(a.model_dir, "decision.json"), encoding="utf-8"))
    lay = Decision2Layout(tokenizers.Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json")), decision)
    path = a.out or os.path.join(os.path.dirname(os.path.abspath(a.model_dir)),
                                 "goldens-%s.jsonl" % os.path.basename(os.path.abspath(a.model_dir)))
    n = 0
    # Written aside and moved into place at the end: a run that dies leaves no partial goldens to pass parity on.
    with open(path + ".part", "w", encoding="utf-8") as f:
        for cid, state, questions in list(cases.all_cases(a.td_limit)) + EXTRA_CASES:
            try:
                rec = record(d, lay, cid, state, questions)
            except (ValueError, AttributeError, KeyError) as e:  # the author's refusal (or a non-object question)
                f.write(json.dumps({"id": cid, "state": state, "questions": questions, "error": type(e).__name__},
                                   ensure_ascii=False) + "\n")
                valid = {}
                for qid, q in (questions.items() if isinstance(questions, dict) else ()):
                    try:
                        d.encode(state, {qid: q})
                        valid[qid] = q
                    except (ValueError, AttributeError, KeyError):
                        pass
                if not valid:
                    continue
                rec = record(d, lay, cid + "#valid", state, valid)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 20 == 0:
                print("%d records" % n, flush=True)
    os.replace(path + ".part", path)
    print("wrote %d records to %s (%.1f MB)" % (n, path, os.path.getsize(path) / 2**20))


if __name__ == "__main__":
    main()
