"""Golden fixtures for the Rust port of `decision-endpoint-v1`, straight from the author's code in fp32.

    uv run python -m ollaya_convert.families.decision.goldens decision-eos out/decision-eos-0.8b [--td-limit 20]

Writes out/goldens-<model>.jsonl, one JSON line per request:
    {"id", "state", "questions",
     "error": null | "<the author's exception class>",   # the whole request is rejected
     "rows": [{"ids": [...], "query": int, "cands": [...]}],   # one row per question, request order
     "plan": [{"qid", "type", "k",
               "option_logits": [...],                 # raw candidate logits, fp32 (TF32 off, exact SDPA)
               "probabilities": [...]}],               # softmax(option_logits / T), float64
     "answers": {...}}                                 # the author's typed_answer of softmax(logits / T)
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
from .layout import DecisionLayout, LayoutError


def softmax(z, t):
    z = np.asarray(z, dtype=np.float64) / t
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
    logits = [z.numpy() for z in d.forward(items)]
    answers = d.answers(rows_up, logits)
    plan = [{"qid": q["qid"], "type": q["type"], "k": q["k"], "option_logits": [float(x) for x in z],
             "probabilities": softmax(z, d.T[q["type"]])} for q, z in zip(pmeta, logits)]
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
    decision = json.load(open(os.path.join(a.model_dir, "decision.json")))
    lay = DecisionLayout(tokenizers.Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json")), decision)
    path = a.out or os.path.join(os.path.dirname(os.path.abspath(a.model_dir)),
                                 "goldens-%s.jsonl" % os.path.basename(os.path.abspath(a.model_dir)))
    n = 0
    with open(path, "w") as f:
        for cid, state, questions in cases.all_cases(a.td_limit):
            try:
                rec = record(d, lay, cid, state, questions)
            except (ValueError, AttributeError) as e:  # the author's ValueError (or a non-object question)
                f.write(json.dumps({"id": cid, "state": state, "questions": questions, "error": type(e).__name__},
                                   ensure_ascii=False) + "\n")
                valid = {}
                for qid, q in questions.items():
                    try:
                        d.encode(state, {qid: q})
                        valid[qid] = q
                    except (ValueError, AttributeError):
                        pass
                if not valid:
                    continue
                rec = record(d, lay, cid + "#valid", state, valid)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print("wrote %d records to %s (%.1f MB)" % (n, path, os.path.getsize(path) / 2**20))


if __name__ == "__main__":
    main()
