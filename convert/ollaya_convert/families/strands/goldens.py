"""Golden fixtures for the Rust port of `strands-pointer-v1`, straight from upstream strands_decider in fp32.

    STRANDS_SRC=/path/to/strands-decider/src uv run --with peft==0.21.0 --with pydantic==2.12.5 \
        python -m ollaya_convert.families.strands.goldens strands-decider-2b out/strands-decider-2b

Writes out/goldens-<model>.jsonl, one JSON line per request:
    {"id", "state", "questions",
     "error": null | "<upstream message>",           # the whole request is rejected (HTTP 422)
     "rows": [{"ids": [...], "decide": int, "opts": [...]}],   # one row per question, request order
     "plan": [{"qid", "type", "k",
               "option_logits": [...],                # raw pointer scores, fp32 (LoRA unmerged, TF32 off)
               "probabilities": [...]}],              # softmax(option_logits / temperature[type])
     "answers": {...}}                                # upstream _to_answer of these scores (4 decimals)
A rejected request is followed by a line "<id>#valid" with the questions upstream accepts on their own.
Every record's rows are also checked against the layout port (layout.py) before they are written.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import tokenizers

from ..llm_common import cases
from . import ref
from .layout import LayoutError, StrandsLayout

# Long questions and many of them: the front cut, the per-run state budget and runs of 32 questions.
LONG_INSTRUCTIONS = "Read the whole policy below before you answer. " * 400
EXTRA_CASES = [
    ("strands/long_question", "A short state.",
     {"q": {"type": "noul", "instructions": LONG_INSTRUCTIONS + "Is the state short?"},
      "r": {"type": "choice", "instructions": "Which word?", "criteria": {"short": None, "long": None}}}),
    ("strands/long_state", "The ledger shows a duplicate charge. " * 900,
     {"dup": {"type": "noul", "instructions": "Is there a duplicate charge?"},
      "lvl": {"type": "score", "instructions": "How severe?", "criteria": ["minor", "moderate", "severe"]}}),
    ("strands/cut_options", "x",
     {"q": {"type": "choice", "instructions": "Pick one.",
            "criteria": {"opt%d" % i: "a long description of option %d " % i * 40 for i in range(40)}}}),
    ("strands/forty_questions", "The order arrived late and damaged; the customer wants a refund. " * 300,
     {"q%02d" % i: {"type": "noul", "instructions": "Statement %d: the customer is unhappy." % i
                    + (" Very long context." * 300 if i == 35 else "")} for i in range(40)}),
    ("strands/whitespace", "  \t\x1c padded state 　 ",
     {"q": {"type": "choice", "instructions": "\n Which? \n",
            "criteria": {" a ": "  spaced\n\ndescription\t ", "b": "", "c": {"k": [1, 2.5, None, True]}}},
      "s": {"type": "score", "instructions": ["one", "two"], "criteria": ["  low\nend ", "", "high"]},
      "n": {"type": "noul", "instructions": "ok?", "criteria": {"true": None}}}),
]


def softmax(z, t):
    z = np.asarray(z, dtype=np.float64) / t
    e = np.exp(z - z.max())
    return (e / e.sum()).tolist()


def record(model, engine, lay, cid, state, questions):
    rows, meta, rendered = ref.encode(engine, state, questions)
    port, _ = lay.encode(state, questions)
    assert port == rows, cid
    scores = ref.forward(model, rows)
    plan = [{"qid": m["qid"], "type": m["type"], "k": m["k"], "option_logits": [float(x) for x in z],
             "probabilities": softmax(z, ref.temperature(model, m["type"]))} for m, z in zip(meta, scores)]
    return {"id": cid, "state": state, "questions": questions, "error": None, "rows": rows, "plan": plan,
            "answers": ref.answers(model, meta, rendered, scores)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(ref.MODELS))
    ap.add_argument("model_dir")
    ap.add_argument("--run", default=None)
    ap.add_argument("--td-limit", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    run = a.run or ref.snapshot(a.model)[0]
    model, engine = ref.load(run, device=a.device)
    decision = json.load(open(os.path.join(a.model_dir, "decision.json"), encoding="utf-8"))
    lay = StrandsLayout(tokenizers.Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json")), decision)
    path = a.out or os.path.join(os.path.dirname(os.path.abspath(a.model_dir)), "goldens-%s.jsonl" % a.model)
    n, t0 = 0, time.time()
    with open(path, "w", encoding="utf-8") as f:
        for cid, state, questions in [*cases.all_cases(a.td_limit), *[(c, cases.wire(s), cases.wire(q)) for c, s, q in EXTRA_CASES]]:
            try:
                rec = record(model, engine, lay, cid, state, questions)
            except ref.RequestError as e:
                try:
                    lay.encode(state, questions)
                    raise AssertionError("%s: upstream rejects (%s), the port accepts" % (cid, e))
                except LayoutError:
                    pass
                f.write(json.dumps({"id": cid, "state": state, "questions": questions, "error": str(e)},
                                   ensure_ascii=False) + "\n")
                valid = {}
                for qid, q in questions.items():
                    try:
                        ref.encode(engine, state, {qid: q})
                        valid[qid] = q
                    except ref.RequestError:
                        pass
                if not valid:
                    continue
                rec = record(model, engine, lay, cid + "#valid", state, valid)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            print("%4d %-50s %5.0fs" % (n, cid, time.time() - t0), flush=True)
    print("wrote %d records to %s (%.1f MB)" % (n, path, os.path.getsize(path) / 2**20))


if __name__ == "__main__":
    main()
