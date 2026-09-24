"""Golden fixtures for the Rust port of `kev-pointer-v1`, straight from upstream kev in fp32.

    KEV_SRC=/path/to/kev uv run --with peft==0.21.0 --with pydantic==2.12.5 \
        python -m ollaya_convert.families.kev.goldens kev-0.8b out/kev-0.8b --run RUN --base BASE

Writes out/goldens-<model>.jsonl, one JSON line per request:
    {"id", "state", "questions",
     "error": null | "<upstream exception class>",    # the whole request is rejected (HTTP 422)
     "rows": [{"ids": [...], "decide": int, "opts": [...]}],   # one row per question, request order
     "plan": [{"qid", "type", "k",
               "option_logits": [...],                 # raw pointer scores, fp32 (LoRA merged, TF32 off)
               "probabilities": [...]}],               # softmax(option_logits / temperature)
     "answers": {...}}                                 # upstream kev.api.to_answers of these scores at T
A rejected request is followed by a line "<id>#valid" with the questions upstream accepts on their own.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import tokenizers

from ..llm_common import cases
from . import ref
from .layout import KevLayout


def softmax(z, t):
    z = np.asarray(z, dtype=np.float64) / t
    e = np.exp(z - z.max())
    return (e / e.sum()).tolist()


def record(ck, tok, m, lay, T, cid, state, questions):
    enc, meta = ref.encode(tok, m, state, questions)
    up_rows = ref.rows(enc)
    rows, pmeta = lay.encode(state, questions)
    assert up_rows == rows, cid
    scores = ref.forward(m, enc)
    answers = ref.answers(meta, scores, T)
    plan = [{"qid": q["qid"], "type": q["type"], "k": q["k"], "option_logits": [float(x) for x in s],
             "probabilities": softmax(s, T)} for q, s in zip(pmeta, scores)]
    return {"id": cid, "state": state, "questions": questions, "error": None, "rows": rows, "plan": plan,
            "answers": answers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(ref.MODELS))
    ap.add_argument("model_dir")
    ap.add_argument("--run", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--td-limit", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    run, base = (a.run, a.base) if a.run and a.base else ref.snapshot(a.model)
    ck, tok, m = ref.load(run, base, device=a.device, merge=True)
    decision = json.load(open(os.path.join(a.model_dir, "decision.json")))
    T = json.load(open(os.path.join(a.model_dir, "calibration.json")))["temperature"][0]
    lay = KevLayout(tokenizers.Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json")), decision)
    path = a.out or os.path.join(os.path.dirname(os.path.abspath(a.model_dir)), "goldens-%s.jsonl" % a.model)
    n = 0
    with open(path, "w") as f:
        for cid, state, questions in cases.all_cases(a.td_limit):
            try:
                rec = record(ck, tok, m, lay, T, cid, state, questions)
            except Exception as e:  # pydantic.ValidationError, ContextOverflow
                if isinstance(e, AssertionError):
                    raise
                f.write(json.dumps({"id": cid, "state": state, "questions": questions, "error": type(e).__name__},
                                   ensure_ascii=False) + "\n")
                valid = {}
                for qid, q in questions.items():
                    try:
                        ref.encode(tok, m, state, {qid: q})
                        valid[qid] = q
                    except Exception:
                        pass
                if not valid:
                    continue
                rec = record(ck, tok, m, lay, T, cid + "#valid", state, valid)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print("wrote %d records to %s (%.1f MB)" % (n, path, os.path.getsize(path) / 2**20))


if __name__ == "__main__":
    main()
