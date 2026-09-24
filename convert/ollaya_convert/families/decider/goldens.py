"""Golden fixtures for the Rust port of `decider-slots-v1`, straight from upstream decider in fp32.

    uv run python -m ollaya_convert.families.decider.goldens decider-0.8b out/decider-0.8b [--td-limit 20]

Writes out/goldens-<model>.jsonl, one JSON line per request:
    {"id", "state", "questions",
     "error": null | "<upstream ValueError>",          # the whole request is rejected (HTTP 422)
     "rows": [{"ids": [...], "slot": int}],             # exact token rows, upstream `decider.prompt.build`
     "plan": [{"qid", "type", "kind": "list"|"iso", "rows": [row indices], "k",
               "label_logits": [[first k (list) or 2 (iso) label logits per row]],   # fp32, TF32 off
               "option_logits": [...],                   # what the runner returns for the question
               "probabilities": [...]}],                 # softmax(option_logits / calibration temperature)
     "answers": {...}}                                   # upstream systemone.assemble of these logits
                                                         # (what system_one returns, rounded to 4 decimals)
A rejected request is followed by a line "<id>#valid" with the questions upstream accepts on their own.
"""
from __future__ import annotations

import argparse
import json
import os

import tokenizers

from ..llm_common import cases
from . import ref
from .layout import DeciderLayout, softmax

QT = {"choice": 0, "score": 1, "noul": 2}


def record(d, lay, calib, cid, state, questions):
    items, index = ref.encode(d, state, questions)
    rows, plan = lay.encode(state, questions)
    assert [it["ids"] for it in items] == [r["ids"] for r in rows], cid
    logits = ref.forward(d, items)
    answers = ref.answers(d, state, questions, logits)
    out_plan = []
    for item in plan:
        z = lay.option_logits(item, logits)
        out_plan.append({**item,
                         "label_logits": [[float(x) for x in logits[r][: rows[r]["k"]]] for r in item["rows"]],
                         "option_logits": z, "probabilities": softmax(z, calib[QT[item["type"]]])})
    return {"id": cid, "state": state, "questions": questions, "error": None,
            "rows": [{"ids": r["ids"], "slot": r["slot"]} for r in rows], "plan": out_plan, "answers": answers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(ref.MODELS))
    ap.add_argument("model_dir")
    ap.add_argument("--root", default=None)
    ap.add_argument("--td-limit", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = ref.load(a.root or ref.snapshot(a.model), device=a.device)
    decision = json.load(open(os.path.join(a.model_dir, "decision.json")))
    calib = json.load(open(os.path.join(a.model_dir, "calibration.json")))["temperature"]
    lay = DeciderLayout(tokenizers.Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json")), decision)
    path = a.out or os.path.join(os.path.dirname(os.path.abspath(a.model_dir)), "goldens-%s.jsonl" % a.model)
    n = 0
    with open(path, "w") as f:
        for cid, state, questions in cases.all_cases(a.td_limit):
            try:
                rec = record(d, lay, calib, cid, state, questions)
            except ValueError as e:
                f.write(json.dumps({"id": cid, "state": state, "questions": questions, "error": str(e)},
                                   ensure_ascii=False) + "\n")
                valid = {}
                for qid, q in questions.items():
                    try:
                        ref.encode(d, state, {qid: q})
                        valid[qid] = q
                    except ValueError:
                        pass
                if not valid:
                    continue
                rec = record(d, lay, calib, cid + "#valid", state, valid)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print("wrote %d records to %s (%.1f MB)" % (n, path, os.path.getsize(path) / 2**20))


if __name__ == "__main__":
    main()
