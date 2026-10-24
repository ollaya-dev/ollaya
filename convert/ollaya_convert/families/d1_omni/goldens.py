"""Golden fixtures for the Rust port of `d1-omni-markers-v1`, straight from the model repo's code in fp32.

    uv run python -m ollaya_convert.families.d1_omni.goldens out/d1-omni-600m [--td-limit 20]

Writes out/goldens-d1-omni-600m.jsonl, one JSON line per request (JSON round-tripped first, as over HTTP):
    {"id", "state", "questions",
     "error": null | "<upstream ValueError>",       # the whole request is rejected (HTTP 422)
     "items": [{"qid", "qtype", "k",
                "ids", "markers",                   # the row, upstream `prompt.encode`
                "logits",                           # the head's raw marker logits, fp32 (TF32 off)
                "temperature",                      # config.temperatures for the question's type and option count
                "probabilities"}],                  # softmax(logits / temperature), Ollaya order ([false, true])
     "answers": {...}}                              # upstream `system_one` of the same request
A rejected request is followed by a line "<id>#valid" with the questions upstream accepts on their own.
"""
from __future__ import annotations

import argparse
import json
import os

import tokenizers

from ... import cases
from . import ref
from .layout import D1OmniLayout, softmax


def wire(x):
    return json.loads(json.dumps(x, ensure_ascii=False))


def record(m, lay, cid, state, questions):
    rows = ref.encode(m, state, questions)
    port = lay.encode(state, questions)
    assert [r["ids"] for r in rows] == [p["ids"] for p in port], cid
    assert [r["markers"] for r in rows] == [p["markers"] for p in port], cid
    logits = ref.forward(m, rows)
    probs = ref.probabilities(m, state, questions)
    items = []
    for r, z, p_up in zip(rows, logits, probs):
        t = ref.temperature(m, r["qtype"], r["k"])
        p = softmax(z, t)
        assert max(abs(a - b) for a, b in zip(p, p_up)) < 1e-5, (cid, r["qid"], p, p_up)
        items.append({"qid": r["qid"], "qtype": r["qtype"], "k": r["k"], "ids": r["ids"], "markers": r["markers"],
                      "logits": z, "temperature": t, "probabilities": p})
    return {"id": cid, "state": state, "questions": questions, "error": None, "items": items,
            "answers": ref.system_one(m, state, questions)["answers"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--out", default=None, help="default out/goldens-d1-omni-600m.jsonl")
    ap.add_argument("--root", default=None)
    ap.add_argument("--td-limit", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    root = a.root or ref.snapshot()
    m = ref.load(root, device=a.device)
    decision = json.load(open(os.path.join(a.model_dir, "decision.json"), encoding="utf-8"))
    tok = tokenizers.Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json"))
    tok.no_truncation()
    tok.no_padding()
    lay = D1OmniLayout(tok, decision)
    out = a.out or os.path.join(os.path.dirname(a.model_dir.rstrip("/")), "goldens-d1-omni-600m.jsonl")
    n = rejected = 0
    with open(out, "w", encoding="utf-8") as f:
        for cid, state, questions in cases.all_cases(a.td_limit):
            state, questions = wire(state), wire(questions)
            try:
                rec = record(m, lay, cid, state, questions)
            except ValueError as e:
                rejected += 1
                f.write(json.dumps({"id": cid, "state": state, "questions": questions, "error": repr(e),
                                    "items": [], "answers": None}, ensure_ascii=False) + "\n")
                valid = {}
                for qid, q in questions.items():
                    try:
                        ref.encode(m, state, {qid: q})
                        valid[qid] = q
                    except ValueError:
                        pass
                if valid:
                    rec = record(m, lay, cid + "#valid", state, valid)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 20 == 0:
                print("  %d requests" % n, flush=True)
    print("wrote %s: %d requests, %d rejected" % (out, n, rejected))


if __name__ == "__main__":
    main()
