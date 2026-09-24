"""Golden fixtures for the Rust port of `schema-pairs-v1`.

    uv run python -m ollaya_convert.families.schema_scorer.goldens --out out/goldens-jev-schema-scorer-deberta-v3-large.jsonl

One JSON line per case (requests JSON round-tripped first):
    {"id", "state", "questions",
     "rejected": {qid: upstream error},                              # answered 400
     "items": [{"qid", "qtype",
                "rows": [{"candidate", "text_a", "text_b", "ids", "score"}],   # fp32 logit per pair
                "probabilities"}]}                                   # softmax over the rows, T = 1
"""
import argparse
import json
import os

import torch

from ... import cases
from . import ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/goldens-jev-schema-scorer-deberta-v3-large.jsonl")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    m = ref.load(a.device)
    n = 0
    with open(a.out, "w") as f:
        for cid, state, questions in cases.all_cases(0 if a.all else 20):
            state, questions = json.loads(json.dumps(state)), json.loads(json.dumps(questions))
            enc = ref.encode(m, state, questions)
            items = []
            if enc["items"]:
                for it, sc in ref.split(enc, ref.forward(m, enc)):
                    items.append({"qid": it["qid"], "qtype": it["qtype"],
                                  "rows": [dict(r, score=float(s[0])) for r, s in zip(it["rows"], sc)],
                                  "probabilities": [float(x) for x in ref.softmax(ref.option_logits(it, sc))]})
            f.write(json.dumps({"id": cid, "state": state, "questions": questions, "rejected": enc["rejected"],
                                "items": items}, ensure_ascii=False) + "\n")
            n += 1
    print("wrote %d cases to %s (%.1f MB)" % (n, a.out, os.path.getsize(a.out) / 2**20))


if __name__ == "__main__":
    main()
