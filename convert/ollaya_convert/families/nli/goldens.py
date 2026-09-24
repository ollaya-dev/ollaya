"""Golden fixtures for the Rust port of `nli-pairs-v1`.

    uv run python -m ollaya_convert.families.nli.goldens deberta-v3-large \
        --out out/goldens-deberta-v3-large-zeroshot-v2.0.jsonl [--all]

One JSON line per case (requests JSON round-tripped first):
    {"id", "state", "questions", "premise", "rejected",       # rejected: {qid: reason}, answered 400
     "items": [{"qid", "qtype", "mode",                       # mode: entail | single
                "rows": [{"hypothesis", "ids", "scores"}],    # scores = fp32 [entailment, not_entailment]
                "option_logits", "probabilities"}]}           # Ollaya order, T = 1
"""
import argparse
import json
import os

import torch

from ... import cases
from . import ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(ref.MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    m = ref.load(a.device, a.model)
    n = 0
    with open(a.out, "w") as f:
        for cid, state, questions in cases.all_cases(0 if a.all else 20):
            state, questions = json.loads(json.dumps(state)), json.loads(json.dumps(questions))
            enc = ref.encode(m, state, questions)
            items = []
            for it, sc in (ref.split(enc, ref.forward(m, enc)) if enc["items"] else []):
                ol = ref.option_logits(it, sc)
                items.append({"qid": it["qid"], "qtype": it["qtype"], "mode": it["mode"],
                              "rows": [{"hypothesis": r["hypothesis"], "ids": r["ids"], "scores": [float(x) for x in s]}
                                       for r, s in zip(it["rows"], sc)],
                              "option_logits": [float(x) for x in ol],
                              "probabilities": [float(x) for x in ref.softmax(ol)]})
            f.write(json.dumps({"id": cid, "state": state, "questions": questions, "premise": enc["premise"],
                                "rejected": enc["rejected"], "items": items}, ensure_ascii=False) + "\n")
            n += 1
    print("wrote %d cases to %s (%.1f MB)" % (n, a.out, os.path.getsize(a.out) / 2**20))


if __name__ == "__main__":
    main()
