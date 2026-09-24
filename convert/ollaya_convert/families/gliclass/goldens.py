"""Golden fixtures for the Rust port of `gliclass-uni-v1`.

    uv run --with gliclass==0.1.20 python -m ollaya_convert.families.gliclass.goldens large \
        --out out/goldens-gliclass-instruct-large.jsonl [--all]

One JSON line per case (requests JSON round-tripped first):
    {"id", "state", "questions", "state_text", "rejected",     # rejected: {qid: reason}, answered 400
     "items": [{"qid", "qtype", "mode",          # mode: softmax | pair | sigmoid
                "labels", "prompt", "text",      # the upstream pipeline input for this question
                "ids", "markers",                # encoder row and <<LABEL>> positions
                "logits",                        # fp32 label logits
                "option_logits", "probabilities"}]}   # Ollaya order, T = 1
"""
import argparse
import json
import os

import torch

from ... import cases
from . import ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=sorted(ref.VARIANTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    m = ref.load(a.device, a.variant)
    n = 0
    with open(a.out, "w") as f:
        for cid, state, questions in cases.all_cases(0 if a.all else 20):
            state, questions = json.loads(json.dumps(state)), json.loads(json.dumps(questions))
            enc = ref.encode(m, state, questions)
            items = []
            for it, lg in zip(enc["items"], ref.forward(m, enc) if enc["items"] else []):
                ol = ref.option_logits(it, lg)
                items.append({k: it[k] for k in ("qid", "qtype", "mode", "labels", "prompt", "text", "ids", "markers")}
                             | {"logits": [float(x) for x in lg], "option_logits": [float(x) for x in ol],
                                "probabilities": [float(x) for x in ref.softmax(ol)]})
            f.write(json.dumps({"id": cid, "state": state, "questions": questions, "state_text": enc["state_text"],
                                "rejected": enc["rejected"], "items": items}, ensure_ascii=False) + "\n")
            n += 1
    print("wrote %d cases to %s (%.1f MB)" % (n, a.out, os.path.getsize(a.out) / 2**20))


if __name__ == "__main__":
    main()
