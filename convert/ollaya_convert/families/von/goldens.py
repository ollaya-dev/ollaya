"""Golden fixtures for the Rust port of `von-option-marker-v1`.

    uv run --with von-sdk==1.1.1 python -m ollaya_convert.families.von.goldens --out out/goldens-von.jsonl [--all]

One JSON line per case (requests are JSON round-tripped first, as they arrive over HTTP):
    {"id", "state", "questions",
     "state_text",            # von's _format_state(state), before [MASK] sanitising
     "state_tokens",          # token count feeding the temperature map
     "items": [{"qid", "qtype", "labels", "zero_shot", "upstream_exact",
                "rows": [{"text", "ids", "markers", "logits"}],   # row 1 = empty-state noul row
                "option_logits",      # Ollaya order ([false, true] for noul), after the debias
                "temperature",        # from the calibration map
                "probabilities"}],    # calibrated, Ollaya order
     "answers"}               # von's own evaluate() output when upstream accepts every question, else null

Numbers are the upstream network in fp32 (TF32 off), one unpadded row at a time.
Adds one case beyond the shared set: a state longer than max_len, for the truncation rule.
"""
import argparse
import json
import os

import torch

from ... import cases
from . import ref


def wire(x):
    return json.loads(json.dumps(x, ensure_ascii=False))


def extra_cases():
    long_state = " ".join(["Order 88213 arrived damaged, the box was crushed and the glass inside shattered."] * 700)
    q = {"damaged": {"type": "noul", "instructions": "The customer received a damaged item.",
                     "criteria": {"true": "The item arrived broken.", "false": "The item is fine."}},
         "next": {"type": "choice", "instructions": "What should support do next?",
                  "criteria": {"refund": "Refund the order.", "replace": "Ship a replacement.", "ask": ""}}}
    return [("von/over_max_len", long_state, q)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/goldens-von.jsonl")
    ap.add_argument("--all", action="store_true", help="all 400 typed-decisions rows, not 20")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    backend = ref.load(a.device)
    tok = backend._get_model().tokenizer
    n = 0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for cid, state, questions in list(cases.all_cases(0 if a.all else 20)) + extra_cases():
            state, questions = wire(state), wire(questions)
            enc = ref.encode(tok, state, questions)
            row_logits = ref.forward_rows(backend, enc)
            items, i = [], 0
            for it in enc["items"]:
                rl = row_logits[i:i + len(it["rows"])]
                i += len(it["rows"])
                lg = ref.option_logits(it, rl)
                t = ref.temperature(backend._calib_map, backend._default_temp, lg, enc["state_tokens"])
                items.append({
                    "qid": it["qid"], "qtype": it["qtype"], "labels": it["labels"], "zero_shot": it["zero_shot"],
                    "upstream_exact": it["upstream_exact"],
                    "rows": [{"text": r["text"], "ids": r["ids"], "markers": r["markers"],
                              "logits": [float(x) for x in l]} for r, l in zip(it["rows"], rl)],
                    "option_logits": [float(x) for x in lg],
                    "temperature": float(t),
                    "probabilities": [float(x) for x in ref.probabilities(backend, it, rl, enc["state_tokens"])],
                })
            answers = None
            if all(it["upstream_exact"] for it in enc["items"]):
                answers = backend.evaluate(state, questions).model_dump()["answers"]
            rec = {"id": cid, "state": state, "questions": questions, "state_text": enc["state_text"],
                   "state_tokens": enc["state_tokens"], "items": items, "answers": answers}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print("wrote %d cases to %s (%.1f MB)" % (n, a.out, os.path.getsize(a.out) / 2**20))


if __name__ == "__main__":
    main()
