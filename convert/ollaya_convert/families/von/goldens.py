"""Golden fixtures for the Rust port of `von-option-marker-v1`.

    uv run --with von-sdk==1.1.1 python -m ollaya_convert.families.von.goldens --out out/goldens-von.jsonl [--all]

Needs a GPU for the rows up to 4096 tokens (about 1 min for the default set) and about 40 GB of RAM
for the two 8192-token rows of `von/over_max_len`, which run in float64 on the CPU.

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

Numbers are the upstream network (same weights, same code) run in float64, one unpadded row at a time:
`ref.Exact`, with the rotary embedding kept in float64 too. `--precision fp32` reproduces the earlier
goldens (upstream's fp32 forward, TF32 off), whose attention ran through PyTorch's CUDA memory-efficient
SDPA kernel; on one row that was 1.05e-3 from the exact value (docs/families/von.md). `answers` is always
von's own `evaluate()` as von runs it (fp32 on `--device`).
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
    ap.add_argument("--precision", choices=["fp64", "fp32"], default="fp64",
                    help="reference numbers: the network in float64 (default), or upstream's fp32 forward")
    ap.add_argument("--long-device", default="cpu",
                    help="where fp64 rows over %d tokens run (fp64 attention at 8192 tokens needs ~25 GB)"
                         % ref.EXACT_LONG_TOKENS)
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    backend = ref.load(a.device)
    tok = backend._get_model().tokenizer
    exact = ref.Exact(backend, a.device, a.long_device) if a.precision == "fp64" else None
    n = 0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for cid, state, questions in list(cases.all_cases(0 if a.all else 20)) + extra_cases():
            state, questions = wire(state), wire(questions)
            enc = ref.encode(tok, state, questions)
            row_logits = ref.forward_rows(backend, enc, exact)
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
            f.flush()
            n += 1
            print("%3d %s" % (n, cid), flush=True)
    print("wrote %d cases to %s (%.1f MB)" % (n, a.out, os.path.getsize(a.out) / 2**20))


if __name__ == "__main__":
    main()
