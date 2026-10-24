"""Golden fixtures for the Rust port of `gliclass-kv-v1`, straight from the `gliclass` package in fp32.

    uv run --with gliclass==0.1.21 python -m ollaya_convert.families.gliclass_kv.goldens gliclass-v4-2b out/gliclass-v4-2b [--td-limit 20]

Writes out/goldens-<model>.jsonl, one JSON line per request (JSON round-tripped first, as over HTTP):
    {"id", "state", "questions", "state_text",
     "items": [{"qid", "qtype", "mode",                  # mode: softmax | pair | sigmoid
                "labels", "prompt",                      # the upstream pipeline call
                "ids", "markers", "section_start", "sep",   # the decoder row, the <<LABEL>> and final <<SEP>> positions
                "steps",                                 # recurrent steps the halting rule ran
                "logits", "uncalibrated",                # calibrated and raw label logits, fp32 (TF32 off)
                "option_logits", "probabilities"}]}      # Ollaya order, softmax (T = 1)
Every record's rows are checked against the layout port (layout.py) before they are written.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import tokenizers

from ... import cases
from . import ref
from .layout import GliclassKvLayout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(ref.MODELS))
    ap.add_argument("model_dir")
    ap.add_argument("--root", default=None)
    ap.add_argument("--td-limit", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    m = ref.load(a.model, a.device, a.root)
    decision = json.load(open(os.path.join(a.model_dir, "decision.json"), encoding="utf-8"))
    tok = tokenizers.Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json"))
    tok.no_truncation()
    tok.no_padding()
    lay = GliclassKvLayout(tok, decision)
    path = a.out or os.path.join(os.path.dirname(os.path.abspath(a.model_dir)), "goldens-%s.jsonl" % a.model)
    n, t0 = 0, time.time()
    with open(path, "w", encoding="utf-8") as f:
        for cid, state, questions in cases.all_cases(a.td_limit):
            state, questions = json.loads(json.dumps(state)), json.loads(json.dumps(questions))
            enc = ref.encode(m, state, questions)
            port = lay.encode(state, questions)
            keys = ("ids", "markers", "section_start", "sep", "labels", "prompt", "mode")
            assert [{k: r[k] for k in keys} for r in port] == [{k: r[k] for k in keys} for r in enc["items"]], cid
            items = []
            for it, o in zip(enc["items"], ref.forward(m, enc)):
                ol = ref.option_logits(it, o["logits"])
                items.append({k: it[k] for k in ("qid", "qtype", "mode", "labels", "prompt", "ids", "markers",
                                                 "section_start", "sep")}
                             | {"steps": o["steps"], "logits": [float(x) for x in o["logits"]],
                                "uncalibrated": [float(x) for x in o["uncalibrated"]],
                                "option_logits": [float(x) for x in ol],
                                "probabilities": [float(x) for x in ref.softmax(ol)]})
            f.write(json.dumps({"id": cid, "state": state, "questions": questions, "state_text": enc["state_text"],
                                "items": items}, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 10 == 0:
                print("%4d %-50s %5.0fs" % (n, cid, time.time() - t0), flush=True)
    print("wrote %d records to %s (%.1f MB)" % (n, path, os.path.getsize(path) / 2**20))


if __name__ == "__main__":
    main()
