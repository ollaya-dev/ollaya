"""Parity of the weightless ONNX export (and of the layout port) against the `gliclass` package in fp32.

    uv run --with gliclass==0.1.21 python -m ollaya_convert.families.gliclass_kv.parity gliclass-v4-2b out/gliclass-v4-2b \\
        --td-limit 50 [--report out/gliclass-v4-2b/parity.json] [--upstream-every 4]

Per case (the shared edge cases + typed-decisions rows, JSON round-tripped):
  1. upstream rows (`prepare_inputs`) vs the port (`layout.py` + `tokenizers`): ids, markers and section identical;
  2. calibrated label logits and the halting steps: upstream `GLiClassModel.forward` in fp32 (GPU, TF32 off, or the CPU)
     vs ONNX Runtime CPU on the padded batch of the request's rows;
  3. probabilities (Ollaya option logits, softmax): argmax agreement, and every `--upstream-every`-th case against the
     public pipeline call (`upstream_scores`).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import tokenizers
import torch

from ... import cases
from ..llm_common.ort_rows import session
from . import ref
from .export import INPUT_NAMES, OUTPUT_NAMES, collate
from .layout import GliclassKvLayout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(ref.MODELS))
    ap.add_argument("model_dir")
    ap.add_argument("--root", default=None)
    ap.add_argument("--td-limit", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=0, help="ONNX Runtime intra-op threads (0 = all)")
    ap.add_argument("--upstream-every", type=int, default=4, help="check the public pipeline on every n-th case (0 = never)")
    ap.add_argument("--report", default=None, help="write a JSON summary here")
    a = ap.parse_args()

    m = ref.load(a.model, a.device, a.root)
    decision = json.load(open(os.path.join(a.model_dir, "decision.json"), encoding="utf-8"))
    tok = tokenizers.Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json"))
    tok.no_truncation()
    tok.no_padding()
    lay = GliclassKvLayout(tok, decision)
    sess = session(os.path.join(a.model_dir, "model.onnx"), a.threads)
    pad = decision["special_tokens"]["pad"]

    case_list = cases.edge_cases() + cases.typed_decisions(a.td_limit)
    st = defaultdict(list)
    n_q = agree = agree_up = n_up = row_mismatch = steps_mismatch = 0
    worst = []
    t_ort = t_ref = 0.0
    keys = ("ids", "markers", "section_start", "sep", "labels", "prompt", "mode")
    for ci, (cid, state, questions) in enumerate(case_list):
        state, questions = json.loads(json.dumps(state)), json.loads(json.dumps(questions))
        enc = ref.encode(m, state, questions)
        port = lay.encode(state, questions)
        for r, p in zip(enc["items"], port):
            if {k: r[k] for k in keys} != {k: p[k] for k in keys}:
                row_mismatch += 1
        t0 = time.perf_counter()
        want = ref.forward(m, enc)
        t_ref += time.perf_counter() - t0
        feed = {n: x.numpy() for n, x in zip(INPUT_NAMES, collate(enc["items"], pad))}
        t0 = time.perf_counter()
        logits, steps = sess.run(OUTPUT_NAMES, feed)
        t_ort += time.perf_counter() - t0
        check_up = a.upstream_every and ci % a.upstream_every == 0
        for i, (it, w) in enumerate(zip(enc["items"], want)):
            k = len(it["labels"])
            z_got = logits[i, :k].astype(np.float64)
            st["logit/" + it["qtype"]].append(float(np.abs(z_got - w["logits"]).max()))
            steps_mismatch += int(steps[i] != w["steps"])
            p_ref = ref.softmax(ref.option_logits(it, w["logits"]))
            p_got = ref.softmax(ref.option_logits(it, z_got))
            dp = float(np.abs(p_ref - p_got).max())
            st["prob/" + it["qtype"]].append(dp)
            same = int(np.argmax(p_ref) == np.argmax(p_got))
            n_q += 1
            agree += same
            worst.append((dp, cid, it["qid"], same, int(steps[i]), w["steps"]))
            if check_up:
                p_up = ref.upstream_scores(m, it, enc["state_text"])
                st["contract_vs_upstream_pipeline"].append(float(np.abs(p_ref - p_up).max()))
                n_up += 1
                agree_up += int(np.argmax(p_got) == np.argmax(p_up))
        if (ci + 1) % 10 == 0:
            print("  %d/%d cases, %d questions, ort %.0fs ref %.0fs" % (ci + 1, len(case_list), n_q, t_ort, t_ref), flush=True)

    summary = {
        "model": a.model, "cases": len(case_list), "questions": n_q, "row_mismatches": row_mismatch,
        "steps_mismatches": steps_mismatch,
        "argmax_agreement_onnx_vs_fp32": agree / max(n_q, 1),
        "upstream_pipeline_questions": n_up, "argmax_agreement_onnx_vs_upstream_pipeline": agree_up / max(n_up, 1),
        "ort_seconds": round(t_ort, 1), "ref_seconds": round(t_ref, 1),
        "stats": {k: {"max": float(np.max(v)), "p99": float(np.quantile(v, 0.99)), "mean": float(np.mean(v))}
                  for k, v in sorted(st.items())},
        "worst": sorted(worst, reverse=True)[:8],
    }
    print(json.dumps(summary, indent=1))
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
