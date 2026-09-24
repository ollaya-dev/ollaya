"""Parity of the weightless kev ONNX export (and of the layout port) against upstream kev in fp32.

    KEV_SRC=/path/to/kev uv run --with peft==0.21.0 --with pydantic==2.12.5 \
        python -m ollaya_convert.families.kev.parity kev-0.8b out/kev-0.8b --run RUN --base BASE --td-limit 50

Per case (Laya's edge cases + typed-decisions rows, JSON round-tripped):
  1. upstream rows (`kev.api.to_record` -> `kev.model.encode` -> `rows_of`) vs the port (`layout.py` +
     `tokenizers` + the repo's tokenizer.json): token ids, decide and option positions must be identical;
     requests upstream rejects (pydantic 422, context overflow) must be rejected by the port too;
  2. raw pointer scores: upstream fp32 (LoRA merged, GPU, TF32 off) vs ONNX Runtime CPU (LoRA unmerged);
  3. probabilities through the runtime contract (scores / temperature, softmax) from both, and against
     upstream `to_answers` (rounded to 4 decimals upstream).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import tokenizers

from ..llm_common import cases
from ..llm_common.ort_rows import run_rows, session
from . import ref
from .layout import KevLayout, LayoutError


def softmax(z, t):
    z = np.asarray(z, dtype=np.float64) / t
    e = np.exp(z - z.max())
    return e / e.sum()


def upstream_probs(answer):
    if answer["type"] == "noul":
        return [1 - answer["noul"], answer["noul"]]
    return list(answer["probabilities"].values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(ref.MODELS))
    ap.add_argument("model_dir")
    ap.add_argument("--run", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--td-limit", type=int, default=50)
    ap.add_argument("--edge", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=0, help="ONNX Runtime intra-op threads (0 = all)")
    ap.add_argument("--torch-threads", type=int, default=0, help="torch threads for a CPU reference")
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--only-decoder", action="store_true", help="only the decoder-specific edge cases")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    if a.torch_threads:
        import torch

        torch.set_num_threads(a.torch_threads)

    run, base = (a.run, a.base) if a.run and a.base else ref.snapshot(a.model)
    ck, tok, m = ref.load(run, base, device=a.device, merge=True)
    decision = json.load(open(os.path.join(a.model_dir, "decision.json")))
    T = json.load(open(os.path.join(a.model_dir, "calibration.json")))["temperature"][0]
    lay = KevLayout(tokenizers.Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json")), decision)
    sess = session(os.path.join(a.model_dir, "model.onnx"), a.threads)
    pad = decision["special_tokens"]["pad"]

    case_list = (cases.edge_cases() if a.edge else []) + cases.typed_decisions(a.td_limit)
    if a.only_decoder:
        case_list = cases.decoder_cases()
    st = defaultdict(list)
    n_q = agree = agree_up = n_rows = mism = both_rej = 0
    rej_mismatch, worst = [], []
    t_ort = 0.0
    for ci, (cid, state, questions) in enumerate(case_list):
        up_err = port_err = None
        try:
            enc, meta = ref.encode(tok, m, state, questions)
        except Exception as e:  # pydantic.ValidationError, kev.model.ContextOverflow
            up_err = type(e).__name__
        try:
            rows, pmeta = lay.encode(state, questions)
        except LayoutError as e:
            port_err = repr(e)
        if up_err or port_err:
            if up_err and port_err:
                both_rej += 1
            else:
                rej_mismatch.append((cid, up_err, port_err))
            # a request with one invalid question: keep the questions both sides accept on their own
            valid = {}
            for qid, q in questions.items():
                try:
                    ref.encode(tok, m, state, {qid: q})
                    lay.encode(state, {qid: q})
                    valid[qid] = q
                except Exception:
                    pass
            if not valid:
                continue
            questions = valid
            enc, meta = ref.encode(tok, m, state, questions)
            rows, pmeta = lay.encode(state, questions)
        up_rows = ref.rows(enc)
        for u, r in zip(up_rows, rows):
            n_rows += 1
            if u["ids"] != r["ids"] or u["decide"] != r["decide"] or u["opts"] != r["opts"]:
                mism += 1
        ref_scores = ref.forward(m, enc)
        answers = ref.system_one(ck, tok, m, state, questions)

        def extra(idx, L):
            K = max(len(rows[i]["opts"]) for i in idx)
            op = np.zeros((len(idx), K), dtype=np.int64)
            for b, i in enumerate(idx):
                op[b, :len(rows[i]["opts"])] = rows[i]["opts"]
            return {"decide_pos": np.array([rows[i]["decide"] for i in idx], dtype=np.int64), "opt_pos": op}

        got, t = run_rows(sess, [r["ids"] for r in rows], extra, pad, a.budget)
        t_ort += t
        for qi, (q, r) in enumerate(zip(pmeta, rows)):
            k = len(r["opts"])
            z_ref = ref_scores[qi][:k]
            z_got = got[qi][:k]
            st["score/" + q["type"]].append(float(np.abs(z_ref - z_got).max()))
            p_ref, p_got = softmax(z_ref, T), softmax(z_got, T)
            p_up = upstream_probs(answers[q["qid"]])
            dp = float(np.abs(p_ref - p_got).max())
            st["prob/" + q["type"]].append(dp)
            st["contract_vs_upstream_answer"].append(float(np.abs(p_ref - np.array(p_up)).max()))
            same = int(p_ref.argmax() == p_got.argmax())
            n_q += 1
            agree += same
            agree_up += int(p_got.argmax() == int(np.argmax(p_up)) or abs(max(p_up) - sorted(p_up)[-2]) < 2e-4)
            worst.append((dp, cid, q["qid"], same))
        if (ci + 1) % 10 == 0:
            print("  %d/%d cases, %d questions, ort %.0fs" % (ci + 1, len(case_list), n_q, t_ort), flush=True)

    summary = {
        "model": a.model, "cases": len(case_list), "questions": n_q, "rows": n_rows,
        "row_token_mismatches": mism, "rejected_by_both": both_rej, "rejection_mismatches": rej_mismatch,
        "argmax_agreement_onnx_vs_fp32": agree / max(n_q, 1),
        "argmax_agreement_onnx_vs_upstream_answers": agree_up / max(n_q, 1),
        "ort_seconds": round(t_ort, 1),
        "stats": {k: {"max": float(np.max(v)), "p99": float(np.quantile(v, 0.99)), "mean": float(np.mean(v))}
                  for k, v in sorted(st.items())},
        "worst": sorted(worst, reverse=True)[:8],
    }
    print(json.dumps(summary, indent=1))
    if a.report:
        with open(a.report, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
