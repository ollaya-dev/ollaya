"""Parity of the weightless Decision 1.0 ONNX export (and of the layout port) against the author's code in fp32.

    uv run python -m ollaya_convert.families.decision.parity decision-eos out/decision-eos-0.8b --td-limit 50

Per case (Laya's edge cases, the decoder edge cases and typed-decisions rows, JSON round-tripped):
  1. the author's rows (`question_row` -> `encode`) vs the port (`layout.py` + `tokenizers` + the repo's
     tokenizer.json): token ids, query and candidate positions must be identical; a request the author
     rejects must be rejected by the port too (and each question on its own exactly when the author
     rejects it);
  2. raw candidate logits: the author's fp32 forward (GPU, TF32 off, exact SDPA) vs ONNX Runtime CPU;
  3. probabilities through the runtime contract (logits / T, softmax) from both, and against the author's
     `typed_answer`.
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
from .layout import DecisionLayout, LayoutError


def softmax(z, t):
    z = np.asarray(z, dtype=np.float64) / t
    e = np.exp(z - z.max())
    return e / e.sum()


def upstream_probs(answer):
    if answer["type"] == "noul":
        return [1 - answer["noul"], answer["noul"]]
    return list(answer["probabilities"].values())


def port_rows(items):
    return [{"ids": it["ids"], "query": it["query_position"], "cands": it["candidate_positions"]} for it in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(ref.MODELS))
    ap.add_argument("model_dir")
    ap.add_argument("--root", default=None)
    ap.add_argument("--td-limit", type=int, default=50)
    ap.add_argument("--edge", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=0, help="ONNX Runtime intra-op threads (0 = all)")
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    d = ref.load(a.root or ref.snapshot(a.model), a.model, device=a.device)
    decision = json.load(open(os.path.join(a.model_dir, "decision.json")))
    T = json.load(open(os.path.join(a.model_dir, "calibration.json")))["temperature"]
    temp = {"choice": T[0], "score": T[1], "noul": T[2]}
    assert all(abs(temp[k] - d.T[k]) < 1e-12 for k in temp), (temp, d.T)
    lay = DecisionLayout(tokenizers.Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json")), decision)
    sess = session(os.path.join(a.model_dir, "model.onnx"), a.threads)
    pad = decision["pad"]

    def accepts(fn):
        try:
            fn()
            return True
        except (ValueError, AttributeError):
            return False

    case_list = (cases.edge_cases() if a.edge else []) + cases.typed_decisions(a.td_limit)
    st = defaultdict(list)
    n_q = agree = agree_up = n_rows = mism = both_rej = 0
    rej_mismatch, worst = [], []
    t_ort = 0.0
    for ci, (cid, state, questions) in enumerate(case_list):
        up_ok = accepts(lambda: d.encode(state, questions))
        port_ok = accepts(lambda: lay.encode(state, questions))
        if not (up_ok and port_ok):
            if up_ok == port_ok:
                both_rej += 1
            else:
                rej_mismatch.append((cid, "request", up_ok, port_ok))
            valid = {}
            for qid, q in questions.items():
                u = accepts(lambda: d.encode(state, {qid: q}))
                p = accepts(lambda: lay.encode(state, {qid: q}))
                if u != p:
                    rej_mismatch.append((cid, qid, u, p))
                if u and p:
                    valid[qid] = q
            if not valid:
                continue
            questions = valid
        rows_up, items = d.encode(state, questions)
        rows, pmeta = lay.encode(state, questions)
        for u, r in zip(port_rows(items), rows):
            n_rows += 1
            if u != r:
                mism += 1
        ref_logits = [z.numpy() for z in d.forward(items)]
        answers = d.answers(rows_up, ref_logits)

        def extra(idx, L):
            K = max(len(rows[i]["cands"]) for i in idx)
            cp = np.zeros((len(idx), K), dtype=np.int64)
            for b, i in enumerate(idx):
                cp[b, :len(rows[i]["cands"])] = rows[i]["cands"]
            return {"query_pos": np.array([rows[i]["query"] for i in idx], dtype=np.int64), "cand_pos": cp}

        got, t = run_rows(sess, [r["ids"] for r in rows], extra, pad, a.budget)
        t_ort += t
        for qi, (q, r) in enumerate(zip(pmeta, rows)):
            k = len(r["cands"])
            z_ref = ref_logits[qi][:k]
            z_got = got[qi][:k]
            st["logit/" + q["type"]].append(float(np.abs(z_ref - z_got).max()))
            p_ref, p_got = softmax(z_ref, temp[q["type"]]), softmax(z_got, temp[q["type"]])
            p_up = upstream_probs(answers[q["qid"]])
            dp = float(np.abs(p_ref - p_got).max())
            st["prob/" + q["type"]].append(dp)
            st["contract_vs_upstream_answer"].append(float(np.abs(p_ref - np.array(p_up)).max()))
            same = int(p_ref.argmax() == p_got.argmax())
            n_q += 1
            agree += same
            agree_up += int(p_got.argmax() == int(np.argmax(p_up)))
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
