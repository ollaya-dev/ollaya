"""Parity of the weightless ONNX export (and of the layout port) against the model repo's code in fp32.

    uv run python -m ollaya_convert.families.d1_omni.parity out/d1-omni-600m --td-limit 50 [--report out/d1-omni-600m/parity.json]

Per case (the shared edge cases + typed-decisions rows, JSON round-tripped):
  1. upstream rows (`prompt.encode`) vs the port (`layout.py` + `tokenizers`): token ids and marker positions must be
     identical; invalid requests must be rejected by both;
  2. marker logits: upstream `DecisionHead` in fp32 (GPU, TF32 off) vs ONNX Runtime CPU;
  3. per-question probabilities through the runtime contract (marker logits + the bucket temperature + softmax) from
     both sets of logits, and against upstream's own `probabilities`.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import tokenizers

from ... import cases
from ..llm_common.ort_rows import run_rows, session
from . import ref
from .layout import QT, D1OmniLayout, LayoutError, softmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--root", default=None)
    ap.add_argument("--td-limit", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=0, help="ONNX Runtime intra-op threads (0 = all)")
    ap.add_argument("--budget", type=int, default=8192, help="padded tokens per ORT batch")
    ap.add_argument("--report", default=None, help="write a JSON summary here")
    a = ap.parse_args()

    root = a.root or ref.snapshot()
    m = ref.load(root, device=a.device)
    decision = json.load(open(os.path.join(a.model_dir, "decision.json"), encoding="utf-8"))
    tok = tokenizers.Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json"))
    tok.no_truncation()
    tok.no_padding()
    lay = D1OmniLayout(tok, decision)
    sess = session(os.path.join(a.model_dir, "model.onnx"), a.threads)
    pad = decision["special_tokens"]["pad"]

    case_list = cases.edge_cases() + cases.typed_decisions(a.td_limit)
    st = defaultdict(list)
    n_q = agree = agree_up = n_rows = id_mismatch = rejected_both = 0
    reject_mismatch, worst = [], []
    t_ort = t_ref = 0.0
    for ci, (cid, state, questions) in enumerate(case_list):
        state, questions = json.loads(json.dumps(state)), json.loads(json.dumps(questions))
        up_err = port_err = None
        try:
            rows = ref.encode(m, state, questions)
        except ValueError as e:
            up_err = repr(e)
        try:
            port = lay.encode(state, questions)
        except LayoutError as e:
            port_err = repr(e)
        if up_err or port_err:
            if up_err and port_err:
                rejected_both += 1
            else:
                reject_mismatch.append((cid, up_err, port_err))
            valid = {}
            for qid, q in questions.items():
                try:
                    ref.encode(m, state, {qid: q})
                    lay.encode(state, {qid: q})
                    valid[qid] = q
                except Exception:
                    pass
            if not valid:
                continue
            questions = valid
            rows, port = ref.encode(m, state, questions), lay.encode(state, questions)
        for r, p in zip(rows, port):
            n_rows += 1
            if r["ids"] != p["ids"] or r["markers"] != p["markers"]:
                id_mismatch += 1
        t0 = time.perf_counter()
        ref_logits = ref.forward(m, rows)
        up_probs = ref.probabilities(m, state, questions)
        t_ref += time.perf_counter() - t0

        def extra(idx, L, rows=rows):
            k = max(len(rows[i]["markers"]) for i in idx)
            pos = np.zeros((len(idx), k), dtype=np.int64)
            mask = np.zeros((len(idx), k), dtype=bool)
            for b, i in enumerate(idx):
                pos[b, : len(rows[i]["markers"])] = rows[i]["markers"]
                mask[b, : len(rows[i]["markers"])] = True
            att = np.zeros((len(idx), L), dtype=np.int64)
            for b, i in enumerate(idx):
                att[b, : len(rows[i]["ids"])] = 1
            return {"attention_mask": att, "marker_pos": pos, "marker_mask": mask,
                    "qtype": np.array([QT[rows[i]["qtype"]] for i in idx], dtype=np.int64)}

        got, t = run_rows(sess, [r["ids"] for r in rows], extra, pad, a.budget)
        t_ort += t
        for r, z_ref, z_got, p_up in zip(rows, ref_logits, got, up_probs):
            k = r["k"]
            z_got = [float(x) for x in z_got[:k]]
            st["logit/" + r["qtype"]].append(float(np.abs(np.array(z_ref) - np.array(z_got)).max()))
            t_q = ref.temperature(m, r["qtype"], k)
            p_ref, p_got = softmax(z_ref, t_q), softmax(z_got, t_q)
            dp = float(np.abs(np.array(p_ref) - np.array(p_got)).max())
            st["prob/" + r["qtype"]].append(dp)
            st["contract_vs_upstream_probabilities"].append(float(np.abs(np.array(p_ref) - np.array(p_up)).max()))
            same = int(np.argmax(p_ref) == np.argmax(p_got))
            n_q += 1
            agree += same
            agree_up += int(np.argmax(p_got) == np.argmax(p_up))
            worst.append((dp, cid, r["qid"], same))
        if (ci + 1) % 10 == 0:
            print("  %d/%d cases, %d questions, ort %.0fs" % (ci + 1, len(case_list), n_q, t_ort), flush=True)

    summary = {
        "model": "d1-omni-600m", "cases": len(case_list), "questions": n_q, "rows": n_rows,
        "row_token_mismatches": id_mismatch, "rejected_by_both": rejected_both,
        "rejection_mismatches": reject_mismatch,
        "argmax_agreement_onnx_vs_fp32": agree / max(n_q, 1),
        "argmax_agreement_onnx_vs_upstream_probabilities": agree_up / max(n_q, 1),
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
