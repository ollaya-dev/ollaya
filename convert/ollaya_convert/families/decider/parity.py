"""Parity of the weightless ONNX export (and of the layout port) against upstream decider in fp32.

    uv run python -m ollaya_convert.families.decider.parity decider-0.8b out/decider-0.8b --td-limit 50

Per case (Laya's edge cases + typed-decisions rows, JSON round-tripped):
  1. upstream rows (`decider.systemone` + `decider.prompt.build`) vs the port (`layout.py` + `tokenizers`):
     token ids and slot positions must be identical; invalid requests must be rejected by both;
  2. label logits: upstream `DecisionModel.slot_logits` in fp32 (GPU, TF32 off) vs ONNX Runtime CPU;
  3. per-question probabilities through the runtime contract (layout.option_logits + calibration
     temperature + softmax) from both sets of logits, and against upstream `system_one`'s own answers
     (rounded to 4 decimals upstream).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import tokenizers

from ..llm_common import cases
from ..llm_common.ort_rows import run_rows, session
from . import ref
from .layout import DeciderLayout, LayoutError, softmax

QT = {"choice": 0, "score": 1, "noul": 2}


def upstream_probs(answer):
    if answer["type"] == "noul":
        return [1 - answer["noul"], answer["noul"]]
    return list(answer["probabilities"].values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(ref.MODELS))
    ap.add_argument("model_dir")
    ap.add_argument("--root", default=None)
    ap.add_argument("--td-limit", type=int, default=50)
    ap.add_argument("--edge", type=int, default=1, help="include the edge cases (1) or not (0)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=0, help="ONNX Runtime intra-op threads (0 = all)")
    ap.add_argument("--torch-threads", type=int, default=0, help="torch threads for a CPU reference")
    ap.add_argument("--budget", type=int, default=8192, help="padded tokens per ORT batch")
    ap.add_argument("--only-decoder", action="store_true", help="only the decoder-specific edge cases")
    ap.add_argument("--report", default=None, help="write a JSON summary here")
    a = ap.parse_args()
    if a.torch_threads:
        import torch

        torch.set_num_threads(a.torch_threads)

    root = a.root or ref.snapshot(a.model)
    d = ref.load(root, device=a.device)
    decision = json.load(open(os.path.join(a.model_dir, "decision.json")))
    calib = json.load(open(os.path.join(a.model_dir, "calibration.json")))["temperature"]
    tok = tokenizers.Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json"))
    lay = DeciderLayout(tok, decision)
    sess = session(os.path.join(a.model_dir, "model.onnx"), a.threads)
    pad = decision["special_tokens"]["pad"]

    case_list = (cases.edge_cases() if a.edge else []) + cases.typed_decisions(a.td_limit)
    if a.only_decoder:
        case_list = cases.decoder_cases()
    st = defaultdict(list)
    n_q = agree = agree_up = n_rows = id_mismatch = rejected_both = 0
    reject_mismatch, worst = [], []
    t_ort = t_ref = 0.0
    for ci, (cid, state, questions) in enumerate(case_list):
        up_err = port_err = None
        try:
            items, index = ref.encode(d, state, questions)
        except (ValueError, KeyError, AttributeError, TypeError) as e:
            up_err = repr(e)
        try:
            rows, plan = lay.encode(state, questions)
        except LayoutError as e:
            port_err = repr(e)
        if up_err or port_err:
            if up_err and port_err:
                rejected_both += 1
            else:
                reject_mismatch.append((cid, up_err, port_err))
            # a request with one invalid question: score the remaining ones individually
            valid = {}
            for qid, q in questions.items():
                try:
                    ref.encode(d, state, {qid: q})
                    lay.encode(state, {qid: q})
                    valid[qid] = q
                except Exception:
                    pass
            if not valid:
                continue
            questions = valid
            items, index = ref.encode(d, state, questions)
            rows, plan = lay.encode(state, questions)
        for it, r in zip(items, rows):
            n_rows += 1
            if it["ids"] != r["ids"] or it["slots"][0] != r["slot"]:
                id_mismatch += 1
        t0 = time.perf_counter()
        ref_logits = ref.forward(d, items)
        answers = ref.system_one(d, state, questions)["answers"]
        t_ref += time.perf_counter() - t0
        got, t = run_rows(sess, [r["ids"] for r in rows],
                          lambda idx, L: {"slot_pos": np.array([rows[i]["slot"] for i in idx], dtype=np.int64)},
                          pad, a.budget)
        t_ort += t
        got = np.stack(got)
        for item in plan:
            for r in item["rows"]:
                kk = rows[r]["k"]
                st["logit/" + item["type"]].append(float(np.abs(ref_logits[r, :kk] - got[r, :kk]).max()))
                st["logit255"].append(float(np.abs(ref_logits[r] - got[r]).max()))
            tq = calib[QT[item["type"]]]
            p_ref = softmax(lay.option_logits(item, ref_logits), tq)
            p_got = softmax(lay.option_logits(item, got), tq)
            p_up = upstream_probs(answers[item["qid"]])
            dp = float(np.abs(np.array(p_ref) - np.array(p_got)).max())
            st["prob/" + item["type"]].append(dp)
            st["contract_vs_upstream_answer"].append(float(np.abs(np.array(p_ref) - np.array(p_up)).max()))
            same = int(np.argmax(p_ref) == np.argmax(p_got))
            n_q += 1
            agree += same
            agree_up += int(np.argmax(p_got) == np.argmax(p_up) or abs(max(p_up) - sorted(p_up)[-2]) < 2e-4)
            worst.append((dp, cid, item["qid"], same))
        if (ci + 1) % 10 == 0:
            print("  %d/%d cases, %d questions, ort %.0fs" % (ci + 1, len(case_list), n_q, t_ort), flush=True)

    summary = {
        "model": a.model, "cases": len(case_list), "questions": n_q, "rows": n_rows,
        "row_token_mismatches": id_mismatch, "rejected_by_both": rejected_both,
        "rejection_mismatches": reject_mismatch,
        "argmax_agreement_onnx_vs_fp32": agree / max(n_q, 1),
        "argmax_agreement_onnx_vs_upstream_answers": agree_up / max(n_q, 1),
        "ort_seconds": round(t_ort, 1), "ref_seconds": round(t_ref, 1),
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
