"""The exported graph under ONNX Runtime (Python) against the goldens, without loading the reference.

    uv run python -m ollaya_convert.families.decision.ort_goldens out/decision-lux-9b out/goldens-decision-lux-9b.jsonl \
        [--keep-bf16] [--provider CUDAExecutionProvider]

For checkpoints whose fp32 reference and ONNX session do not fit in memory together (Lux). The token rows
come from the goldens (the author's), so this checks the graph only; `parity.py` and the Rust parity
examples check the layout. `--keep-bf16` disables ONNX Runtime's constant folding, so the BF16 -> fp32
Casts run per request instead of being folded into fp32 copies of every weight at load: the same
arithmetic (an exact upcast), about half the memory. Same tolerances as the Rust parity example.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time

import numpy as np

from ..llm_common.ort_rows import run_rows

LOGIT_TOL = 1e-3


def softmax(z, t):
    z = np.asarray(z, dtype=np.float64) / t
    e = np.exp(z - z.max())
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("goldens")
    ap.add_argument("--keep-bf16", action="store_true")
    ap.add_argument("--provider", default="CPUExecutionProvider")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    import onnxruntime as ort

    decision = json.load(open(os.path.join(a.model_dir, "decision.json")))
    T = json.load(open(os.path.join(a.model_dir, "calibration.json")))["temperature"]
    temp = {"choice": T[0], "score": T[1], "noul": T[2]}
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if a.threads:
        so.intra_op_num_threads = a.threads
    if a.provider != "CPUExecutionProvider":
        so.enable_mem_reuse = False  # the CUDA EP workaround (docs/families/decider.md)
    t0 = time.time()
    sess = ort.InferenceSession(os.path.join(a.model_dir, "model.onnx"), so, providers=[a.provider],
                                disabled_optimizers=["ConstantFolding"] if a.keep_bf16 else None)
    load = time.time() - t0
    print("session %.0fs, peak RSS %.1f GB" % (load, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20),
          flush=True)
    logit_max, prob_max, disagree, n_q, t_run = 0.0, 0.0, 0, 0, 0.0
    worst = []
    for line in open(a.goldens):
        rec = json.loads(line)
        if rec["error"]:
            continue
        rows = rec["rows"]

        def extra(idx, L):
            K = max(len(rows[i]["cands"]) for i in idx)
            cp = np.zeros((len(idx), K), dtype=np.int64)
            for b, i in enumerate(idx):
                cp[b, :len(rows[i]["cands"])] = rows[i]["cands"]
            return {"query_pos": np.array([rows[i]["query"] for i in idx], dtype=np.int64), "cand_pos": cp}

        got, t = run_rows(sess, [r["ids"] for r in rows], extra, decision["pad"], a.budget)
        t_run += t
        for p, z in zip(rec["plan"], got):
            k = p["k"]
            d = float(np.abs(np.asarray(z[:k], dtype=np.float64) - np.array(p["option_logits"])).max())
            logit_max = max(logit_max, d)
            pr = softmax(z[:k], temp[p["type"]])
            prob_max = max(prob_max, float(np.abs(pr - np.array(p["probabilities"])).max()))
            disagree += int(pr.argmax() != int(np.argmax(p["probabilities"])))
            n_q += 1
            worst.append((d, rec["id"], p["qid"]))
    summary = {"model_dir": a.model_dir, "provider": a.provider, "keep_bf16": a.keep_bf16, "questions": n_q,
               "logit_max": logit_max, "prob_max": prob_max, "decisions_differ": disagree,
               "session_seconds": round(load, 1), "run_seconds": round(t_run, 1),
               "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
               "worst": sorted(worst, reverse=True)[:5], "pass": logit_max <= LOGIT_TOL and disagree == 0}
    print(json.dumps(summary, indent=1))
    if a.report:
        with open(a.report, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
