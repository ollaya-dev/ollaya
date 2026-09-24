"""End-to-end `llm-logits-v1` on a stock llama-server: checks, decisions, typed-decisions quality and
per-type temperature fitting for one GGUF.

    uv run python -m ollaya_convert.families.llm_logits.demo \
        --server /path/llama-server --gguf /path/model.gguf --slug qwen3-4b-instruct-2507-q8_0 \
        --repo ggml-org/Qwen3-4B-Instruct-2507-Q8_0-GGUF --revision <sha> --file <name>.gguf --td-limit 400

Writes convert/out/llm-logits-<slug>/{decision.json, calibration.json, report.json}:
  * exactness of the bias trick: candidate log-softmax from `candidate_logprobs` vs the same candidates
    read from the pre-sampling full-vocabulary top-n (`top_logprobs`), on prompts where all are present;
  * a few end-to-end decisions (Laya presets on edge states);
  * typed-decisions accuracy / soft cross-entropy / ECE at T=1 and with per-type temperatures fitted on
    one half of the rows and evaluated on the other; the shipped calibration is the fit on all rows.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from ..llm_common import cases, quality
from ..llm_common.llama_server import LlamaServer
from ..llm_common.onnx_export import write_json
from .ref import LAYOUT, SYSTEM, FINAL, LlmLogitsLayout, decide


def _logsm(x):
    x = np.asarray(x, dtype=np.float64)
    return x - x.max() - np.log(np.exp(x - x.max()).sum())


def bias_check(srv, lay, n_cases=40):
    d_lp, d_p, w_lp, w_p, mass, top1, checked, skipped = [], [], [], [], [], [], 0, 0
    for cid, state, qs in cases.edge_cases()[:n_cases]:
        try:
            items = lay.encode(state, qs)
        except Exception:
            continue
        for it in items:
            if len(it["label_ids"]) < 2:
                continue
            cold, _ = srv.candidate_logprobs(it["ids"], it["label_ids"], cache_prompt=False)
            top = srv.top_logprobs(it["ids"], 400, cache_prompt=False)
            mass.append(float(sum(np.exp(top[i]) for i in it["label_ids"] if i in top)))
            top1.append(max(top, key=top.get) in it["label_ids"])
            warm, _ = srv.candidate_logprobs(it["ids"][:-1] + [it["ids"][-1]], it["label_ids"], cache_prompt=True)
            a = _logsm(cold)
            w = _logsm(warm)
            w_lp.append(float(np.abs(a - w).max()))
            w_p.append(float(np.abs(np.exp(a) - np.exp(w)).max()))
            if not all(i in top for i in it["label_ids"]):
                skipped += 1
                continue
            b = _logsm([top[i] for i in it["label_ids"]])
            d_lp.append(float(np.abs(a - b).max()))
            d_p.append(float(np.abs(np.exp(a) - np.exp(b)).max()))
            checked += 1
    q = lambda v: {"max": max(v), "p99": float(np.quantile(v, 0.99)), "mean": float(np.mean(v))} if v else None
    return {"checked": checked, "skipped_not_in_top400": skipped,
            "label_mass": {"mean": float(np.mean(mass)), "min": float(np.min(mass)),
                           "top1_is_a_label": float(np.mean(top1))},
            "cold_bias_vs_cold_fullvocab": {"logp": q(d_lp), "p": q(d_p)},
            "warm_cache_vs_cold": {"logp": q(w_lp), "p": q(w_p)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True, help="llama-server binary")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--sha256", default=None)
    ap.add_argument("--td-limit", type=int, default=400)
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--assistant-prefix", default="")
    ap.add_argument("--check-cases", type=int, default=40)
    ap.add_argument("--checks-only", action="store_true")
    ap.add_argument("--out-root", default=os.path.join(os.path.dirname(__file__), "..", "..", "..", "out"))
    a = ap.parse_args()

    out = os.path.join(os.path.abspath(a.out_root), "llm-logits-" + a.slug)
    os.makedirs(out, exist_ok=True)
    srv = LlamaServer.start(a.server, a.gguf, port=a.port, ctx=a.ctx, parallel=a.parallel,
                            log=os.path.join(out, "llama-server.log"), timeout=900)
    report = {"model": a.slug, "layout": LAYOUT}
    try:

        lay = LlmLogitsLayout(srv, assistant_prefix=a.assistant_prefix)
        report["template_pieces"] = list(lay.template)
        report["labels"] = {"choice": len(lay.labels.choice), "add_bos": lay.add_bos}
        print("labels: %d choice labels; template tail %r" % (len(lay.labels.choice), lay.template[2]), flush=True)

        # 1. exactness of the bias trick against the full-vocabulary pre-sampling distribution, both cold
        #    (cache_prompt=False: one single-split evaluation each), and how much llama.cpp's own numbers
        #    move when the prompt is served from a cached prefix (warm) instead
        report["bias_trick"] = bias_check(srv, lay, a.check_cases)
        print("bias trick:", report["bias_trick"], flush=True)
        if a.checks_only:
            write_json(os.path.join(out, "checks.json"), report)
            return

        # 2. a few decisions
        demo = {}
        for cid, state, qs in cases.edge_cases():
            if cid in ("preset/triage/tr_billing", "preset/email/email_dict", "preset/guard/en_injection",
                       "edge/many_options_40"):
                r = decide(lay, state, qs)
                demo[cid] = {q: {"probabilities": [round(x, 4) for x in v["probabilities"]]} for q, v in r.items()}
        report["demo_decisions"] = demo

        # 3. typed-decisions quality + temperature fitting
        rows = cases.typed_decisions_gold(a.td_limit)
        t0 = time.time()

        def scorer(state, questions):
            return {q: v["option_logits"] for q, v in decide(lay, state, questions).items()}

        items = quality.collect(scorer, rows, progress=50)
        quality.dump(items, os.path.join(out, "typed-decisions-logits.jsonl"))
        report["typed_decisions"] = quality.cross_fit(items)
        report["typed_decisions"]["seconds"] = round(time.time() - t0, 1)
        report["typed_decisions"]["rows"] = len(rows)
        print(json.dumps(report["typed_decisions"], indent=1), flush=True)
        temps = report["typed_decisions"]["fit_all"]

        decision = {
            "engine": "llama.cpp",
            "family": "llm-logits",
            "layout": LAYOUT,
            "gguf": {"repo": a.repo, "revision": a.revision, "path": a.file, "sha256": a.sha256,
                     "url": "https://huggingface.co/%s/resolve/%s/%s" % (a.repo, a.revision, a.file)},
            "system": SYSTEM,
            "user_template": "State:\n{state}\n\nQuestion: {instructions}\nOptions:\n{label}. {option}\n...\n" + FINAL,
            "chat_template": "from the GGUF (tokenizer.chat_template), rendered by llama.cpp",
            "chat_template_kwargs": lay.kwargs,
            "assistant_prefix": a.assistant_prefix,
            "labels": lay.labels.to_json(),
            "add_bos": lay.add_bos,
            "max_state_tokens": lay.max_state,
            "n_ctx": a.ctx,
            "limits": {"choice_options": [1, len(lay.labels.choice)], "score_levels": [2, 10]},
        }
        write_json(os.path.join(out, "decision.json"), decision)
        write_json(os.path.join(out, "calibration.json"),
                   {"temperature": [temps["choice"], temps["score"], temps["noul"]], "temperature_by_options": {},
                    "source": "fitted on typed-decisions test (%d rows, soft-label cross-entropy)" % len(rows)})
        write_json(os.path.join(out, "report.json"), report)
    finally:
        srv.stop()


if __name__ == "__main__":
    main()
