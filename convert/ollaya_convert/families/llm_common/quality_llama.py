"""Typed-decisions quality of a GGUF decision model, as Ollaya runs it.

    python -m ollaya_convert.families.llm_common.quality_llama out/winnow-12b-q8_0 \\
        --server <b11146 build>/llama-server --gguf .../Winnow-12B-Q8_0.gguf [--device CUDA0]

The model's layout (`decision.json`) builds each question's prompt, and the pinned llama-server
evaluates it with the runtime's fixed plan (`plan.py`), the path the goldens take and the Rust
runtime matches (parity). Every one of the 400 typed-decisions test rows is scored; the report
gives accuracy against the gold label (T-independent), and cross-entropy and ECE at the shipped
temperatures (`calibration.json`), at T=1, and cross-fitted (`quality.cross_fit`). It writes
typed-decisions.json next to decision.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

from . import cases, quality
from .export_llama import LlmLogits, Winnow, engine_form, post_to
from .llama_server import LlamaServer
from .plan import FixedPlan, server_args


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--server", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--device", default="CUDA0")
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--limit", type=int, default=0, help="typed-decisions rows (0: all 400)")
    a = ap.parse_args()

    decision = json.load(open(os.path.join(a.model_dir, "decision.json")))
    cal = json.load(open(os.path.join(a.model_dir, "calibration.json")))
    llama = decision["llama"]
    dev = None if a.device == "cpu" else a.device
    srv = LlamaServer.start(a.server, a.gguf, port=a.port,
                            argv=server_args(llama["n_ctx"], llama.get("swa_full", False), dev),
                            log=os.path.join(a.model_dir, "llama-server-quality.log"), timeout=1800)
    try:
        args = SimpleNamespace(upstream_commit=decision.get("upstream", {}).get("commit", ""),
                               assistant_prefix=decision.get("assistant_prefix", ""))
        lay = (Winnow if decision["layout"] == "winnow-v1" else LlmLogits)(srv, args)
        plan = FixedPlan(post_to(srv.url))

        def scorer(state, questions):
            _, _, rows = lay.encode(state, engine_form(questions))
            z = {}
            for qid, ids, p, cands, wire in rows:
                lp = plan.ask(ids, p, cands) if len(cands) > 1 else [0.0]
                z[qid] = [lp[j] for j in wire]
            return z

        t0 = time.time()
        rows = cases.typed_decisions_gold(a.limit)
        items = quality.collect(scorer, rows, progress=50)
        t = cal["temperature"]
        shipped = {"choice": t[0], "score": t[1], "noul": t[2]}
        report = {"rows": len(rows), "questions": len(items), "device": a.device,
                  "shipped_temperatures": shipped, "shipped": quality.metrics(items, shipped),
                  **quality.cross_fit(items), "seconds": round(time.time() - t0, 1)}
        quality.dump(items, os.path.join(a.model_dir, "typed-decisions-logits.jsonl"))
        with open(os.path.join(a.model_dir, "typed-decisions.json"), "w") as f:
            json.dump(report, f, indent=1)
        print(json.dumps({"acc": report["shipped"]["all"]["acc"], "ece_shipped": report["shipped"]["all"]["ece"],
                          "ece_T1": report["raw_T1"]["all"]["ece"], "n": report["questions"]}))
    finally:
        srv.stop()


if __name__ == "__main__":
    main()
