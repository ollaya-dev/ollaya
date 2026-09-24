"""How much do llama.cpp's option probabilities depend on how the prompt is split into evaluation
batches? (Prompt caching changes the split: a cached prefix plus the new suffix, vs one cold pass.)

    uv run python -m ollaya_convert.families.llm_logits.determinism --server BIN --gguf MODEL [--cpu]

For each prompt (Laya presets on the edge states, llm-logits-v1 layout), with the same server:
  cold      cache_prompt=false: the whole prompt in one pass (llama.cpp splits it by n_batch)
  cold2     the same again: run-to-run determinism of a fixed split
  last      the full prompt again with cache on: only the last token is re-evaluated
  split     cache warmed with the prompt up to the end of the state, then the full prompt: the question
            suffix is evaluated as one batch on top of the cached state (Ollaya's shared-prefix plan)
  split2    the split again after an unrelated prompt: determinism of the split plan
Reports max / p99 / mean |dp| between them.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from ..llm_common import cases
from ..llm_common.llama_server import LlamaServer
from .ref import LlmLogitsLayout


def probs(lp):
    lp = np.asarray(lp, dtype=np.float64)
    e = np.exp(lp - lp.max())
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--cpu", action="store_true", help="no GPU offload (-ngl 0)")
    ap.add_argument("--cases", type=int, default=24)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    srv = LlamaServer.start(a.server, a.gguf, port=a.port, ctx=8192, parallel=1, ngl=0 if a.cpu else 999,
                            log=(a.out or "/dev/null") + ".server.log" if a.out else None)
    try:
        lay = LlmLogitsLayout(srv)
        pairs = {"cold_vs_cold2": [], "cold_vs_last": [], "cold_vs_split": [], "split_vs_split2": []}
        flips = {k: 0 for k in pairs}
        n = 0
        for cid, state, qs in cases.edge_cases()[: a.cases]:
            try:
                items = lay.encode(state, qs)
            except Exception:
                continue
            # the shared prefix: everything before "\n\nQuestion:" is identical across the request's questions
            common = items[0]["ids"]
            for it in items[1:]:
                k = 0
                while k < min(len(common), len(it["ids"])) and common[k] == it["ids"][k]:
                    k += 1
                common = common[:k]
            for it in items:
                if len(it["label_ids"]) < 2 or len(common) >= len(it["ids"]) - 1:
                    continue
                c = probs(srv.candidate_logprobs(it["ids"], it["label_ids"], cache_prompt=False)[0])
                c2 = probs(srv.candidate_logprobs(it["ids"], it["label_ids"], cache_prompt=False)[0])
                last = probs(srv.candidate_logprobs(it["ids"], it["label_ids"], cache_prompt=True)[0])
                srv.candidate_logprobs(common, it["label_ids"], cache_prompt=False)
                s = probs(srv.candidate_logprobs(it["ids"], it["label_ids"], cache_prompt=True)[0])
                srv.candidate_logprobs([it["ids"][0]] * 8, it["label_ids"], cache_prompt=False)
                srv.candidate_logprobs(common, it["label_ids"], cache_prompt=False)
                s2 = probs(srv.candidate_logprobs(it["ids"], it["label_ids"], cache_prompt=True)[0])
                for key, (x, y) in {"cold_vs_cold2": (c, c2), "cold_vs_last": (c, last), "cold_vs_split": (c, s),
                                    "split_vs_split2": (s, s2)}.items():
                    pairs[key].append(float(np.abs(x - y).max()))
                    flips[key] += int(x.argmax() != y.argmax())
                n += 1
        res = {"prompts": n, "backend": "cpu" if a.cpu else "cuda"}
        for k, v in pairs.items():
            res[k] = {"max": max(v), "p99": float(np.quantile(v, 0.99)), "mean": float(np.mean(v)),
                      "argmax_flips": flips[k]}
        print(json.dumps(res, indent=1))
        if a.out:
            with open(a.out, "w") as f:
                json.dump(res, f, indent=1)
    finally:
        srv.stop()


if __name__ == "__main__":
    main()
