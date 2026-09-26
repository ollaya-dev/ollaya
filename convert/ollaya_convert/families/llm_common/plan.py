"""The fixed evaluation plan Ollaya's llama runner uses on a stock llama-server (reference).

llama.cpp's logits depend on how a prompt is split into evaluation batches (a cached prefix plus
the rest, vs one cold pass). A fixed split is bit-reproducible, so every question is evaluated the
same way, whatever ran before it:

    cold(ids[:P]) + one pass over ids[P:]

P is the question's split point: its shared state prefix (winnow-v1: the prefix tokens; llm-logits-v1:
the tokens the question shares with a reference prompt for the same state, see `split_point`).

On the server (`-np 1 --cache-ram 0 --ctx-checkpoints 0`, and `--swa-full` for sliding-window models):
  * `prefix`   ids[:P] with cache_prompt=false: a cold pass; the slot then holds exactly ids[:P]
  * `barrier`  ids[:P] + [T] with cache_prompt=true: cuts the slot back to ids[:P] (T differs from the
               token after P in both the slot and the next question, so the common prefix is P)
  * `question` ids with cache_prompt=true: the server reuses exactly P tokens and evaluates the rest
Each cached step must report `timings.cache_n == P`; otherwise the prefix is re-evaluated cold and the
step repeated. A question with P == 0 is one cold pass.
"""
from __future__ import annotations

import math

BIAS = 100.0
BARRIER_TOKENS = (0, 1, 2)


def server_args(n_ctx, swa_full, device=None):
    """llama-server's arguments after `--port`, exactly as the runtime passes them
    (`crates/ollaya-server/src/llama/process.rs`, `server_args`). `device` is a llama.cpp device
    (`CUDA0`, `MTL0`) or None for the CPU."""
    args = ["-c", str(n_ctx), "-np", "1", "--cache-ram", "0", "--ctx-checkpoints", "0",
            "-b", "2048", "-ub", "512", "--fit", "off", "--no-ui", "--offline"]
    if swa_full:
        args.append("--swa-full")
    return args + (["-ngl", "all", "--device", device] if device else ["-ngl", "0", "--device", "none"])


def lcp(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


def split_point(ids, ref_ids):
    """P for a question: its common prefix with the reference prompt, leaving >= 1 token to evaluate."""
    return min(lcp(ids, ref_ids), len(ids) - 1)


def barrier_token(slot_next, question_next):
    return next(t for t in BARRIER_TOKENS if t != slot_next and t != question_next)


def completion_body(ids, candidates=None, cache_prompt=True, bias=BIAS):
    """The /completion request of one step. Candidates: the bias trick (llm-logits.md)."""
    body = {"prompt": list(ids), "n_predict": 1, "cache_prompt": cache_prompt, "stream": False,
            "samplers": ["top_k"], "top_k": 1, "n_probs": 0}
    if candidates:
        k = len(candidates)
        body.update({"n_probs": k, "post_sampling_probs": True,
                     "logit_bias": [[int(c), bias] for c in candidates],
                     "samplers": ["top_k", "temperature"], "top_k": k, "temperature": 1.0,
                     "top_p": 1.0, "min_p": 0.0, "typical_p": 1.0, "repeat_penalty": 1.0,
                     "presence_penalty": 0.0, "frequency_penalty": 0.0, "dry_multiplier": 0.0,
                     "xtc_probability": 0.0})
    return body


class FixedPlan:
    """Runs the plan against a server (`post(path, body) -> json`). `trace` records every step."""

    def __init__(self, post):
        self.post = post
        self.prefix = None  # the cold prefix the slot holds
        self.slot = None    # every token the slot holds (prefix + a tail that gets cut)
        self.trace = []

    def reset(self):
        """Forget the slot (another client used the server)."""
        self.prefix = self.slot = None

    def _step(self, kind, ids, candidates=None, cache_prompt=True, bias=BIAS):
        self.trace.append({"step": kind, "n": len(ids), "cache_prompt": cache_prompt})
        r = self.post("/completion", completion_body(ids, candidates, cache_prompt, bias))
        return r, r.get("timings", {}).get("cache_n")

    def _cold_prefix(self, pre):
        self._step("prefix", pre, cache_prompt=False)
        self.prefix, self.slot = list(pre), list(pre)

    def _read(self, r, candidates):
        top = {t["id"]: t["prob"] for t in r["completion_probabilities"][0]["top_probs"]}
        if any(c not in top for c in candidates):
            return None
        return [math.log(top[c]) if top[c] > 0 else -1e4 for c in candidates]

    def ask(self, ids, P, candidates):
        """Label log-probabilities (logits up to a constant) at the last token of `ids`."""
        for bias in (BIAS, 1000.0):
            lp = self._ask(ids, P, candidates, bias)
            if lp is not None:
                return lp
        raise RuntimeError("llama-server did not return every candidate")

    def _ask(self, ids, P, candidates, bias):
        ids = list(ids)
        if P == 0:
            r, _ = self._step("cold", ids, candidates, cache_prompt=False, bias=bias)
            self.prefix, self.slot = [], ids
            return self._read(r, candidates)
        pre = ids[:P]
        if self.prefix != pre:
            self._cold_prefix(pre)
        elif len(self.slot) > P and lcp(self.slot, ids) > P:
            t = barrier_token(self.slot[P], ids[P])
            _, n = self._step("barrier", pre + [t])
            if n == P:
                self.slot = pre + [t]
            else:
                self._cold_prefix(pre)
        r, n = self._step("question", ids, candidates, bias=bias)
        if n != P:
            self._cold_prefix(pre)
            r, n = self._step("question", ids, candidates, bias=bias)
            if n != P:
                raise RuntimeError("llama-server reused %s cached tokens, the plan needs %d" % (n, P))
        self.slot = ids
        return self._read(r, candidates)
