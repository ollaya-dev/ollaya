"""A minimal client for a stock `llama-server` (llama.cpp) used as a logit reader.

Stock llama-server cannot return the logits of an arbitrary token set: `n_probs` returns the top-n of
the full-vocabulary distribution, and the candidates may not be in it. `candidate_logprobs` gets them
exactly with the sampler chain instead:

    logit_bias = +B on every candidate   (a shared shift: relative logits unchanged)
    samplers   = [top_k, temperature], top_k = K, temperature = 1
    post_sampling_probs = true, n_probs = K

With B large enough that every candidate outranks every other token, top_k keeps exactly the K
candidates and the returned post-sampling probabilities are softmax over the candidates' own logits.
B = 100 keeps fp32 resolution near 1e-5 on the shifted logits (a real next-token logit gap above 100
nats does not occur; Gemma's final soft-cap bounds logits to +-30). If a non-candidate still makes the
top-k, a candidate is missing from the reply; the call then retries once with B = 1000. `log(p)` equals the
raw logits up to one additive constant per position, which temperature scaling and softmax ignore.
`top_logprobs` (pre-sampling, full vocabulary) is kept for cross-checking.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import time
import urllib.request

import numpy as np

BIAS = 100.0


class LlamaServer:
    def __init__(self, url: str, proc=None):
        self.url = url.rstrip("/")
        self.proc = proc

    # ---- process management -------------------------------------------------------------------
    @classmethod
    def start(cls, binary: str, model: str, port: int = 8093, ctx: int = 8192, parallel: int = 4,
              ngl: int = 999, extra=(), log=None, timeout: float = 300):
        env = dict(os.environ)
        libdir = os.path.dirname(os.path.abspath(binary))
        env["LD_LIBRARY_PATH"] = libdir + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        cmd = [binary, "-m", model, "--port", str(port), "--host", "127.0.0.1", "-c", str(ctx),
               "-np", str(parallel), "-ngl", str(ngl), "--no-webui", *extra]
        out = open(log, "w") if log else subprocess.DEVNULL
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env)
        srv = cls("http://127.0.0.1:%d" % port, proc)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if proc.poll() is not None:
                raise RuntimeError("llama-server exited with %s (see %s)" % (proc.returncode, log))
            try:
                if srv._get("/health").get("status") == "ok":
                    return srv
            except Exception:
                pass
            time.sleep(0.5)
        proc.kill()
        raise TimeoutError("llama-server did not become healthy")

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None

    # ---- HTTP ---------------------------------------------------------------------------------
    def _get(self, path):
        with urllib.request.urlopen(self.url + path, timeout=600) as r:
            return json.loads(r.read())

    def _post(self, path, body):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.loads(r.read())

    def props(self):
        return self._get("/props")

    def tokenize(self, text: str, add_special=False, parse_special=False):
        r = self._post("/tokenize", {"content": text, "add_special": add_special, "parse_special": parse_special})
        return [t if isinstance(t, int) else t["id"] for t in r["tokens"]]

    def pieces(self, ids):
        return [self._post("/detokenize", {"tokens": [i]})["content"] for i in ids]

    def detokenize(self, ids):
        return self._post("/detokenize", {"tokens": list(ids)})["content"]

    def apply_template(self, messages, **template_kwargs):
        body = {"messages": messages}
        if template_kwargs:
            body["chat_template_kwargs"] = template_kwargs
        return self._post("/apply-template", body)["prompt"]

    # ---- logit reading ------------------------------------------------------------------------
    def candidate_logprobs(self, ids, candidates, bias: float = BIAS, cache_prompt: bool = True):
        """log softmax over `candidates` of the next-token logits after `ids` (a token id list).

        llama.cpp's numbers depend on how the prompt was split into batches: reusing a cached prefix
        and evaluating only the rest gives slightly different logits than one cold pass (measured up to
        ~0.5 nats on near-zero options of a Q8_0 model). cache_prompt=False makes every call a cold,
        single-split evaluation (reproducible, slower)."""
        k = len(candidates)
        body = {"prompt": list(ids), "n_predict": 1, "n_probs": k, "post_sampling_probs": True,
                "logit_bias": [[int(c), bias] for c in candidates], "samplers": ["top_k", "temperature"],
                "top_k": k, "temperature": 1.0, "top_p": 1.0, "min_p": 0.0, "typical_p": 1.0,
                "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
                "dry_multiplier": 0.0, "xtc_probability": 0.0, "cache_prompt": cache_prompt, "stream": False}
        r = self._post("/completion", body)
        top = r["completion_probabilities"][0]["top_probs"]
        p = {t["id"]: t["prob"] for t in top}
        missing = [c for c in candidates if c not in p]
        if missing:
            if bias < 1000.0:
                return self.candidate_logprobs(ids, candidates, 1000.0, cache_prompt)
            raise RuntimeError("llama-server did not return candidates %s" % missing)
        return np.array([math.log(p[c]) if p[c] > 0 else -1e4 for c in candidates]), r

    def top_logprobs(self, ids, n: int = 100, cache_prompt: bool = True):
        """Pre-sampling full-vocabulary log-probabilities of the top-n next tokens: {id: logprob}."""
        body = {"prompt": list(ids), "n_predict": 1, "n_probs": n, "post_sampling_probs": False,
                "temperature": 0.0, "cache_prompt": cache_prompt, "stream": False}
        r = self._post("/completion", body)
        top = r["completion_probabilities"][0]["top_logprobs"]
        return {t["id"]: t["logprob"] for t in top}
