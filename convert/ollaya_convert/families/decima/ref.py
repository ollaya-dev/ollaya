"""The PyTorch reference for Decima (amyrmahdy/decima-small, -base, -base-clean, -agent, -guard and -guard-small): the author's
own code (github.com/amyrmahdy/decima at v2.3.0, pinned below) on each model repository's `pytorch/` checkpoint, in fp32.

    d = ref.load("cpu", "decima-base")                   # upstream decima.model.Decima (fp32, eval)
    code, message = ref.verdict(d, state, questions)     # serve.py's checks: None, INVALID_REQUEST or STATE_TRUNCATED
    items = ref.encode(d, state, questions)              # per question: texts, token ids, truncation
    out = ref.forward(d, item)                           # raw scores, ordinal projections, log-probabilities
    answers = ref.answers(d, state, questions)           # decima.systemone.system_one, as /v1/systemone answers

Decima is a late-interaction decision model: an encoder (intfloat/multilingual-e5-small for small, jhu-clsp/mmBERT-base
for base, agent and guard, jhu-clsp/mmBERT-small for guard-small, all MIT) fine-tuned with a two-layer scorer (Apache-2.0). Per question, every text is encoded
on its own (the prefixes are e5's "query: " and "passage: " for small, empty for mmBERT):

    state row     state_prefix  + normalize(instructions + "\\n" + state)  (question_in_state)
    option rows   choice_prefix + normalize(instructions + " " + option)   (question_in_choices), one per option,
                  rendered by decima/render.py (below)
    scorer        per option row: [self-attention, cross-attention over the state row's tokens, FFN] x 2, the
                  masked mean, a linear score, plus the bi-encoder skip sim_scale * (cos(mean state, mean option)
                  - sim_center)  ->  one raw score s_k per option; options never see each other
    answer        choice and noul: softmax(s / T); score: the cumulative-link ordinal head on s / T and the pooled
                  option vectors; T is decima.json's fitted temperature

Delegated to upstream code, so it cannot drift:
  * question mapping       decima.systemone.to_question: choice "label: description" (the label alone without
                           one), noul "yes"/"no" with "True if / False if" lines from its criteria, score levels,
                           the question id when there are no instructions
  * texts                  DecimaConfig.state_of / choices_of (prefixes, where the question goes, normalize, and
                           render.choice_texts: with decima.json's choice_fit an option row that would not fit
                           max_choice_tokens keeps the option whole and shortens the question's copy beside it;
                           without it, the right-hand cut unless that makes two options' rows identical)
  * tokenization           the checkpoint's tokenizer through transformers, truncated as Decima._encode does
  * the network            DecimaModel.choice_scores and log_probs, as Decima.logits runs them (fp32)
  * request checks         decima.systemone (to_question, system_one's own checks)
  * answers                decima.systemone.system_one (rounding, confidence, noul = P(yes), expected score)
Re-stated here, cited against decima/serve.py at v2.3.0 (`do_POST`, without its opt-in "long" and
"min_confidence" request fields):
  * the state as text      `state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)`
  * STATE_TRUNCATED        a question whose state row is over max_state_tokens tokens is a 422, checked in
                           question order before anything is answered. model.py itself truncates (keeps the
                           first max_state_tokens tokens), which is what Ollaya's /api/decide answers with.
"""
from __future__ import annotations

import json
import os
import sys
import tarfile
import urllib.request

import numpy as np
import torch

MODELS = {
    # The tag v1.1.1 (an annotated tag object, 83d3a5c, on this commit): "Scoring head as safetensors (identical to
    # head.pt)", 2026-10-03.
    "decima-small": {"repo": "amyrmahdy/decima-small", "revision": "2e7f4d0757df0215f48f2a9b2b589e1f3a6348ed",
                     "tag": "v1.1.1", "base": "intfloat/multilingual-e5-small (MIT)"},
    # The tag v2.0, 2026-10-04: the mmBERT-base generation.
    "decima-base": {"repo": "amyrmahdy/decima-base", "revision": "2468005d5e48e95eb74072c32a6d9df164578071",
                    "tag": "v2.0", "base": "jhu-clsp/mmBERT-base (MIT)"},
    # The tag v2.3: decima-base fine-tuned for agent decisions, states up to 8,192 tokens, "choice_fit": true. The tag
    # v2.2 (2026-10-08, 2,048 tokens) was 5c14a460ff368b5bf3aaf086d1ab46292d668e32; v2.1 (2026-10-05, without
    # choice_fit) 86a07aab1c340fa5869bdb754e57d3851a1d288a.
    "decima-agent": {"repo": "amyrmahdy/decima-agent", "revision": "652d90ac9d33088edd730692c95f1da381cb9174",
                     "tag": "v2.3", "base": "amyrmahdy/decima-base, on jhu-clsp/mmBERT-base (MIT)"},
    # The tag v1.0-preview: guards for LLM apps (inputs, tool results, replies), states up to 2,048 tokens,
    # "choice_fit": true.
    "decima-guard": {"repo": "amyrmahdy/decima-guard", "revision": "98d3f734c8323cb390d9ad750e1a606a8dd56f09",
                     "tag": "v1.0-preview", "base": "amyrmahdy/decima-base-clean, on jhu-clsp/mmBERT-base (MIT)"},
    # The tag v1.0: decima-base's network trained without non-commercial data and without BTZSC data, "choice_fit":
    # true, states up to 512 tokens. main (a37e006f) is one commit later, with only README.md changed.
    "decima-base-clean": {"repo": "amyrmahdy/decima-base-clean", "revision": "35eb218bdeeb8c450baba4159ebc08b67c4d632c",
                          "tag": "v1.0", "base": "jhu-clsp/mmBERT-base (MIT)"},
    "decima-guard-small": {"repo": "amyrmahdy/decima-guard-small",
                           "revision": "ac11024887d51e9b7b9522b3b8ccc2d78ce7353d",
                           "tag": "v1.0-preview", "base": "jhu-clsp/mmBERT-small (MIT)"},
}
DEFAULT = "decima-small"
CHECKPOINT = "pytorch"   # encoder/ (config, model.safetensors, tokenizer), head.safetensors, decima.json
FILES = ["LICENSE", "pytorch/decima.json", "pytorch/head.safetensors", "pytorch/encoder/config.json",
         "pytorch/encoder/model.safetensors", "pytorch/encoder/tokenizer.json", "pytorch/encoder/tokenizer_config.json"]
# The release of decima-agent 2.3 and decima-guard (the annotated tag object 6e3d2b92 on this commit). Against v2.2.0
# (efaefe2a), under decima/ only render.py's choice_texts changed (the token count alone decides whether an option's
# full text fits; the 62-character shortcut is gone) and systemone.py's opt-in "ordinal" choice, which no request
# here sends. v2.2.0's render.py was v2.1.0's (097d9eb3); the rest of what this reference runs gives v1.1.1's
# (2df60942) rows and numbers, with which small, base and agent 2.1 were exported. system_one batches a request's
# questions (Decima.decide_many), which upstream states gives the same numbers as one by one.
CODE = {"repo": "https://github.com/amyrmahdy/decima", "tag": "v2.3.0",
        "commit": "1b07634e0385c71c97bbd0d74ce8a8345b0e7d0a"}
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "out")


class RequestError(ValueError):
    """The request is invalid for this model (HTTP 422); `code` is upstream's error code."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def snapshot(slug=DEFAULT):
    """The model repository at its pinned revision (only the PyTorch checkpoint and the license)."""
    from huggingface_hub import snapshot_download

    m = MODELS[slug]
    return os.environ.get("DECIMA_MODEL") or snapshot_download(m["repo"], revision=m["revision"], allow_patterns=FILES)


def checkpoint(slug=DEFAULT, snap=None):
    return os.path.join(snap or snapshot(slug), CHECKPOINT)


def code():
    """The author's package at CODE["commit"], on sys.path: DECIMA_SRC, or the GitHub tarball of that commit."""
    src = os.environ.get("DECIMA_SRC")
    if not src:
        src = os.path.abspath(os.path.join(OUT, "upstream", "decima-" + CODE["commit"][:12]))
        if not os.path.exists(os.path.join(src, "decima", "model.py")):
            os.makedirs(src, exist_ok=True)
            url = "https://codeload.github.com/amyrmahdy/decima/tar.gz/" + CODE["commit"]
            tmp = src + ".tar.gz"
            urllib.request.urlretrieve(url, tmp)
            with tarfile.open(tmp) as t:
                for m in t.getmembers():
                    parts = m.name.split("/", 1)
                    if len(parts) == 2 and parts[1] and (m.isfile() or m.isdir()):
                        m.name = parts[1]
                        t.extract(m, src, filter="data")
            os.remove(tmp)
    with open(os.path.join(src, "pyproject.toml"), encoding="utf-8") as f:
        if 'version = "2.3.0"' not in f.read():
            raise SystemExit("%s is not decima 2.3.0 (%s at %s)" % (src, CODE["repo"], CODE["commit"]))
    if src not in sys.path:
        sys.path.insert(0, src)
    return src


def load(device="cpu", slug=DEFAULT, snap=None):
    """Upstream `decima.model.Decima` on the PyTorch checkpoint (fp32, eval; TF32 off)."""
    code()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from decima.model import Decima
    from safetensors.torch import load_file

    path = checkpoint(slug, snap)
    d = Decima(path, device)
    # DecimaModel.load uses load_state_dict(strict=False): prove every head tensor landed, and nothing else is missing.
    head = load_file(os.path.join(path, "head.safetensors"))
    params = dict(d.model.named_parameters())
    missing = [n for n in params if not n.startswith("encoder.") and n not in head]
    if missing or any(k not in params or not torch.equal(params[k].detach().cpu(), v) for k, v in head.items()):
        raise SystemExit("head.safetensors does not load into DecimaModel (missing %s)" % missing)
    return d


def state_string(state):
    """serve.py / system_one: a string as is, anything else as json.dumps(ensure_ascii=False)."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


class _Null:
    """A model for system_one that answers nothing: running it applies upstream's request checks only."""

    def decide(self, state, q):
        from decima.types import Decision

        return Decision(probs=[1.0 / len(q.choices)] * len(q.choices), choices=list(q.choices))


def verdict(d, state, questions):
    """serve.py do_POST after the body is parsed: (None, None) for a 200, else (code, message) of its 422.

    Its order: the state as text; per question in order, to_question (INVALID_REQUEST) and the state row's length
    (STATE_TRUNCATED); then system_one, whose own checks (1-256 questions, the state's JSON type) come first."""
    from decima.systemone import SystemOneError, system_one, to_question

    try:
        s = state_string(state)
        for qid, spec in (questions or {}).items():
            _, q, _ = to_question(qid, spec)
            if len(d.tok(d.cfg.state_of(s, q.text, q.lang))["input_ids"]) > d.cfg.max_state_tokens:
                return "STATE_TRUNCATED", "state: part of state would be dropped to fit the context"
        system_one(_Null(), state, questions)
    except SystemOneError as e:
        return e.code, str(e)
    return None, None


def encode(d, state, questions):
    """Every question's texts and token ids, as Decima.logits builds them. Raises RequestError for a question
    to_question rejects. A state row over max_state_tokens keeps its first tokens (model.py's truncation)."""
    from decima.systemone import SystemOneError, to_question

    s = state_string(state)
    items = []
    for qid, spec in questions.items():
        try:
            t, q, keys = to_question(qid, spec)
        except SystemOneError as e:
            raise RequestError(e.code, str(e)) from e
        state_text = d.cfg.state_of(s, q.text, q.lang)
        option_texts = d.cfg.choices_of(d.tok, q.text, q.choices, q.lang)
        full = d.tok(state_text)["input_ids"]
        items.append({
            "qid": qid, "type": t, "question": q, "keys": keys, "state": s,
            "state_text": state_text, "option_texts": option_texts,
            "state_ids": d.tok(state_text, truncation=True, max_length=d.cfg.max_state_tokens)["input_ids"],
            "option_ids": [d.tok(o, truncation=True, max_length=d.cfg.max_choice_tokens)["input_ids"]
                           for o in option_texts],
            "state_truncated": len(full) > d.cfg.max_state_tokens,
        })
    return items


@torch.no_grad()
def forward(d, item):
    """Decima.logits for one question, step by step: the raw scores s_k (before the temperature), the ordinal
    head's per-option projections ord_g(z_k) and ord_gap(z_k), and the calibrated log-probabilities. Upstream
    order: a choice's labels, score levels, noul [yes, no]."""
    q = item["question"]
    hc, mc = d._choices(q)
    hs, ms = d._encode([item["state_text"]], d.cfg.max_state_tokens)
    n = len(q.choices)
    owner = torch.zeros(n, dtype=torch.long, device=hs.device)
    s, z = d.model.choice_scores(hs, ms, hc, mc, owner)
    lp = d.model.log_probs(s, z, owner, 1, [q.kind], d.cfg.temperature)[0]
    # The same numbers as upstream's own entry point, to the bit.
    whole = torch.from_numpy(d.logits([item["state"]], q)[0])
    if not torch.equal(whole, lp.float().cpu()):
        raise AssertionError("%s: step-by-step log-probabilities differ from Decima.logits" % item["qid"])
    return {
        "scores": s.double().cpu().numpy(),
        "ordinal_g": d.model.ord_g(z).squeeze(-1).double().cpu().numpy(),
        "ordinal_gap": d.model.ord_gap(z).squeeze(-1).double().cpu().numpy(),
        "log_probs": lp.double().cpu().numpy(),
    }


def answers(d, state, questions):
    """decima.systemone.system_one on the PyTorch reference (Decima.decide_many, as serve.py's DecimaOnnx): the answers
    object of upstream's /v1/systemone."""
    from decima.systemone import system_one

    return system_one(d, state, questions)


def main():
    d = load("cpu", sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
    state = "My card was charged twice"
    qs = {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "charges, refunds",
                                                                                 "tech": "bugs"}},
          "urgent": {"type": "noul", "instructions": "The customer needs an answer today."},
          "anger": {"type": "score", "instructions": "How upset is the customer?",
                    "criteria": ["calm", "annoyed", "furious"]}}
    print(verdict(d, state, qs))
    for it in encode(d, state, qs):
        print(it["qid"], {k: np.round(v, 4).tolist() for k, v in forward(d, it).items()})
    print(json.dumps(answers(d, state, qs)))


if __name__ == "__main__":
    main()
