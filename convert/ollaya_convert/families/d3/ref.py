"""The PyTorch reference for vllm-sr's d3 (Decision 3.0): the runtime the repository ships, in fp32.

Each repository holds the whole checkpoint: a fine-tuned `Qwen3_5Model` (keys `language_model.*` and `visual.*`),
the decision head `readout.safetensors` (`weight` [255, hidden], fp32), `decision_config.json` (answer codes,
attention mode, temperature) and the inference code (`d3_runtime.py`, `d3_format.py`), which this module imports
from the snapshot.

The reference is upstream's plain text path (`D3.prepare` and `D3.logits`): one sequence per question, the
full-attention layers see the whole prompt (`noncausal_full_attention`, the Gated DeltaNet layers stay causal) and
the readout reads the last position. Upstream loads BF16; the reference widens the backbone to fp32 (TF32 off),
turns the ROCm fast path off and runs every question alone, unpadded.

    d3 = ref.load(run_dir, device="cuda")
    prepared = ref.encode(d3, state, questions)   # upstream prepare; RequestError when it rejects a question
    logits = ref.forward(d3, prepared)            # {qid: raw readout logits of its options}
    answers = ref.answers(d3, prepared, logits)   # upstream respond at the checkpoint's temperature
"""
from __future__ import annotations

import os
import sys

import torch

MODELS = {
    "d3-edge": {"repo": "vllm-sr/d3-edge", "revision": "c454ee39b862248d70abd7f84195d2f54ffe6d43",
                "shards": ["model-00001-of-00001.safetensors"]},
    "d3-lite": {"repo": "vllm-sr/d3-lite", "revision": "03560c26f5199c3c04129a2ccb38960c913635e0",
                "shards": ["model-00001-of-00001.safetensors"]},
    "d3-nano": {"repo": "vllm-sr/d3-nano", "revision": "462efc366229ea268cf4eed3738b3fc4415e4817",
                "shards": ["model.safetensors"]},
    "d3-mini": {"repo": "vllm-sr/d3-mini", "revision": "bd0090dd758d32ba59ef5c749910a8e87fc4391f",
                "shards": ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]},
    "d3-flash": {"repo": "vllm-sr/d3-flash", "revision": "13dc1cc8ef23caefa765907e1d764751e5629f7d",
                 "shards": ["model-0000%d-of-00004.safetensors" % i for i in range(1, 5)]},
}
# Ollaya's window per question. Upstream has no limit (`max_length` null); the graph materializes full attention,
# so a sequence of n tokens takes heads * n^2 * 4 bytes per such layer.
MAX_LENGTH = 8192


class RequestError(ValueError):
    """Upstream rejects a question of the request (invalid_question or max_length_exceeded)."""


def snapshot(slug):
    from huggingface_hub import snapshot_download

    m = MODELS[slug]
    return os.environ.get("D3_RUN") or snapshot_download(m["repo"], revision=m["revision"])


def runtime(run_dir):
    if run_dir not in sys.path:
        sys.path.insert(0, run_dir)
    import d3_format  # noqa: F401
    import d3_runtime  # noqa: F401

    return sys.modules["d3_runtime"], sys.modules["d3_format"]


def load(run_dir, device="cpu"):
    """Upstream `D3` (plain path) with the backbone and the readout in fp32."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ["D3_FAST"] = "0"
    rt, _ = runtime(run_dir)
    d3 = rt.D3.from_pretrained(run_dir, device=device, verify="none", max_length=MAX_LENGTH)
    d3.backbone.float()
    d3.readout = d3.readout.float()
    return d3


def encode(d3, state, questions):
    """Upstream `prepare` (text path): the request's questions, each with its token ids; a question upstream
    rejects (invalid_question, max_length_exceeded) rejects the request, as Ollaya's parser does."""
    try:
        prepared = d3.prepare(state, questions)
    except ValueError as e:
        raise RequestError(str(e)) from e
    if prepared.errors:
        qid, err = next(iter(prepared.errors.items()))
        raise RequestError("question %r: %s" % (qid, err["message"]))
    return prepared


def rows(prepared):
    """One sequence per question, in request order: ids, the readout position (the last token), the option count."""
    return [{"ids": [int(x) for x in prepared.sequences[k]], "markers": [len(prepared.sequences[k]) - 1],
             "counts": [len(prepared.questions[k].keys)], "keys": [k]} for k in prepared.keys]


@torch.no_grad()
def forward(d3, prepared):
    """{qid: raw readout logits of its options}: upstream `logits` on each question alone."""
    out = {}
    for k in prepared.keys:
        n = len(prepared.questions[k].keys)
        out[k] = d3.logits([prepared.sequences[k]], [n])[0, :n].tolist()
    return out


def answers(d3, prepared, logits):
    """Upstream `respond` on these logits at the checkpoint's temperature."""
    probs = {k: torch.softmax(torch.tensor(v, dtype=torch.float64) / d3.temperature, -1).tolist()
             for k, v in logits.items()}
    return d3.respond(prepared, probs, 0)["answers"]
