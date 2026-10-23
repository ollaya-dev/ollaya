"""The PyTorch reference for perplexity-ai/pplx-decider-flash-v1.2: the package the repository ships, in fp32.

The repository holds the whole checkpoint: a post-trained `Qwen3_5Model` (Qwen3.5-4B, two BF16 shards, keys
`language_model.*` and `visual.*`), the decision head `readout.safetensors` (`weight` [255, 2560]), the fitted
temperature (`decision_config.json`), the request protocol (`joint_config.json`) and the inference code
(`source/src/pplx_decider`), which this module imports from the snapshot.

The reference is upstream's exact path (`Decider(fast=False)`): `Encoder.plan` builds one sequence per request
(bisecting the questions when it is longer than `max_length`), and `Decider.reference_logits` runs it through the
unmodified Transformers backbone with full-attention layers that see the whole sequence (linear-attention layers
stay causal) and applies the readout at each question's `<|fim_suffix|>` marker. Upstream loads BF16 on CUDA;
the reference widens the backbone and the head to fp32 (TF32 off) and runs every sequence alone, unpadded.

    decider = ref.load(run_dir, device="cuda")
    chunks = ref.encode(decider, state, questions)        # upstream Encoder.plan at MAX_LENGTH
    logits = ref.forward(decider, chunks)                 # {qid: raw readout logits of its valid options}
    answers = ref.answers(decider, state, questions, logits, chunks)   # upstream _respond at the fitted temperature
"""
from __future__ import annotations

import os
import sys

import torch

MODELS = {
    "pplx-decider-flash": {"repo": "perplexity-ai/pplx-decider-flash-v1.2",
                           "revision": "6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d",
                           "shards": ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]},
    # Ollaya's own: Qwen3.5-0.8B-Base distilled from pplx-decider-flash in its joint format (train/sura.py).
    "sura-0.8b": {"repo": "ollaya-dev/sura-0.8b",
                  "revision": "9e991faa3250b255cc41d9cd8f5422f1ca9f7694",
                  "shards": ["model.safetensors"]},
}
# Ollaya's window per sequence. Upstream's default is the backbone's 262,144 positions; the graph materializes
# full attention (8 of 32 layers), so a sequence of n tokens takes 16 * n^2 * 4 bytes per such layer.
MAX_LENGTH = 8192


class RequestError(ValueError):
    """Upstream rejects the request (RequestError, or LengthError: one question with the state over the window)."""


def snapshot(slug):
    from huggingface_hub import snapshot_download

    m = MODELS[slug]
    return os.environ.get("PPLX_RUN") or snapshot_download(m["repo"], revision=m["revision"])


def package(run_dir):
    src = os.path.join(run_dir, "source", "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import pplx_decider.model  # noqa: F401
    import pplx_decider.protocol  # noqa: F401

    return sys.modules["pplx_decider.model"], sys.modules["pplx_decider.protocol"]


def load(run_dir, device="cpu"):
    """Upstream `Decider` (reference path) with the backbone and the readout in fp32."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, _ = package(run_dir)
    from pathlib import Path

    decider = model.Decider(Path(run_dir), device=device, fast=False, max_length=MAX_LENGTH,
                            cpu_threads=os.cpu_count() or 8)   # upstream's default is 8
    decider.backbone.float()
    decider.readout.float()
    return decider


def encode(decider, state, questions):
    """Upstream's plan: the request's sequences (`protocol.Encoded`), each with its question keys and markers."""
    _, protocol = package(decider.checkpoint)
    try:
        return decider.encoder.plan({"state": state, "questions": questions}, MAX_LENGTH)
    except (protocol.RequestError, protocol.LengthError) as e:
        raise RequestError(str(e)) from e
    except (AttributeError, TypeError) as e:   # upstream assumes the documented shapes (e.g. a dict question)
        raise RequestError("%s: %s" % (type(e).__name__, e)) from e


def rows(chunks):
    """The sequences as plain lists: ids, marker positions, option counts and question keys."""
    return [{"ids": [int(x) for x in c.input_ids], "markers": [int(x) for x in c.markers], "counts": list(c.counts),
             "keys": list(c.keys)} for c in chunks]


@torch.no_grad()
def forward(decider, chunks):
    """{qid: raw readout logits of its options}: upstream reference_logits on each sequence alone."""
    out = {}
    for c in chunks:
        for key, values in zip(c.keys, decider.reference_logits([c])[0], strict=True):
            out[key] = values
    return out


def answers(decider, state, questions, logits, chunks):
    return decider._respond({"state": state, "questions": questions}, logits, chunks, decider.temperature)["answers"]
