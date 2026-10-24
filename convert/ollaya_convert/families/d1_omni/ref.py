"""The PyTorch reference for d1-omni: the model repo's own code (`modeling_d1.py`, `encoder.py`, `prompt.py`),
loaded with `trust_remote_code`, run in fp32 with TF32 off.

    root = ref.snapshot()                       # or D1_OMNI_ROOT=/path/to/local/copy
    m = ref.load(root, device="cuda")
    rows = ref.encode(m, state, questions)      # upstream prompt.encode: ids and marker positions per question
    logits = ref.forward(m, rows)               # raw marker logits [rows, k], fp32, before the temperature
    probs = ref.probabilities(m, state, questions)   # upstream's own `probabilities` (temperature applied)

Everything that decides what the model sees is upstream code: `prompt.render_options`, `prompt.encode` and the
network (`encoder.Trunk`, `encoder.DecisionHead`). `forward` repeats the text branch of `D1OmniModel._forward`
up to the head's logits, because upstream only returns probabilities; `parity.py` checks the two agree.
"""
from __future__ import annotations

import importlib
import os
from typing import Any, Dict, List

import torch

REPO = "LiquidAI/d1-omni-600M"
REVISION = "02b55d7076f15129e59ab3f94783f32c4b088674"
BASE = "LiquidAI/LFM2.5-Encoder-350M"

QTYPES = {"choice": 0, "score": 1, "noul": 2}


def snapshot() -> str:
    env = os.environ.get("D1_OMNI_ROOT")
    if env:
        return env
    from huggingface_hub import snapshot_download

    return snapshot_download(REPO, revision=REVISION, allow_patterns=["*.py", "*.json", "model.safetensors", "LICENSE",
                                                                     "README.md"])


def load(root: str, device: str = "cpu"):
    """`AutoModel.from_pretrained(root, trust_remote_code=True)` in fp32, eval, TF32 off."""
    from transformers import AutoModel

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m = AutoModel.from_pretrained(root, trust_remote_code=True, dtype=torch.float32).to(device).eval()
    m.tokenizer  # noqa: B018  (cached_property: load it once, here)
    return m


def prompt(m):
    """The repo's `prompt` module, as transformers imported it next to `modeling_d1`."""
    return importlib.import_module(type(m).__module__.rsplit(".", 1)[0] + ".prompt")


def encode(m, state: Any, questions: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Upstream rows of a text request, in question order: {"qid", "qtype", "k", "ids", "markers", "q"}.
    Raises ValueError where upstream does (`as_question`, `encode`)."""
    p = prompt(m)
    out = []
    for qid, qdef in questions.items():
        q = p.as_question(qdef)
        ids, markers = p.encode(m.tokenizer, "" if state is None else state, q, m.config.max_length, None, False)
        out.append({"qid": qid, "qtype": q.type, "k": q.options, "ids": ids, "markers": markers, "q": q})
    return out


@torch.no_grad()
def forward(m, rows: List[Dict[str, Any]], max_tokens: int = 65536) -> List[List[float]]:
    """The head's raw logit at every marker of every row (the first `k` of each), fp32. Rows are batched by
    padded-token budget as `D1OmniModel._run` does; this is `_forward`'s text branch without the temperature."""
    p = prompt(m)
    device = m.device
    out: List[List[float]] = [None] * len(rows)  # type: ignore[list-item]
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ids"]), reverse=True)
    while order:
        longest, size = len(rows[order[0]]["ids"]), 1
        while size < len(order) and (size + 1) * longest <= max_tokens:
            size += 1
        batch, order = order[:size], order[size:]
        seqs = [m.encoder.embed_tokens(torch.tensor(rows[i]["ids"], device=device)) for i in batch]
        lengths = [len(rows[i]["ids"]) for i in batch]
        h = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        total = torch.tensor(lengths, device=device)
        pad = torch.arange(h.shape[1], device=device)[None] < total[:, None]
        h = m.encoder(h, pad, torch.zeros(len(batch), dtype=torch.long, device=device))
        k = max(len(rows[i]["markers"]) for i in batch)
        marker_pos = torch.zeros(len(batch), k, dtype=torch.long, device=device)
        marker_mask = torch.zeros(len(batch), k, dtype=torch.bool, device=device)
        for b, i in enumerate(batch):
            marker_pos[b, : len(rows[i]["markers"])] = torch.tensor(rows[i]["markers"], device=device)
            marker_mask[b, : len(rows[i]["markers"])] = True
        qtype = torch.tensor([p.QTYPES[rows[i]["qtype"]] for i in batch], device=device)
        logits = m.head(h, pad, marker_pos, marker_mask, qtype)
        for b, i in enumerate(batch):
            out[i] = logits[b, : rows[i]["k"]].float().cpu().tolist()
    return out


@torch.no_grad()
def probabilities(m, state: Any, questions: Dict[str, Any]) -> List[List[float]]:
    """Upstream `probabilities`, in Ollaya's option order ([false, true] for a noul; upstream returns [yes, no])."""
    probs = m.probabilities("" if state is None else state, list(questions.values()))
    return [list(reversed(pr)) if qdef.get("type") == "noul" else list(pr) for pr, qdef in zip(probs, questions.values())]


def system_one(m, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
    return m.system_one("" if state is None else state, questions)


def temperature(m, qtype: str, k: int) -> float:
    """`config.temperatures[temperature_key(q)]`, else the type's, else 1.0 (as `_forward` reads it)."""
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    t = m.config.temperatures
    return float(t.get("%s:%s" % (qtype, size), t.get(qtype, 1.0)))
