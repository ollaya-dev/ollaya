"""The PyTorch reference for Mapika/decider: upstream's own `decider` package, run in fp32.

Each model repo ships the inference subset of https://github.com/Mapika/decider under `decider/`;
this module imports that copy from the pinned snapshot, so the reference cannot drift from what the
authors serve. Everything that decides what the model sees (state rendering, question rendering,
isolated score rows, prompt building) is upstream code; only the forward is called with all 255
label logits unmasked, since the ONNX graph returns them all and the runtime slices.

    root = ref.snapshot("decider-0.8b")          # or DECIDER_ROOT_<SLUG>=/path/to/local/copy
    d = ref.load(root, device="cuda")            # fp32, TF32 off, eager (no CUDA graphs)
    items, plan = ref.encode(d, state, questions)
    logits = ref.forward(d, items)               # [rows, 255] fp32
    answers = ref.system_one(d, state, questions)
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict

import torch

MODELS = {
    "decider-0.8b": {"repo": "Mapika/decider-0.8b", "revision": "a0a01d6f8135298f400a8c856b355793012ae971",
                     "base": "Qwen/Qwen3.5-0.8B-Base"},
    "decider-2b": {"repo": "Mapika/decider-2b", "revision": "9839cc9d908be16c5988c0d041034b5fdf82c7a2",
                   "base": "Qwen/Qwen3.5-2B-Base"},
}


def snapshot(slug: str) -> str:
    env = os.environ.get("DECIDER_ROOT_" + slug.upper().replace("-", "_").replace(".", "_"))
    if env:
        return env
    from huggingface_hub import snapshot_download

    m = MODELS[slug]
    return snapshot_download(m["repo"], revision=m["revision"])


def load(root: str, device: str = "cpu"):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if root not in sys.path:
        sys.path.insert(0, root)
    from decider.infer import Decider

    return Decider(root, device=device, dtype=torch.float32, use_graphs=False)


class _Keep:
    def shuffle(self, x):
        pass

    def sample(self, xs, k):
        return xs[:k]


def encode(d, state: Any, questions: Dict[str, Any], max_state_tokens: int = 32768):
    """Upstream rows for a system_one request (independent=True, the default): (items, index)."""
    from decider.infer import Example, Q, neutralize_options
    from decider.prompt import MAX_OPTIONS, build
    from decider.systemone import plan_rows, render_question, render_state

    ctx = render_state(state)
    rqs = {k: render_question(v) for k, v in questions.items()}
    opts = (lambda r: neutralize_options(r["options"])[0]) if d.neutralize_none else (lambda r: list(r["options"]))
    flat, index = plan_rows(rqs, d.isolated_levels)
    items = [build(Example(ctx, [Q(r["question"], opts(r), 0)]), d.m.tok, _Keep(), max_options=MAX_OPTIONS,
                   max_ctx_tokens=max_state_tokens, layout="state_first") for r in flat]
    return items, index


@torch.no_grad()
def forward(d, items, max_tokens: int = 8192):
    """All 255 label logits at every row's slot, fp32, in row order (rows batched up to max_tokens)."""
    from decider.model import collate
    from decider.prompt import MAX_OPTIONS

    out = []
    i = 0
    while i < len(items):
        j = i + 1
        longest = len(items[i]["ids"])
        while j < len(items) and max(longest, len(items[j]["ids"])) * (j + 1 - i) <= max_tokens:
            longest = max(longest, len(items[j]["ids"]))
            j += 1
        b = collate(items[i:j], d.m.tok.pad_token_id)
        nopts = torch.full_like(b["nopts"], MAX_OPTIONS)
        lg = d.m.slot_logits(b["input_ids"].to(d.dev), b["attention_mask"].to(d.dev), b["slot_idx"].to(d.dev),
                             b["slot_batch"].to(d.dev), nopts.to(d.dev))
        out.append(lg.float().cpu())
        i = j
    return torch.cat(out).numpy()


def system_one(d, state, questions):
    return d.system_one(state, questions)


def answers(d, state, questions, logits):
    """Upstream's answer assembly (decider.systemone.assemble) applied to `logits` from forward(): the
    same computation system_one does after its own forward pass, without running the model again."""
    from decider.systemone import assemble, plan_rows, render_question

    rqs = {k: render_question(v) for k, v in questions.items()}
    flat, index = plan_rows(rqs, d.isolated_levels)
    probs = []
    for r, row in zip(logits, flat):
        z = torch.tensor(r[: len(row["options"])]) / d.T
        probs.append(torch.softmax(z, -1).tolist())
    return assemble(rqs, index, probs)
