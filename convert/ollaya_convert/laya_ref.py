"""The PyTorch reference: load a Laya checkpoint and encode requests exactly as `laya` does.

Everything that decides what the model sees (option rendering, sequence layout, collation) is
delegated to the `laya` package itself, so the reference cannot drift from upstream.
"""
import os
from typing import Any, Dict, List

import laya
import torch
from laya.common import QTYPES, build_sequence, collate_items, render_options

# Checkpoint name -> subfolder of the bundled Laya repo (None = repo root).
CHECKPOINTS = {
    "en": None,
    "multilingual": "multilingual",
    "typed-decisions": "typed-decisions",
}

DEFAULT_ROOT = os.environ.get("LAYA_ROOT", os.path.expanduser("~/models/laya"))


def load(name: str, root: str = DEFAULT_ROOT, device: str = "cpu") -> laya.Agent:
    """Load a checkpoint in fp32 on `device` (CPU keeps the reference deterministic)."""
    return laya.load(root, subfolder=CHECKPOINTS[name], device=device)


def encode(agent: laya.Agent, state: Any, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The exact tensors `Agent.system_one` feeds the network, plus the per-question items."""
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    items: List[Dict[str, Any]] = []
    for qid, qdef in questions.items():
        agent._check_question(qid, qdef)
        q = agent._to_internal(qdef)
        seq, markers = build_sequence(agent.tok, state, q, max_len, head_max_len)
        if len(markers) != len(render_options(q)):
            raise ValueError("question %r options exceed head_max_len=%d" % (qid, head_max_len))
        items.append({"qid": qid, "ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
    batch = collate_items([items], agent.tok.pad_token_id)
    return {"items": items, "batch": batch}


@torch.no_grad()
def forward(agent: laya.Agent, batch: Dict[str, Any]):
    """Raw (logits, act_logits) in fp32, no autocast: the numbers an fp32 ONNX export must match."""
    dev = agent.device
    logits, act = agent.model(
        batch["input_ids"].to(dev),
        batch["attention_mask"].to(dev),
        batch["marker_pos"].to(dev),
        batch["marker_mask"].to(dev),
        batch["qtype"].to(dev),
    )
    return logits.float().cpu().numpy(), act.float().cpu().numpy()
