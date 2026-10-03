"""The PyTorch reference: load a Laya checkpoint and encode requests exactly as `laya` does.

Everything that decides what the model sees (option rendering, sequence layout, collation) is
delegated to the `laya` package itself, so the reference cannot drift from upstream.
"""
import copy
import os
from typing import Any, Dict, List, Optional

import laya
import torch
from laya.common import QTYPES, build_sequence, collate_items, render_options

from .families.von.ref import fp64_rotary
from .model_paths import DEFAULT_ROOT

# Checkpoint name -> subfolder of the bundled Laya repo (None = repo root).
CHECKPOINTS = {
    "en": None,
    "multilingual": "multilingual",
    "typed-decisions": "typed-decisions",
}


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
def forward(agent: laya.Agent, batch: Dict[str, Any], exact: Optional["Exact"] = None):
    """Raw (logits, act_logits), no autocast: upstream's fp32 forward, or the `exact` network's."""
    if exact is not None:
        logits, act = exact(*(batch[n] for n in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")))
        return logits.double().numpy(), act.double().numpy()
    dev = agent.device
    logits, act = agent.model(
        batch["input_ids"].to(dev),
        batch["attention_mask"].to(dev),
        batch["marker_pos"].to(dev),
        batch["marker_mask"].to(dev),
        batch["qtype"].to(dev),
    )
    return logits.float().cpu().numpy(), act.float().cpu().numpy()


class Exact(torch.nn.Module):
    """The golden reference: `DecisionModel.forward` with the same weights, kept in float64 end to end.

    Upstream's forward casts the scorer logits and the pooled state to fp32 (`.float()`), and
    transformers' ModernBERT applies its rotary embedding in fp32 whatever the model's dtype, so a
    `.double()` model alone still rounds in fp32. This copy of the forward keeps the model's dtype
    at every step (inside `fp64_rotary`), which is the only change; with `dtype=torch.float32` it
    reproduces upstream's fp32 numbers. On the laya:en set the fp32 goldens sit at most 6.2e-5 in
    probability (5.2e-4 in logit) from this network, on `preset/guard/email_dict` `jailbreak`.
    """

    def __init__(self, agent: laya.Agent, device: str = "cpu", dtype: torch.dtype = torch.float64):
        super().__init__()
        self.m = copy.deepcopy(agent.model).to(device=device, dtype=dtype).eval()
        self.device, self.dtype = torch.device(device), dtype

    @torch.no_grad()
    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        m, dev = self.m, self.device
        input_ids, attention_mask = input_ids.to(dev), attention_mask.to(dev)
        marker_pos, marker_mask, qtype = marker_pos.to(dev), marker_mask.to(dev), qtype.to(dev)
        with fp64_rotary():
            h = m.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + m.type_emb(qtype)[:, None, :]
        pad = ~attention_mask.bool()
        for layer in m.head.layers:
            h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        logits = m.scorer(torch.gather(h, 1, idx)).squeeze(-1).masked_fill(~marker_mask, -1e4)
        p = torch.softmax(logits, -1)
        k = marker_mask.sum(-1).clamp(min=2).to(p.dtype)
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        if p.size(-1) >= 2:
            top2 = p.topk(2, -1).values
        else:
            top1 = p.topk(1, -1).values
            top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        act = m.act_head(torch.cat([h[:, 0], feats], -1))
        return logits.cpu(), act.cpu()


def system_one_exact(agent: laya.Agent, exact: Exact, state: Any, questions: Dict[str, Dict[str, Any]]):
    """`Agent.system_one` (upstream's code, including its answer rounding) on the `exact` network."""
    model = agent.model
    agent.model = exact
    try:
        return agent.system_one(state, questions)
    finally:
        agent.model = model
