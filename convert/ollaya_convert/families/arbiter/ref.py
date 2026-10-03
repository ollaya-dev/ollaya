"""PyTorch reference for hiteshluke/arbiter-4b: an inline implementation of the family (no external
`arbiter` package).

Arbiter is a LoRA on `unsloth/gemma-3-4b-it` plus a fixed `nn.Linear(2560, 24)` pointer head read at
the final-position hidden state of the prompt. The head's rows are initialized from the base LM-head
verbalizer rows for the 24 tokens listed in `layout.VERBALIZERS` and jointly trained with the LoRA; the
checkpoint carries the EMA average of the last 5 eval heads.

    ck, tok, m = ref.load(run_dir, base_dir, device="cuda")
    enc = ref.encode(tok, m, state, questions)
    scores = ref.forward(m, enc)        # list of per-row 24-dim slot score vectors
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .layout import ArbiterLayout, VERBALIZERS, NUM_SLOTS

HIDDEN_SIZE = 2560
PAD_FALLBACK = 0

MODELS = {
    "arbiter-4b": {
        "repo": "hiteshluke/arbiter-4b",
        "revision": "0c44271c59f89758e3cae17b032e98a9140093e9",
        "base": "unsloth/gemma-3-4b-it",
        "base_revision": "bf46152c47f5dd20b896357cb51abc4c03b8ee8c",
        "base_files": ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
    },
}


def snapshot(slug: str) -> Tuple[str, str]:
    """Resolve local snapshot directories for the checkpoint and the base model."""
    from huggingface_hub import snapshot_download

    m = MODELS[slug]
    run = os.environ.get("ARBITER_RUN") or snapshot_download(m["repo"], revision=m["revision"])
    base = os.environ.get("ARBITER_BASE") or snapshot_download(m["base"], revision=m["base_revision"])
    return run, base


def _verbalizer_slot_ids(tok) -> List[int]:
    """Resolve the 24 verbalizer token ids by tokenizing each character on its own. Slot 1 (F in T/F)
    and slot 7 (F in A..P) are the same token id; keep both entries so the slot order is preserved."""
    ids: List[int] = []
    for ch in VERBALIZERS:
        enc = tok(ch, add_special_tokens=False)
        tid = enc["input_ids"] if isinstance(enc, dict) else enc.ids
        if len(tid) != 1:
            raise ValueError("verbalizer %r did not tokenize to a single id (got %r)" % (ch, tid))
        ids.append(int(tid[0]))
    return ids


class ArbiterHead(nn.Module):
    """Fixed 24-slot pointer head over the final-position hidden state."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE, num_slots: int = NUM_SLOTS):
        super().__init__()
        self.proj = nn.Linear(hidden_size, num_slots, bias=True)

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        return self.proj(h_last)


class ArbiterModel(nn.Module):
    """Wraps the Gemma 3 text model + LoRA + the fixed 24-slot head.

    `forward_hidden(ids, attn)` returns `[rows, hidden]`, the last valid position's hidden state per
    row (indexed by `attn.sum(-1) - 1`).
    """

    def __init__(self, text_model, head: ArbiterHead):
        super().__init__()
        self.text_model = text_model
        self.head = head

    @torch.no_grad()
    def forward_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.text_model(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1] if hasattr(out, "hidden_states") and out.hidden_states is not None \
            else out.last_hidden_state
        last = attention_mask.sum(-1) - 1
        rows = torch.arange(h.shape[0], device=h.device)
        return h[rows, last].float()

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_hidden(input_ids, attention_mask))


class Checkpoint:
    """Minimal checkpoint descriptor; carries meta plus the resolved base pin for the export check."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.meta: Dict[str, Any] = {}
        meta_path = os.path.join(run_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.meta = json.load(f)
        self.upstream_base: Optional[Tuple[str, Optional[str]]] = None


def _load_tokenizer(base_dir: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_dir)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else PAD_FALLBACK
    return tok


def _load_base_text_model(base_dir: str, dtype: torch.dtype, device: str):
    """Load Gemma 3 4B IT in text-only form (we never run the vision tower)."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=dtype)
    model.to(device)
    # Gemma 3 4B IT exposes its decoder as model.model; keep the whole causal LM so output_hidden_states works.
    return model


def _wrap_lora(base_model, run_dir: str, merge: bool):
    from peft import PeftModel
    peft_model = PeftModel.from_pretrained(base_model, run_dir)
    if merge:
        peft_model = peft_model.merge_and_unload()
    return peft_model


def _load_head(run_dir: str, device: str) -> ArbiterHead:
    head = ArbiterHead()
    head_path = os.path.join(run_dir, "head.pt")
    state = torch.load(head_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Accept either {'proj.weight', 'proj.bias'} or raw {'weight', 'bias'}.
    sd = {}
    for k, v in state.items():
        if k.startswith("proj."):
            sd[k] = v
        elif k in ("weight", "bias"):
            sd["proj." + k] = v
    head.load_state_dict(sd)
    head.to(device).eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


def load(run_dir: str, base_dir: str, device: str = "cpu", merge: bool = True):
    """Load tokenizer + Gemma 3 text model + LoRA + 24-slot head in fp32. Returns (ck, tok, m)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ck = Checkpoint(run_dir)
    ck.upstream_base = (ck.meta.get("base", "unsloth/gemma-3-4b-it"),
                        ck.meta.get("base_revision"))
    tok = _load_tokenizer(base_dir)
    base = _load_base_text_model(base_dir, dtype=torch.float32, device=device)
    peft = _wrap_lora(base, run_dir, merge=merge)
    head = _load_head(run_dir, device=device)
    m = ArbiterModel(peft, head).to(device).eval()
    return ck, tok, m


def encode(tok, m: ArbiterModel, state, questions):
    """-> (enc, meta). `enc` is a dict with batched tensors ready for forward_hidden."""
    decision = {"max_state_tokens": 8192, "max_row_tokens": 8192}
    lay = ArbiterLayout(tok, decision)
    row_dicts, meta = lay.encode(state, questions)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else PAD_FALLBACK
    T = max(len(r["ids"]) for r in row_dicts)
    R = len(row_dicts)
    ids = torch.full((R, T), pad_id, dtype=torch.long)
    attn = torch.zeros((R, T), dtype=torch.long)
    for i, r in enumerate(row_dicts):
        n = len(r["ids"])
        ids[i, :n] = torch.tensor(r["ids"], dtype=torch.long)
        attn[i, :n] = 1
    enc = {"input_ids": ids, "attention_mask": attn, "rows": row_dicts}
    return enc, meta


def rows(enc) -> List[Dict[str, Any]]:
    """Per-row dicts {'ids', 'last_pos', 'slots'}."""
    return enc["rows"]


@torch.no_grad()
def forward(m: ArbiterModel, enc) -> List["torch.Tensor"]:
    """Raw 24-slot scores per row (no temperature, no slot masking)."""
    device = next(m.parameters()).device
    ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    scores = m(ids, attn).float().cpu()
    return [scores[i].numpy() for i in range(scores.shape[0])]
