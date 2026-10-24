"""`gliclass-kv-v1`: the request -> decoder rows layout of GLiClass v4 (decoder-KV), written the way the Rust port is
(HF `tokenizers` and `decision.json` only, no `gliclass` package). `parity.py` checks it id for id against upstream
`DecoderKVZeroShotClassificationPipeline.prepare_inputs`.

One row per question (Ollaya's mapping of a typed question onto GLiClass's `labels` / `prompt` call is
`families.gliclass.ref.question_call`, the same as for the uni-encoder models):

    context = sanitize(prompt) + sanitize(state_text)              # prompt_first, no separator (upstream)
    labels  = "<<SEP>>" + "".join(label + "<<LABEL>>" for label in labels) + "<<SEP>>"
    row     = enc(context)[:max(1, max_len - len(enc(labels)))] + enc(labels)   # no special tokens, right truncation

The label section starts after the `<<SEP>>` that follows the context and ends with the final `<<SEP>>`; the
scorer reads label j at its `<<LABEL>>` and the text at the final `<<SEP>>`. The graph returns the calibrated label
logits; option logits are those logits (choice, score, a noul with both descriptions) or `[0, z]` for a one-label noul.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..gliclass.ref import question_call, sanitize
from laya.common import serialize_state


class GliclassKvLayout:
    def __init__(self, tok, decision: Dict[str, Any]):
        self.tok = tok
        self.max_len = int(decision["max_len"])
        s = decision["special_tokens"]
        self.label, self.sep, self.pad = s["label"], s["sep"], s["pad"]
        self.label_text, self.sep_text = s["label_text"], s["sep_text"]

    def enc(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode(self, state: Any, questions: Dict[str, Any]):
        """-> rows [{"qid", "qtype", "mode", "labels", "prompt", "ids", "markers", "section_start", "sep"}]."""
        text = sanitize(serialize_state(state))
        rows = []
        for qid, qdef in questions.items():
            call = question_call(qdef)
            labels = [sanitize(l) for l in call["labels"]]
            prompt = sanitize(call["prompt"])
            label_ids = self.enc(self.sep_text + "".join(l + self.label_text for l in labels) + self.sep_text)
            ctx = self.enc(prompt + text)[: max(1, self.max_len - len(label_ids))]
            ids = ctx + label_ids
            markers = [len(ctx) + i for i, t in enumerate(label_ids) if t == self.label]
            if len(markers) != len(labels):
                raise ValueError("question %r: a label spells the model's %s marker" % (qid, self.label_text))
            rows.append({"qid": qid, "qtype": qdef["type"], "mode": call["mode"], "labels": labels, "prompt": prompt,
                         "ids": ids, "markers": markers, "section_start": len(ctx) + 1, "sep": len(ids) - 1,
                         "state_truncated": len(ctx) < len(self.enc(prompt + text))})
        return rows
