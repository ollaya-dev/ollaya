"""The PyTorch reference for GLiClass v4 (decoder-KV), driven through the `gliclass` package in fp32.

    uv run --with gliclass==0.1.21 python -m ollaya_convert.families.gliclass_kv.ref 2b

Upstream (Apache-2.0): the `gliclass` package 0.1.21 (PyPI; github.com/Knowledgator/GLiClass @ 6a00f80), checkpoints
`knowledgator/gliclass-v4-2b` (Qwen/Qwen3.5-2B) and `knowledgator/gliclass-v4-9b` (Qwen/Qwen3.5-9B).

GLiClass is a classifier, not a typed-decision model; the question mapping is Ollaya's and the same as for the
uni-encoder models (`families.gliclass.ref.question_call`). The model call is upstream's classic (non-streaming)
`DecoderKVZeroShotClassificationPipeline`:
  * input ids      `prepare_inputs`: context = prompt + text, tokenized with right truncation to
                   max_length - len(label ids), no special tokens; then `<<SEP>>label1<<LABEL>>...<<SEP>>`
  * network        GLiClassModel.forward: the Qwen3.5 decoder over the whole row (causal), the label section
                   (after the last <<SEP>> before the first <<LABEL>>) through DecoderKVScorer: DeBERTa encoder,
                   text repr at the final <<SEP>>, label reprs at the <<LABEL>> tokens, MLP; then
                   recurrent_steps - 1 gated steps, each example halting once no label's sigmoid probability
                   moves by recurrent_halt_threshold; then the calibrator: logit' = beta * logit + bias
  * scoring        single-label = softmax over the calibrated label logits, multi-label = per-label sigmoid
                   (`_postprocess_logits`)
The special markers "<<LABEL>>", "<<SEP>>", "<<EXAMPLE>>" are replaced by " " in every user string, as for the
uni-encoder models.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np
import torch

from ..gliclass.ref import MARKERS, QTYPES, option_logits, question_call, sanitize, softmax  # noqa: F401
from laya.common import serialize_state

MODELS = {
    "gliclass-v4-2b": {"repo": "knowledgator/gliclass-v4-2b", "revision": "40fe023b45e8d97d1ae4f7f9342279ac1c76317d",
                       "base": "Qwen/Qwen3.5-2B"},
    "gliclass-v4-9b": {"repo": "knowledgator/gliclass-v4-9b", "revision": "14d11a92b5e0bab4a166e1fce4cc0777cac27182",
                       "base": "Qwen/Qwen3.5-9B"},
}
UPSTREAM = "gliclass==0.1.21"
GLICLASS_GIT = {"repo": "https://github.com/Knowledgator/GLiClass", "commit": "6a00f80"}
# The model card's usage (`ZeroShotClassificationPipeline(..., max_length=8192)`); the pipeline's own default is 1024.
MAX_LEN = 8192


def snapshot(slug: str) -> str:
    env = os.environ.get("GLICLASS_ROOT_" + slug.upper().replace("-", "_"))
    if env:
        return env
    from huggingface_hub import snapshot_download

    m = MODELS[slug]
    return snapshot_download(m["repo"], revision=m["revision"], allow_patterns=["*.json", "*.safetensors", "*.jinja",
                                                                                "README.md"])


class Model:
    def __init__(self, slug: str, device: str, root: str | None = None):
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline
        from transformers import AutoTokenizer

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.slug = slug
        self.repo, self.revision = MODELS[slug]["repo"], MODELS[slug]["revision"]
        self.root = root or snapshot(slug)
        self.model = GLiClassModel.from_pretrained(self.root, dtype=torch.float32).to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(self.root)
        self.device = device
        pipe = ZeroShotClassificationPipeline(self.model, self.tok, max_length=MAX_LEN,
                                              classification_type="single-label", device=device, progress_bar=False)
        self.pipe = pipe.pipe  # DecoderKVZeroShotClassificationPipeline
        cfg = self.model.config
        for tok_text, idx in (("<<LABEL>>", cfg.class_token_index), ("<<SEP>>", cfg.sep_token_index),
                              ("<<EXAMPLE>>", cfg.example_token_index)):
            if self.tok.convert_tokens_to_ids(tok_text) != idx:
                raise RuntimeError("%s is not token %d in %s" % (tok_text, idx, self.repo))
        assert cfg.architecture_type == "decoder-kv" and cfg.prompt_first and not cfg.scorer_full_sequence
        assert cfg.recurrent_inference and not cfg.recurrent_read_text and cfg.recurrent_steps > 1
        assert cfg.use_calibrator and self.model.calibrator is not None
        assert cfg.problem_type == "multi_label_classification" and not cfg.normalize_features
        assert self.tok.pad_token_id == cfg.pad_token_id


def load(slug: str = "gliclass-v4-2b", device: str = "cpu", root: str | None = None) -> Model:
    return Model(slug, device, root)


def encode(m: Model, state: Any, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """One decoder row per question: the upstream input ids, the label section and the <<LABEL>> positions."""
    text = sanitize(serialize_state(state))
    items = []
    for qid, qdef in questions.items():
        call = question_call(qdef)
        labels = [sanitize(l) for l in call["labels"]]
        prompt = sanitize(call["prompt"])
        inputs = m.pipe.prepare_inputs([text], labels, same_labels=True, prompt=prompt)
        ids = inputs["input_ids"][0].tolist()
        cfg = m.model.config
        markers = [i for i, t in enumerate(ids) if t == cfg.class_token_index]
        seps = [i for i, t in enumerate(ids) if t == cfg.sep_token_index]
        start = max(s for s in seps if s < markers[0]) + 1   # first token of the label section
        items.append({"qid": qid, "qtype": qdef["type"], "mode": call["mode"], "labels": labels, "prompt": prompt,
                      "ids": ids, "markers": markers, "section_start": start, "sep": seps[-1]})
    return {"state_text": text, "items": items}


@torch.no_grad()
def forward(m: Model, enc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Upstream forward, one unpadded row per question: calibrated label logits (what the pipeline scores), the
    uncalibrated ones, every recurrent step's raw logits and the steps the halting rule took."""
    out = []
    for it in enc["items"]:
        ids = torch.tensor([it["ids"]], device=m.device)
        o = m.model(input_ids=ids, attention_mask=torch.ones_like(ids), output_recurrent_logits=True,
                    max_num_classes=len(it["labels"]))
        k = len(it["labels"])
        out.append({"logits": o.logits[0, :k].float().cpu().numpy(),
                    "uncalibrated": o.uncalibrated_logits[0, :k].float().cpu().numpy(),
                    "steps": int(o.recurrent_num_steps[0].item()),
                    "step_logits": [z[0, :k].float().cpu().numpy() for z in o.recurrent_logits]})
    return out


def upstream_scores(m: Model, item: Dict[str, Any], state_text: str) -> np.ndarray:
    """The same question through the public pipeline call, in Ollaya option order."""
    cls = "multi-label" if item["mode"] == "sigmoid" else "single-label"
    res = m.pipe(state_text, item["labels"], prompt=item["prompt"] or None, classification_type=cls,
                 threshold=0.0, return_hierarchical=True)[0]
    p = np.array([res[l] for l in item["labels"]])
    return np.array([1 - p[0], p[0]]) if item["mode"] == "sigmoid" else p


def main():
    import sys

    m = load("gliclass-v4-" + (sys.argv[1] if len(sys.argv) > 1 else "2b"), "cuda" if torch.cuda.is_available() else "cpu")
    state = {"subject": "Invoice #4411", "body": "We were billed twice for March, please refund."}
    qs = {"dept": {"type": "choice", "instructions": "Which team handles this?",
                   "criteria": {"billing": "payments", "technical": "bugs", "other": ""}},
          "refund": {"type": "noul", "instructions": "The user asks for a refund."},
          "angry": {"type": "noul", "instructions": "The customer is angry.",
                    "criteria": {"true": "strong negative emotion", "false": "calm"}}}
    enc = encode(m, state, qs)
    print(json.dumps(enc["items"][0]["ids"][-12:]), enc["items"][0]["markers"], enc["items"][0]["section_start"])
    for it, o in zip(enc["items"], forward(m, enc)):
        print(it["qid"], o["steps"], softmax(option_logits(it, o["logits"])), upstream_scores(m, it, enc["state_text"]))


if __name__ == "__main__":
    main()
