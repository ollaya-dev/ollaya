"""The PyTorch reference for vllm-sr's Decision 2.0 models: the author's own code, run in fp32.

Each model repo ships its runtime (`decision2/`, with the training code it was scored with vendored under
`decision2/_vendor/dev2model`). This module imports that package from the pinned snapshot, so the reference
cannot drift from what the author serves, and replicates the exact path of `decision2/qwen.py`
(`QwenDecision.system_one` with `share_context` off) in fp32:

  * the request checks of `system_one`, then the author's `question_to_row` and `encode` per question;
  * `collate` pads the request's questions into one batch, as `system_one` does on the CPU;
  * `DecisionModel.from_checkpoint` holds every tensor in fp32 (upstream's CPU path; its GPU path runs the
    backbone under BF16 autocast), TF32 off, SDPA on its exact math kernel;
  * Kai's per-level score offsets (`score_bias.json`, bound to the weights' fingerprint) are added, and answers
    are the author's `product_answer` at temperature 1 (no package has a calibration file).

Any question upstream answers with an error rejects the whole request here, as it does in Ollaya.

    d = ref.load(ref.snapshot("decision2-eos"), "decision2-eos", device="cuda")
    rows, items = d.encode(state, questions)       # author's question_to_row + encode
    logits = d.forward(items)                      # list of [k] fp32 candidate logits
    answers = d.answers(rows, items, logits)       # author's product_answer
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

MODELS = {
    "decision2-kai": {"repo": "vllm-sr/Decision-2.0-Kai-0.6B", "revision": "cc7f30cda9e7c7b28c5f71ef33f01e09385f1c4b"},
    "decision2-eos": {"repo": "vllm-sr/Decision-2.0-Eos-0.8B", "revision": "52cad37ef21f598302fe964cbfa164739cb965d5"},
    "decision2-sol": {"repo": "vllm-sr/Decision-2.0-Sol-2B", "revision": "df30237f9c1ced5765fdbb499dd1f2c7dca478ad"},
    "decision2-nox": {"repo": "vllm-sr/Decision-2.0-Nox-4B", "revision": "ae269ecf34a0b3a89026bc1afd9c995279212fad"},
    "decision2-lux": {"repo": "vllm-sr/Decision-2.0-Lux-9B", "revision": "1dcd1e7f240eab37ac09a6d0b0f77654b59411f9"},
}


def snapshot(slug: str) -> str:
    env = os.environ.get("DECISION2_ROOT_" + slug.upper().replace("-", "_"))
    if env:
        return env
    from huggingface_hub import snapshot_download

    m = MODELS[slug]
    return snapshot_download(m["repo"], revision=m["revision"], ignore_patterns=["assets/*"])


def package(root: str) -> dict:
    """The repo's package pointer (`config.json`) and manifest."""
    read = lambda name: json.load(open(os.path.join(root, name), encoding="utf-8"))  # noqa: E731
    return {"pointer": read("config.json"), "manifest": read("MODEL_MANIFEST.json"),
            "decision": read("decision_config.json")}


class Reference:
    def __init__(self, root: str, slug: str, device: str = "cpu", load_model: bool = True):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.root, self.slug, self.device = root, slug, device
        pkg = package(root)
        self.manifest = pkg["manifest"]
        assert self.manifest["profile"] == "qwen-full" and self.manifest["calibration"] is None, self.manifest["profile"]
        # One model per process: the package is imported from this snapshot under its own name.
        sys.path.insert(0, root)
        from decision2._vendor.dev2model import data, decision_model, infer, score_bias

        assert os.path.realpath(decision_model.__file__).startswith(os.path.realpath(root)), decision_model.__file__
        self.data, self.dm, self.infer = data, decision_model, infer
        self.cap = self.manifest["max_input_tokens"]
        self.offsets = None
        entry = self.manifest.get("score_bias")
        if entry:
            self.offsets, _ = score_bias.load_score_bias(Path(root) / entry["file"],
                                                         self.manifest["identity"]["model_sha256"])
            assert {str(k): v for k, v in self.offsets.items()} == entry["offsets"]
        self.apply_bias = score_bias.apply
        # The root config.json names the repo's Transformers remote code; upstream answers AutoTokenizer's
        # remote-code prompt "no" at once and loads the tokenizer from tokenizer_config.json.
        from decision2.api import _without_remote_code_prompt

        if load_model:
            self.model, self.tok = _without_remote_code_prompt(decision_model.DecisionModel.from_checkpoint)(Path(root))
            self.model = self.model.float().to(device).eval()
        else:
            from transformers import AutoTokenizer

            self.model = None
            self.tok = _without_remote_code_prompt(AutoTokenizer.from_pretrained)(root, local_files_only=True)
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id

    def rows(self, state, questions):
        """`system_one`'s request checks and `question_to_row` per question (ValueError when one fails)."""
        if not isinstance(questions, dict) or not questions or any(not isinstance(k, str) or not k for k in questions):
            raise ValueError("questions must be a nonempty mapping of question IDs")
        if not self.infer._api_json_payload(state):
            raise ValueError("state must be text, an object, or an array")
        self.data.canonical(state)
        item = {"id": "request", "state": state}
        return [self.infer.question_to_row(item, qid, q) for qid, q in questions.items()]

    def encode(self, state, questions):
        """-> (rows, items): the author's rows and encoded items (ids, candidate and query positions)."""
        rows = self.rows(state, questions)
        return rows, [self.dm.encode(r, self.tok, self.cap) for r in rows]

    @torch.no_grad()
    def forward(self, items):
        """Raw candidate logits per item (fp32, no autocast), the request's questions in one batch."""
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.MATH):
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                     for k, v in self.dm.collate(items, self.pad).items()}
            logits = self.model(**batch)
        return [values[:len(item["keys"])].float().cpu() for item, values in zip(items, logits)]

    def biased(self, row, values):
        """The logits `product_answer` reads: Kai's score offsets added, as in `system_one`."""
        values = [float(v) for v in values]
        if self.offsets is not None and row["task_type"] == "score":
            values = self.apply_bias(self.offsets, values, len(row["options"]))
        return values

    def answers(self, rows, items, logits):
        return {r["id"].split("/", 1)[1]: self.infer.product_answer(r["task_type"], it["keys"], self.biased(r, z), 1.0,
                                                                   [o["description"] for o in r["options"]])
                for r, it, z in zip(rows, items, logits)}


def load(root: str, slug: str, device: str = "cpu", load_model: bool = True) -> Reference:
    return Reference(root, slug, device, load_model)
