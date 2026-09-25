"""The PyTorch reference for llm-semantic-router's Decision 1.0 models: the author's own code, run in fp32.

Each model repo ships its inference code: the model and prompt (`decision_model.py`, identical in every
repo, sha256 d3e28489...) and the typed request/answer adapter (Eos: `decision/types.py`; Nox and Lux:
`code/decision_api.py`). This module loads those files from the pinned snapshot, so the reference cannot
drift from what the author serves, and replicates the adapters' `predict_rows` loop in fp32:

  * the author's `question_row` builds each question's candidates, the author's `encode` (Nox and Lux:
    `encode_request`, their segment-cached tokenization) builds the token rows, and `collate` pads them
    in physical batches of 8 (the production batch size);
  * `DecisionModel.from_checkpoint(dtype=float32)` runs the forward without autocast (upstream serves
    a BF16 body under autocast; Ollaya computes in fp32 from the same BF16 weights), TF32 off, and
    SDPA on its exact math kernel;
  * probabilities are the adapter's `softmax(logits / T)` and answers its `typed_answer`.

Lux in fp32 (32 GB) does not fit a 24 GB GPU: with `DECISION_REF_GPU_LAYERS=N` the first N decoder
layers, the final norm and the head run on the GPU and the rest on the CPU, still in fp32 (`split`;
on Eos it reproduces the plain reference's logits to 8.6e-6).

    d = ref.load(ref.snapshot("decision-eos"), "decision-eos", device="cuda")
    rows, items = d.encode(state, questions)       # author's question_row + encode
    logits = d.forward(items)                      # list of [k] fp32 candidate logits
    answers = d.answers(rows, logits)              # author's typed_answer of softmax(logits / T)
"""
from __future__ import annotations

import importlib.util
import json
import os

import torch

MODELS = {
    "decision-eos": {"repo": "llm-semantic-router/Decision-1.0-Eos-0.8B",
                     "revision": "3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd",
                     "base": "Qwen/Qwen3.5-0.8B", "model_code": "decision/model.py", "api": "decision/types.py",
                     "temperature": ("runtime.json", None), "segment_cache": False,
                     "backbone": ["backbone/model.safetensors"]},
    "decision-nox": {"repo": "llm-semantic-router/Decision-1.0-Nox-4B",
                     "revision": "0bb833504965c0eabdb9630b7bbd385cb2fe5cd4",
                     "base": "Qwen/Qwen3.5-4B", "model_code": "code/decision_model.py", "api": "code/decision_api.py",
                     "temperature": ("temperature.json", "temperatures"), "segment_cache": True,
                     "backbone": ["backbone/model-0000%d-of-00003.safetensors" % i for i in (1, 2, 3)]},
    "decision-lux": {"repo": "llm-semantic-router/Decision-1.0-Lux-9B",
                     "revision": "bd45a30aee8c84032791c245c70f86dee5389cc8",
                     "base": "Qwen/Qwen3.5-9B", "model_code": "code/decision_model.py", "api": "code/decision_api.py",
                     "temperature": ("temperature.json", "temperatures"), "segment_cache": True,
                     "backbone": ["backbone/model-0000%d-of-00004.safetensors" % i for i in (1, 2, 3, 4)]},
}
BATCH = 8           # the production physical batch size (bundle-manifest.json, engine.py)
MAX_LENGTH = 16384  # the complete-input limit (runtime.json / bundle-manifest.json)


def snapshot(slug: str) -> str:
    env = os.environ.get("DECISION_ROOT_" + slug.upper().replace("-", "_"))
    if env:
        return env
    from huggingface_hub import snapshot_download

    m = MODELS[slug]
    return snapshot_download(m["repo"], revision=m["revision"],
                             ignore_patterns=["assets/*", "architecture-source/*", "*.pdf", "*.png"])


def _module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def temperatures(root: str, slug: str) -> dict:
    """The released temperature per question type."""
    fname, key = MODELS[slug]["temperature"]
    cfg = json.load(open(os.path.join(root, fname)))
    if key is None:
        t = float(cfg["temperature"])
        return {"choice": t, "noul": t, "score": t}
    return {k: float(v) for k, v in cfg[key].items()}


def choice_null_description(root: str, slug: str) -> str:
    """How the adapter renders a null choice description: "key" (Nox 1.3.2) or "null"."""
    api = open(os.path.join(root, MODELS[slug]["api"])).read()
    return "key" if "'description':key if value is None else value" in api else "null"


def _to(x, dev):
    if torch.is_tensor(x):
        return x.to(dev)
    if isinstance(x, (tuple, list)):
        return type(x)(_to(v, dev) for v in x)
    if isinstance(x, dict):
        return {k: _to(v, dev) for k, v in x.items()}
    return x


def _follow_inputs(mod):
    """Move a module's inputs to the device of its parameters before it runs."""
    dev = next(mod.parameters()).device
    mod.register_forward_pre_hook(lambda m, args, kwargs: (_to(args, dev), _to(kwargs, dev)), with_kwargs=True)


def split(model, device: str, gpu_layers: int):
    """Spread an fp32 model that does not fit `device` over it and the CPU: the embedding and the
    decoder layers from `gpu_layers` on stay on the CPU, everything else goes to `device`; each module
    moves its inputs to its own device. The model is loaded in BF16 (the author's default) and every
    module is upcast to fp32 where it lands, which is exact and keeps host memory near the BF16 size.
    Numerically this is `from_checkpoint(dtype=float32)`: the same fp32 weights and the same forward."""
    bb = model.backbone
    for i, layer in enumerate(bb.layers):
        layer.to(device if i < gpu_layers else "cpu", torch.float32)
        _follow_inputs(layer)
    bb.embed_tokens.to("cpu", torch.float32)
    # Rotary frequencies as an fp32 load computes them (a BF16 load may have cast the buffer).
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    bb.rotary_emb = Qwen3_5TextRotaryEmbedding(config=bb.config)
    # The backbone builds its masks and positions next to the embedding, on the CPU.
    bb.register_forward_pre_hook(lambda m, args, kwargs: (_to(args, "cpu"), _to(kwargs, "cpu")), with_kwargs=True)
    bb.norm.to(device, torch.float32)
    _follow_inputs(bb.norm)
    model.head.to(device, torch.float32)
    _follow_inputs(model.head)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    return model


class Reference:
    def __init__(self, root: str, slug: str, device: str = "cpu", load_model: bool = True):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.root, self.slug, self.device = root, slug, device
        spec = MODELS[slug]
        self.module = _module(os.path.join(root, spec["model_code"]), "decision_model_" + slug.replace("-", "_"))
        self.api = _module(os.path.join(root, spec["api"]), "decision_api_" + slug.replace("-", "_"))
        self.T = temperatures(root, slug)
        self.segment_cache = spec["segment_cache"]
        gpu_layers = os.environ.get("DECISION_REF_GPU_LAYERS")  # fp32 models larger than the GPU
        if load_model and gpu_layers is not None and device != "cpu":
            self.model, self.tok = self.module.DecisionModel.from_checkpoint(root, dtype=torch.bfloat16)
            self.model = split(self.model, device, int(gpu_layers)).eval()
        elif load_model:
            self.model, self.tok = self.module.DecisionModel.from_checkpoint(root, dtype=torch.float32)
            self.model = self.model.to(device).eval()
        else:
            from transformers import AutoTokenizer

            self.model, self.tok = None, AutoTokenizer.from_pretrained(root, local_files_only=True)
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id

    def rows(self, state, questions):
        """The adapter's request checks and question_row per question (ValueError when it rejects)."""
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a nonempty mapping")
        if not all(isinstance(q, dict) for q in questions.values()):
            raise ValueError("Each question must be an object")
        return [self.api.question_row(state, name, q) for name, q in questions.items()]

    def encode(self, state, questions):
        """-> (rows, items): the author's rows and encoded items (ids, candidate and query positions)."""
        rows = self.rows(state, questions)
        if self.segment_cache:
            items = self.api.encode_request(rows, self.tok, self.module, MAX_LENGTH)
        else:
            items = [self.module.encode(r, self.tok, MAX_LENGTH) for r in rows]
        return rows, items

    @torch.no_grad()
    def forward(self, items):
        """Raw candidate logits per item (fp32, no autocast), in physical batches of 8 as upstream."""
        from torch.nn.attention import SDPBackend, sdpa_kernel

        out = []
        with sdpa_kernel(SDPBackend.MATH):
            for start in range(0, len(items), BATCH):
                chunk = items[start:start + BATCH]
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                         for k, v in self.module.collate(chunk, self.pad).items()}
                logits = self.model(**batch)
                for item, values in zip(chunk, logits):
                    out.append(values[:item["nopts"]].float().cpu())
        return out

    def probabilities(self, row, values):
        """The adapter's float32 softmax(values / T)."""
        return (torch.as_tensor(values, dtype=torch.float32) / self.T[row["task_type"]]).softmax(-1).tolist()

    def answers(self, rows, logits):
        return {r["id"]: self.api.typed_answer(r, self.probabilities(r, z)) for r, z in zip(rows, logits)}


def load(root: str, slug: str, device: str = "cpu", load_model: bool = True) -> Reference:
    return Reference(root, slug, device, load_model)
