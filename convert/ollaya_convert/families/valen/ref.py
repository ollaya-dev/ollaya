"""The PyTorch reference for Valen-Decision-1.5: the authors' own code from the model repository at the pinned
revision (`modeling_valen.py`, `compiler.py`, `block_attention.py`, `heads.py`), loaded with `trust_remote_code` and
run in fp32.

    model, processor = ref.load(snapshot, device="cuda")
    compiled, logits = ref.forward(model, processor, state, questions, image=png_bytes_or_None)

`forward` is the body of upstream `ValenQwenForDecisionMaking.predict` (compile the request, one backbone forward
for every question, the Mixer head) without the answer rendering: it returns upstream's compiled state and one
[k] logit vector per question, in upstream's candidate order (noul [true, false], a choice's keys in the caller's
order, score levels). Probabilities are `softmax(logits)`: the checkpoints ship no fitted temperature.

Three choices are Ollaya's, because its API carries a state and at most one image where upstream takes a string or
a message list:

* **A state that is not a string** is written as `json.dumps(state, ensure_ascii=False, sort_keys=True)`, the way
  the authors' own benchmark harness feeds structured states to the model (`evaluation/jevbench/dataset.py` in
  github.com/Valen-Team/Valen). Upstream `predict` itself takes only a string or `{"messages": [...]}`.
* **An image** is one user message, the image first and the state after it:
  `[{"type": "image_url", ...}, {"type": "text", "text": state}]`.
* **Image preprocessing** uses the PIL backend of Qwen's image processor (`Qwen2VLImageProcessorPil`), which is
  platform-independent and portable to Rust value for value (crates/ollaya-runner/src/vision.rs). Upstream's
  default, the torchvision backend, resizes with PyTorch's uint8 kernels, which differ from PIL by 1 in a few
  pixels. The size limits are upstream's (`valen_media_kwargs` in config.json).
"""
from __future__ import annotations

import json
import os
import tempfile

import torch

REPO = "Valen-Team/Valen-Decision-1.5-%s"
# tag -> (repository suffix, pinned commit, weight files)
MODELS = {
    "0.8b": ("0.8B", "e079395fe9b8e0763b384af403e636e91c1a1559", ["model.safetensors"]),
    "2b": ("2B", "1b890606578ba0f40bb70ed46891ab65f778e76d", ["model.safetensors"]),
    "4b": ("4B", "9b0cacc976d7fb72f252ee1adb1564f5f15e462c",
           ["model-%05d-of-00002.safetensors" % i for i in (1, 2)]),
}
CODE = "modeling_valen.py, compiler.py, schema.py, block_attention.py and heads.py in the model repository"


class RequestError(ValueError):
    """The request is invalid for this model (HTTP 422)."""


def repo(tag):
    return REPO % MODELS[tag][0]


def snapshot(tag):
    """The model repository at the pinned revision (VALEN_MODEL: a local copy of it)."""
    env = os.environ.get("VALEN_MODEL")
    if env:
        return env
    from huggingface_hub import snapshot_download

    return snapshot_download(repo(tag), revision=MODELS[tag][1])


def load(root, device="cuda"):
    """Upstream's model in fp32 (TF32 off) and its processor with the PIL image backend."""
    from transformers import AutoModel, AutoProcessor
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModel.from_pretrained(root, trust_remote_code=True, dtype=torch.float32).to(device).eval()
    if model.config.valen_decision_version != "1.5" or model.config.valen_full_attention_mask != "bbox_blocks":
        raise SystemExit("%s is not a Valen-Decision-1.5 checkpoint" % root)
    processor = AutoProcessor.from_pretrained(root, trust_remote_code=True)
    processor.image_processor = Qwen2VLImageProcessorPil.from_pretrained(root)
    return model, processor


def state_text(state):
    """The state as the model reads it: a string as is, anything else as sorted-key JSON (see the module doc)."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True, allow_nan=False)


def upstream_state(state, image_path=None):
    """Upstream's `state` for an Ollaya request."""
    text = state_text(state)
    if image_path is None:
        return text
    return {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_path}},
                                                      {"type": "text", "text": text}]}]}


def compile_request(model, processor, state, questions, image=None):
    """Upstream `Compiler.compile` on the request. Anything upstream raises on is a rejection."""
    compiler = model.make_compiler(processor=processor)
    with tempfile.TemporaryDirectory(prefix="valen-ref-") as tmp:
        path = None
        if image is not None:
            path = os.path.join(tmp, "image.png")
            with open(path, "wb") as f:
                f.write(image)
        try:
            if not isinstance(questions, dict) or not questions:
                raise ValueError("questions must be a non-empty object")
            compiled = compiler.compile({"request": {"state": upstream_state(state, path), "questions": questions}})
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            raise RequestError("%s: %s" % (type(e).__name__, e)) from e
    return compiled


@torch.inference_mode()
def forward(model, processor, state, questions, image=None):
    """-> (upstream's CompiledState, one float64 [k] logit array per question, upstream's candidate order)."""
    compiled = compile_request(model, processor, state, questions, image)
    logits = model.forward_state(compiled)
    return compiled, [z.double().cpu().numpy() for z in logits]


def rope_positions(model, compiled):
    """Upstream's interleaved M-RoPE position ids for the compiled row: [3][tokens]."""
    inp = compiled.inputs
    device = next(model.parameters()).device
    grid = inp.get("image_grid_thw")
    pos, _ = model.backbone.get_rope_index(
        inp["input_ids"].to(device), inp["mm_token_type_ids"].to(device) if "mm_token_type_ids" in inp else None,
        image_grid_thw=grid.to(device) if grid is not None else None,
        attention_mask=inp["attention_mask"].to(device))
    return pos[:, 0].tolist()


def demo_png():
    """A small scene for the export check (a red square and a blue disc on white)."""
    import io

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (256, 240), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 140, 140], fill="red")
    d.ellipse([160, 120, 230, 200], fill="blue")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()
