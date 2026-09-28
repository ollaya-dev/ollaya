"""`decider-vision-v1`: Mapika/decider-2b-vision, the reference (upstream `decider/vision.py`, fp32).

    uv run --with pillow --with torchvision python -m ollaya_convert.families.decider_vision.ref ROOT

decider-2b-vision is the Qwen3.5-2B vision-language model with decider-2b v5 text weights. It reads an
image plus lettered options and answers at an `Answer: (` slot, like the text decider. What the model
sees is upstream code: `VisionDecisionModel.prepare` (its v5 `prompt.build`, the image placeholder,
the Qwen processor) and `slot_logits`. Two choices are Ollaya's:

* **Typed questions.** The repo has no TypeSafe mapping for this model, so each question is rendered
  the way Ollaya's `decider` layout renders it (`decider.layout.render_question`: choice options as
  `label` or `label: description`, noul as `no`/`yes`, score levels as `i: level`), one row per
  question with the image in every row, at most 10 options (the model's letters A-J).
* **Image preprocessing.** The PIL backend of Qwen's image processor (`Qwen2VLImageProcessorPil`):
  platform-independent and portable to Rust value for value. Upstream's default, the torchvision
  backend, resizes with PyTorch's uint8 kernels, which differ from PIL by 1 in a few pixels and
  between CPU architectures.
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import torch

REPO = "Mapika/decider-2b-vision"
REVISION = "863e290863655f1d6b69324d77d09ac972d21609"
MAX_OPTIONS = 10
MAX_CTX_TOKENS = 1536


def snapshot():
    env = os.environ.get("DECIDER_VISION_ROOT")
    if env:
        return env
    from huggingface_hub import snapshot_download
    return snapshot_download(REPO, revision=REVISION)


def load(root, device="cuda"):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if root not in sys.path:
        sys.path.insert(0, root)
    from decider.vision import VisionDecisionModel
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

    m = VisionDecisionModel(root, dtype=torch.float32, grad_ckpt=False)
    m.proc.image_processor = Qwen2VLImageProcessorPil.from_pretrained(root)
    return m.to(device).eval()


def image_from_bytes(data):
    """The image as upstream reads it: PIL, converted to RGB (alpha dropped, not composited)."""
    from PIL import Image
    return Image.open(io.BytesIO(data)).convert("RGB")


def examples(state, questions):
    """One upstream Example per question, in request order, and the per-question plan."""
    from decider.infer import Example, Q

    from ..decider.layout import render_question, render_state
    ctx = render_state(state)
    exs, plan = [], []
    for qid, spec in questions.items():
        rq = render_question(spec, MAX_OPTIONS, MAX_OPTIONS)
        exs.append(Example(ctx, [Q(rq["question"], rq["options"], 0)]))
        plan.append({"qid": qid, "type": rq["type"], "k": len(rq["options"])})
    return exs, plan


@torch.no_grad()
def forward(m, image_bytes, state, questions):
    """-> (inputs, plan, option logits per question [k]). image_bytes None: text only."""
    exs, plan = examples(state, questions)
    img = image_from_bytes(image_bytes) if image_bytes is not None else None
    inp = m.prepare([(img, ex) for ex in exs], max_ctx_tokens=MAX_CTX_TOKENS)
    lg = m.slot_logits(inp).double().cpu().numpy()
    return inp, plan, [lg[i, :p["k"]].tolist() for i, p in enumerate(plan)]


def demo_png():
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (256, 240), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 140, 140], fill="red")
    d.ellipse([160, 120, 230, 200], fill="blue")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


if __name__ == "__main__":
    m = load(sys.argv[1] if len(sys.argv) > 1 else snapshot())
    qs = {"square": {"type": "choice", "instructions": "What color is the square?",
                     "criteria": {"red": None, "green": None, "blue": None, "yellow": None}},
          "circle": {"type": "noul", "instructions": "Is there a circle in the image?"}}
    inp, plan, lg = forward(m, demo_png(), "This is a visual question about the image.", qs)
    print({k: tuple(v.shape) for k, v in inp.items() if hasattr(v, "shape")})
    for p, z in zip(plan, lg):
        e = np.exp(np.array(z) - max(z))
        print(p["qid"], np.round(e / e.sum(), 4).tolist())
