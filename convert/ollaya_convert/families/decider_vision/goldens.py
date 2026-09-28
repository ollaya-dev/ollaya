"""Golden fixtures for the Rust port of `decider-vision-v1` (ref.py: upstream decider/vision.py in fp32).

    uv run --with pillow --with torchvision python -m ollaya_convert.families.decider_vision.goldens \
        out/decider-2b-vision [--root ROOT]

Writes out/goldens-decider-2b-vision.jsonl, one JSON line per request:
    {"id", "image": null | base64 PNG, "state", "questions",
     "error": null | "<exception class>",
     "resized": null | {"height", "width", "rgb": base64 uint8 HxWx3},    # the image after PIL's resize
     "grid": null | [t, h, w],
     "rows": [{"ids": [...], "slot": int, "k": int}],                     # one row per question
     "positions": [[[t...], [h...], [w...]] per row],                      # upstream get_rope_index
     "plan": [{"qid", "type", "k", "option_logits", "probabilities"}]}
The images are synthetic PNGs drawn here (shapes, colours, counts, text), in sizes that exercise
the resize (down, up to the minimum pixel count, already a multiple of 32) and the PNG colour modes
upstream converts to RGB (RGBA, grayscale, palette).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os

import numpy as np
import torch

from ..llm_common import cases
from . import ref


def _png(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def images():
    from PIL import Image, ImageDraw
    out = []

    def scene(w, h, seed):
        rng = np.random.default_rng(seed)
        img = Image.new("RGB", (w, h), tuple(int(x) for x in rng.integers(180, 256, 3)))
        d = ImageDraw.Draw(img)
        shapes = []
        for i in range(int(rng.integers(1, 5))):
            x0, y0 = int(rng.integers(0, w * 3 // 4)), int(rng.integers(0, h * 3 // 4))
            s = int(rng.integers(max(8, min(w, h) // 8), max(9, min(w, h) // 3)))
            color = ["red", "green", "blue", "yellow", "black"][int(rng.integers(0, 5))]
            kind = ["square", "circle"][int(rng.integers(0, 2))]
            (d.rectangle if kind == "square" else d.ellipse)([x0, y0, x0 + s, y0 + s], fill=color)
            shapes.append((kind, color))
        return img, shapes

    for i, (w, h) in enumerate([(256, 240), (320, 200), (640, 480), (173, 100), (1000, 750), (40, 30), (512, 512)]):
        img, shapes = scene(w, h, i)
        out.append(("scene_%dx%d" % (w, h), _png(img), shapes))
    img, shapes = scene(300, 220, 11)
    out.append(("rgba_300x220", _png(img.convert("RGBA")), shapes))
    img, shapes = scene(300, 220, 12)
    out.append(("gray_300x220", _png(img.convert("L")), shapes))
    img, shapes = scene(300, 220, 13)
    out.append(("palette_300x220", _png(img.convert("P", palette=Image.ADAPTIVE, colors=16)), shapes))
    text = Image.new("RGB", (400, 120), "white")
    ImageDraw.Draw(text).text((20, 40), "REFUND APPROVED", fill="black")
    out.append(("text_400x120", _png(text), []))
    return out


QUESTIONS = {
    "shape": {"type": "choice", "instructions": "Which shape is the most prominent in the image?",
              "criteria": {"square": None, "circle": None, "none": "no shapes, only text"}},
    "color": {"type": "choice", "instructions": "What is the dominant colour of the shapes?",
              "criteria": {"red": None, "green": None, "blue": None, "yellow": None, "black": None}},
    "many": {"type": "noul", "instructions": "Are there more than two shapes?"},
    "coverage": {"type": "score", "instructions": "How much of the image do the shapes cover?",
                 "criteria": ["almost nothing", "a small part", "about half", "most of it"]},
    "text": {"type": "noul", "instructions": "Does the image contain written words?",
             "criteria": {"true": "there is readable text", "false": None}},
}


def record(m, cid, image, state, questions):
    inp, plan, logits = ref.forward(m, image, state, questions)
    lm = m.lm
    pos, _ = lm.model.get_rope_index(inp["input_ids"].to(lm.device), inp["mm_token_type_ids"].to(lm.device),
                                     image_grid_thw=inp["image_grid_thw"].to(lm.device) if image else None,
                                     attention_mask=inp["attention_mask"].to(lm.device))
    rows, positions = [], []
    for i in range(inp["input_ids"].shape[0]):
        n = int(inp["attention_mask"][i].sum())
        rows.append({"ids": inp["input_ids"][i, :n].tolist(), "slot": int(inp["slot_idx"][i]), "k": plan[i]["k"]})
        positions.append(pos[:, i, :n].tolist())
    resized, grid = None, None
    if image is not None:
        ip = m.proc.image_processor
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        pil = ref.image_from_bytes(image)
        h, w = smart_resize(pil.height, pil.width, factor=ip.patch_size * ip.merge_size,
                            min_pixels=ip.size["shortest_edge"], max_pixels=ip.size["longest_edge"])
        from PIL import Image
        rgb = np.asarray(pil.resize((w, h), Image.BICUBIC) if (w, h) != pil.size else pil, dtype=np.uint8)
        resized = {"height": h, "width": w, "rgb": base64.b64encode(rgb.tobytes()).decode()}
        grid = inp["image_grid_thw"][0].tolist()
    plan = [dict(p, option_logits=[float(x) for x in z],
                 probabilities=(lambda e: (e / e.sum()).tolist())(np.exp(np.array(z) - max(z))))
            for p, z in zip(plan, logits)]
    return {"id": cid, "image": base64.b64encode(image).decode() if image else None, "state": state,
            "questions": questions, "error": None, "resized": resized, "grid": grid, "rows": rows,
            "positions": positions, "plan": plan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--root", default=None)
    ap.add_argument("--td-limit", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    m = ref.load(a.root or ref.snapshot(), a.device)
    path = os.path.join(os.path.dirname(os.path.abspath(a.model_dir)), "goldens-decider-2b-vision.jsonl")
    n = 0
    with open(path, "w") as f:
        def emit(cid, image, state, questions):
            nonlocal n
            try:
                rec = record(m, cid, image, state, questions)
            except (ValueError, AssertionError) as e:
                if isinstance(e, AssertionError) and "options required" not in str(e):
                    raise
                rec = {"id": cid, "image": base64.b64encode(image).decode() if image else None, "state": state,
                       "questions": questions, "error": type(e).__name__}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

        for name, png, shapes in images():
            emit("img/" + name, png, "A picture from a visual question-answering test.", QUESTIONS)
        # A JSON state with an image, and the same image with one question.
        png = images()[0][1]
        emit("img/json_state", png, {"source": "camera 3", "note": "check the scene", "count": 4}, {"many": QUESTIONS["many"]})
        # Text-only rows through the vision model (the shared cases, as far as they fit 10 options).
        for cid, state, questions in cases.all_cases(a.td_limit):
            emit("text/" + cid, None, state, questions)
    print("wrote %d records to %s (%.1f MB)" % (n, path, os.path.getsize(path) / 2**20))


if __name__ == "__main__":
    main()
