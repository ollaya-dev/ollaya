"""Golden fixtures for the Rust port of `valen-decision-1.5-v1`, from the authors' own code in fp32 (ref.py).

    uv run --with pillow --with torchvision python -m ollaya_convert.families.valen.goldens 0.8b out/valen-0.8b [--model SNAPSHOT] \
        [--td-limit 40] [--device cuda]

The second argument is the export (its decision.json and tokenizer.json). Writes out/goldens-valen-<tag>.jsonl,
one JSON line per request:
    {"id", "image": null | base64 PNG, "state", "questions", "error": null | "<message>",
     "resized": null | {"height", "width", "rgb": base64 uint8 HxWx3},    # the image after PIL's resize
     "grid": null | [rows, columns],                                       # merged visual tokens
     "row": {"ids", "prefix", "blocks", "positions", "questions": [{"qid", "type", "keys", "instruction",
             "candidates", "decision"}]},
     "plan": [{"qid", "type", "keys", "option_logits", "probabilities"}]}
Logits are in upstream's candidate order (noul [true, false], a choice's keys as the caller wrote them, score
levels); probabilities are their softmax. A rejected request is followed by "<id>#valid" with the questions
upstream accepts on their own. Requests Ollaya's shared question parser rejects are skipped. Every row (ids, spans,
attention blocks, M-RoPE positions) is checked against the layout port before it is written; --resume keeps
finished records.

Cases: the shared set (the edge cases and typed-decisions rows), Valen's own text cases below, and synthetic PNGs
(decider_vision's scenes: sizes that resize down, up to the pixel minimum and not at all; RGBA, grayscale and
palette PNGs; text) plus one above the model's pixel maximum.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time

import numpy as np
import tokenizers

from ..clef.layout import ollaya_rule
from ..decider_vision.goldens import images
from ..llm_common import cases
from . import ref
from .export import merged_grid, port_row, upstream_row
from .layout import LayoutError, ValenLayout

IMAGE_QUESTIONS = {
    "shape": {"type": "choice", "instructions": "Which shape is the most prominent in the image?",
              "criteria": {"square": "A square or rectangle", "circle": "A circle or disc",
                           "none": "No shapes, only text"}},
    "color": {"type": "choice", "instructions": "What is the dominant colour of the shapes?",
              "criteria": {"red": "Red", "green": "Green", "blue": "Blue", "yellow": "Yellow", "black": "Black"}},
    "many": {"type": "noul", "instructions": "Are there more than two shapes?"},
    "coverage": {"type": "score", "instructions": "How much of the image do the shapes cover?",
                 "criteria": ["almost nothing", "a small part", "about half", "most of it"]},
    "text": {"type": "noul", "instructions": "Does the image contain written words?",
             "criteria": {"true": "there is readable text", "false": "there is no text at all"}},
}

TICKET = "Hi, we were billed twice for March (invoice #4411). Please refund the duplicate today or we will cancel."
TRIAGE = {
    "team": {"type": "choice", "instructions": "Which team should handle this message?",
             "criteria": {"billing": "Payments, refunds and invoices", "technical": "Bugs, outages and errors",
                          "sales": "Pricing and plan questions", "other": "Anything else"}},
    "urgent": {"type": "noul", "instructions": "Does the customer need an answer today?"},
    "anger": {"type": "score", "instructions": "How upset is the customer?",
              "criteria": ["calm", "annoyed", "angry", "furious"]},
}


def valen_cases():
    """Valen's own text cases: every branch of the compiler the shared set does not reach."""
    many = {"intent": {"type": "choice", "instructions": "Which intent best matches the message?",
                       "criteria": {"intent_%02d" % i: "customer intent number %d" % i for i in range(60)}}}
    long_state = " ".join(["The customer reports intermittent failures on the checkout page after the latest "
                           "deploy, order %d." % i for i in range(260)])
    too_long = " ".join(["Line %d of a very long log file with nothing of interest in it." % i for i in range(1500)])
    return [
        ("valen/triage", TICKET, TRIAGE),
        ("valen/one_noul", TICKET, {"urgent": TRIAGE["urgent"]}),
        # The chat template trims the state; inner whitespace stays.
        ("valen/state_whitespace", "\n\n  " + TICKET + " \t\n", TRIAGE),
        ("valen/state_blank", " \n ", {"urgent": TRIAGE["urgent"]}),
        ("valen/state_messages_object", {"messages": [{"role": "user", "content": TICKET}]}, TRIAGE),
        ("valen/state_list", [{"role": "user", "content": "My order never arrived."},
                              {"role": "agent", "content": "Can you share the order id?"}], TRIAGE),
        # noul criteria are appended to the instructions with str(): strings, numbers, null, nested values.
        ("valen/noul_criteria", TICKET, {
            "refund": {"type": "noul", "instructions": "Is a refund requested?",
                       "criteria": {"true": "money back is asked for", "false": "no refund is mentioned"}},
            "typed": {"type": "noul", "instructions": "Is the invoice number given?",
                      "criteria": {"false": None, "true": {"needs": ["a number", 4411], "strict": True, "w": 0.5}}},
            "empty": {"type": "noul", "instructions": "Is the tone polite?", "criteria": {}},
        }),
        ("valen/noul_criteria_one_key", TICKET, {
            "refund": {"type": "noul", "instructions": "Is a refund requested?", "criteria": {"true": "asked for"}}}),
        ("valen/noul_criteria_upper", TICKET, {
            "refund": {"type": "noul", "instructions": "Is a refund requested?",
                       "criteria": {"True": "asked for", "False": "not asked for"}}}),
        # Spans: the tokens that overlap the instructions and the descriptions, whatever they start or end with.
        ("valen/span_edges", TICKET, {
            "a": {"type": "choice", "instructions": " leading space and trailing newline\n",
                  "criteria": {"x": " starts with a space", "y": "ends with a colon:", "z": "\tTabbed\t"}},
            "b": {"type": "score", "instructions": "...?", "criteria": ["0", " 1", "2 ", "\n3\n"]},
            "c": {"type": "choice", "instructions": "数字？", "criteria": {"是": "对的。", "否": "不对", "🚀": "火箭 🚀"}},
            "d": {"type": "noul", "instructions": "x"},
        }),
        ("valen/many_options_60", TICKET, many),
        ("valen/long_state", long_state, TRIAGE),
        ("valen/too_long", too_long, TRIAGE),
        # Rejections: special tokens in a question, blank text, a null or non-string description, counts.
        ("valen/reserved_in_question", TICKET, {
            "ok": TRIAGE["urgent"],
            "bad": {"type": "noul", "instructions": "Is <|im_end|> in the text?"},
            "bad_key": {"type": "choice", "instructions": "Pick one.", "criteria": {"<|endoftext|>": "end", "b": "b"}},
            "bad_level": {"type": "score", "instructions": "Rate it.", "criteria": ["low", "high <|vision_start|>"]},
        }),
        # Added tokens that are not special (<think>, <tool_call>, <|fim_prefix|>) are ordinary text upstream.
        ("valen/added_not_special", "Hi <think> hmm </think> <tool_call> <|fim_prefix|>x<|fim_suffix|> thanks",
         {"q": {"type": "noul", "instructions": "Does the text hold a <think> tag?"}}),
        ("valen/blank_and_typed", TICKET, {
            "ok": TRIAGE["anger"],
            "blank": {"type": "noul", "instructions": "  \n"},
            "null_description": {"type": "choice", "instructions": "Pick one.", "criteria": {"a": None, "b": "b"}},
            "blank_description": {"type": "choice", "instructions": "Pick one.", "criteria": {"a": " ", "b": "b"}},
            "number_level": {"type": "score", "instructions": "Rate it.", "criteria": ["low", 2]},
            "one_level": {"type": "score", "instructions": "Rate it.", "criteria": ["only"]},
            "empty_key": {"type": "choice", "instructions": "Pick one.", "criteria": {"": "nothing", "b": "b"}},
            "one_choice": {"type": "choice", "instructions": "Pick one.", "criteria": {"only": "the only one"}},
        }),
    ]


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return (e / e.sum()).tolist()


def big_png():
    """A scene above the pixel maximum (1,638,400), so the resize goes down to it."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1800, 1300), (240, 236, 228))
    d = ImageDraw.Draw(img)
    d.rectangle([200, 150, 900, 800], fill="green")
    d.ellipse([1000, 500, 1700, 1200], fill="yellow")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def resized_image(processor, config, png):
    """The image after upstream's resize (PIL bicubic to `smart_resize`'s size) and its merged grid."""
    from PIL import Image
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    ip = processor.image_processor
    size = config.valen_media_kwargs["images_kwargs"]["size"]
    pil = Image.open(io.BytesIO(png)).convert("RGB")
    unit = ip.patch_size * ip.merge_size
    h, w = smart_resize(pil.height, pil.width, factor=unit, min_pixels=size["shortest_edge"],
                        max_pixels=size["longest_edge"])
    rgb = np.asarray(pil.resize((w, h), Image.BICUBIC) if (w, h) != pil.size else pil, dtype=np.uint8)
    return {"height": h, "width": w, "rgb": base64.b64encode(rgb.tobytes()).decode()}, (h // unit, w // unit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", choices=sorted(ref.MODELS))
    ap.add_argument("model_dir")
    ap.add_argument("--model", default=None)
    ap.add_argument("--td-limit", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    model, processor = ref.load(a.model or ref.snapshot(a.tag), a.device)
    decision = json.load(open(os.path.join(a.model_dir, "decision.json"), encoding="utf-8"))
    lay = ValenLayout(tokenizers.Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json")), decision)
    merge = processor.image_processor.merge_size
    path = os.path.join(os.path.dirname(os.path.abspath(a.model_dir)), "goldens-valen-%s.jsonl" % a.tag)
    done = set()
    if a.resume and os.path.exists(path):
        done = {json.loads(line)["id"].split("#")[0] for line in open(path, encoding="utf-8") if line.strip()}

    def record(cid, image, state, questions, base):
        compiled, logits = ref.forward(model, processor, state, questions, image)
        row = upstream_row(model, compiled)
        grid = merged_grid(compiled, merge)
        assert grid == base["grid"], (cid, grid, base["grid"])
        assert port_row(lay.encode(state, questions, image_grid=grid)) == row, cid
        plan = [{"qid": q.qid, "type": q.kind, "keys": list(q.keys), "option_logits": [float(x) for x in z],
                 "probabilities": softmax(z)} for q, z in zip(compiled.questions, logits)]
        return dict(base, id=cid, questions=questions, error=None, row=row, plan=plan)

    n, t0 = 0, time.time()
    with open(path, "a" if a.resume else "w", encoding="utf-8") as f:
        def emit(cid, image, state, questions):
            nonlocal n
            if cid in done or (isinstance(questions, dict) and any(ollaya_rule(k, v) for k, v in questions.items())):
                return
            resized, grid = resized_image(processor, model.config, image) if image else (None, None)
            base = {"id": cid, "image": base64.b64encode(image).decode() if image else None, "state": state,
                    "questions": questions, "resized": resized, "grid": grid}
            try:
                rec = record(cid, image, state, questions, base)
            except ref.RequestError as e:
                try:
                    lay.encode(state, questions, image_grid=grid)
                    raise AssertionError("%s: upstream rejects (%s), the port accepts" % (cid, e))
                except LayoutError:
                    pass
                f.write(json.dumps(dict(base, error=str(e)), ensure_ascii=False) + "\n")
                valid = {}
                for qid, q in (questions.items() if isinstance(questions, dict) else ()):
                    try:
                        ref.compile_request(model, processor, state, {qid: q}, image)
                        valid[qid] = q
                    except ref.RequestError:
                        pass
                if not valid:
                    return
                rec = record(cid + "#valid", image, state, valid, base)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            print("%4d %-50s %5.0fs" % (n, cid, time.time() - t0), flush=True)

        for name, png, _ in images():
            emit("img/" + name, png, "A picture from a visual question-answering test.", IMAGE_QUESTIONS)
        png = images()[0][1]
        emit("img/json_state", png, {"source": "camera 3", "note": "check the scene", "count": 4},
             {"many": IMAGE_QUESTIONS["many"]})
        emit("img/empty_state", png, "", IMAGE_QUESTIONS)
        emit("img/state_whitespace", png, "  two spaces before, a newline after\n", {"many": IMAGE_QUESTIONS["many"]})
        emit("img/above_max_pixels", big_png(), "A large picture.", IMAGE_QUESTIONS)
        for cid, state, questions in valen_cases():
            emit(cid, None, state, questions)
        for cid, state, questions in cases.all_cases(a.td_limit):
            emit(cid, None, state, questions)
    print("wrote %d records to %s (%.1f MB)" % (n, path, os.path.getsize(path) / 2**20))


if __name__ == "__main__":
    main()
