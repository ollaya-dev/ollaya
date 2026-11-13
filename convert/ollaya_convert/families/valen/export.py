"""Export a Valen-Decision-1.5 checkpoint to two weightless ONNX graphs + tokenizer + decision/calibration config.

    OLLAYA_SCRATCH=<big disk> uv run --with pillow --with torchvision python -m ollaya_convert.families.valen.export 0.8b out/valen-0.8b \
        [--model SNAPSHOT]

Graphs (layout `valen-decision-1.5-v1`; graphs.py, layout.py and docs/families/valen.md):
    vision.onnx   patches [N, 1536], pos_idx [N, 4], pos_w [N, 4], rot_ids [N, 2] -> image_embeds [N/4, H]
    model.onnx    input_ids [1, T], position_ids [3, 1, T], image_embeds [M, H], image_pos [M], blocks [1, T],
                  instruction_spans [Q, 2], decision_pos [Q], candidate_spans [O, 2], candidate_question [O]
                  -> logits [O]
Both reference the author's safetensors (BF16 backbone, F32 head) by byte offset; nothing is copied. The export
checks itself against the reference (ref.py: the authors' code, fp32) on a request with an image and on a
text-only request, first in PyTorch and then on ONNX Runtime.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import torch

from ..llm_common import onnx_export as ox
from ..llm_common.qwen35 import CHUNK
from ...weightless_sharded import safetensors_source
from . import ref
from .graphs import DecoderGraph, VisionGraph
from .layout import LAYOUT, ValenLayout, graph_inputs, prompt_for

VISION_INPUTS = ["patches", "pos_idx", "pos_w", "rot_ids"]
VISION_OUTPUTS = ["image_embeds"]
INPUTS = ["input_ids", "position_ids", "image_embeds", "image_pos", "blocks", "instruction_spans", "decision_pos",
          "candidate_spans", "candidate_question"]
OUTPUTS = ["logits"]

SAMPLE_STATE = "A picture from a product listing, sent with the note: please check it before we publish."
SAMPLE = {
    "square": {"type": "choice", "instructions": "What color is the square?",
               "criteria": {"red": "The square is red", "green": "The square is green", "blue": "The square is blue"}},
    "circle": {"type": "noul", "instructions": "Is there a circle in the image?"},
    "busy": {"type": "score", "instructions": "How cluttered is the picture?",
             "criteria": ["empty", "a few shapes", "many shapes", "crowded"]},
}


def rename(name):
    if name.startswith("v."):
        return ["backbone.visual." + name[2:]]
    if name.startswith("trunk.m."):
        return ["backbone.language_model." + name[len("trunk.m."):]]
    return [name]


def decision_for(model, processor):
    """The layout's fields of decision.json, from upstream's config and tokenizer."""
    cfg, tok = model.config, processor.tokenizer
    return {
        "prompt": prompt_for(),
        "reserved_tokens": list(tok.all_special_tokens),
        "max_tokens": cfg.valen_max_length,
        "min_levels": 2,
        "max_options": 255,
        "tokens": {"image": cfg.image_token_id, "pad": tok.pad_token_id},
        "hidden_size": cfg.text_config.hidden_size,
    }


def upstream_row(model, compiled):
    """Upstream's compiled state in the port's shape (layout.ValenLayout.encode)."""
    import sys

    block_attention = sys.modules[type(model).__module__.rsplit(".", 1)[0] + ".block_attention"]
    first = compiled.questions[0].branches[0]
    kwargs = block_attention.forward_kwargs(model.full_attention_mask, [first], "cpu")
    return {
        "ids": compiled.inputs["input_ids"][0].tolist(),
        "prefix": int(kwargs["valen_prefix_lengths"][0]),
        "blocks": kwargs["valen_segments"][0].tolist(),
        "positions": ref.rope_positions(model, compiled),
        "questions": [{"qid": q.qid, "type": q.kind, "keys": list(q.keys),
                       "instruction": list(q.branches[0].instruction_span),
                       "candidates": [list(s) for s in q.branches[0].candidate_spans],
                       "decision": int(q.branches[0].decision_position)} for q in compiled.questions],
    }


def port_row(row):
    return {k: row[k] for k in ("ids", "prefix", "blocks", "positions", "questions")}


def merged_grid(compiled, merge):
    grid = compiled.inputs.get("image_grid_thw")
    if grid is None:
        return None
    _, h, w = grid[0].tolist()
    return (h // merge, w // merge)


def vision_inputs(vis, compiled):
    """The vision graph's inputs for the request's image, as upstream's tower computes its tables."""
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids

    grid = compiled.inputs["image_grid_thw"][:1]
    pv = compiled.inputs["pixel_values"][: int(grid.prod())]
    idx, wt = get_vision_interpolation_indices_and_weights(grid, vis.num_grid_per_side, mode="bilinear",
                                                            align_corners=True,
                                                            spatial_merge_size=vis.spatial_merge_size)
    rot = get_vision_position_ids(grid, vis.spatial_merge_size)
    return pv, idx, wt, rot


def ort_check(out_dir, vis_args, cases):
    """The two exported graphs on ONNX Runtime (CPU) against the reference logits."""
    import onnxruntime as onnxrt

    so = onnxrt.SessionOptions()
    vs = onnxrt.InferenceSession(os.path.join(out_dir, "vision.onnx"), so, providers=["CPUExecutionProvider"])
    ds = onnxrt.InferenceSession(os.path.join(out_dir, "model.onnx"), so, providers=["CPUExecutionProvider"])
    emb = vs.run(None, {k: v.numpy() for k, v in zip(VISION_INPUTS, vis_args)})[0]
    worst = 0.0
    for name, inputs, want in cases:
        feed = dict(inputs)
        if name == "image":
            feed["image_embeds"] = emb
        got = ds.run(None, feed)[0]
        diff = float(np.abs(got - want).max())
        print("onnxruntime vs upstream (%s): %.2e" % (name, diff))
        worst = max(worst, diff)
    return worst


def export(tag, out_dir, root):
    import tokenizers

    model, processor = ref.load(root, "cpu")
    cfg = model.config
    lm, vis = model.backbone.language_model, model.backbone.visual
    ip = processor.image_processor
    layout = decision_for(model, processor)
    tok_json = os.path.join(root, "tokenizer.json")
    lay = ValenLayout(tokenizers.Tokenizer.from_file(tok_json), layout)
    hidden, pad = layout["hidden_size"], layout["tokens"]["pad"]

    vg = VisionGraph(vis).eval()
    dg = DecoderGraph(lm, model.head, cfg.text_config.rope_parameters["mrope_section"]).eval()
    cases, vis_args = [], None
    for name, image in (("image", ref.demo_png()), ("text", None)):
        compiled, want = ref.forward(model, processor, SAMPLE_STATE, SAMPLE, image=image)
        want = np.concatenate(want)
        row = lay.encode(SAMPLE_STATE, SAMPLE, image_grid=merged_grid(compiled, ip.merge_size))
        assert port_row(row) == upstream_row(model, compiled), "layout port differs from upstream on the sample"
        emb = None
        if image is not None:
            vis_args = vision_inputs(vis, compiled)
            with torch.no_grad():
                emb = vg(*vis_args).numpy()
        inputs = graph_inputs(row, pad, CHUNK, hidden, emb)
        with torch.no_grad():
            got = dg(*(torch.from_numpy(inputs[k]) for k in INPUTS)).numpy()
        diff = float(np.abs(got - want).max())
        print("eager graphs vs upstream (%s): %.2e" % (name, diff))
        assert diff < 1e-3, diff
        cases.append((name, inputs, want))

    tmp = ox.scratch_dir("valen-export-")
    Nt = torch.export.Dim("tokens", min=1, max=16384)
    secs = ox.export_graph(vg, vis_args, VISION_INPUTS, VISION_OUTPUTS,
                           {"patches": {0: 4 * Nt}, "pos_idx": {0: 4 * Nt}, "pos_w": {0: 4 * Nt}, "rot_ids": {0: 4 * Nt}},
                           os.path.join(tmp, "vision", "model.onnx"))
    print("vision exported in %.0fs" % secs)
    Nc = torch.export.Dim("chunks", min=1, max=cfg.valen_max_length // CHUNK + 1)
    M = torch.export.Dim("image_tokens", min=1, max=65536)
    Q = torch.export.Dim("questions", min=1, max=4096)
    O = torch.export.Dim("candidates", min=1, max=65536)
    args = tuple(torch.from_numpy(cases[0][1][k]) for k in INPUTS)
    secs = ox.export_graph(dg, args, INPUTS, OUTPUTS,
                           {"input_ids": {1: CHUNK * Nc}, "position_ids": {2: CHUNK * Nc}, "image_embeds": {0: M},
                            "image_pos": {0: M}, "blocks": {1: CHUNK * Nc}, "instruction_spans": {0: Q},
                            "decision_pos": {0: Q}, "candidate_spans": {0: O}, "candidate_question": {0: O}},
                           os.path.join(tmp, "decoder", "model.onnx"))
    print("decoder exported in %.0fs" % secs)

    repo, revision, shards = ref.repo(tag), ref.MODELS[tag][1], ref.MODELS[tag][2]
    src = [safetensors_source(f, os.path.join(root, f), repo=repo, revision=revision, filename=f) for f in shards]
    os.makedirs(out_dir, exist_ok=True)
    vdir = os.path.join(out_dir, "vision")
    rep_v = ox.weightless(os.path.join(tmp, "vision"), vdir, src, rename)
    rep_d = ox.weightless(os.path.join(tmp, "decoder"), out_dir, src, rename)
    shutil.move(os.path.join(vdir, "model.onnx"), os.path.join(out_dir, "vision.onnx"))
    shutil.rmtree(vdir)
    ox.cleanup(tmp)
    shutil.copy(tok_json, os.path.join(out_dir, "tokenizer.json"))

    decision = {
        "engine": "onnx",
        "family": "valen",
        "layout": LAYOUT,
        "upstream": {"repo": repo, "revision": revision, "code": ref.CODE},
        "graphs": {"vision": "vision.onnx", "decoder": "model.onnx"},
        "contract": {
            "inputs": {"input_ids": {"dtype": "int64", "shape": [1, "seq"],
                                     "note": "the request's ids; seq a multiple of 64 with at least one padding "
                                             "position; right-pad with the pad id"},
                       "position_ids": {"dtype": "int64", "shape": [3, 1, "seq"], "note": "interleaved M-RoPE (t, h, w)"},
                       "image_embeds": {"dtype": "float32", "shape": ["image_tokens", "hidden"],
                                        "note": "the vision graph's output; one zero row for a text-only request"},
                       "image_pos": {"dtype": "int64", "shape": ["image_tokens"],
                                     "note": "where the rows replace token embeddings; a padding position for the "
                                             "zero row"},
                       "blocks": {"dtype": "int64", "shape": [1, "seq"],
                                  "note": "-1 prefix, 0 shared glue, 1 the decision questions, -2 padding"},
                       "instruction_spans": {"dtype": "int64", "shape": ["questions", 2],
                                             "note": "instruction tokens [start, end) per question"},
                       "decision_pos": {"dtype": "int64", "shape": ["questions"], "note": "each question's readout token"},
                       "candidate_spans": {"dtype": "int64", "shape": ["candidates", 2],
                                           "note": "description tokens [start, end), every question's candidates in order"},
                       "candidate_question": {"dtype": "int64", "shape": ["candidates"],
                                              "note": "the question index of each candidate"}},
            "outputs": {"logits": {"dtype": "float32", "shape": ["candidates"], "note": "raw candidate logits"}},
            "attention": "full-attention layers: upstream's bbox_blocks mask, from `blocks`; Gated DeltaNet layers: causal"},
        "chunk": CHUNK,
        **layout,
        "candidate_logits": {"noul": "[true, false]", "choice": "keys in the caller's order", "score": "levels in order"},
        "image": {"patch_size": ip.patch_size, "temporal_patch_size": ip.temporal_patch_size,
                  "merge_size": ip.merge_size,
                  "min_pixels": cfg.valen_media_kwargs["images_kwargs"]["size"]["shortest_edge"],
                  "max_pixels": cfg.valen_media_kwargs["images_kwargs"]["size"]["longest_edge"],
                  "resample": "PIL bicubic", "rescale_factor": ip.rescale_factor, "image_mean": list(ip.image_mean),
                  "image_std": list(ip.image_std), "position_table_side": vis.num_grid_per_side,
                  "position_interpolation": "bilinear, align_corners"},
        "mrope_section": cfg.text_config.rope_parameters["mrope_section"],
        "opset": ox.OPSET,
        "precision": "fp32 compute; backbone weights BF16 (widened by Cast), head F32",
        "weights_in_memory": ox.weights_in_memory(rep_d),
    }
    calibration = {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {},
                   "source": "none: upstream answers with a plain softmax of the head's logits (temperature 1.0)"}
    graph = lambda name: {"role": "graph", "path": name, "hosted_by": "ollaya",  # noqa: E731
                          "bytes": os.path.getsize(os.path.join(out_dir, name)),
                          "sha256": ox.sha256_file(os.path.join(out_dir, name))}
    files = {"model": "valen-" + tag, "layers": [
        graph("model.onnx"), graph("vision.onnx"),
        *[ox.file_entry("weights", repo, revision, f, os.path.join(root, f), location=f) for f in shards],
        ox.file_entry("tokenizer", repo, revision, "tokenizer.json", tok_json),
        {"role": "decision", "path": "decision.json", "hosted_by": "ollaya"},
        {"role": "calibration", "path": "calibration.json", "hosted_by": "ollaya"},
        ox.file_entry("license", repo, revision, "LICENSE", os.path.join(root, "LICENSE"))],
        "weightless": {"vision": {k: v for k, v in rep_v.items() if k != "unused"},
                       "decoder": {k: v for k, v in rep_d.items() if k != "unused"}},
        "unused_checkpoint_tensors": {"vision": {k: len(v) for k, v in rep_v["unused"].items()},
                                      "decoder": {k: len(v) for k, v in rep_d["unused"].items()}}}
    for name, obj in (("decision.json", decision), ("calibration.json", calibration), ("files.json", files)):
        ox.write_json(os.path.join(out_dir, name), obj)
    print(json.dumps({"vision": rep_v["stats"], "decoder": rep_d["stats"]}))

    del model, dg, vg
    worst = ort_check(out_dir, vis_args, cases)
    assert worst < 1e-3, worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag", choices=sorted(ref.MODELS))
    ap.add_argument("out")
    ap.add_argument("--model", default=None, help="a local copy of the model repository at the pinned revision")
    a = ap.parse_args()
    export(a.tag, a.out, a.model or ref.snapshot(a.tag))


if __name__ == "__main__":
    main()
