"""Export Mapika/decider-2b-vision to two weightless ONNX graphs + tokenizer + decision/calibration config.

    uv run --with pillow --with torchvision python -m ollaya_convert.families.decider_vision.export \
        --out out/decider-2b-vision [--root ROOT]

Graphs (layout `decider-vision-v1`, see graphs.py, ref.py and docs/families/decider.md):
    vision.onnx   patches [N, 1536], pos_idx [N, 4], pos_w [N, 4], rot_ids [N, 2] -> image_embeds [N/4, 2048]
    model.onnx    input_ids [R, T], position_ids [3, R, T], image_embeds [M, 2048], image_pos [M], slot_pos [R]
                  -> letter_logits [R, 10]
Both reference the author's `model.safetensors` (BF16) by byte offset; nothing is copied.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch

from ..llm_common import onnx_export as ox
from ..llm_common.qwen35 import CHUNK
from ...weightless_sharded import safetensors_source
from . import ref
from .graphs import DecoderGraph, VisionGraph

VISION_INPUTS = ["patches", "pos_idx", "pos_w", "rot_ids"]
VISION_OUTPUTS = ["image_embeds"]
INPUTS = ["input_ids", "position_ids", "image_embeds", "image_pos", "slot_pos"]
OUTPUTS = ["letter_logits"]


def rename(name):
    if name.startswith("v."):
        return ["model.visual." + name[2:]]
    if name.startswith("trunk.m."):
        return ["model.language_model." + name[len("trunk.m."):]]
    if name == "lm_head.weight":
        return ["model.language_model.embed_tokens.weight"]
    return [name]


def export(out_dir, root):
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids

    m = ref.load(root, "cpu")
    lm = m.lm
    vis = lm.model.visual
    cfg = lm.config
    sys.path.insert(0, root)
    from decider.prompt import letter_ids
    letters = letter_ids(m.tok)

    qs = {"square": {"type": "choice", "instructions": "What color is the square?",
                     "criteria": {"red": None, "green": None, "blue": None, "yellow": None}},
          "circle": {"type": "noul", "instructions": "Is there a circle in the image?"}}
    inp, plan, want = ref.forward(m, ref.demo_png(), "This is a visual question about the image.", qs)
    grid = inp["image_grid_thw"][:1]
    pv = inp["pixel_values"][: int(grid.prod())]
    idx, wt = get_vision_interpolation_indices_and_weights(grid, vis.num_grid_per_side, mode="bilinear",
                                                            align_corners=True, spatial_merge_size=vis.spatial_merge_size)
    rot = get_vision_position_ids(grid, vis.spatial_merge_size)
    vg = VisionGraph(vis).eval()
    with torch.no_grad():
        emb = vg(pv, idx, wt, rot)

    ids, am, mm = inp["input_ids"], inp["attention_mask"], inp["mm_token_type_ids"]
    pos, _ = lm.model.get_rope_index(ids, mm, image_grid_thw=inp["image_grid_thw"], attention_mask=am)
    R, T = ids.shape
    T2 = -(-(T + 1) // CHUNK) * CHUNK
    ids2 = torch.zeros((R, T2), dtype=torch.long)
    ids2[:, :T] = ids
    pos2 = torch.zeros((3, R, T2), dtype=torch.long)
    pos2[:, :, :T] = pos
    where = (ids2 == cfg.image_token_id).nonzero()
    image_pos = where[:, 0] * T2 + where[:, 1]
    image_embeds = torch.cat([emb] * R)
    dg = DecoderGraph(lm.model.language_model, lm.lm_head, letters, cfg.text_config.rope_parameters["mrope_section"]).eval()
    with torch.no_grad():
        got = dg(ids2, pos2, image_embeds, image_pos, inp["slot_idx"])
    diff = max(float((got[i, :p["k"]] - torch.tensor(want[i])).abs().max()) for i, p in enumerate(plan))
    print("eager graphs vs upstream: %.2e" % diff)

    tmp = ox.scratch_dir("decider-vision-export-")
    Nt = torch.export.Dim("tokens", min=1, max=16384)
    secs = ox.export_graph(vg, (pv, idx, wt, rot), VISION_INPUTS, VISION_OUTPUTS,
                           {"patches": {0: 4 * Nt}, "pos_idx": {0: 4 * Nt}, "pos_w": {0: 4 * Nt}, "rot_ids": {0: 4 * Nt}},
                           os.path.join(tmp, "vision", "model.onnx"))
    print("vision exported in %.0fs" % secs)
    Rd = torch.export.Dim("rows", min=1, max=1024)
    Nc = torch.export.Dim("chunks", min=1, max=64)
    M = torch.export.Dim("image_tokens", min=1, max=65536)
    secs = ox.export_graph(dg, (ids2, pos2, image_embeds, image_pos, inp["slot_idx"]), INPUTS, OUTPUTS,
                           {"input_ids": {0: Rd, 1: CHUNK * Nc}, "position_ids": {1: Rd, 2: CHUNK * Nc},
                            "image_embeds": {0: M}, "image_pos": {0: M}, "slot_pos": {0: Rd}},
                           os.path.join(tmp, "decoder", "model.onnx"))
    print("decoder exported in %.0fs" % secs)

    src = [safetensors_source("model.safetensors", os.path.join(root, "model.safetensors"), repo=ref.REPO,
                              revision=ref.REVISION, filename="model.safetensors")]
    os.makedirs(out_dir, exist_ok=True)
    vdir = os.path.join(out_dir, "vision")
    rep_v = ox.weightless(os.path.join(tmp, "vision"), vdir, src, rename)
    rep_d = ox.weightless(os.path.join(tmp, "decoder"), out_dir, src, rename)
    shutil.move(os.path.join(vdir, "model.onnx"), os.path.join(out_dir, "vision.onnx"))
    shutil.rmtree(vdir)
    ox.cleanup(tmp)
    shutil.copy(os.path.join(root, "tokenizer.json"), os.path.join(out_dir, "tokenizer.json"))

    ip = m.proc.image_processor
    decision = {
        "engine": "onnx",
        "family": "decider",
        "layout": "decider-vision-v1",
        "upstream": {"repo": ref.REPO, "revision": ref.REVISION, "code": "decider/vision.py + decider/prompt.py (v5)"},
        "graphs": {"vision": "vision.onnx", "decoder": "model.onnx"},
        # The text layout of Ollaya's `decider` (decider-slots-v1), narrowed to v5's ten letters.
        "max_ctx_tokens": ref.MAX_CTX_TOKENS,
        "max_options": ref.MAX_OPTIONS,
        "min_options": 2,
        "max_levels": ref.MAX_OPTIONS,
        "isolated_levels": False,
        "isolated_row_temperature": 1.0,
        "neutralize_none": False,
        "independent_rows": True,
        "index_arrays_min_len": 8,
        "labels": {"narrow": ref.MAX_OPTIONS, "strings": list("ABCDEFGHIJ"), "ids": letters,
                   "open_ids": m.tok.encode("\n(", add_special_tokens=False)},
        "special_tokens": {"pad": 0},
        "retokenize": "upstream decodes each row's text ids and tokenizes the string again (with the image "
                      "placeholder in front); rows are encode(decode(ids))",
        "chunk": CHUNK,
        "tokens": {"image": cfg.image_token_id, "vision_start": cfg.vision_start_token_id,
                   "vision_end": cfg.vision_end_token_id, "pad": 0},
        "image": {"patch_size": ip.patch_size, "temporal_patch_size": ip.temporal_patch_size,
                  "merge_size": ip.merge_size, "min_pixels": ip.size["shortest_edge"],
                  "max_pixels": ip.size["longest_edge"], "resample": "PIL bicubic",
                  "rescale_factor": ip.rescale_factor, "image_mean": list(ip.image_mean), "image_std": list(ip.image_std),
                  "position_table_side": vis.num_grid_per_side, "position_interpolation": "bilinear, align_corners"},
        "mrope_section": cfg.text_config.rope_parameters["mrope_section"],
        "opset": ox.OPSET,
        "weights_in_memory": ox.weights_in_memory(rep_d),
    }
    calibration = {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {},
                   "source": "none upstream (decider_config.json has no temperature for the vision model)"}
    files = {"model": "decider-2b-vision", "weightless": {"vision": rep_v["stats"], "decoder": rep_d["stats"]},
             "unused_checkpoint_tensors": {"vision": {k: len(v) for k, v in rep_v["unused"].items()},
                                           "decoder": {k: len(v) for k, v in rep_d["unused"].items()}}}
    ox.write_json(os.path.join(out_dir, "decision.json"), decision)
    ox.write_json(os.path.join(out_dir, "calibration.json"), calibration)
    ox.write_json(os.path.join(out_dir, "files.json"), files)
    print(json.dumps(files))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    export(a.out, a.root or ref.snapshot())


if __name__ == "__main__":
    main()
