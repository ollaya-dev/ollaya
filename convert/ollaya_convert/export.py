"""Export a Laya checkpoint (encoder + decision head + marker gather) to a single ONNX graph.

    uv run python -m ollaya_convert.export en --out out/laya-en

Inputs (all dynamic along questions `q`, sequence `s` and markers `k`):
    input_ids      int64 [q, s]
    attention_mask int64 [q, s]
    marker_pos     int64 [q, k]
    marker_mask    bool  [q, k]
    qtype          int64 [q]
Outputs:
    logits         float32 [q, k]   raw option scores; temperature and softmax stay in the runtime
    act_logits     float32 [q, 2]   act/escalate head
"""
import argparse
import json
import os
import shutil

import numpy as np
import onnx
import torch

from . import laya_ref

INPUT_NAMES = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
OUTPUT_NAMES = ["logits", "act_logits"]
OPSET = int(os.environ.get("OLLAYA_OPSET", "20"))
# DecisionModel.forward picks topk(2) vs a zero-padded topk(1) with a Python `if` on the marker
# width, so the exported graph always takes topk(2). Runtimes pad the marker axis to at least this
# many slots; a masked slot scores -1e4, so probabilities, entropy and k are unchanged.
MIN_MARKERS = 2


def onnx_feed(batch):
    """Numpy inputs for the exported graph, with the marker axis padded to MIN_MARKERS."""
    feed = {n: batch[n].numpy() for n in INPUT_NAMES}
    pad = MIN_MARKERS - feed["marker_pos"].shape[1]
    if pad > 0:
        feed["marker_pos"] = np.pad(feed["marker_pos"], ((0, 0), (0, pad)))
        feed["marker_mask"] = np.pad(feed["marker_mask"], ((0, 0), (0, pad)))
    return feed


def expand_attention_masks(model):
    """Give every opset-23 `Attention` node a mask with the full query dimension.

    The decision head's `nn.TransformerEncoderLayer` passes a key-padding mask shaped
    [B, 1, 1, S]. The ONNX spec lets it broadcast over queries, but ONNX Runtime 1.28's kernel
    requires the query dimension spelled out ([B, 1, S_q, S]). Expanding it is exact.
    """
    from onnx import helper

    nodes, added = [], 0
    for node in model.graph.node:
        if node.op_type == "Attention" and len(node.input) > 3 and node.input[3]:
            q, mask = node.input[0], node.input[3]
            p = "%s_mask" % node.name
            nodes += [
                helper.make_node("Shape", [q], [p + "_qshape"], start=2, end=3),
                helper.make_node("Constant", [], [p + "_ones"],
                                 value=helper.make_tensor(p + "_ones_v", onnx.TensorProto.INT64, [2], [1, 1])),
                helper.make_node("Constant", [], [p + "_one"],
                                 value=helper.make_tensor(p + "_one_v", onnx.TensorProto.INT64, [1], [1])),
                helper.make_node("Concat", [p + "_ones", p + "_qshape", p + "_one"], [p + "_target"], axis=0),
                helper.make_node("Expand", [mask, p + "_target"], [p + "_full"]),
            ]
            node.input[3] = p + "_full"
            added += 1
        nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return added


class Graph(torch.nn.Module):
    """DecisionModel.forward with a stable, export-friendly signature."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        logits, act = self.model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        return logits.float(), act.float()


def sample_inputs(agent):
    """A real two-question request, so tracing follows the path production traffic takes."""
    state = {"subject": "Invoice #4411", "body": "We were billed twice for March, please refund."}
    questions = {
        "department": {"type": "choice", "instructions": "Which team handles this?",
                       "criteria": {"billing": "payments", "technical": "bugs", "other": "else"}},
        "refund": {"type": "noul", "instructions": "Does the user ask for a refund?"},
    }
    b = laya_ref.encode(agent, state, questions)["batch"]
    return tuple(b[n] for n in INPUT_NAMES)


def export(name: str, out_dir: str, root: str) -> str:
    # nn.TransformerEncoderLayer's fused fast path (aten::_transformer_encoder_layer_fwd) has no
    # ONNX lowering; the decomposed path is numerically the same module.
    torch.backends.mha.set_fastpath_enabled(False)
    agent = laya_ref.load(name, root=root, device="cpu")
    graph = Graph(agent.model).eval()
    args = sample_inputs(agent)

    q = torch.export.Dim("q", min=1, max=512)
    s = torch.export.Dim("s", min=8, max=agent.cfg.get("max_len", 512))
    k = torch.export.Dim("k", min=1, max=255)
    dynamic_shapes = {
        "input_ids": {0: q, 1: s},
        "attention_mask": {0: q, 1: s},
        "marker_pos": {0: q, 1: k},
        "marker_mask": {0: q, 1: k},
        "qtype": {0: q},
    }

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "model.onnx")
    with torch.no_grad():
        program = torch.onnx.export(
            graph, args, dynamo=True, opset_version=OPSET,
            input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
            dynamic_shapes=dynamic_shapes, optimize=os.environ.get("OLLAYA_ONNX_OPTIMIZE", "1") == "1",
        )
    program.save(path, external_data=False)
    if OPSET >= 23:
        model = onnx.load(path, load_external_data=True)
        expand_attention_masks(model)
        onnx.save(model, path, save_as_external_data=True, location="model.onnx.data")
    onnx.checker.check_model(path, full_check=True)

    # The tokenizer and decision config travel with the graph as their own layers.
    src = os.path.join(root, laya_ref.CHECKPOINTS[name] or "")
    shutil.copy(os.path.join(src, "tokenizer", "tokenizer.json"), os.path.join(out_dir, "tokenizer.json"))
    with open(os.path.join(src, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    tok = agent.tok
    decision = {
        "engine": "onnx",
        "family": "laya",
        "layout": "laya-markers-v1",
        "encoder": cfg["encoder"],
        "max_len": cfg.get("max_len", 512),
        "head_max_len": cfg.get("head_max_len", 192),
        "special_tokens": {
            "cls": tok.cls_token_id, "sep": tok.sep_token_id,
            "mask": tok.mask_token_id, "pad": tok.pad_token_id, "mask_text": tok.mask_token,
        },
        "inputs": INPUT_NAMES,
        "outputs": OUTPUT_NAMES,
        "min_markers": MIN_MARKERS,
        "opset": OPSET,
    }
    calibration = {
        "temperature": cfg.get("temperature", [1.0, 1.0, 1.0]),
        "temperature_by_options": cfg.get("temperature_by_options", {}),
    }
    with open(os.path.join(out_dir, "decision.json"), "w") as f:
        json.dump(decision, f, indent=2)
    with open(os.path.join(out_dir, "calibration.json"), "w") as f:
        json.dump(calibration, f, indent=2)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", choices=sorted(laya_ref.CHECKPOINTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=laya_ref.DEFAULT_ROOT)
    a = ap.parse_args()
    path = export(a.checkpoint, a.out, a.root)
    print("wrote %s (%.0f MB)" % (path, os.path.getsize(path) / 2**20))


if __name__ == "__main__":
    main()
