"""Export Cloudflare/clef-flash (Qwen3.5-9B + joint schema head) to a weightless ONNX graph, `clef-joint-v1`.

    OLLAYA_SCRATCH=<big disk> uv run python -m ollaya_convert.families.clef.export --out OUT [--model SNAPSHOT]

Graph (one request per run, every question in it; docs/families/clef.md):
    inputs   input_ids        int64   [1, seq]  the request's ids (layout.py), right-padded to a multiple of 64
             token_positions  int64   [n]       0..n-1: the unpadded positions
             question_spans   int64   [q, 2]    each question's instruction tokens, [start, end)
             question_types   int64   [q]       noul 0, choice 1, score 2
             option_spans     int64   [o, 2]    each option's tokens, [start, end), questions in order
             option_question  int64   [o]       the question each option belongs to
    outputs  logits           float32 [o]       one raw logit per option; a softmax per question gives probabilities

The backbone is transformers' Qwen3.5 text model (the repository's text_config) recomputed by
llm_common.qwen35. The head is the authors' `JointSchemaHead` (its own modules and weights) with the loops over
questions and options written as masked matrix products: span means as a span-mask product, the per-question
option softmax as a masked softmax, gathers by `option_question`. `check_head` compares it with the authors'
forward on the sample before export. The lexical option vectors read rows of the untied LM head, which stays
BF16 (a Gather, then a Cast). Every initializer references the repository's four BF16 shards or
`joint_head.safetensors` by byte offset.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil

import torch
import torch.nn.functional as F

from ..llm_common import onnx_export as ox
from ..llm_common.qwen35 import CHUNK, Qwen35Trunk, half_trunk
from ...weightless_sharded import safetensors_source
from . import ref
from .check import decision_for
from .layout import TYPE_INDEX, ClefLayout, graph_inputs

INPUT_NAMES = ["input_ids", "token_positions", "question_spans", "question_types", "option_spans", "option_question"]
OUTPUT_NAMES = ["logits"]
PREFIX = "model.language_model."


def span_mean(values, positions, spans):
    """values [n, d], spans [s, 2] -> the mean of values[start:end] per span, [s, d]."""
    inside = (positions[None, :] >= spans[:, :1]) & (positions[None, :] < spans[:, 1:])
    m = inside.to(values.dtype)
    return (m @ values) / m.sum(dim=-1, keepdim=True)


def attention(mha, query, key, value):
    """torch.nn.MultiheadAttention (batch_first, no masks, no dropout) on unbatched [L, E] inputs."""
    e = query.shape[-1]
    h = mha.num_heads
    d = e // h
    w, b = mha.in_proj_weight, mha.in_proj_bias
    q = F.linear(query, w[:e], b[:e]).view(-1, h, d).transpose(0, 1)
    k = F.linear(key, w[e:2 * e], b[e:2 * e]).view(-1, h, d).transpose(0, 1)
    v = F.linear(value, w[2 * e:], b[2 * e:]).view(-1, h, d).transpose(0, 1)
    a = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(d), dim=-1) @ v
    return mha.out_proj(a.transpose(0, 1).reshape(-1, e))


def head_scores(hd, hidden, token_rows, positions, question_spans, question_types, option_spans, option_question):
    """JointSchemaHead.forward for one record, vectorized. hidden [n, H] (unpadded); token_rows [n, H], the LM head's
    row for each token -> logits [o]."""
    x = hd.hidden_norm(hidden)
    memory = hd.memory_projection(x)
    global_vector = x[-1]
    question_vectors = span_mean(x, positions, question_spans)
    contexts = span_mean(x, positions, option_spans)
    lexical = span_mean(token_rows, positions, option_spans)
    routed = (hd.option_context_projection(contexts) + hd.option_lexical_projection(lexical)
              + hd.option_question_projection(question_vectors).index_select(0, option_question))
    for layer in hd.evidence_layers:
        mem = layer.memory_norm(memory)
        routed = routed + attention(layer.attention, layer.query_norm(routed), mem, mem)
        routed = routed + layer.feedforward(layer.feedforward_norm(routed))

    base_fields = hd.question_projection(question_vectors)
    member = option_question[None, :] == torch.arange(question_spans.shape[0], device=hidden.device)[:, None]
    scores = (base_fields @ routed.transpose(0, 1)) / math.sqrt(routed.shape[-1])
    summaries = torch.softmax(scores.masked_fill(~member, float("-inf")), dim=-1) @ routed
    fields = (base_fields + hd.option_summary_norm(summaries) + hd.global_projection(global_vector)[None]
              + hd.type_embedding(question_types))
    for layer in hd.layers:   # TransformerDecoderLayer, norm_first, GELU, no masks
        y = layer.norm1(fields)
        fields = fields + attention(layer.self_attn, y, y, y)
        y = layer.norm2(fields)
        fields = fields + attention(layer.multihead_attn, y, memory, memory)
        fields = fields + layer.linear2(F.gelu(layer.linear1(layer.norm3(fields))))
    fields = hd.field_norm(fields)

    anchor = F.normalize(question_vectors + global_vector, dim=-1).index_select(0, option_question)
    prior = hd.prior_logit_scale.clamp(max=math.log(100.0)).exp() * (F.normalize(lexical, dim=-1) * anchor).sum(-1)
    options = hd.option_norm(routed)
    field = fields.index_select(0, option_question)
    cosine = F.cosine_similarity(field, options, dim=-1)
    features = torch.cat([field, options, field * options, torch.abs(field - options)], dim=-1)
    residual = hd.residual_scorer(features).squeeze(-1)
    joint = hd.joint_logit_scale.clamp(max=math.log(100.0)).exp() * cosine + residual
    return prior + torch.sigmoid(hd.residual_gate) * joint


class ClefGraph(torch.nn.Module):
    def __init__(self, text_model, lm_head, head):
        super().__init__()
        self.trunk = Qwen35Trunk(text_model)
        self.lm_head = lm_head   # BF16 [vocab, hidden], rows gathered then widened
        self.head = head

    def scores(self, hidden, input_ids, token_positions, question_spans, question_types, option_spans, option_question):
        h = hidden.index_select(0, token_positions)
        ids = input_ids[0].index_select(0, token_positions)
        lexical = F.embedding(ids, self.lm_head).float()
        return head_scores(self.head, h, lexical, token_positions, question_spans, question_types, option_spans,
                           option_question)

    def forward(self, input_ids, token_positions, question_spans, question_types, option_spans, option_question):
        hidden = self.trunk(input_ids)[0].float()
        return self.scores(hidden, input_ids, token_positions, question_spans, question_types, option_spans,
                           option_question)


def text_model(model_dir):
    """transformers' Qwen3.5 text model with Clef's backbone weights (fp32), and the LM head (BF16)."""
    from safetensors.torch import load_file
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        config = Qwen3_5TextConfig(**json.load(f)["text_config"])
    # Built on the meta device and filled by assignment, so the fp32 weights (33 GB) exist once.
    with torch.device("meta"):
        m = Qwen3_5TextModel(config)
    state, lm_head = {}, None
    for f in ref.SHARDS:
        for k, v in load_file(os.path.join(model_dir, f)).items():
            if k.startswith(PREFIX):
                state[k[len(PREFIX):]] = v.float()
            elif k == "lm_head.weight":
                lm_head = v
            del v
    missing, unexpected = m.load_state_dict(state, strict=False, assign=True)
    del state
    if missing or unexpected or lm_head is None:
        raise SystemExit("weights do not fit the text model: missing %s, unexpected %s" % (missing[:5], unexpected[:5]))
    m.rotary_emb = type(m.rotary_emb)(config=config)   # its inv_freq buffer is not in the checkpoint
    return m.eval(), torch.nn.Parameter(lm_head, requires_grad=False)


def load_head(jsm, model_dir):
    from safetensors.torch import load_file

    head = jsm.JointSchemaHead(**json.load(open(os.path.join(model_dir, "joint_head_config.json"), encoding="utf-8")))
    head.load_state_dict(load_file(os.path.join(model_dir, ref.HEAD)), strict=True)
    return head.float().eval()


class Rows:
    """The LM head as upstream's head reads it (`weight[token_ids]`), widened row by row, not as a 4 GB fp32 copy."""

    def __init__(self, weight):
        self.weight = weight

    def __getitem__(self, ids):
        return self.weight[ids].float()


def check_head(jsm, graph, hidden, row, inputs, tokenizer):
    """The vectorized head against the authors' JointSchemaHead.forward on the same hidden states."""
    n = len(row["ids"])
    rec = jsm.encode_record(tokenizer, SAMPLE)
    assert list(rec.input_ids) == row["ids"], "layout port differs from upstream on the sample"
    t = {k: torch.from_numpy(v) for k, v in inputs.items()}
    with torch.no_grad():
        ours = graph.scores(hidden, t["input_ids"], t["token_positions"], t["question_spans"], t["question_types"],
                            t["option_spans"], t["option_question"])
        theirs = graph.head(hidden[None, :n], t["input_ids"][:, :n], torch.ones(1, n, dtype=torch.long), [rec],
                            Rows(graph.lm_head))[0]
    theirs = torch.cat(theirs)
    diff = float((ours - theirs).abs().max())
    print("head port vs upstream on the sample: max |logit diff| %.2e" % diff)
    print("sample logits:", [round(float(x), 4) for x in ours])
    assert diff < 1e-4, diff


SAMPLE = {
    "state": {"ticket": "The customer was charged twice for order A-104 and wants the duplicate refunded.",
              "plan": "pro", "amount": 49.0},
    "questions": {
        "team": {"type": "choice", "instructions": "Which team handles this?",
                 "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages", "sales": None}},
        "upset": {"type": "score", "instructions": "How upset is the customer?",
                  "criteria": ["calm", "annoyed", "angry", "furious"]},
        "refund": {"type": "noul", "instructions": "Is a refund requested?"},
    },
}


def export(out_dir, model_dir, fp16=False):
    import tokenizers
    from transformers import AutoTokenizer

    jsm = ref.upstream(model_dir)
    tok_json = os.path.join(model_dir, "tokenizer.json")
    decision_layout = decision_for(jsm)
    lay = ClefLayout(tokenizers.Tokenizer.from_file(tok_json), decision_layout)
    pad = tokenizers.Tokenizer.from_file(tok_json).token_to_id("<|endoftext|>")
    row = lay.encode(SAMPLE["state"], SAMPLE["questions"])
    inputs = graph_inputs(row, pad, CHUNK)

    tm, lm_head = text_model(model_dir)
    graph = ClefGraph(tm, lm_head, load_head(jsm, model_dir)).eval()
    if fp16:
        half_trunk(graph.trunk)   # the head and the LM-head rows stay fp32
    with torch.no_grad():
        hidden = graph.trunk(torch.from_numpy(inputs["input_ids"]))[0].float()
    check_head(jsm, graph, hidden, row, inputs, AutoTokenizer.from_pretrained(model_dir))
    del hidden

    args = tuple(torch.from_numpy(inputs[k]) for k in INPUT_NAMES)
    # The graph takes upstream's full 16,384 tokens; decision.json's max_tokens (ref.MAX_LEN) is the runtime's cap.
    N = torch.export.Dim("chunks", min=1, max=16384 // CHUNK)
    T = torch.export.Dim("tokens", min=2, max=16384)
    Q = torch.export.Dim("questions", min=1, max=4096)
    O = torch.export.Dim("options", min=2, max=65536)
    dyn = {"input_ids": {1: CHUNK * N}, "token_positions": {0: T}, "question_spans": {0: Q}, "question_types": {0: Q},
           "option_spans": {0: O}, "option_question": {0: O}}
    tmp = ox.scratch_dir("clef-export-")
    secs = ox.export_graph(graph, args, INPUT_NAMES, OUTPUT_NAMES, dyn, os.path.join(tmp, "model.onnx"))
    print("exported in %.0fs" % secs)
    del graph, tm, lm_head
    gc.collect()

    def rename(name):
        if name.startswith("trunk.m."):
            return [PREFIX + name[len("trunk.m."):]]
        if name.startswith("head."):
            return [name[len("head."):]]
        if name == "lm_head":
            return ["lm_head.weight"]
        return [name]

    sources = [safetensors_source(f, os.path.join(model_dir, f), repo=ref.REPO, revision=ref.REVISION, filename=f)
               for f in ref.SHARDS + [ref.HEAD]]
    report = ox.weightless(tmp, out_dir, sources, rename)
    ox.cleanup(tmp)
    shutil.copy(tok_json, os.path.join(out_dir, "tokenizer.json"))
    decision = {
        "engine": "onnx", "family": "clef", "layout": "clef-joint-v1",
        "upstream": {"repo": ref.REPO, "revision": ref.REVISION,
                     "code": "joint_schema_model.py in the model repository (encode_record, JointSchemaHead)"},
        "contract": {
            "inputs": {"input_ids": {"dtype": "int64", "shape": [1, "seq"],
                                     "note": "the request's ids; seq a multiple of 64; right-pad with the pad id"},
                       "token_positions": {"dtype": "int64", "shape": ["tokens"], "note": "0..tokens-1"},
                       "question_spans": {"dtype": "int64", "shape": ["questions", 2],
                                          "note": "instruction tokens [start, end) per question"},
                       "question_types": {"dtype": "int64", "shape": ["questions"], "note": "type_index"},
                       "option_spans": {"dtype": "int64", "shape": ["options", 2],
                                        "note": "option tokens [start, end), every question's options in order"},
                       "option_question": {"dtype": "int64", "shape": ["options"],
                                           "note": "the question index of each option"}},
            "outputs": {"logits": {"dtype": "float32", "shape": ["options"], "note": "raw option logits"}},
            "seq_multiple": CHUNK, "positions": "0..seq-1, implicit", "attention": "causal; no mask input"},
        **decision_layout,
        "pad": pad,
        "type_index": TYPE_INDEX,
        "option_logits": {"noul": "[true, false]", "choice": "labels sorted by code point", "score": "levels in order"},
        "opset": ox.OPSET,
        "precision": ox.FP16_PRECISION if fp16 else "fp32 compute; weights BF16 (widened by Cast)",
        "weights_in_memory": ox.weights_in_memory(report),
    }
    calibration = {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {},
                   "source": "none: upstream answers with a plain softmax of the head's logits"}
    files = {"model": "clef-flash", "layers": [
        {"role": "graph", "path": "model.onnx", "hosted_by": "ollaya", "bytes": os.path.getsize(os.path.join(out_dir, "model.onnx")),
         "sha256": ox.sha256_file(os.path.join(out_dir, "model.onnx"))},
        *[ox.file_entry("weights", ref.REPO, ref.REVISION, f, os.path.join(model_dir, f), location=f) for f in ref.SHARDS],
        ox.file_entry("weights/head", ref.REPO, ref.REVISION, ref.HEAD, os.path.join(model_dir, ref.HEAD), location=ref.HEAD),
        ox.file_entry("tokenizer", ref.REPO, ref.REVISION, "tokenizer.json", tok_json),
        {"role": "decision", "path": "decision.json", "hosted_by": "ollaya"},
        {"role": "calibration", "path": "calibration.json", "hosted_by": "ollaya"},
        ox.file_entry("license", ref.REPO, ref.REVISION, "LICENSE", os.path.join(model_dir, "LICENSE"))],
        "weightless": {k: v for k, v in report.items() if k != "unused"},
        "unused_checkpoint_tensors": {k: len(v) for k, v in report["unused"].items()}}
    for name, obj in (("decision.json", decision), ("calibration.json", calibration), ("files.json", files)):
        ox.write_json(os.path.join(out_dir, name), obj)
    print(json.dumps(files["weightless"]["stats"]), "graph MB %.1f" % (files["layers"][0]["bytes"] / 2**20),
          "inline", files["weightless"]["inline_bytes"], "unused", files["unused_checkpoint_tensors"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--fp16", action="store_true", help="fp16 compute in the trunk (the -fp16 GPU tags)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    export(a.out, a.model or ref.snapshot(), fp16=a.fp16)


if __name__ == "__main__":
    main()
