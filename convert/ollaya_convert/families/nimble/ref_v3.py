"""The PyTorch reference for bespokelabs/Bespoke-Nimble-9B-v3: the author's Decision Index engine, run in fp32.

The model repo holds only the LoRA adapter (r=64) on Qwen/Qwen3.5-9B and its contract (`schema_config.json`:
the 255 option codes and their token ids, and the sha256 of the two prompt files). The code ships with the
author's Decision Index run, the dataset `bespokelabs/bespoke-nimble-9b-v3-decision-index` (pinned below):
`code/nimble/evaluation/{parallel_schema,extended_schema,decision_index_engine}.py`. This module imports those
files from the snapshot, so the prompts, the TypeSafe mapping and the readout are upstream's code, not a copy.
The prompt files are byte-identical to v2's (`a0a0f94d...`, `fb51ac3d...`).

How v3 differs from v2 (`ref.py`), all in `decision_index_engine.NimbleEngine.__call__`:
* `native_schema`: a choice or noul side without a description has none (v2 wrote the label, or Yes/No);
* `native_question`: noul criteria keyed exactly {yes, no} read as {true, false};
* a blank string state is passed as `json.dumps(state)`;
* prompts may run to 32,768 tokens (v2's serving contract: 8,192), and nothing is truncated;
* probabilities are softmax(candidate logits) at temperature 1 (no temperature file);
* the tokenizer is `AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")`, which transformers 5 rebuilds from the
  vocabulary with its own pre-tokenizer: byte-identical to the `tokenizer.json` v2 ships, not to the base
  repo's file (which splits combining marks differently). The export ships v2's file.

The engine itself takes requests the Decision Index harness has already validated against TypeSafe's request
models; `_typesafe` applies those shapes first (a score's levels are a list, a noul's criteria an object, a
list of choice labels means labels without descriptions), as v2's reference does with openjev's models.

    tok, model = ref_v3.load(adapter_dir, base_dir, device="offload")
    prepared, meta = ref_v3.prepare(tok, state, questions)   # the engine's mapping + extended_schema.prepare_prompts
    logits = ref_v3.forward(model, prepared)                 # [k] raw candidate logits per question
    answers = ref_v3.answers(meta, logits, 1.0)              # TypeSafe answers from softmax(logits)
"""
from __future__ import annotations

import json
import os
import sys
import types

import torch

from . import ref as ref_v2

REPO = "bespokelabs/Bespoke-Nimble-9B-v3"
REVISION = "8e927b9b4afdbb14479fac10a7364d1a695be208"
BASE, BASE_REVISION, BASE_FILES = ref_v2.BASE, ref_v2.BASE_REVISION, ref_v2.BASE_FILES
CODE = {"repo": "bespokelabs/bespoke-nimble-9b-v3-decision-index", "repo_type": "dataset",
        "revision": "654b124d3508a7da0d849405744bde80c4bb1a1b", "path": "code/nimble/evaluation"}
# transformers 5's AutoTokenizer for the base at BASE_REVISION, byte for byte (see the module docs)
TOKENIZER = {"repo": ref_v2.REPO, "revision": ref_v2.REVISION, "file": "tokenizer.json"}
MAX_TOKENS = 32768   # NimbleEngine(max_tokens=32768), the setting of the published run

RequestError = ref_v2.RequestError


def snapshot():
    from huggingface_hub import snapshot_download

    adapter = os.environ.get("NIMBLE_ADAPTER") or snapshot_download(REPO, revision=REVISION)
    base = os.environ.get("NIMBLE_BASE") or snapshot_download(BASE, revision=BASE_REVISION)
    return adapter, base


def code_dir():
    """The code as plain files: the engine hashes the prompt files next to its own resolved path, which in
    the hub cache's symlink layout is the blob store."""
    from huggingface_hub import snapshot_download

    local = os.path.expanduser("~/.cache/ollaya_convert/%s-%s" % (CODE["repo"].split("/")[1], CODE["revision"][:12]))
    return os.environ.get("NIMBLE_CODE") or snapshot_download(CODE["repo"], repo_type="dataset", revision=CODE["revision"],
                                                              allow_patterns=["code/**"], local_dir=local)


def upstream():
    """The author's modules: (extended_schema, decision_index_engine)."""
    root = os.path.join(code_dir(), "code")
    if root not in sys.path:
        sys.path.insert(0, root)
    if "decision_index.engines" not in sys.modules:
        # the harness package, of which the engine imports only these two names
        engines = types.ModuleType("decision_index.engines")
        engines.Engine = type("Engine", (), {})
        engines.Unsupported = type("Unsupported", (Exception,), {})
        sys.modules["decision_index"] = types.ModuleType("decision_index")
        sys.modules["decision_index.engines"] = engines
    from nimble.evaluation import decision_index_engine, extended_schema

    return extended_schema, decision_index_engine


def contract(adapter_dir):
    return json.load(open(os.path.join(adapter_dir, "schema_config.json"), encoding="utf-8"))


def load(adapter_dir, base_dir, device="cpu"):
    """The engine's model in fp32 (Qwen3_5ForConditionalGeneration + PeftModel, unmerged) and its tokenizer."""
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    c = contract(adapter_dir)
    if (c["model"], c["revision"]) != (BASE, BASE_REVISION):
        raise SystemExit("schema_config.json names %s@%s, not the pinned base" % (c["model"], c["revision"]))
    _, engine = upstream()
    tok = AutoTokenizer.from_pretrained(base_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    engine.validate_contract(c, tok)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(base_dir, dtype=torch.float32, attn_implementation="sdpa")
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, adapter_dir).eval()
    if device == "offload":   # as ref.load: fp32 9B (36 GB) on a 24 GB GPU
        from accelerate import dispatch_model, infer_auto_device_map

        device_map = infer_auto_device_map(model, max_memory={0: os.environ.get("NIMBLE_GPU_MEM", "20GiB"), "cpu": "200GiB"},
                                           no_split_module_classes=base._no_split_modules)
        model = dispatch_model(model, device_map=device_map)
        model.offload_device_map = device_map
        return tok, model
    return tok, model.to(device)


def _typesafe(qid, q):
    """TypeSafe's request shapes, which the Decision Index harness validates before the engine runs."""
    if not isinstance(q, dict) or "instructions" not in q:
        raise RequestError("question %r: instructions are required" % qid)
    t, crit = q.get("type"), q.get("criteria")
    if t == "noul":
        if crit is not None and not isinstance(crit, dict):
            raise RequestError("question %r: noul criteria must be an object" % qid)
    elif t == "choice":
        if isinstance(crit, list):
            if not all(isinstance(x, str) for x in crit):
                raise RequestError("question %r: choice labels must be strings" % qid)
            q = dict(q, criteria=dict.fromkeys(crit))
        elif not isinstance(crit, dict):
            raise RequestError("question %r: choice criteria must be an object or a list of labels" % qid)
    elif t == "score":
        if not isinstance(crit, list):
            raise RequestError("question %r: score criteria must be a list of levels" % qid)
    else:
        raise RequestError("question %r: unknown type %r" % (qid, t))
    return q


def prepare(tok, state, questions, adapter_dir=None):
    """-> (PreparedPrompts, meta): NimbleEngine.__call__ up to its forward pass."""
    extended_schema, engine = upstream()
    if not isinstance(questions, dict) or not questions:
        raise RequestError("questions must contain at least one question")
    questions = {qid: _typesafe(qid, q) for qid, q in questions.items()}
    if isinstance(state, str) and not state.strip():
        state = json.dumps(state)
    try:
        body = {"state": state, "questions": {k: engine.native_question(q) for k, q in questions.items()}}
        context, schema = engine.native_schema(body)
        prepared = extended_schema.prepare_prompts(tok, context, schema, MAX_TOKENS)
    except (ValueError, engine.Unsupported) as e:
        raise RequestError(str(e)) from e
    meta = [{"qid": n, "type": questions[n]["type"], "k": len(c), "criteria": questions[n].get("criteria")}
            for n, c in zip(prepared.names, prepared.choices)]
    return prepared, meta


@torch.no_grad()
def forward(model, prepared, adapter_dir=None):
    """Raw candidate logits (fp32), one [k] array per question: the engine's candidate_logits, one row at a time."""
    _, engine = upstream()
    device = "cuda:0" if getattr(model, "offload_device_map", None) else next(model.parameters()).device
    out = []
    for ids, cand in zip(prepared.full_ids, prepared.candidate_ids):
        batch = {k: v.to(device) for k, v in engine.collate([{"input_ids": ids, "candidate_ids": cand}], 0).items()}
        out.append(engine.candidate_logits(model, batch)[0].double().cpu().numpy())
    return out


def answers(meta, logits, temperature=1.0):
    """TypeSafe answers from softmax(logits / T): the engine's probabilities (noul: P(true); choice: the
    argmax and the distribution), and a score's expected level, as for v2."""
    return ref_v2.answers(meta, logits, temperature)
