"""The PyTorch reference for StrandsAgents/strands-decider-* (Qwen3.5-2B and Gemma 4 E2B/E4B): upstream's own package, in fp32.

The model repo holds the LoRA adapter (`lora/`), the pointer head (`head.safetensors`), its config
(`strands_decider_config.json`, with the fitted temperatures) and a tokenizer; the inference code is
https://github.com/strands-labs/strands-decider (pinned below). Point STRANDS_SRC at that commit's `src/`.

The reference is upstream's serving path: `StrandsDeciderModel.load` and `SystemOneEngine` with its default
`EngineConfig` (questions fitted in runs of 32, a question reserve of at most 75% of the 4,096-token window,
no strict window), with the torso built in fp32 from its BF16 weights, TF32 off, and the batched path (no
prefix cache; upstream pins the two equal to float tolerance). Scores come out raw, before the per-type
temperature, which the runtime applies.

    model, engine = ref.load(run_dir, device="cuda")
    rows, meta, rendered = ref.encode(engine, state, questions)   # upstream _chunks / _fit / _option_idx
    scores = ref.forward(model, rows)                              # one [k] array of raw scores per question
    answers = ref.answers(model, meta, rendered, scores)           # upstream _to_answer at the fitted temperatures
"""
from __future__ import annotations

import os
import sys

import torch

STRANDS_GIT = {"repo": "https://github.com/strands-labs/strands-decider", "commit": "930e6d5de16309be32579f5631a7369a3322140c"}
MODELS = {
    # v1 2610: the "soup" of three Qwen3.5-2B-Base runs (provenance.json run_id v1-qwen3.5-2b-soup).
    "strands-decider-2b": {"repo": "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610",
                           "revision": "216d1ccb46a9e9b036eb9187b248db98480f77d4",
                           "base": "Qwen/Qwen3.5-2B-Base", "base_revision": "b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
                           "base_files": ["model.safetensors-00001-of-00001.safetensors"]},
    # v1 2610 on Gemma 4 (provenance.json run_id g26-e2b-code-soup3 / g26-e4b-code-soup3, 2026-10-09). The base
    # revision is the Hub main upstream's loader resolved when the models were made (provenance.json names it).
    "strands-decider-e2b": {"repo": "StrandsAgents/strands-decider-E2B-gemma4-v1-2610",
                            "revision": "d383ecf8534b13d49b3b567a1b66f88fca6d50ba",
                            "base": "google/gemma-4-E2B-it", "base_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
                            "base_files": ["model.safetensors"]},
    "strands-decider-e4b": {"repo": "StrandsAgents/strands-decider-E4B-gemma4-v1-2610",
                            "revision": "1344d44397c457f3be34b8ff15c593b41248c6bf",
                            "base": "google/gemma-4-E4B-it", "base_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
                            "base_files": ["model.safetensors"]},
}


class RequestError(ValueError):
    """Upstream rejects the request (pydantic validation, or a prompt truncated through its options)."""


def _import():
    src = os.environ.get("STRANDS_SRC")
    if not src:
        raise SystemExit("set STRANDS_SRC to the src/ of %s at %s" % (STRANDS_GIT["repo"], STRANDS_GIT["commit"]))
    if src not in sys.path:
        sys.path.insert(0, src)
    import strands_decider.infer  # noqa: F401
    import strands_decider.modeling  # noqa: F401
    import strands_decider.prompting  # noqa: F401
    import strands_decider.schema  # noqa: F401
    m = sys.modules
    return (m["strands_decider.infer"], m["strands_decider.modeling"], m["strands_decider.prompting"],
            m["strands_decider.schema"])


def snapshot(slug):
    """Local snapshots of the model repo and of the pinned base (upstream's loader reads the base from the hub cache)."""
    from huggingface_hub import snapshot_download

    m = MODELS[slug]
    run = os.environ.get("STRANDS_RUN") or snapshot_download(m["repo"], revision=m["revision"])
    base = os.environ.get("STRANDS_BASE") or snapshot_download(m["base"], revision=m["base_revision"])
    return run, base


def load(run_dir, device="cpu"):
    """Upstream StrandsDeciderModel.load with the torso in fp32, and a SystemOneEngine on `device`."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    infer, modeling, _, _ = _import()
    from_json = modeling.StrandsDeciderConfig.from_json

    def fp32(path):  # load() builds the torso in the config's dtype (bfloat16); the reference widens it
        cfg = from_json(path)
        cfg.torch_dtype = "float32"
        return cfg

    modeling.StrandsDeciderConfig.from_json = staticmethod(fp32)
    try:
        model = modeling.StrandsDeciderModel.load(run_dir)
    finally:
        modeling.StrandsDeciderConfig.from_json = from_json
    engine = infer.SystemOneEngine(model, infer.EngineConfig(device=device, use_prefix_cache=False))
    return model, engine


def encode(engine, state, questions):
    """-> (rows, meta, rendered): one row per question, request order, as upstream's batched path builds it.

    rows[q] = {"ids": state + question token ids, "decide": the last position (`<answer>`),
               "opts": the last token of each option's line}."""
    import pydantic

    _, _, prompting, schema = _import()
    try:
        req = schema.SystemOneRequest(state=state, questions=questions)
    except pydantic.ValidationError as e:
        raise RequestError("ValidationError: %s" % e.errors()[0]["msg"]) from e
    names = list(req.questions)
    rendered = [prompting.render_question(req.questions[n]) for n in names]
    rows = []
    for chunk, s, q, offsets in engine._chunks(prompting.render_state(req.state), rendered):
        rs = rendered[chunk]
        try:
            idx = engine._option_idx(rs, len(s), offsets).tolist()
        except ValueError as e:  # truncation removed an option's line
            raise RequestError(str(e)) from e
        for rq, qi, ix in zip(rs, q, idx, strict=True):
            ids = list(s) + list(qi)
            rows.append({"ids": ids, "decide": len(ids) - 1, "opts": ix[: rq.n_slots]})
    meta = [{"qid": n, "type": rq.kind, "k": rq.n_slots} for n, rq in zip(names, rendered)]
    return rows, meta, rendered


@torch.no_grad()
def forward(model, rows):
    """Raw pointer scores (before temperature), one [k] array per row: upstream forward, one row at a time."""
    _, modeling, _, _ = _import()
    device = next(model.head.parameters()).device
    out = []
    for r in rows:
        ids = torch.tensor([r["ids"]], device=device)
        mask = torch.ones_like(ids)
        hidden = model.encode(ids, mask)
        pooled = modeling.pool_last_token(hidden, mask).to(torch.float32)
        options = modeling.gather_options(hidden, torch.tensor([r["opts"]], device=device)).to(torch.float32)
        out.append(model.head(pooled, options)[0].float().cpu().numpy())
    return out


def temperature(model, kind):
    by_kind = model.config.temperature_by_kind or {}
    return float(by_kind.get(kind, model.config.temperature))


def answers(model, meta, rendered, scores):
    """Upstream _to_answer on softmax(scores / T[kind]): what the server returns for these scores (4 decimals)."""
    infer, _, _, _ = _import()
    out = {}
    for m, rq, z in zip(meta, rendered, scores, strict=True):
        p = torch.softmax(torch.tensor(z, dtype=torch.float32) / temperature(model, rq.kind), -1).tolist()
        out[m["qid"]] = infer._to_answer(rq, p, ordinal_smoothing=model.config.ordinal_smoothing).model_dump()
    return out
