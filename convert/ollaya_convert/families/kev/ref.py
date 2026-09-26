"""The PyTorch reference for jaredpalmer/kev: upstream's own `kev` package, run in fp32.

The model repo holds only the adapter, `head.pt` and a tokenizer; the inference code lives at
https://github.com/jaredpalmer/kev (pinned below). Point KEV_SRC at a checkout of that commit. The
reference is upstream's exact path (`Checkpoint.load` with fp32 and the LoRA merged, the path every
reported number uses; `DecisionModel.forward` in row form, no prefix cache), with the head temperature
set to 1.0 so the raw pointer scores come out; the fitted temperature is applied by the runtime.

    ck, tok, m = ref.load(run_dir, base_dir, device="cuda")
    enc = ref.encode(tok, m, state, questions)        # upstream encode() of upstream to_record()
    logits = ref.forward(m, enc)                      # list of [k] raw scores, one per question
    answers = ref.system_one(ck, tok, m, state, questions)
"""
from __future__ import annotations

import os
import sys

import torch

KEV_GIT = {"repo": "https://github.com/jaredpalmer/kev", "commit": "234e5a7498f82f253de34e67b9fa99aefb5f20f5"}
# Each checkpoint names its base and base revision in head.pt; the export checks them against these pins.
MODELS = {
    # Kev-0.8B round 15 (documents + skills delta). Round 7 + dates was 54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8.
    "kev-0.8b": {"repo": "jaredpalmer/kev-0.8b", "revision": "9a45d25eb2ab761841196625383fa1dff0e56c1e",
                 "base": "Qwen/Qwen3.5-0.8B-Base", "base_revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
                 "base_files": ["model.safetensors-00001-of-00001.safetensors"]},
    # Kev-4B round 10 (skills delta on round 8).
    "kev-4b": {"repo": "jaredpalmer/kev-4b", "revision": "139fdd94f1b6a6ad80cc15e08fcb99cac885a101",
               "base": "Qwen/Qwen3.5-4B-Base", "base_revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
               "base_files": ["model.safetensors-00001-of-00002.safetensors",
                              "model.safetensors-00002-of-00002.safetensors"]},
    # Kev-9B (decision-v7 recipe + dates/unknowable delta; head.pt carries the fitted temperature).
    "kev-9b": {"repo": "jaredpalmer/kev-9b", "revision": "2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c",
               "base": "Qwen/Qwen3.5-9B-Base", "base_revision": "68c46c4b3498877f3ef123c856ecfde50c39f404",
               "base_files": ["model.safetensors-%05d-of-00004.safetensors" % i for i in range(1, 5)]},
}


def _import_kev():
    src = os.environ.get("KEV_SRC")
    if not src:
        raise SystemExit("set KEV_SRC to a checkout of %s at %s" % (KEV_GIT["repo"], KEV_GIT["commit"]))
    if src not in sys.path:
        sys.path.insert(0, src)
    import kev.api  # noqa: F401
    import kev.checkpoint  # noqa: F401
    import kev.model  # noqa: F401
    return sys.modules["kev.api"], sys.modules["kev.checkpoint"], sys.modules["kev.model"]


def snapshot(slug):
    from huggingface_hub import snapshot_download

    m = MODELS[slug]
    run = os.environ.get("KEV_RUN") or snapshot_download(m["repo"], revision=m["revision"])
    base = os.environ.get("KEV_BASE") or snapshot_download(m["base"], revision=m["base_revision"])
    return run, base


def load(run_dir: str, base_dir: str, device: str = "cpu", merge: bool = True):
    """Upstream Checkpoint.load in fp32. The base is read from a local snapshot of the pinned base revision."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, kc, _ = _import_kev()
    ck = kc.Checkpoint(run_dir)
    ck.upstream_base = (ck.meta.base, ck.meta.base_revision)  # what head.pt names, before the local override
    ck.meta.base, ck.meta.base_revision = base_dir, None   # same bytes as the pinned base revision, read locally
    tok, m = ck.load(device, kc.LoadOptions(dtype=torch.float32, merge=merge))
    return ck, tok, m


def request(state, questions):
    api, _, _ = _import_kev()
    return api.SystemOneRequest(state=state, questions=questions)


def encode(tok, m, state, questions):
    """-> (enc, meta): upstream encode() of upstream to_record(), serving limits."""
    api, _, km = _import_kev()
    rec, meta = api.to_record(request(state, questions))
    enc = m.encode(tok, rec, max_state=km.SERVE_MAX_STATE, max_branch=km.SERVE_MAX_BRANCH)
    return enc, meta


def rows(enc):
    """Upstream rows_of(): the causal rows (state + branch) with decide/option positions in the row."""
    _, _, km = _import_kev()
    S, _, brs = km.rows_of(enc)
    return [{"ids": S + r["ids"], "decide": len(S) + r["decide"], "opts": [len(S) + o for o in r["opts"]]} for r in brs]


@torch.no_grad()
def forward(m, enc):
    """Raw pointer scores (temperature 1.0), one [k] array per question."""
    t = m.head.temperature
    m.head.temperature = 1.0
    try:
        return [z.float().cpu().numpy() for z in m.forward(enc)]
    finally:
        m.head.temperature = t


def system_one(ck, tok, m, state, questions):
    """What kev.serve answers (minus its prefix cache, which is exact up to fp reassociation)."""
    api, _, km = _import_kev()
    rec, meta = api.to_record(request(state, questions))
    ps = m.probs(m.encode(tok, rec, max_state=km.SERVE_MAX_STATE, max_branch=km.SERVE_MAX_BRANCH))
    return api.to_answers([p.tolist() for p in ps], meta)


def answers(meta, scores, temperature):
    """Upstream kev.api.to_answers on softmax(scores / T): what kev.serve returns for these scores."""
    api, _, _ = _import_kev()
    ps = [torch.softmax(torch.tensor(z, dtype=torch.float32) / temperature, -1).tolist() for z in scores]
    return api.to_answers(ps, meta)
