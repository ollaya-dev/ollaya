"""The PyTorch reference for PostHog/jeeves: the authors' own code (github.com/PostHog/jeeves, pinned below),
run in fp32, in its no-thinking mode.

The model repository holds the fused weights (Qwen3.5-9B with the LoRA merged), the pointer head
(`head.pt`) and `export.json` (temperature 1.522 for v2, format `markers-v3-plainchains`). A question is the
upstream `Encoder`'s prompt with an empty thought, then the option block again and the decide marker:

    ids  = enc(chat(STATE + state + "\\n" + Q + instructions + "\\n" + options)) + enc("\\n")
           + enc("</think>\\n\\n" + options + DECIDE)
    options = "".join(OPT + option + OPT_END + "\\n" for option in question.options())

The pointer head scores option j as k(h[opt_end_j]) . q(h[decide]) / 16, at the option markers after
`</think>`; the fitted temperature is applied by the runtime.

    model, head, enc = ref.load(snapshot, device="cpu")
    rows, meta = ref.encode(enc, state, questions)       # upstream parse_request + Encoder
    scores = ref.forward(model, head, enc, rows)         # [k] raw scores per question
"""
from __future__ import annotations

import os
import sys

import torch

REPO = "PostHog/jeeves"
# Jeeves v2 (main; the Hub has no v2 tag). v1 was 8622b7d1652a9dcb8629486b84dce9e8d690c5cd (Hub tag v1, README aside).
REVISION = "04b9f87a79484fda54131de3b030b579d399e704"
JEEVES_GIT = {"repo": "https://github.com/PostHog/jeeves", "commit": "6151619c14fcffb2406830f09fd4f09fdd22200e"}
SHARDS = ["model-%05d-of-00005.safetensors" % i for i in range(1, 6)]
MAX_LEN = 8192
# The runtime's tokenizer. transformers 5 does not load Jeeves' tokenizer.json verbatim (its Qwen2Tokenizer
# rewrites the pre-tokenizer, `[\\p{L}\\p{M}]+` to `\\p{L}+`, as for Kev), and upstream tokenizes through
# transformers. Kev-9B's tokenizer.json is that tokenizer as transformers saves it, identical in every field
# to what transformers builds from Jeeves' repository, so the runtime loads it from there.
TOKENIZER = ("jaredpalmer/kev-9b", "2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c", "tokenizer.json")


class RequestError(ValueError):
    """The request is invalid for this model (HTTP 422)."""


def _src():
    src = os.environ.get("JEEVES_SRC")
    if not src:
        raise SystemExit("set JEEVES_SRC to a checkout of %s at %s" % (JEEVES_GIT["repo"], JEEVES_GIT["commit"]))
    if src not in sys.path:
        sys.path.insert(0, src)
    return src


def snapshot():
    from huggingface_hub import snapshot_download

    return os.environ.get("JEEVES_MODEL") or snapshot_download(REPO, revision=REVISION, ignore_patterns=["drafter_*"])


def load(path, device="cpu"):
    """Upstream export.load_export in fp32, the head's temperature set to 1 (raw scores)."""
    _src()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from export import load_export

    model, head, enc = load_export(path, device=device, dtype=torch.float32)
    head.temperature = 1.0
    return model.eval(), head.eval(), enc


def encoder(path):
    _src()
    from loader.dataloader import Encoder

    return Encoder(path)


def encode(enc, state, questions):
    """-> (rows, meta): upstream parse_request, then the Encoder's no-thinking ids and readout positions."""
    _src()
    from inference.api import parse_request
    from inference.types import Options
    from loader.dataloader import readout_positions

    try:
        record, _, _ = parse_request({"state": state, "questions": questions}, Options())
    except ValueError as e:
        raise RequestError(str(e)) from e
    rows, meta = [], []
    for q in record.questions:
        ids = enc.encode(enc.prompt_text(record, q)) + list(enc.empty_think) + enc.encode(enc.remainder_text(q))
        if len(ids) > MAX_LEN:
            raise RequestError("question %r: %d tokens; the model reads at most %d" % (q.id, len(ids), MAX_LEN))
        opts, decide = readout_positions(ids, enc.markers)
        rows.append({"ids": ids, "decide": decide, "opts": opts})
        meta.append({"qid": q.id, "type": q.type, "k": len(q.options())})
    return rows, meta


@torch.no_grad()
def forward(model, head, rows):
    """Raw pointer scores (temperature 1), one [k] array per row, one row at a time."""
    device = next(model.parameters()).device
    out = []
    for r in rows:
        h = model.model(torch.tensor([r["ids"]], device=device))[0]
        z = head(h[r["opts"]][None], h[r["decide"]][None])[0]
        out.append(z.double().cpu().numpy())
    return out
