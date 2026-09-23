"""What the public library ships: one entry per model, one variant per tag.

Every variant pins an upstream Hugging Face repository to a commit. `checkpoint` is a local copy
of that commit's weights, used only to verify and rewrite the exported graphs; `package.py`
refuses to run if its sha256 differs from the upstream file.
"""
import os

from .laya_ref import DEFAULT_ROOT

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
LICENSE_APACHE = open(os.path.join(os.path.dirname(__file__), "..", "..", "LICENSE")).read()

LAYA_REPO = "convaiinnovations/laya"
LAYA_COMMIT = "aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8"


def _laya(sub, description, params, ctx, languages):
    prefix = sub + "/" if sub else ""
    slug = sub or "en"
    return {
        "repo": LAYA_REPO,
        "commit": LAYA_COMMIT,
        "weights": prefix + "model.safetensors",
        "tokenizer": prefix + "tokenizer/tokenizer.json",
        "checkpoint": os.path.join(DEFAULT_ROOT, prefix, "model.safetensors"),
        "prefix": "model.",
        "exports": {
            "fp32": os.path.join(OUT, "laya-" + slug),
            "fp16": os.path.join(OUT, "laya-%s-fp16" % slug),
        },
        "parameter_size": params,
        "context_length": ctx,
        "languages": languages,
        "description": description,
    }


CATALOG = {
    "laya": {
        "namespace": "library",
        "model": "laya",
        "family": "laya",
        "author": "Convai Innovations",
        "license": "Apache-2.0",
        "license_text": "Laya by Convai Innovations (https://huggingface.co/convaiinnovations/laya)\n"
                        "Licensed under the Apache License, Version 2.0.\n\n" + LICENSE_APACHE,
        "tags": {
            "en": _laya("", "English decision model (ModernBERT-large): guardrails, email and ticket triage.",
                        "421M", 512, ["en"]),
            "multilingual": _laya("multilingual", "Decision model for 100+ languages (mmBERT-base).",
                                  "322M", 1024, ["multilingual"]),
            "typed-decisions": _laya("typed-decisions",
                                     "Fine-tuned on the typed-decisions workflows (0.766 accuracy).",
                                     "421M", 1024, ["en"]),
        },
        "routers": {
            "latest": {
                "description": "Routes each request to laya:en or laya:multilingual by the text's script and language.",
                "languages": ["en", "multilingual"],
                "router": {"strategy": "script", "default": "english",
                           "routes": {"english": "laya:en", "multilingual": "laya:multilingual"}},
            },
        },
    },
}
