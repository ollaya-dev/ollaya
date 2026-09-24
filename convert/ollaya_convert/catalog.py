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


def _wl(slug, repo, commit, description, params, ctx, languages, license=None, license_text=None, wl_dir=None):
    """A model converted under `families/`, whose weightless graph (`out/<slug>-wl`) already
    references the upstream checkpoint by its file name."""
    return {
        "kind": "wl",
        "repo": repo,
        "commit": commit,
        "wl_dir": wl_dir or os.path.join(OUT, slug + "-wl"),
        "weights": {"model.safetensors": "model.safetensors"},  # graph location -> upstream path
        "tokenizer": "tokenizer.json",
        "parameter_size": params,
        "context_length": ctx,
        "languages": languages,
        "description": description,
        "license": license,
        "license_text": license_text,
    }


LICENSE_MIT_NLI = ("DeBERTa-v3-large zero-shot v2.0 by Moritz Laurer "
                   "(https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0), MIT License.\n"
                   "Note from the model card: part of the training data carries non-commercial licenses.\n")

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
    "nli": {
        "namespace": "library",
        "model": "nli",
        "family": "nli",
        "author": "Moritz Laurer",
        "license": "MIT",
        "license_text": LICENSE_MIT_NLI,
        "tags": {
            "deberta-v3-large": _wl(
                "deberta-v3-large-zeroshot-v2.0", "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
                "cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2",
                "Zero-shot NLI classifier (DeBERTa-v3-large): each option is a hypothesis scored for entailment.",
                "435M", 512, ["en"]),
            "modernbert-large": _wl(
                "modernbert-large-zeroshot-v2.0", "MoritzLaurer/ModernBERT-large-zeroshot-v2.0",
                "a51e07b524299e309dd2b88d48b0cfa2bd9ec598",
                "Zero-shot NLI classifier (ModernBERT-large, Apache-2.0): faster, slightly less accurate.",
                "396M", 512, ["en"], license="Apache-2.0",
                license_text="ModernBERT-large zero-shot v2.0 by Moritz Laurer "
                             "(https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0)\n"
                             "Licensed under the Apache License, Version 2.0.\n\n" + LICENSE_APACHE),
        },
        "aliases": {"latest": "deberta-v3-large"},
        "parity": "Ollaya's Rust runtime matches the Python reference exactly on 483 questions "
                  "(1,513 hypothesis rows) per model. The token ids are identical, and so is the "
                  "decision on every question. Probabilities are within 2.5e-5, on CPU and CUDA.",
    },
    "gliclass": {
        "namespace": "library",
        "model": "gliclass",
        "family": "gliclass",
        "author": "Knowledgator",
        "license": "Apache-2.0",
        "license_text": "GLiClass instruct large v1.0 by Knowledgator "
                        "(https://huggingface.co/knowledgator/gliclass-instruct-large-v1.0)\n"
                        "Licensed under the Apache License, Version 2.0.\n\n" + LICENSE_APACHE,
        "tags": {
            "large": _wl(
                "gliclass-instruct-large", "knowledgator/gliclass-instruct-large-v1.0",
                "825e5478c1bf4bffbf297690517097ccbdb2e006",
                "Instruction-following zero-shot classifier: all options scored in one pass.",
                "439M", 1024, ["en"]),
        },
        "aliases": {"latest": "large"},
        "parity": "Ollaya's Rust runtime matches the Python reference exactly on 482 questions. "
                  "The token ids and label positions are identical, and so is the decision on every "
                  "question. Probabilities are within 1e-5, on CPU and CUDA.",
    },
    "decider": {
        "namespace": "library",
        "model": "decider",
        "family": "decider",
        "author": "Mapika",
        "license": "Apache-2.0",
        "license_text": "decider by Mapika (https://huggingface.co/Mapika/decider-2b)\n"
                        "Licensed under the Apache License, Version 2.0.\n\n" + LICENSE_APACHE,
        "tags": {
            "2b": _wl("decider-2b", "Mapika/decider-2b", "9839cc9d908be16c5988c0d041034b5fdf82c7a2",
                      "Decoder decision model (Qwen3.5-2B base): reads option-letter logits at an answer slot. "
                      "Highest accuracy of the open models Ollaya ships.",
                      "1.9B", 32768, ["en"], wl_dir=os.path.join(OUT, "decider-2b")),
            "0.8b": _wl("decider-0.8b", "Mapika/decider-0.8b", "a0a01d6f8135298f400a8c856b355793012ae971",
                        "Smaller decider (Qwen3.5-0.8B base): faster, less accurate.",
                        "0.75B", 32768, ["en"], wl_dir=os.path.join(OUT, "decider-0.8b")),
        },
        "aliases": {"latest": "2b"},
        "parity": "Ollaya's Rust runtime matches the Python reference exactly on 479 questions (902 rows) "
                  "per model. The token ids and answer-slot positions are identical, and so is the "
                  "decision on every question. Probabilities are within 6.3e-6, on CPU and CUDA.",
    },
}
