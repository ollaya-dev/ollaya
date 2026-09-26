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


def _laya(sub, description, params, ctx, languages, mlx=False):
    """`mlx`: publish an arch layer (the tag passed its parity gate on Metal)."""
    prefix = sub + "/" if sub else ""
    slug = sub or "en"
    arch = {"family": "laya", "config": prefix + "encoder/config.json",
            "agent_config": prefix + "rl_agent_config.json"} if mlx else None
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
        "arch": arch,
    }


def _wl(slug, repo, commit, description, params, ctx, languages, license=None, license_text=None, wl_dir=None,
        weights=None, arch=None):
    """A model converted under `families/`, whose weightless graph (`out/<slug>-wl`) already
    references the upstream checkpoint by its file name.

    `weights` maps each graph location to its upstream file: a path in `repo`, or a
    `(repo, commit, path)` triple for a file in another repository (a LoRA's base model).
    `arch`: `{"family": ..., "config": <path in repo>}` for an arch layer (MLX; see `arch.py`), only
    for tags that passed their parity gate on Metal."""
    return {
        "kind": "wl",
        "repo": repo,
        "commit": commit,
        "wl_dir": wl_dir or os.path.join(OUT, slug + "-wl"),
        "weights": weights or {"model.safetensors": "model.safetensors"},  # graph location -> upstream file
        "tokenizer": "tokenizer.json",
        "parameter_size": params,
        "context_length": ctx,
        "languages": languages,
        "description": description,
        "license": license,
        "license_text": license_text,
        "arch": arch,
    }


def _kev_weights(base, base_commit, shards):
    """Graph location -> upstream file of a Kev checkpoint: the base model's shards from Qwen's repository, the
    adapter and head.pt from Kev's."""
    files = ["model.safetensors-%05d-of-%05d.safetensors" % (i, shards) for i in range(1, shards + 1)]
    return {**{f: (base, base_commit, f) for f in files},
            "adapter_model.safetensors": "adapter_model.safetensors", "head.pt": "head.pt"}


def _kev_license(repo, base):
    return ("Kev by Jared Palmer (https://huggingface.co/jaredpalmer/%s)\n"
            "LoRA adapter and pointer head: Apache-2.0, per the model card.\n"
            "Base model: %s by the Qwen team (https://huggingface.co/Qwen/%s), Apache-2.0.\n"
            "Licensed under the Apache License, Version 2.0.\n\n" % (repo, base, base)) + LICENSE_APACHE


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
                        "421M", 512, ["en"], mlx=True),
            "multilingual": _laya("multilingual", "Decision model for 100+ languages (mmBERT-base).",
                                  "322M", 1024, ["multilingual"], mlx=True),
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
                arch={"family": "sequence-classification", "config": "config.json"},
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
    "kev": {
        "namespace": "library",
        "model": "kev",
        "family": "kev",
        "author": "Jared Palmer (adapter and pointer head) and the Qwen team (base model)",
        "license": "Apache-2.0",
        "license_text": _kev_license("kev-0.8b", "Qwen3.5-0.8B-Base"),
        "tags": {
            # Round 15 (2026-09-24); round 7 + dates was 54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8.
            "0.8b": _wl("kev-0.8b", "jaredpalmer/kev-0.8b", "9a45d25eb2ab761841196625383fa1dff0e56c1e",
                        "Pointer-head decision model (LoRA on Qwen3.5-0.8B base): each option is scored at its own "
                        "span, all in one forward pass per question.",
                        "0.76B", 8192, ["en"], wl_dir=os.path.join(OUT, "kev-0.8b-r15"),
                        license_text=_kev_license("kev-0.8b", "Qwen3.5-0.8B-Base"),
                        weights=_kev_weights("Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68", 1)),
            # Round 10 (2026-09-24): skills delta on round 8.
            "4b": _wl("kev-4b", "jaredpalmer/kev-4b", "139fdd94f1b6a6ad80cc15e08fcb99cac885a101",
                      "Kev on Qwen3.5-4B base (LoRA and pointer head): more accurate than 0.8b out of domain. "
                      "Needs about 9 GB of memory.",
                      "4.2B", 8192, ["en"], wl_dir=os.path.join(OUT, "kev-4b"),
                      license_text=_kev_license("kev-4b", "Qwen3.5-4B-Base"),
                      weights=_kev_weights("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b", 2)),
            "9b": _wl("kev-9b", "jaredpalmer/kev-9b", "2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c",
                      "Largest Kev (Qwen3.5-9B base, LoRA and pointer head). Needs about 17 GB of memory; best on a "
                      "24 GB GPU.",
                      "7.9B", 8192, ["en"], wl_dir=os.path.join(OUT, "kev-9b"),
                      license_text=_kev_license("kev-9b", "Qwen3.5-9B-Base"),
                      weights=_kev_weights("Qwen/Qwen3.5-9B-Base", "68c46c4b3498877f3ef123c856ecfde50c39f404", 4)),
        },
        "aliases": {"latest": "0.8b"},
        "parity": "Ollaya's Rust runtime matches upstream Kev (PyTorch fp32) exactly on 480 questions from 117 "
                  "requests per checkpoint, and rejects the same 16 requests upstream rejects. The token rows and "
                  "option positions are identical, and so is the decision on every question. Probabilities are "
                  "within 2.8e-6 (0.8b), 3.1e-5 (4b) and 3.8e-6 (9b), on CPU and CUDA, and the TypeSafe answers "
                  "equal upstream's to its 4-decimal rounding.",
    },
    "qwen3guard": {
        "namespace": "library",
        "model": "qwen3guard",
        "family": "qwen3guard",
        "author": "the Qwen team, Alibaba Cloud",
        "license": "Apache-2.0",
        "license_text": "Qwen3Guard-Gen-0.6B by the Qwen team, Alibaba Cloud (https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B)\n"
                        "Licensed under the Apache License, Version 2.0.\n\n" + LICENSE_APACHE,
        "tags": {
            "0.6b": _wl("qwen3guard-gen-0.6b", "Qwen/Qwen3Guard-Gen-0.6B", "fada3b2f655b89601929198343c94cd2f64d93cc",
                        "Safety guard (Qwen3Guard-Gen-0.6B) with built-in questions: safe, controversial or unsafe, "
                        "and the unsafe category.",
                        "0.6B", 32768, ["multilingual"], wl_dir=os.path.join(OUT, "qwen3guard-gen-0.6b")),
        },
        "aliases": {"latest": "0.6b"},
        "parity": "Ollaya's Rust runtime matches the transformers fp32 reference exactly on 74 texts (148 rows, "
                  "296 built-in questions). The token ids are identical, and so is the decision on every "
                  "question. Probabilities are within 1.4e-5, on CPU and CUDA.",
    },
    "von": {
        "namespace": "library",
        "model": "von",
        "family": "von",
        "author": "Victor Hugo Panisa",
        "license": "Apache-2.0",
        "license_text": "Von by Victor Hugo Panisa (https://huggingface.co/wfzyx/von)\n"
                        "Licensed under the Apache License, Version 2.0.\n\n" + LICENSE_APACHE,
        "tags": {
            # Von 1.1. The weights exist only in option_marker.pt, a torch.save zip whose tensors are
            # stored uncompressed: the graph reads them in place by byte offset, nothing is unpickled.
            "1.1": _wl("von", "wfzyx/von", "d8bb5e0745d8ee1fb65d536d6d4892d54d5a93fd",
                       "Von 1.1 (ModernBERT-large): every option is scored at its own [MASK] marker, "
                       "all options of a question in one pass.",
                       "395M", 8192, ["en"], weights={"option_marker.pt": "option_marker.pt"}),
        },
        "aliases": {"latest": "1.1"},
        "parity": "Ollaya's Rust runtime matches upstream Von, run in float64, on 485 questions (653 rows). The "
                  "token ids and marker positions are identical, and so is the decision on every question. Logits "
                  "are within 4.4e-4 and probabilities within 4.7e-5 on x86-64 CPU and CUDA; on Apple silicon's CPU "
                  "one of the 653 rows is 1.1e-3 off, and every decision is still the same.",
    },
}
