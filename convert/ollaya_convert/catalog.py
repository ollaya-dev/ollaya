"""What the public library ships: one entry per model, one variant per tag.

Every variant pins an upstream Hugging Face repository to a commit. `checkpoint` is a local copy
of that commit's weights, used only to verify and rewrite the exported graphs; `package.py`
refuses to run if its sha256 differs from the upstream file.
"""
import os

from .model_paths import DEFAULT_ROOT

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
LICENSE_APACHE = open(os.path.join(os.path.dirname(__file__), "..", "..", "LICENSE"), encoding="utf-8").read()

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
            # fp32: the opset-23 export (fused Attention, CPU); fp16: derived from the opset-20 one.
            "fp32": os.path.join(OUT, "laya-%s-o23" % slug),
            "fp16": os.path.join(OUT, "laya-%s-fp16" % slug),
        },
        "parameter_size": params,
        "context_length": ctx,
        "languages": languages,
        "description": description,
        "arch": arch,
    }


def _gguf(export, repo, commit, gguf, description, params, languages, notice=None):
    """A GGUF model run by llama.cpp: the author's GGUF, unmodified, plus the decision.json and
    calibration.json that `llm_common/export_llama.py` derived from it (`out/<export>`)."""
    return {
        "kind": "gguf",
        "repo": repo,
        "commit": commit,
        "gguf": gguf,
        "export_dir": os.path.join(OUT, export),
        "parameter_size": params,
        "languages": languages,
        "description": description,
        "notice": notice,
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


def _decima(slug, commit, description, params, ctx):
    """A Decima tag: the encoder and the head from the repository's pytorch/ checkpoint, and its tokenizer."""
    return dict(_wl(slug, "amyrmahdy/" + slug, commit, description, params, ctx, ["multilingual"],
                    wl_dir=os.path.join(OUT, slug),
                    weights={"model.safetensors": "pytorch/encoder/model.safetensors",
                             "head.safetensors": "pytorch/head.safetensors"}),
                tokenizer="pytorch/encoder/tokenizer.json")


def _kev_weights(base, base_commit, shards):
    """Graph location -> upstream file of a Kev checkpoint: the base model's shards from Qwen's repository, the
    adapter and head.pt from Kev's."""
    files = ["model.safetensors-%05d-of-%05d.safetensors" % (i, shards) for i in range(1, shards + 1)]
    return {**{f: (base, base_commit, f) for f in files},
            "adapter_model.safetensors": "adapter_model.safetensors", "head.pt": "head.pt"}


def _arbiter_weights(base, base_commit, shards):
    """Graph location -> upstream file of an Arbiter checkpoint: the base model's shards from the
    Gemma 3 repository, the adapter and head.pt from the Arbiter repository."""
    files = ["model-%05d-of-%05d.safetensors" % (i, shards) for i in range(1, shards + 1)]
    return {**{f: (base, base_commit, f) for f in files},
            "adapter_model.safetensors": "adapter_model.safetensors",
            "head.pt": "head.pt"}


ARBITER_LICENSE = "Apache-2.0 (LoRA adapter and head) and the Gemma Terms of Use (Gemma 3 base model)"


def _arbiter_license(repo, base):
    return ("Arbiter by Codekins Pvt Ltd · Zyot Lab (https://huggingface.co/%s)\n"
            "LoRA adapter and 24-slot pointer head: Apache-2.0.\n"
            "Base model: %s by Google DeepMind (https://huggingface.co/google/%s), under the Gemma Terms of Use "
            "(https://ai.google.dev/gemma/terms) and the Gemma Prohibited Use Policy "
            "(https://ai.google.dev/gemma/prohibited_use_policy).\n"
            "Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms.\n"
            "The adapter and the head are licensed under the Apache License, Version 2.0:\n\n"
            % (repo, base, base)) + LICENSE_APACHE


def _kev_license(repo, base):
    return ("Kev by Jared Palmer (https://huggingface.co/jaredpalmer/%s)\n"
            "LoRA adapter and pointer head: Apache-2.0, per the model card.\n"
            "Base model: %s by the Qwen team (https://huggingface.co/Qwen/%s), Apache-2.0.\n"
            "Licensed under the Apache License, Version 2.0.\n\n" % (repo, base, base)) + LICENSE_APACHE


LICENSE_MIT_NLI = ("DeBERTa-v3-large zero-shot v2.0 by Moritz Laurer "
                   "(https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0), MIT License.\n"
                   "Note from the model card: part of the training data carries non-commercial licenses.\n")

CATALOG = {
    "d1": {
        "namespace": "library", "model": "d1", "family": "d1", "author": "Liquid AI",
        "license": "LFM Open License v1.0",
        "license_text": "d1-3B by Liquid AI (https://huggingface.co/LiquidAI/d1-3B).\n\n" +
            open(os.path.join(os.path.dirname(__file__), "families", "d1", "LICENSE"), encoding="utf-8").read(),
        "tags": {
            "3b": _gguf("d1-3b-q8_0", "LiquidAI/d1-3B-GGUF",
                "bb1e436ea78eb96a3f1acb6da865f70c2fbeb563", "d1-3B-Q8_0.gguf",
                "LiquidAI d1-3B: typed decisions with max-pooled answer forms, Q8_0.", "3B", ["multilingual"]),
            "3b-vision": dict(_gguf("d1-3b-q8_0", "LiquidAI/d1-3B-GGUF",
                "bb1e436ea78eb96a3f1acb6da865f70c2fbeb563", "d1-3B-Q8_0.gguf",
                "LiquidAI d1-3B with its matching Q8_0 image projector.", "3B", ["multilingual"]),
                mmproj="mmproj-d1-3B-Q8_0.gguf"),
        },
        "aliases": {"latest": "3b"},
    },
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
            # v2.1: temperature_by_type (choice 1.110, noul 1.560, score 1.287) in decision.json/calibration.json.
            "4b": _wl("decider-4b", "Mapika/decider-4b", "eb5fbdfc9448473ec25e399882912863afbdb70e",
                      "Largest decider (Qwen3.5-4B base, v2.1): reads option-letter logits at an answer slot, with one "
                      "fitted temperature per answer type. Needs about 9 GB of memory.",
                      "4.2B", 32768, ["en"], wl_dir=os.path.join(OUT, "decider-4b")),
            "2b": _wl("decider-2b", "Mapika/decider-2b", "9839cc9d908be16c5988c0d041034b5fdf82c7a2",
                      "Decoder decision model (Qwen3.5-2B base): reads option-letter logits at an answer slot. "
                      "Highest accuracy of the open models Ollaya ships.",
                      "1.9B", 32768, ["en"], wl_dir=os.path.join(OUT, "decider-2b")),
            # v5 text weights in the Qwen3.5-2B vision-language model: `decider-vision-v1`, two graphs
            # (vision.onnx, the image tower; model.onnx, the decoder with M-RoPE positions).
            "2b-vision": _wl("decider-2b-vision", "Mapika/decider-2b-vision", "863e290863655f1d6b69324d77d09ac972d21609",
                             "Vision decider (Qwen3.5-2B vision-language, v5 text weights): answers questions about an "
                             "image and a text state, reading option-letter logits at an answer slot. One PNG image per "
                             "request (`images` in /api/decide, `--image` in the CLI); up to 10 options per question.",
                             "2.2B", 32768, ["en"], wl_dir=os.path.join(OUT, "decider-2b-vision")),
            "0.8b": _wl("decider-0.8b", "Mapika/decider-0.8b", "a0a01d6f8135298f400a8c856b355793012ae971",
                        "Smaller decider (Qwen3.5-0.8B base): faster, less accurate.",
                        "0.75B", 32768, ["en"], wl_dir=os.path.join(OUT, "decider-0.8b")),
        },
        "aliases": {"latest": "2b"},
        "parity": "Ollaya's Rust runtime matches the Python reference exactly on 479 questions (902 rows) "
                  "per model. The token ids and answer-slot positions are identical, and so is the "
                  "decision on every question. Probabilities are within 6.3e-6, on CPU and CUDA.",
    },
    "clm": {
        "namespace": "library",
        "model": "clm",
        "family": "clm",
        "author": "Contrastive-LM (projection heads) and the Qwen team (Qwen3-8B encoder)",
        "license": "Apache-2.0",
        "license_text": ("CLM-v0.1-8B by Contrastive-LM (https://huggingface.co/Contrastive-LM/CLM-v0.1-8B), "
                         "state and action projection heads: Apache-2.0.\n"
                         "Encoder: Qwen3-8B by the Qwen team (https://huggingface.co/Qwen/Qwen3-8B), Apache-2.0.\n"
                         "Licensed under the Apache License, Version 2.0.\n\n") + LICENSE_APACHE,
        "tags": {
            "8b": dict(_wl("clm-8b", "Contrastive-LM/CLM-v0.1-8B", "e939398d4556fcd9400c76fa8c5a513202f42b0a",
                           "Contrastive decision model: the Qwen3-8B encoder embeds the state and each option, and "
                           "two small heads score every option by its similarity to the state. Repeated options "
                           "and questions are cached.",
                           "8.2B", 2048, ["en"], wl_dir=os.path.join(OUT, "clm-8b"),
                           weights={**{"model-%05d-of-00005.safetensors" % i:
                                       ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218",
                                        "model-%05d-of-00005.safetensors" % i) for i in range(1, 6)},
                                    "CLM_v0.1-8B.pt": "CLM_v0.1-8B.pt"}),
                       tokenizer=("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", "tokenizer.json")),
        },
        "aliases": {"latest": "8b"},
        "parity": "Ollaya's Rust runtime matches the reference (upstream clm.schema and heads, the Qwen3-8B "
                  "encoder in fp32 on its BF16 weights) on 480 questions from 117 requests, on CPU and CUDA: the "
                  "same state and option texts and token ids, the same rejections, the same decision on every "
                  "question, option logits within 1.3e-4 and probabilities within 2.9e-5.",
    },
    "nimble": {
        "namespace": "library",
        "model": "nimble",
        "family": "nimble",
        "author": "Bespoke Labs (adapter) and the Qwen team (base model)",
        "license": "Apache-2.0",
        "license_text": ("Bespoke-Nimble-9B-v2 by Bespoke Labs (https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2)\n"
                         "LoRA adapter: Apache-2.0.\n"
                         "Base model: Qwen3.5-9B by the Qwen team (https://huggingface.co/Qwen/Qwen3.5-9B), Apache-2.0.\n"
                         "Licensed under the Apache License, Version 2.0.\n\n") + LICENSE_APACHE,
        "tags": {
            "9b": _wl("nimble-9b-v2", "bespokelabs/Bespoke-Nimble-9B-v2", "4b8c04d1ac2cea3e41e5e3c4d2130bcead2c0abe",
                      "Bespoke Labs' Nimble v2: a LoRA on Qwen3.5-9B that answers every question from the next-token "
                      "logits of its option codes, calibrated with the author's temperature. Up to 255 options. "
                      "Needs about 18 GB of memory; best on a 24 GB GPU.",
                      "9B", 8192, ["en"], wl_dir=os.path.join(OUT, "nimble-9b-v2"),
                      weights={**{"model.safetensors-%05d-of-00004.safetensors" % i:
                                  ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
                                   "model.safetensors-%05d-of-00004.safetensors" % i) for i in range(1, 5)},
                               "adapter_model.safetensors": "adapter_model.safetensors"}),
        },
        "aliases": {"latest": "9b"},
        "parity": "Ollaya's Rust runtime matches the author's reference code (serving_schema.prepare_prompts and "
                  "inference.candidate_logits, PyTorch fp32 with the LoRA unmerged) on 492 questions from 104 "
                  "requests: identical token rows, the same 4 rejected requests, the same decision on every "
                  "question, option logits within 1.1e-4 and probabilities within 6.5e-6 (CUDA).",
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
        "aliases": {"latest": "4b"},
        "parity": "Ollaya's Rust runtime matches upstream Kev (PyTorch fp32) exactly on 480 questions from 117 "
                  "requests per checkpoint, and rejects the same 16 requests upstream rejects. The token rows and "
                  "option positions are identical, and so is the decision on every question. Probabilities are "
                  "within 2.8e-6 (0.8b), 3.1e-5 (4b) and 3.8e-6 (9b), on CPU and CUDA, and the TypeSafe answers "
                  "equal upstream's to its 4-decimal rounding.",
    },
    "arbiter": {
        "namespace": "library",
        "model": "arbiter",
        "family": "arbiter",
        "author": "Codekins Pvt Ltd · Zyot Lab",
        "license": ARBITER_LICENSE,
        # The Hugging Face card's `license: other` (publish_hf).
        "hf_license": ("apache-2.0-and-gemma", "https://ai.google.dev/gemma/terms"),
        "license_text": _arbiter_license("hiteshluke/arbiter-4b", "gemma-3-4b-it"),
        "tags": {
            # The tokenizer is the base's: the arbiter repository's tokenizer.json has truncation (255) on.
            "4b": dict(_wl("arbiter-4b", "hiteshluke/arbiter-4b", "0c44271c59f89758e3cae17b032e98a9140093e9",
                           "Arbiter v3.3: a LoRA and a fixed 24-slot head on Gemma 3 4B, one forward pass per "
                           "question. Answers noul, choices of up to 16 options and scores of exactly 6 levels.",
                           "4.3B", 8192, ["multilingual"], license=ARBITER_LICENSE,
                           wl_dir=os.path.join(OUT, "arbiter-4b"),
                           license_text=_arbiter_license("hiteshluke/arbiter-4b", "gemma-3-4b-it"),
                           weights=_arbiter_weights("unsloth/gemma-3-4b-it",
                                                    "bf46152c47f5dd20b896357cb51abc4c03b8ee8c", 2)),
                       tokenizer=("unsloth/gemma-3-4b-it", "bf46152c47f5dd20b896357cb51abc4c03b8ee8c",
                                  "tokenizer.json")),
        },
        "aliases": {"latest": "4b"},
        "parity": "Ollaya's Rust runtime matches the reference (transformers' Gemma 3 with the authors' LoRA and head, "
                  "fp32, the training script's prompts) on 420 questions from 127 requests, on CPU and CUDA (RTX "
                  "4090): identical token rows, the same 121 rejected requests, the same decision on every question, "
                  "slot scores within 8.5e-5 and probabilities within 1.0e-5.",
    },
    "decision": {
        "namespace": "library",
        "model": "decision",
        "family": "decision",
        "author": "the vLLM Semantic Router contributors",
        "license": "Apache-2.0",
        "license_text": "Decision 1.0 by the vLLM Semantic Router contributors "
                        "(https://huggingface.co/llm-semantic-router)\n"
                        "A fine-tune of Qwen3.5 by the Qwen team (https://huggingface.co/Qwen), Apache-2.0.\n"
                        "Licensed under the Apache License, Version 2.0.\n\n" + LICENSE_APACHE,
        "tags": {
            # The backbone and the head are read in place from the author's safetensors.
            "eos": _wl("decision-eos-0.8b", "llm-semantic-router/Decision-1.0-Eos-0.8B",
                       "3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd",
                       "Decision 1.0 Eos (fully fine-tuned Qwen3.5-0.8B) with an endpoint head: every option is "
                       "scored at its own last token against the question, in one forward pass per question.",
                       "0.75B", 16384, ["en", "zh"], wl_dir=os.path.join(OUT, "decision-eos-0.8b"),
                       weights={"model.safetensors": "backbone/model.safetensors",
                                "decision_head.safetensors": "decision_head.safetensors"}),
        },
        "aliases": {"latest": "eos"},
        "parity": "Ollaya's Rust runtime matches the author's code (PyTorch fp32) exactly on 466 questions from 117 "
                  "requests, and rejects the same 17 requests the author rejects. The token rows and option "
                  "positions are identical, and so is the decision on every question. Probabilities are within "
                  "3.2e-6, on CPU and CUDA, and the TypeSafe answers equal the author's up to 4-decimal rounding.",
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
    "credence": {
        "namespace": "library",
        "model": "credence",
        "family": "winnow",
        "author": "Txoka",
        "license": "Apache-2.0",
        "license_text": "Credence v1 by Txoka, a MiCA refinement of EldanRing's Winnow-E4B "
                        "on Google DeepMind's Gemma 4 E4B. Preserve upstream notices and "
                        "individual training-data attribution.\n\n" + LICENSE_APACHE,
        "tags": {
            "e4b": _gguf("winnow-credence-e4b", "Txoka/Credence-v1-Gemma4-E4B",
                         "7d5ffc84145f34a3a76f7074e8c70bf1c1efce56", "accuracy/model-Q8_0.gguf",
                         "Credence v1 Gemma4 E4B, Q8_0, accuracy-focused MiCA checkpoint; folded calibration, "
                         "external temperature 1. Typed calibration is worse than original Winnow.",
                         "7.5B", ["multilingual"], notice="accuracy/NOTICE"),
            "e4b-calibrated": _gguf("winnow-credence-e4b-calibrated", "Txoka/Credence-v1-Gemma4-E4B",
                         "7d5ffc84145f34a3a76f7074e8c70bf1c1efce56", "calibrated/model-Q8_0.gguf",
                         "Credence v1 Gemma4 E4B, Q8_0, separate zero-synthetic checkpoint with "
                         "validation-fitted temperature; stronger public calibration, lower public accuracy.",
                         "7.5B", ["multilingual"], notice="calibrated/NOTICE"),
            "e4b-vision": dict(_gguf("winnow-credence-e4b", "Txoka/Credence-v1-Gemma4-E4B",
                         "7d5ffc84145f34a3a76f7074e8c70bf1c1efce56", "accuracy/model-Q8_0.gguf",
                         "Credence v1 Gemma4 E4B accuracy-focused, Q8_0 with Winnow-E4B's unchanged vision projector: PNG image decisions through libmtmd.",
                         "7.5B", ["multilingual"], notice="accuracy/NOTICE"),
                         mmproj=("EldanRing/Winnow-E4B", "734302fe5fbfeb3f21a7ece62653c9539be4aaf3",
                                 "gguf/mmproj-Winnow-E4B.gguf")),
            "e4b-calibrated-vision": dict(_gguf("winnow-credence-e4b-calibrated", "Txoka/Credence-v1-Gemma4-E4B",
                         "7d5ffc84145f34a3a76f7074e8c70bf1c1efce56", "calibrated/model-Q8_0.gguf",
                         "Credence v1 Gemma4 E4B validation-calibrated, Q8_0 with Winnow-E4B's unchanged vision projector: PNG image decisions through libmtmd.",
                         "7.5B", ["multilingual"], notice="calibrated/NOTICE"),
                         mmproj=("EldanRing/Winnow-E4B", "734302fe5fbfeb3f21a7ece62653c9539be4aaf3",
                                 "gguf/mmproj-Winnow-E4B.gguf")),
        },
        "aliases": {"latest": "e4b"},
        "parity": "Ollaya's runner matches stock llama-server of the pinned build (b11146) on the same GGUF and the "
                  "same device: 505/505 decisions for both checkpoints, option logits within 7.6e-6, on the x86-64 CPU "
                  "and an RTX 4070 (the contributor) and an RTX 4090 (Ollaya); the vision tags also match it on images.",
    },
    "winnow": {
        "namespace": "library",
        "model": "winnow",
        "family": "winnow",
        "author": "EldanRing",
        "license": "Apache-2.0",
        "license_text": "Winnow by EldanRing (https://huggingface.co/EldanRing), a fine-tune of Google DeepMind's "
                        "Gemma 4.\nLicensed under the Apache License, Version 2.0.\n\n" + LICENSE_APACHE,
        "tags": {
            # Q8_0, the quantization the author measured and recommends. The GGUF holds the weights,
            # tokenizer and chat template; llama.cpp loads it as the author published it.
            "12b": _gguf("winnow-12b-q8_0", "EldanRing/Winnow-12B", "b6ac22b0d51b69b18200acacb3fbdd98073fffe8",
                         "gguf/Winnow-12B-Q8_0.gguf",
                         "Winnow-12B (Gemma 4 12B IT fine-tune), Q8_0 GGUF on llama.cpp: option-label logits "
                         "after Winnow's own prompt.",
                         "12B", ["multilingual"], notice="NOTICE"),
            "e4b": _gguf("winnow-e4b-q8_0", "EldanRing/Winnow-E4B", "734302fe5fbfeb3f21a7ece62653c9539be4aaf3",
                         "gguf/Winnow-E4B-Q8_0.gguf",
                         "Winnow-E4B (Gemma 4 E4B IT fine-tune), Q8_0 GGUF on llama.cpp: option-label logits "
                         "after Winnow's own prompt, with the author's fitted temperature.",
                         "7.5B", ["multilingual"], notice="NOTICE"),
            # Opt-in images: each text tag's GGUF, decision and calibration plus its matching projector from the
            # same revision, read through libmtmd. The text tags keep their downloads and inference path.
            "12b-vision": dict(_gguf("winnow-12b-q8_0", "EldanRing/Winnow-12B", "b6ac22b0d51b69b18200acacb3fbdd98073fffe8",
                                     "gguf/Winnow-12B-Q8_0.gguf",
                                     "Winnow 12B Q8_0 with the author's matching vision projector: PNG image decisions "
                                     "through libmtmd.",
                                     "12B", ["multilingual"], notice="NOTICE"),
                               mmproj="gguf/mmproj-Winnow-12B.gguf"),
            "e4b-vision": dict(_gguf("winnow-e4b-q8_0", "EldanRing/Winnow-E4B", "734302fe5fbfeb3f21a7ece62653c9539be4aaf3",
                                     "gguf/Winnow-E4B-Q8_0.gguf",
                                     "Winnow E4B Q8_0 with the author's matching vision projector: PNG image decisions "
                                     "through libmtmd.",
                                     "7.5B", ["multilingual"], notice="NOTICE"),
                               mmproj="gguf/mmproj-Winnow-E4B.gguf"),
        },
        "aliases": {"latest": "12b"},
        "parity": "Ollaya's runner matches stock llama-server of the pinned build (b11146) on the same GGUF and the "
                  "same device: 505 text questions per model, every decision the same, option logits within 1.3e-5 "
                  "and probabilities within 3.0e-6, on CUDA (RTX 4090, Linux and Windows) and, for e4b, the x86-64 "
                  "CPU. With the vision projector, 65 image questions per model, every decision the same, option "
                  "logits within 1.2e-5 (RTX 4070, RTX 5090 and the x86-64 CPU).",
    },
    "cygnet": {
        "namespace": "library",
        "model": "cygnet",
        "family": "cygnet",
        "author": "blockbrain-ai (recipe) and Google DeepMind (Gemma 4)",
        "license": "Apache-2.0",
        "license_text": "Cygnet by blockbrain-ai (https://github.com/blockbrain-ai/cygnet-recipe), a prompt and "
                        "calibration for Gemma 4 12B IT by Google DeepMind, MIT. The weights are "
                        "google/gemma-4-12B-it (Apache-2.0), as ggml-org's GGUF conversion of the revision Cygnet pins.\n"
                        "Licensed under the Apache License, Version 2.0.\n\n" + LICENSE_APACHE,
        "tags": {
            # ggml-org's Q8_0 conversion of google/gemma-4-12B-it@707f0a3b (its .src_sha), the revision
            # Cygnet's published runs pin. Cygnet itself is a prompt and one temperature, no weights.
            "12b": _gguf("cygnet-12b-q8_0", "ggml-org/gemma-4-12B-it-GGUF", "e3e681731089efaa3f0917336944ac64752db8ba",
                         "gemma-4-12B-it-Q8_0.gguf",
                         "Cygnet: frozen Gemma 4 12B IT (Q8_0 GGUF) reading one option letter after Cygnet's prompt, "
                         "with its calibration temperature 3.4. Up to 20 options.",
                         "12B", ["multilingual"]),
        },
        "aliases": {"latest": "12b"},
        "parity": "Ollaya's runner matches stock llama-server of the pinned build (b11146) on the same GGUF, CUDA "
                  "(RTX 4090): 502 questions, every decision the same, option logits within 7.7e-6 and probabilities "
                  "within 4.1e-7. The user messages are identical to Cygnet's own shim on 1,364 test prompts.",
    },
    "snap": {
        "namespace": "library",
        "model": "snap",
        "family": "snap",
        "author": "logitlab (snap1-2b, with emnlmn's snap) and OpenBMB (MiniCPM5)",
        "license": "Apache-2.0",
        "license_text": "snap1-2b by logitlab (https://huggingface.co/logitlab/snap1-2b-GGUF), openbmb/MiniCPM5-2B "
                        "(Apache-2.0) fine-tuned with a LoRA and merged, Apache-2.0. Its prompt is snap's "
                        "(https://github.com/emnlmn/snap, MIT), ported to Ollaya's runtime.\n"
                        "Licensed under the Apache License, Version 2.0.\n\n" + LICENSE_APACHE,
        "tags": {
            # Q8_0 of snap1-2b at the GGUF repo's pinned commit; snap's own default is Q4_K_M, which the
            # author measured the same on typed-decisions (0.654 against 0.655).
            "2b": _gguf("snap-v1-snap1-2b-q8_0", "logitlab/snap1-2b-GGUF", "39321915452e54428e18558ba882c771bc856110",
                        "snap1-2b-q8_0.gguf",
                        "snap1-2b (MiniCPM5-2B with a merged LoRA), Q8_0 GGUF on llama.cpp: the option letters' "
                        "logits after snap's own prompt, raw probabilities. Up to 26 options.",
                        "2B", ["en", "it"]),
            # snap's own default file, from the same commit: 1.56 GB, measured with its own goldens.
            "2b-q4_k_m": _gguf("snap-v1-snap1-2b-q4_k_m", "logitlab/snap1-2b-GGUF",
                               "39321915452e54428e18558ba882c771bc856110", "snap1-2b-q4_k_m.gguf",
                               "snap1-2b (MiniCPM5-2B with a merged LoRA), Q4_K_M GGUF, snap's own default file: "
                               "the same prompt and readout as snap:2b in 1.6 GB.",
                               "2B", ["en", "it"]),
        },
        "aliases": {"latest": "2b"},
        "parity": "Ollaya's runner matches stock llama-server of the pinned build (b11146) on the same GGUF, CPU "
                  "and CUDA (RTX 5090 for Q8_0, RTX 4090 for Q4_K_M): 573 questions per file, every decision the "
                  "same, option logits within 7.4e-6 and probabilities within 1.9e-6. The prompt token ids are "
                  "snap's own, from its export-prompts.",
    },
    "jeeves": {
        "namespace": "library",
        "model": "jeeves",
        "family": "jeeves",
        "author": "PostHog (fused weights and pointer head) and the Qwen team (base model)",
        "license": "Apache-2.0",
        "license_text": ("Jeeves-9B by PostHog (https://huggingface.co/PostHog/jeeves, https://github.com/PostHog/jeeves): "
                         "Qwen3.5-9B with its LoRA merged, and a pointer head, Apache-2.0.\n"
                         "Base model: Qwen3.5-9B by the Qwen team, Apache-2.0.\n"
                         "Licensed under the Apache License, Version 2.0.\n\n") + LICENSE_APACHE,
        "tags": {
            # No-thinking mode: the pointer head reads the option markers after an empty thought.
            "9b": dict(_wl("jeeves-9b", "PostHog/jeeves", "8622b7d1652a9dcb8629486b84dce9e8d690c5cd",
                           "PostHog's Jeeves-9B without its reasoning chain: Qwen3.5-9B (LoRA merged) and a pointer head "
                           "that scores every option at its own marker, calibrated. Needs about 18 GB of memory.",
                           "9B", 8192, ["en"], wl_dir=os.path.join(OUT, "jeeves-9b"),
                           weights={**{"model-%05d-of-00005.safetensors" % i: "model-%05d-of-00005.safetensors" % i
                                       for i in range(1, 6)}, "head.pt": "head.pt"}),
                       tokenizer=("jaredpalmer/kev-9b", "2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c", "tokenizer.json")),
        },
        "aliases": {"latest": "9b"},
        "parity": "Ollaya's Rust runtime matches the authors' own code (their Qwen3.5 model and pointer head, fp32, no "
                  "thinking) on 430 questions from 107 requests, on CUDA: identical token rows and option positions, the "
                  "same 16 rejected requests, the same decision on every question, scores within 1.9e-4 and "
                  "probabilities within 1.4e-5.",
    },
    "clef": {
        "namespace": "library",
        "model": "clef",
        "family": "clef",
        "author": "Cloudflare (post-trained model and joint schema head) and the Qwen team (base model)",
        "license": "Apache-2.0",
        "license_text": ("Clef-Flash by Cloudflare (https://huggingface.co/Cloudflare/clef-flash): Qwen3.5-9B, fully "
                         "post-trained, with a joint schema head, Apache-2.0.\n"
                         "Base model: Qwen3.5-9B by the Qwen team, Apache-2.0.\n"
                         "Licensed under the Apache License, Version 2.0.\n\n") + LICENSE_APACHE,
        "tags": {
            # Text only: the vision tower (in the last shard) is not exported. Clef (27B) does not fit the GPUs
            # the parity gate runs on, so only Clef-Flash is converted.
            "flash": dict(_wl("clef-flash", "Cloudflare/clef-flash", "17f0b0ad64efb65d273590632833508766b2aae6",
                              "Cloudflare's Clef-Flash: Qwen3.5-9B post-trained with a joint schema head that scores "
                              "every option of every question together, in one forward pass per request. Needs about "
                              "19 GB of memory.",
                              "9B", 4096, ["en"], wl_dir=os.path.join(OUT, "clef-flash"),
                              weights={**{"model-%05d-of-00004.safetensors" % i: "model-%05d-of-00004.safetensors" % i
                                          for i in range(1, 5)},
                                       "joint_head.safetensors": "joint_head.safetensors"})),
        },
        "aliases": {"latest": "flash"},
        "parity": "Ollaya's Rust runtime matches the authors' own code (joint_schema_model.py: their encoder, Qwen3.5 "
                  "model and joint schema head, fp32) on 571 questions from 131 requests, on CUDA: identical token ids "
                  "and spans, the same 13 rejected requests, the same decision on every question, logits within "
                  "4.3e-5 and probabilities within 6.3e-6.",
    },
    "jeb": {
        "namespace": "library",
        "model": "jeb",
        "family": "jebadiah",
        "author": "Jason Brashear, AINode (frontier-infra)",
        "license": "Apache-2.0",
        "license_text": "Jebadiah by Jason Brashear and AINode (https://huggingface.co/frontier-infra, "
                        "https://github.com/getainode/jebadiah): rank-16 LoRA merges into Qwen3.5-4B, Qwen3.5-9B and "
                        "Qwen3.8-27B by the Qwen team (Apache-2.0), published by the authors as GGUF.\n"
                        "Licensed under the Apache License, Version 2.0.\n\n" + LICENSE_APACHE,
        "tags": {
            # The authors' own GGUF files, pinned; Q8_0 for 4b and 9b (the files they checked against bf16:
            # 256/260 and 257/260), Q4_K_M for 27b so it fits a 24 GB GPU (29 GB at Q8_0).
            "4b": _gguf("jebadiah-4b-q8_0", "frontier-infra/jebadiah-4b-v2-GGUF", "7f671f9a31827257c26401000484e154b39417e6",
                        "jebadiah-4b-v2-Q8_0.gguf",
                        "Jebadiah 4B v2 (Qwen3.5-4B, LoRA merged), Q8_0 GGUF on llama.cpp: the option letters' logits "
                        "after AINode's decision prompt, with the authors' per-type temperatures.",
                        "4B", ["en"]),
            "9b": _gguf("jebadiah-9b-q8_0", "frontier-infra/jebadiah-9b-v2-GGUF", "adaec6b3d1f0421706deb49fa275ab49093982ff",
                        "jebadiah-9b-v2-Q8_0.gguf",
                        "Jebadiah 9B v2 (Qwen3.5-9B, LoRA merged), Q8_0 GGUF: the authors' recommended local model.",
                        "9B", ["en"]),
            "27b": _gguf("jebadiah-27b-q4_k_m", "frontier-infra/jebadiah-27b-GGUF", "7451e611a20ce3f56dd7d87f78c8a25b736c08f5",
                         "jebadiah-27b-Q4_K_M.gguf",
                         "Jebadiah 27B (Qwen3.8-27B, LoRA merged), Q4_K_M GGUF (17 GB, fits a 24 GB GPU): the most "
                         "accurate Jebadiah.",
                         "27B", ["en"]),
        },
        "aliases": {"latest": "9b"},
        "parity": "Ollaya's runner matches stock llama-server of the pinned build (b11146) on the same GGUF, CUDA "
                  "(RTX 4090): 494 questions per model, every decision the same, option logits within 7.7e-6 and "
                  "probabilities within 2.4e-6. The prompts are identical to the authors' jebadiah_prompt.Renderer "
                  "on 1,349 test prompts.",
    },
    "jevk5": {
        "namespace": "library",
        "model": "jevk5",
        "family": "jevk5",
        "author": "alibiserikbay",
        "license": "Apache-2.0",
        "license_text": "JevK5 by the JevK5 authors (https://github.com/allebee, https://huggingface.co/alibiserikbay), "
                        "a fine-tune of Qwen3.5-4B by the Qwen team (Apache-2.0). Its decision prompt and one-pass "
                        "readout are adapted from SemIf by TheoLeeCJ (MIT).\n"
                        "Licensed under the Apache License, Version 2.0.\n\n" + LICENSE_APACHE,
        "tags": {
            # JevK5 v0.3 (4B), Q8_0: the file the author checked against bf16 (229 of 231 JevBench
            # answers the same). The GGUF holds the weights, tokenizer and chat template.
            "4b": _gguf("jevk5-4b-q8_0", "alibiserikbay/JevK5-GGUF", "ec67b0bfce5119a8b11a2cdb430bb43e3fa3e82a",
                        "jevk5-4b-v0.3-Q8_0.gguf",
                        "JevK5 v0.3 (Qwen3.5-4B fine-tune), Q8_0 GGUF on llama.cpp: the answer letters' logits "
                        "after JevK5's own prompt, with the author's temperature 1.22. Up to 16 options.",
                        "4B", ["en"]),
        },
        "aliases": {"latest": "4b"},
        "parity": "Ollaya's runner matches stock llama-server of the pinned build (b11146) on the same GGUF, "
                  "CUDA (RTX 4090): 593 questions, every decision the same, option logits within 7.7e-6 and "
                  "probabilities within 1.6e-6. The prompts are byte-identical to the author's jevk5.prompt.",
    },
    "decima": {
        "namespace": "library",
        "model": "decima",
        "family": "decima",
        "author": "A. M. Madani (amyrmahdy)",
        "license": "Apache-2.0",
        "license_text": ("Decima-small, Decima-base and Decima-agent by A. M. Madani "
                         "(https://huggingface.co/amyrmahdy, https://github.com/amyrmahdy/decima), Apache-2.0.\n"
                         "Base models: multilingual-e5-small by intfloat "
                         "(https://huggingface.co/intfloat/multilingual-e5-small) for Decima-small, and mmBERT-base "
                         "by JHU CLSP (https://huggingface.co/jhu-clsp/mmBERT-base) for Decima-base and "
                         "Decima-agent, both MIT License.\n"
                         "Licensed under the Apache License, Version 2.0.\n\n") + LICENSE_APACHE,
        # The fp32 PyTorch checkpoint under pytorch/ of each repository, the encoder and the late-interaction
        # head in two files; the author's int8 ONNX exports are not used. The code is the GitHub tag v1.1.1
        # (2df60942); decima/ is unchanged at v2.0.0, which released base and agent.
        "tags": {
            "small": _decima("decima-small", "2e7f4d0757df0215f48f2a9b2b589e1f3a6348ed",  # HF tag v1.1.1
                             "A. M. Madani's Decima-small 1.1: multilingual-e5-small with a late-interaction "
                             "scorer that reads every option against the state, and an ordinal head for "
                             "scores, with the author's temperature. 122M parameters; fast on a CPU.",
                             "122M", 512),
            "base": _decima("decima-base", "2468005d5e48e95eb74072c32a6d9df164578071",  # HF tag v2.0
                            "A. M. Madani's Decima-base 2.0: mmBERT-base with Decima's late-interaction scorer "
                            "and ordinal head, with the author's temperature. 321M parameters, multilingual.",
                            "321M", 512),
            "agent": _decima("decima-agent", "86a07aab1c340fa5869bdb754e57d3851a1d288a",  # HF tag v2.1
                             "A. M. Madani's Decima-agent 2.1: Decima-base fine-tuned for the decisions inside "
                             "a coding agent's loop (secret and command gates, tool, command and model choice). "
                             "States up to 2,048 tokens.",
                             "321M", 2048),
        },
        "aliases": {"latest": "base"},
        "parity": "Ollaya's Rust runtime matches the author's own code (decima/model.py and systemone.py, fp32) on "
                  "581 questions from 122 requests for each of small, base and agent, on CPU and CUDA (RTX 4090 and "
                  "RTX 5090): identical token rows and truncation, the same 18 rejected requests, the same decision "
                  "on every question, scores within 2.0e-5 and probabilities within 5.5e-6.",
    },
}
