"""Publish Ollaya's derived files for a model to its Hugging Face repository.

    uv run python -m ollaya_convert.publish_hf laya [--org ollaya-dev] [--dry-run out/hf]

Uploads, to `huggingface.co/<org>/<model>`, what Ollaya derives from the upstream checkpoint:
the ONNX graphs (weightless: they reference the author's `model.safetensors` by byte offset),
the decision and calibration configs, and a model card crediting the original authors.
Weights are never uploaded; `ollaya pull` fetches them from the author's repository.

The files come from `registry/` (run `package.py` first), so what is published is exactly what
the registry serves, digest for digest.
"""
import argparse
import json
import os
import shutil

from .catalog import CATALOG
from .package import MEDIA, REGISTRY

CARD = """---
license: {license_id}
base_model:
{base_models}
library_name: onnx
tags:
- ollaya
- onnx
- decision-model
- system-one
pipeline_tag: text-classification
---

# {model} for Ollaya

[Ollaya](https://github.com/ollaya-dev/ollaya) package of {base_links}{by}.
Ollaya runs open decision models locally, the way Ollama runs LLMs: typed questions in,
calibrated answers out, behind a TypeSafe-compatible API.

```sh
ollaya run {model}
```

## What is in this repository

This repository holds only the files Ollaya derives, with no weights. Each graph is an ONNX export of the
original model whose weights **reference the author's own `model.safetensors` by byte offset**,
so `ollaya pull` downloads the weights from the upstream repository, unmodified and pinned to a
commit, and verifies their sha256.

| Tag | Upstream | Files |
|---|---|---|
{rows}

{graphs_note} Each tag also has `decision.json` (sequence layout, special tokens) and
`calibration.json` (temperatures).

## Parity

{parity}

## License

{license_text_note}
"""


LAYA_PARITY = """The exports are checked against the PyTorch reference on 2,383 questions per checkpoint.
The checks use typed-decisions plus multilingual and edge cases:

- **fp32:** the same decision on 100% of questions, with probabilities within 1.1e-4.
- **fp16:** the same decision on 99.1–99.6% of questions. Nearly all of the differences are
  near-ties between the top two options."""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(CATALOG))
    ap.add_argument("--org", default="ollaya-dev")
    ap.add_argument("--dry-run", metavar="DIR", help="write the repository to DIR instead of uploading")
    a = ap.parse_args()
    spec = CATALOG[a.model]
    ns, model = spec["namespace"], spec["model"]

    stage = a.dry_run or os.path.join(os.path.dirname(__file__), "..", "out", "hf", model)
    if os.path.exists(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)
    rows, base_models, any_fp16 = [], [], False
    for tag in spec["tags"]:
        with open(os.path.join(REGISTRY, "v2", ns, model, "manifests", tag)) as f:
            manifest = json.load(f)
        os.makedirs(os.path.join(stage, tag))
        names = []
        for layer in manifest["layers"]:
            blob = os.path.join(REGISTRY, "blobs", "sha256-" + layer["digest"].split(":", 1)[1])
            kind = layer["mediaType"].rsplit(".", 1)[-1]
            name = {"onnx": "model-%s.onnx" % layer.get("annotations", {}).get("org.ollaya.precision", "fp32"),
                    "decision": "decision.json", "calibration": "calibration.json"}.get(kind)
            if name and os.path.exists(blob):
                shutil.copy(blob, os.path.join(stage, tag, name))
                names.append(name)
                any_fp16 = any_fp16 or name == "model-fp16.onnx"
        v = spec["tags"][tag]
        if v["repo"] not in base_models:
            base_models.append(v["repo"])
        upstream = "[%s@%s](https://huggingface.co/%s/tree/%s)" % (v["repo"], v["commit"][:7], v["repo"], v["commit"])
        rows.append("| `%s:%s` | %s | %s |" % (model, tag, upstream, ", ".join("`%s/%s`" % (tag, n) for n in names)))
    by = spec.get("author", "")
    base_model = base_models[0]
    card = CARD.format(
        license_id=spec["license"].lower(), model=model,
        base_models="\n".join("- " + b for b in base_models),
        base_links=" and ".join("**[%s](https://huggingface.co/%s)**" % (b, b) for b in base_models),
        by=(" by " + by) if by else "", rows="\n".join(rows),
        graphs_note=("Each tag has an fp32 graph (CPU) and an fp16 graph (GPU)." if any_fp16
                     else "Each tag has an fp32 graph, used on CPU and GPU."),
        parity=spec.get("parity", LAYA_PARITY),
        license_text_note="Same as the upstream model (%s). Ollaya itself is Apache-2.0." % spec["license"],
    )
    with open(os.path.join(stage, "README.md"), "w") as f:
        f.write(card)
    print("staged %s" % os.path.abspath(stage))
    if a.dry_run:
        return

    from huggingface_hub import HfApi

    api = HfApi()
    repo = "%s/%s" % (a.org, model)
    api.create_repo(repo, repo_type="model", exist_ok=True)
    info = api.upload_folder(repo_id=repo, folder_path=stage, commit_message="Ollaya package for %s" % base_model)
    print("published https://huggingface.co/%s (%s)" % (repo, info.oid[:7] if hasattr(info, "oid") else "ok"))


if __name__ == "__main__":
    main()
