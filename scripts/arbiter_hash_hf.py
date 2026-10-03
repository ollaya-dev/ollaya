"""Hash the Hugging Face files that the arbiter manifests pin.

    uv run python scripts/arbiter_hash_hf.py

Downloads each file from its pinned revision, streams it through sha256, prints a JSON
map {filename: {"sha256": ..., "size": ...}} and deletes the local copy. Needs an HF
token with read access to `hiteshluke/arbiter-4b` (via `huggingface-cli login` or the
`HF_TOKEN` env var).

The values match the registry/v2/library/arbiter/manifests entries; rerun to verify.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    sys.exit("huggingface_hub is required: pip install huggingface_hub")


FILES = [
    ("unsloth/gemma-3-4b-it",   "bf46152c47f5dd20b896357cb51abc4c03b8ee8c", "model-00001-of-00002.safetensors"),
    ("unsloth/gemma-3-4b-it",   "bf46152c47f5dd20b896357cb51abc4c03b8ee8c", "model-00002-of-00002.safetensors"),
    ("hiteshluke/arbiter-4b",   "0c44271c59f89758e3cae17b032e98a9140093e9", "adapter_model.safetensors"),
    ("hiteshluke/arbiter-4b",   "0c44271c59f89758e3cae17b032e98a9140093e9", "head.pt"),
    ("hiteshluke/arbiter-4b",   "0c44271c59f89758e3cae17b032e98a9140093e9", "tokenizer.json"),
    ("hiteshluke/arbiter-4b",   "0c44271c59f89758e3cae17b032e98a9140093e9", "README.md"),
]


def sha256_stream(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    out: dict[str, dict[str, object]] = {}
    token = os.environ.get("HF_TOKEN")
    for repo, revision, filename in FILES:
        local = hf_hub_download(
            repo_id=repo, revision=revision, filename=filename,
            token=token, local_dir=None,
        )
        try:
            digest = sha256_stream(local)
            size = os.path.getsize(local)
            out[filename] = {"repo": repo, "revision": revision,
                             "sha256": digest, "size": size}
            print("  %s  %s  %d bytes" % (filename, digest, size), file=sys.stderr)
        finally:
            try:
                os.remove(local)
            except OSError:
                pass
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
