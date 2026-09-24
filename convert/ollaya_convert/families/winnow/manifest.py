"""Write the Ollaya config for EldanRing/Winnow-12B (engine llama.cpp): decision.json, calibration.json
and files.json. No weights are downloaded or copied; the GGUF is referenced in the author's repo.

    uv run python -m ollaya_convert.families.winnow.manifest --out out/winnow-12b
"""
from __future__ import annotations

import argparse
import os

from ..llm_common.onnx_export import hub_file_info, write_json
from .ref import LAYOUT, MAX_LABELS, SYSTEM, THOUGHT

REPO = "EldanRing/Winnow-12B"
REVISION = "b6ac22b0d51b69b18200acacb3fbdd98073fffe8"
GGUF = {"q8_0": "gguf/Winnow-12B-Q8_0.gguf", "bf16": "gguf/Winnow-12B-BF16.gguf"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    variants = {}
    for tag, path in GGUF.items():
        info = hub_file_info(REPO, REVISION, path) or {}
        variants[tag] = {"repo": REPO, "revision": REVISION, "path": path, "sha256": info.get("sha256"),
                         "bytes": info.get("size"),
                         "url": "https://huggingface.co/%s/resolve/%s/%s" % (REPO, REVISION, path)}
    decision = {
        "engine": "llama.cpp",
        "family": "winnow",
        "layout": LAYOUT,
        "gguf": variants,
        "upstream": {"inference": "https://github.com/EldanRing/winnow-inference",
                     "commit": "6c2b3c04e248a319f2cb43832628eba03e55fe38", "source": "native/protocol.h"},
        "system": SYSTEM,
        "templates": {
            "prefix": "<|turn>system\n{system}<turn|>\n<|turn>user\nState:\n{safe(state)}\n",
            "suffix": "\nQuestion: {safe(instructions or '')}\nOptions:\n{label}: {safe(rendered)}\n...Return the correct letter label."
                      "<turn|>\n<|turn>model\n" + THOUGHT + "Answer:\n",
            "safe": "nlohmann ordered_json dump (compact, UTF-8 kept) with '<' -> \\u003c",
            "tokenize": "prefix: add_bos, parse_special; suffix: no bos, parse_special; concatenated",
        },
        "labels": {"candidates": "A..Z then AA..ZZ, single tokens whose piece equals the label", "max": MAX_LABELS},
        "limits": {"questions": [1, 256], "alternatives": [2, MAX_LABELS]},
        "option_logits": "label logits at the last prompt token, key order (noul: [false, true])",
        "softcap": "Gemma final logit soft-cap 30*tanh(x/30), applied by llama.cpp",
    }
    write_json(os.path.join(a.out, "decision.json"), decision)
    write_json(os.path.join(a.out, "calibration.json"),
               {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {},
                "source": "Winnow default decision temperature 1.0; no fitted post-hoc map (release-manifest.json)"})
    write_json(os.path.join(a.out, "files.json"), {"model": "winnow-12b", "layers": [
        {"role": "weights/" + t, **v} for t, v in variants.items()] + [
        {"role": "license", "repo": REPO, "revision": REVISION, "path": "LICENSE"},
        {"role": "notice", "repo": REPO, "revision": REVISION, "path": "NOTICE"},
        {"role": "decision", "path": "decision.json", "hosted_by": "ollaya"},
        {"role": "calibration", "path": "calibration.json", "hosted_by": "ollaya"}]})
    print({t: (v["sha256"], v["bytes"]) for t, v in variants.items()})


if __name__ == "__main__":
    main()
