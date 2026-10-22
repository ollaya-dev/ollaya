"""A model's Hugging Face repository, staged as a directory to upload.

    python hub.py <checkpoint> <ollaya model dir> <out-dir> --card <README.md> [--packed]

The checkpoint is Laya's format (rl_agent_config.json, model.safetensors, encoder/config.json, tokenizer/), which
`laya` loads as it is and Ollaya's registry points its weights layer at. The Ollaya model dir is what
`ollaya_convert.package` makes of it (model.onnx reading the weights by their blob name, sha256-<hex>, plus
tokenizer.json, decision.json, calibration.json; for a ternary model also model.packed from `ternary.py pack`).
Its files go in too, so the repository alone runs the model in a browser (web/ollaya.js `fromHub`).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil

import onnx

HERE = os.path.dirname(os.path.abspath(__file__))
ETTIN = """The encoder is initialised from Ettin (https://huggingface.co/jhu-clsp), under the MIT License:

MIT License

Copyright (c) 2025 Johns Hopkins University Center for Language and Speech Processing

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The training data's sources and their licenses are listed in DATA_SOURCES.tsv.
"""


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("model_dir")
    p.add_argument("out")
    p.add_argument("--card", required=True)
    p.add_argument("--packed", action="store_true")
    a = p.parse_args()

    weights = os.path.join(a.checkpoint, "model.safetensors")
    blob = "sha256-" + sha256(weights)
    graph = os.path.join(a.model_dir, "model.onnx")
    locations = {e.value for t in onnx.load(graph, load_external_data=False).graph.initializer
                 for e in t.external_data if e.key == "location"}
    assert locations == {blob}, f"the graph reads {locations}, the checkpoint is {blob}"

    os.makedirs(os.path.join(a.out, "onnx"), exist_ok=True)
    for f in ("model.safetensors", "rl_agent_config.json", "encoder", "tokenizer"):
        src, dst = os.path.join(a.checkpoint, f), os.path.join(a.out, f)
        (shutil.copytree(src, dst, dirs_exist_ok=True) if os.path.isdir(src) else shutil.copy(src, dst))
    shutil.copy(graph, os.path.join(a.out, "onnx", "model.onnx"))
    for f in ("tokenizer.json", "decision.json", "calibration.json") + (("model.packed",) if a.packed else ()):
        shutil.copy(os.path.join(a.model_dir, f), os.path.join(a.out, f))
    shutil.copy(a.card, os.path.join(a.out, "README.md"))
    shutil.copy(os.path.join(HERE, "..", "LICENSE"), os.path.join(a.out, "LICENSE"))
    with open(os.path.join(a.out, "NOTICE"), "w") as f:
        f.write(ETTIN)
    # The sources the model learned from, with their licenses; the in-domain ones as nimble's modules state them.
    with open(os.path.join(HERE, "data", "licenses.tsv")) as f, open(os.path.join(a.out, "DATA_SOURCES.tsv"), "w") as o:
        rows = [line.rstrip("\n").split("\t") for line in f]
        o.write("source\tlicense\trows\n")
        o.writelines(f"{r[0]}\t{r[1]}\t{r[8]}\n" for r in rows[1:] if r[2] == "keep")
        o.writelines(f"{s}\t{lic}\t{n}\n" for s, lic, n in INDOMAIN)
    print("staged", a.out, "weights", blob)


INDOMAIN = [
    ("indomain/aegis2", "CC BY 4.0", 20000),
    ("indomain/boolq", "CC BY-SA 3.0", 9263),
    ("indomain/civil_comments", "CC0 1.0", 20000),
    ("indomain/helpsteer2", "CC BY 4.0", 20000),
    ("indomain/massive", "CC BY 4.0", 11512),
    ("indomain/multinli", "Mixed per genre: OANC public domain; fiction and other sources CC BY-SA 3.0 or similar", 20000),
    ("indomain/paws", "Google PAWS license: may be freely used for any purpose", 20000),
    ("indomain/pubmedqa", "MIT", 20000),
    ("indomain/squad2", "CC BY-SA 4.0", 20000),
    ("indomain/vitaminc", "CC BY-SA 3.0", 20000),
]

if __name__ == "__main__":
    main()
