"""A model's Hugging Face repository, staged as a directory to upload.

    python hub.py <checkpoint> <out-dir> --card card.md --var name=… --var title=… … [--web <dir>]

The checkpoint is Laya's format (rl_agent_config.json, model.safetensors, encoder/config.json, tokenizer/), which
`laya` loads as it is and Ollaya's registry points its weights layer at. For a ternary model, `--web` adds the
browser's files (web/ollaya.js `fromHub`): the graph from webgraph.py as onnx/model.onnx, and Ollaya's
tokenizer.json, decision.json and calibration.json from `ollaya_convert.export`.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("out")
    p.add_argument("--card", required=True, help="the model card's template; {{key}} takes --var key=value")
    p.add_argument("--var", action="append", default=[], help="key=value for the card (a value may be @file)")
    p.add_argument("--web", help="a dir with webgraph.py's model.onnx and the export's tokenizer, decision, calibration")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    for f in ("model.safetensors", "rl_agent_config.json", "encoder", "tokenizer"):
        src, dst = os.path.join(a.checkpoint, f), os.path.join(a.out, f)
        (shutil.copytree(src, dst, dirs_exist_ok=True) if os.path.isdir(src) else shutil.copy(src, dst))
    if a.web:
        os.makedirs(os.path.join(a.out, "onnx"), exist_ok=True)
        shutil.copy(os.path.join(a.web, "model.onnx"), os.path.join(a.out, "onnx", "model.onnx"))
        for f in ("tokenizer.json", "decision.json", "calibration.json"):
            shutil.copy(os.path.join(a.web, f), os.path.join(a.out, f))
    card = open(a.card).read()
    values = [(k, open(v[1:]).read().rstrip("\n") if v.startswith("@") else v) for k, v in (kv.split("=", 1) for kv in a.var)]
    for _ in range(2):  # a value may hold {{keys}} itself (card-web.md)
        for k, v in values:
            card = card.replace("{{%s}}" % k, v)
    card = re.sub(r"\n{3,}", "\n\n", card)  # an empty value (web= for an fp model) leaves no gap
    left = set(re.findall(r"{{(\w+)}}", card))
    assert not left, f"the card still has {sorted(left)}"
    with open(os.path.join(a.out, "README.md"), "w") as f:
        f.write(card)
    shutil.copy(os.path.join(HERE, "..", "LICENSE"), os.path.join(a.out, "LICENSE"))
    with open(os.path.join(a.out, "NOTICE"), "w") as f:
        f.write(ETTIN)
    # The sources the model learned from, with their licenses; the in-domain ones as nimble's modules state them.
    with open(os.path.join(HERE, "data", "licenses.tsv")) as f, open(os.path.join(a.out, "DATA_SOURCES.tsv"), "w") as o:
        rows = [line.rstrip("\n").split("\t") for line in f]
        o.write("source\tlicense\trows\n")
        o.writelines(f"{r[0]}\t{r[1]}\t{r[8]}\n" for r in rows[1:] if r[2] == "keep")
        o.writelines(f"{s}\t{lic}\t{n}\n" for s, lic, n in INDOMAIN)
    print("staged", a.out)


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
