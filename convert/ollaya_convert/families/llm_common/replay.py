"""Replay a model's goldens on another llama.cpp build or device: the reference numbers for that device.

    python -m ollaya_convert.families.llm_common.replay out/winnow-12b-q8_0 --server <build>/llama-server \
        --gguf .../Winnow-12B-Q8_0.gguf --from goldens-cuda.jsonl --device cpu|MTL0|CUDA0

llama.cpp's logits depend on the backend (CUDA, Metal and the CPU kernels round differently), so each
device the runtime supports gets its own goldens. The prompts do not depend on the device: this takes
the reference prompts (token ids, split points, candidates) from an existing goldens file and evaluates
them with the fixed evaluation plan (`plan.py`) on a llama-server started with the runtime's exact
arguments for `--device`. It writes goldens-<device>.jsonl next to the source, in the same format.

Standard library only, so it also runs on a Mac without the convert environment.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

from .llama_server import LlamaServer
from .plan import FixedPlan, server_args


def post_to(url):
    import urllib.request

    def post(path, body):
        req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.loads(r.read())
    return post


def device_class(device):
    return "cuda" if device.startswith("CUDA") else "metal" if device.startswith("MTL") else "cpu"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", help="out/<layout>-<slug>: decision.json and the goldens")
    ap.add_argument("--server", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--from", dest="source", default="goldens-cuda.jsonl")
    ap.add_argument("--device", required=True, help="llama.cpp device (CUDA0, MTL0), or cpu")
    ap.add_argument("--port", type=int, default=8096)
    a = ap.parse_args()

    decision = json.load(open(os.path.join(a.model_dir, "decision.json")))
    llama = decision["llama"]
    dev = None if a.device == "cpu" else a.device
    argv = server_args(llama["n_ctx"], llama.get("swa_full", False), dev)
    name = "goldens-%s" % device_class(a.device)
    if os.path.join(a.model_dir, name + ".jsonl") == os.path.join(a.model_dir, a.source):
        raise SystemExit("the source and the output are the same file")
    srv = LlamaServer.start(a.server, a.gguf, port=a.port, argv=argv,
                            log=os.path.join(a.model_dir, "llama-server-%s.log" % device_class(a.device)),
                            timeout=1800)
    try:
        version = subprocess.run([a.server, "--version"], capture_output=True, text=True, timeout=60)
        version = " ".join(l.strip() for l in (version.stdout + version.stderr).splitlines()
                           if l.strip().startswith(("version", "built")))
        with open(os.path.join(a.model_dir, name + ".meta.json"), "w") as f:
            json.dump({"server": version, "device": a.device, "args": argv,
                       "gguf_sha256": decision["gguf"]["sha256"], "prompts_from": a.source,
                       "reference": "the prompts of %s, evaluated through llm_common.plan" % a.source}, f, indent=1)
        plan = FixedPlan(post_to(srv.url))
        n = 0
        with open(os.path.join(a.model_dir, a.source)) as src, \
                open(os.path.join(a.model_dir, name + ".jsonl"), "w") as out:
            for line in src:
                rec = json.loads(line)
                for row in rec.get("rows", []):
                    cands = row["candidates"]
                    if len(cands) > 1:
                        lp = plan.ask(row["ids"], row["P"], cands)
                        row["option_logits"] = [lp[j] for j in row["wire_order"]]
                    n += 1
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("%s/%s.jsonl: %d questions on %s" % (a.model_dir, name, n, a.device))
    finally:
        srv.stop()


if __name__ == "__main__":
    main()
