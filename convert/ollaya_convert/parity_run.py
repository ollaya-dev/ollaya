"""Run the runtime parity gate for library tags and record it (docs/model-lifecycle.md, stage 2).

    python -m ollaya_convert.parity_run --machine rtx-4090 --device cuda kev:4b kev:4b-fp16

For each tag, catalog.py gives the export directory and the family. The family's example
(`crates/ollaya-runner/examples/parity_<family>.rs`) runs on the goldens `out/goldens-<dir>.jsonl`; an `-fp16`
directory uses its base's goldens and the fp16 gate (`--fp16`, ADR 0008). The example's `record:` line, its exit
code (the gate) and the sha256 of the graph it checked go into `results/runs/<date>-parity-<machine>-<device>.json`,
one file per day, machine and device; a tag checked again that day replaces its row. `bench_plan` reads these files
to find the parity result of a changed graph.

Only the examples that print a `record:` line can be recorded (the Qwen3.5 families today); add the line to an
example when its family is next touched. A CUDA run needs the CUDA libraries on LD_LIBRARY_PATH, as any CUDA parity run.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

from .catalog import CATALOG, OUT

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "results")

PROTOCOL = {
    "reference": "goldens from each family's goldens.py: the authors' code in PyTorch fp32 (TF32 off) on the "
                 "checkpoint the manifest names",
    "gate": "fp32 tags: identical encodings and decisions, option logits within 1e-3, wire answers as the authors' "
            "code; -fp16 tags (docs/decisions/0008-fp16-decoder-tags.md): identical encodings, at least 99.5 % of "
            "decisions the same, differences reported",
    "runner": "convert/ollaya_convert/parity_run.py with crates/ollaya-runner/examples/parity_<family>.rs",
}


def wl_tags():
    """tag -> (family, export directory) for every tag converted under families/."""
    out = {}
    for e in CATALOG.values():
        for t, v in e["tags"].items():
            if v.get("kind") == "wl":
                out["%s:%s" % (e["model"], t)] = (e["family"], os.path.normpath(v["wl_dir"]))
    return out


def command(family, d, device):
    """The cargo command that runs the parity example of export directory `d`, and its gate."""
    name = os.path.basename(d)
    fp16 = name.endswith("-fp16")
    base = name[: -len("-fp16")] if fp16 else name
    example = "parity_" + family + ("_vision" if base.endswith("-vision") else "")
    cmd = ["cargo", "run", "--release", "-q", "-p", "ollaya-runner", "--example", example]
    if device == "cuda":
        cmd += ["--features", "cuda"]
    cmd += ["--", d, os.path.join(OUT, "goldens-%s.jsonl" % base), device] + (["--fp16"] if fp16 else [])
    return cmd, "fp16" if fp16 else "fp32"


def parse(tag, stdout, returncode):
    """The example's record line, or None when it stopped before its numbers (an encoding mismatch)."""
    lines = [line for line in stdout.splitlines() if line.startswith("record: ")]
    if not lines:
        return None
    return {"model": tag, **json.loads(lines[-1][len("record: "):]), "pass": returncode == 0}


def export_sha256(d):
    with open(os.path.join(d, "model.onnx"), "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def version():
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    pkg = next(line.split('"')[1] for line in open(os.path.join(ROOT, "Cargo.toml"), encoding="utf-8")
               if line.startswith("version"))
    return "%s at %s%s" % (pkg, sha, " with local changes" if dirty else "")


def merge(path, rows, machine, device, date, ollaya):
    """Write `rows` into the day's run file, replacing earlier rows of the same tags."""
    run = (json.load(open(path, encoding="utf-8")) if os.path.exists(path) else
           {"schema": 1, "suite": "parity", "date": date, "machine": machine, "device": device,
            "ollaya": ollaya, "protocol": PROTOCOL, "results": []})
    tags = {r["model"] for r in rows}
    run["results"] = sorted([r for r in run["results"] if r["model"] not in tags] + rows, key=lambda r: r["model"])
    run["ollaya"] = ollaya
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=1)
        f.write("\n")
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--machine", required=True, help="the machine's id in results/machines.json")
    ap.add_argument("--device", required=True, choices=["cuda", "cpu"])
    a = ap.parse_args()
    known = wl_tags()
    unknown = [t for t in a.tags if t not in known]
    if unknown:
        sys.exit("not converted under families/ (catalog.py kind 'wl'): %s" % " ".join(unknown))
    if a.machine not in json.load(open(os.path.join(RESULTS, "machines.json"), encoding="utf-8")):
        sys.exit("%s is not in results/machines.json; add the machine first" % a.machine)

    date = datetime.date.today().isoformat()
    path = os.path.join(RESULTS, "runs", "%s-parity-%s-%s.json" % (date, a.machine.replace("-", ""), a.device))
    failed = []
    for tag in a.tags:
        family, d = known[tag]
        cmd, gate = command(family, d, a.device)
        print("== %s: %s" % (tag, " ".join(cmd)), flush=True)
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        print("\n".join(line for line in p.stdout.splitlines() if not line.startswith("record: ")))
        row = parse(tag, p.stdout, p.returncode)
        if row is None:
            print(p.stderr[-3000:], file=sys.stderr)
            failed.append(tag)
            continue
        row.update(device=a.device, kind="runtime", gate=gate, export_sha256=export_sha256(d))
        merge(path, [row], a.machine, a.device, date, version())
        if not row["pass"]:
            print(p.stderr[-3000:], file=sys.stderr)
            failed.append(tag)
    print("recorded in %s" % os.path.relpath(path, ROOT))
    if failed:
        sys.exit("failed: %s" % " ".join(failed))


if __name__ == "__main__":
    main()
