"""What a release must measure for Ollaya Bench, from what changed (docs/decisions/0009-measurement-per-release.md).

    python -m ollaya_convert.bench_plan --registry ../registry --out plan.json [--cpu]

Compares every tag in REGISTRY (registry/v2/library/<model>/manifests/<tag>) with the digest the current edition
(results/editions/, its runs and updates) measured, layer by layer, and lists per tag the suites to run:

    new tag, or new weights or tokenizer    typed, public, latency-cuda, latency-cpu
    new -fp16 tag (fp16 graphs only)         typed, latency-cuda        (shown under its base tag, not ranked)
    graph or config changed                  latency-cuda (and latency-cpu with --cpu); accuracy is carried when the
                                             tag passed the fp32 parity gate since the edition's last update, typed
                                             as well when it passed the fp16 gate, and the plan fails when it has no
                                             parity record (`python -m ollaya_convert.parity_run`)
    same layers                              nothing

It prints the plan with an estimate from the edition's own timings and writes it as JSON for `bench_run --plan`.
Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Tags the bench leaves out: image tags (the bench is text only) and laya's single-precision tags (laya:en runs the
# same graphs, fp16 on a GPU and fp32 on a CPU).
OUT_OF_SCOPE = re.compile(r"-vision$|^laya:.+-fp(16|32)$")
# Models that answer only their built-in questions: timed, never scored.
NO_ACCURACY = {"qwen3guard:0.6b"}
# The router answers through laya:en or laya:multilingual, which are timed themselves.
NO_LATENCY = {"laya"}
UPSTREAM = ("application/vnd.ollaya.weights", "application/vnd.ollaya.weights.gguf", "application/vnd.ollaya.tokenizer")
FULL = ["typed", "public", "latency-cuda", "latency-cpu"]
# Minutes for a model the edition never timed (shortcut: flat guesses; a new model's own numbers replace them next time).
GUESS = {"typed": 10, "public": 60, "latency-cuda": 3, "latency-cpu": 15}


def short(tag):
    return tag[: -len(":latest")] if tag.endswith(":latest") else tag


def registry_tags(registry):
    """tag -> manifest path, for every tag in the registry except aliases (a tag whose manifest is another tag's)."""
    out = {}
    base = os.path.join(registry, "v2", "library")
    for model in sorted(os.listdir(base)):
        d = os.path.join(base, model, "manifests")
        tags = sorted(t for t in os.listdir(d) if not t.startswith("sha256-"))
        bodies = {t: open(os.path.join(d, t), "rb").read() for t in tags}
        for t in tags:
            if t == "latest" and any(bodies[o] == bodies[t] for o in tags if o != t):
                continue
            out[short("%s:%s" % (model, t))] = os.path.join(d, t)
    return out


def edition_state(results):
    """The current edition, with per model the newest digest and timings its runs measured, and when it last changed."""
    eds = sorted(os.listdir(os.path.join(results, "editions")))
    edition = json.load(open(os.path.join(results, "editions", eds[-1]), encoding="utf-8"))
    digests, timings = {}, {}
    for name in edition["runs"]:  # in order: the edition's runs, then each update's
        run = json.load(open(os.path.join(results, "runs", name), encoding="utf-8"))
        suite = run["suite"] if run["suite"] != "latency-triage" else "latency-" + run["device"]
        for r in run["results"]:
            m = short(r["model"])
            if r.get("digest"):
                digests[m] = r["digest"]
            t = timings.setdefault(m, {})
            if suite.startswith("latency-") and r.get("p50_ms") is not None:
                n = 25 if suite == "latency-cuda" else 12  # warm-up plus timed requests
                t[suite] = ((r.get("load_ms") or 0) + n * r["p50_ms"]) / 60000
            elif suite == "typed-decisions" and r.get("median_ms"):
                t["typed"] = r["rows"] * r["median_ms"] / 60000
            elif suite == "public-benchmark" and r.get("median_ms"):
                t["public"] = r["records"] * r["median_ms"] / 60000
    last = max([edition["date"]] + [u["date"] for u in edition.get("updates", [])])
    return edition, digests, timings, last


def parity_rows(results, since):
    """tag -> the newest runtime parity row on CUDA dated on or after `since`."""
    out = {}
    d = os.path.join(results, "runs")
    for name in sorted(os.listdir(d)):
        if "-parity-" not in name or name[:10] < since[:10]:
            continue
        run = json.load(open(os.path.join(d, name), encoding="utf-8"))
        for r in run["results"]:
            if r.get("device") == "cuda" and r.get("kind") == "runtime":
                out[short(r["model"])] = r
    return out


def layers(manifest):
    return {(l["mediaType"], l["digest"]) for l in manifest["layers"]}


def decide(tag, manifest, old, parity, cpu):
    """(suites, why) for one tag; `old` is the edition's manifest of the tag (None when it is new)."""
    graphs = [l for l in manifest["layers"] if l["mediaType"] == "application/vnd.ollaya.graph.onnx"]
    if old is None:
        if graphs and all((l.get("annotations") or {}).get("org.ollaya.precision") == "fp16" for l in graphs):
            return ["typed", "latency-cuda"], "new fp16 tag"
        return list(FULL), "new tag"
    new, was = layers(manifest), layers(old)
    if new == was:
        return [], "same layers"
    if {x for x in new if x[0] in UPSTREAM} != {x for x in was if x[0] in UPSTREAM}:
        return list(FULL), "new weights or tokenizer"
    suites = ["latency-cuda"] + (["latency-cpu"] if cpu else [])
    if not graphs:
        return suites, "config changed"
    p = parity.get(tag)
    if p is None:
        return None, "graph changed, no parity record since the edition's last update"
    if not p.get("pass"):
        return None, "graph changed, parity failed (%s gate)" % p.get("gate", "fp32")
    if p.get("gate") == "fp16":
        return ["typed"] + suites, "graph changed, fp16 gate passed"
    return suites, "graph changed, fp32 parity passed: accuracy carried"


def plan(registry, results, cpu=False, only=None):
    edition, digests, timings, last = edition_state(results)
    parity = parity_rows(results, last)
    entries, blocked = {}, {}
    for tag, path in registry_tags(registry).items():
        if OUT_OF_SCOPE.search(tag) or (only and tag not in only):
            continue
        body = open(path, "rb").read()
        manifest = json.loads(body)
        if tag in edition["models"] and digests.get(tag) is None:
            continue  # measured without a digest (the router): nothing to compare
        old = None
        if tag in digests:
            hexdigest = digests[tag].split(":")[-1]  # the server reports the bare hex
            if hexdigest == hashlib.sha256(body).hexdigest():
                continue
            kept = os.path.join(os.path.dirname(path), "sha256-" + hexdigest)
            if not os.path.exists(kept):
                entries[tag] = {"suites": list(FULL), "why": "the edition's manifest is not kept in the registry",
                                "minutes": sum(GUESS.values())}
                continue
            old = json.load(open(kept, encoding="utf-8"))
        suites, why = decide(tag, manifest, old, parity, cpu)
        if suites is None:
            blocked[tag] = why
            continue
        suites = [s for s in suites if not (tag in NO_ACCURACY and s in ("typed", "public"))
                  and not (tag in NO_LATENCY and s.startswith("latency"))]
        if suites:
            base = tag[: -len("-fp16")] if tag.endswith("-fp16") else tag
            t = timings.get(tag) or timings.get(base) or {}
            entries[tag] = {"suites": suites, "why": why,
                            "minutes": round(sum(t.get(s, GUESS[s]) for s in suites), 1)}
    return {"edition": edition["id"], "since": last, "registry": os.path.abspath(registry),
            "models": entries, "blocked": blocked, "minutes": round(sum(e["minutes"] for e in entries.values()))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=os.path.join(ROOT, "registry"))
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--cpu", action="store_true", help="time changed graphs on the CPU too")
    ap.add_argument("--models", nargs="*", help="plan only these tags")
    ap.add_argument("--out", help="write the plan here, for bench_run --plan")
    a = ap.parse_args()
    p = plan(a.registry, a.results, a.cpu, set(a.models or []))
    print("Edition %s, changes since %s" % (p["edition"], p["since"]))
    for tag, e in p["models"].items():
        print("  %-28s %-40s %5.0f min  %s" % (tag, " ".join(e["suites"]), e["minutes"], e["why"]))
    print("About %d minutes on the bench machine." % p["minutes"])
    for tag, why in p["blocked"].items():
        print("  BLOCKED %-20s %s" % (tag, why))
    if a.out:
        json.dump(p, open(a.out, "w", encoding="utf-8"), indent=1)
    if p["blocked"]:
        sys.exit("Run parity for the blocked tags first (docs/model-lifecycle.md, stage 2).")


if __name__ == "__main__":
    main()
