"""Package exported models into Ollaya's static registry.

    uv run python -m ollaya_convert.package laya [--origin https://ollaya.dev]

Writes, under `registry/` at the repository root (copied into the website at build time):

    registry/v2/<namespace>/<model>/manifests/<tag>   Docker v2 manifests
    registry/blobs/sha256-<hex>                       files Ollaya derives (graphs, configs)

Weights and tokenizers are never copied. Their manifest layers point at the author's Hugging
Face repository, pinned to a commit, and carry the file's sha256 (the Git LFS object id), which
`ollaya pull` verifies. Graphs name the weights by their blob file name (`sha256-<hex>`), so the
blob store needs no links to load them.

A tag whose catalog entry has `arch` also gets an `application/vnd.ollaya.arch` layer: the network
the MLX engine builds from the same weights file (`arch.py`), derived from the author's config at
the pinned commit. Only tags that pass their parity gate on Metal have one
(docs/decisions/0001-mlx-engine.md).
"""
import argparse
import hashlib
import json
import os
import urllib.request

import onnx

from . import arch as mlx_arch
from . import weightless
from .catalog import CATALOG

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
REGISTRY = os.environ.get("OLLAYA_REGISTRY_OUT", os.path.join(REPO_ROOT, "registry"))
MANIFEST_V2 = "application/vnd.docker.distribution.manifest.v2+json"
MEDIA = {
    "config": "application/vnd.ollaya.config.v1+json",
    "graph": "application/vnd.ollaya.graph.onnx",
    "weights": "application/vnd.ollaya.weights",
    "tokenizer": "application/vnd.ollaya.tokenizer",
    "decision": "application/vnd.ollaya.decision",
    "calibration": "application/vnd.ollaya.calibration",
    "router": "application/vnd.ollaya.router",
    "params": "application/vnd.ollaya.params",
    "questions": "application/vnd.ollaya.questions",
    "license": "application/vnd.ollaya.license",
    "arch": "application/vnd.ollaya.arch",
    # A GGUF file (weights, tokenizer and chat template) that llama.cpp loads, unmodified from the
    # author's repository.
    "gguf": "application/vnd.ollaya.weights.gguf",
}
HF = "https://huggingface.co"


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def hf_paths_info(repo, commit, paths):
    req = urllib.request.Request(
        "%s/api/models/%s/paths-info/%s" % (HF, repo, commit),
        data=json.dumps({"paths": paths}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return {f["path"]: f for f in json.load(r)}


def hf_file(repo, commit, path):
    """Descriptor fields for a file in an upstream repo: url, size, sha256."""
    info = hf_paths_info(repo, commit, [path])[path]
    url = "%s/%s/resolve/%s/%s" % (HF, repo, commit, path)
    lfs = info.get("lfs")
    if lfs:
        return url, info["size"], lfs["oid"]
    with urllib.request.urlopen(url) as r:  # small git file: hash it ourselves
        data = r.read()
    return url, len(data), sha256(data)


class Blobs:
    """Derived files, content-addressed, served from `<origin>/blobs/sha256-<hex>`."""

    def __init__(self, origin):
        self.origin = origin.rstrip("/")
        os.makedirs(os.path.join(REGISTRY, "blobs"), exist_ok=True)

    def put(self, media_type, data, annotations=None):
        h = sha256(data)
        path = os.path.join(REGISTRY, "blobs", "sha256-" + h)
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(data)
        d = {"mediaType": media_type, "digest": "sha256:" + h, "size": len(data),
             "urls": ["%s/blobs/sha256-%s" % (self.origin, h)]}
        if annotations:
            d["annotations"] = annotations
        return d


def hf_commit_date(repo, commit):
    """YYYY-MM-DD of the pinned upstream commit."""
    with urllib.request.urlopen("%s/api/models/%s/revision/%s" % (HF, repo, commit)) as r:
        return json.load(r)["lastModified"][:10]


def upstream(media_type, repo, commit, path):
    url, size, oid = hf_file(repo, commit, path)
    return {"mediaType": media_type, "digest": "sha256:" + oid, "size": size, "urls": [url]}


def hf_json(repo, commit, path):
    with urllib.request.urlopen("%s/%s/resolve/%s/%s" % (HF, repo, commit, path)) as r:
        return json.load(r)


def arch_layers(v, blobs, weights_path):
    """The arch layer for MLX, when the catalog entry asks for one: `arch.family` and the author's
    `arch.config` (and, for laya, `arch.agent_config`) at the pinned commit, checked against the
    local copy of the weights file."""
    spec = v.get("arch")
    if not spec:
        return []
    config = hf_json(v["repo"], v["commit"], spec["config"])
    agent = hf_json(v["repo"], v["commit"], spec["agent_config"]) if spec.get("agent_config") else None
    layer = mlx_arch.build(spec["family"], config, weights_path, agent)
    return [blobs.put(MEDIA["arch"], mlx_arch.dumps(layer))]


def write_manifest(namespace, model, tag, config, layers):
    manifest = {"schemaVersion": 2, "mediaType": MANIFEST_V2, "config": config, "layers": layers}
    d = os.path.join(REGISTRY, "v2", namespace, model, "manifests")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, tag), "w") as f:
        json.dump(manifest, f, indent=2)
    print("  %s/%s:%s  %d layers" % (namespace, model, tag, len(layers)))


def graph_bytes(export_dir, checkpoint, prefix, weights_oid):
    """The exported graph, rewritten to reference the weights blob, serialized."""
    model = onnx.load(os.path.join(export_dir, "model.onnx"), load_external_data=True)
    stats = weightless.rewrite(model, checkpoint, prefix, "sha256-" + weights_oid)
    big = [k for k in ("inline",) if stats.get(k, 0) > 64]
    if big:
        raise SystemExit("%s: too many inline initializers %s" % (export_dir, stats))
    return model.SerializeToString(), stats


def graph_from_wl(wl_dir, oids):
    """A weightless graph from `families/`, its external-data locations renamed from upstream file
    names to the blob names (`sha256-<oid>`) the weights have in the store."""
    model = onnx.load(os.path.join(wl_dir, "model.onnx"), load_external_data=False)
    n = 0
    for t in model.graph.initializer:
        for kv in t.external_data:
            if kv.key == "location":
                kv.value = "sha256-" + oids[kv.value]
                n += 1
    return model.SerializeToString(), {"external": n, "inline": len(model.graph.initializer) - n}


def package_wl(spec, tag, v, blobs):
    """One tag of a model converted under `families/` (fp32 graph; runs fp32 on every device)."""
    repo, commit = v["repo"], v["commit"]
    oids, weights, locals_ = {}, [], []
    for location, source in v["weights"].items():
        w_repo, w_commit, path = source if isinstance(source, tuple) else (repo, commit, source)
        d = upstream(MEDIA["weights"], w_repo, w_commit, path)
        oid = d["digest"].split(":", 1)[1]
        local = os.path.join(v["wl_dir"], location)
        with open(local, "rb") as f:  # the local copy must be the pinned upstream file
            if hashlib.file_digest(f, "sha256").hexdigest() != oid:
                raise SystemExit("%s does not match %s@%s:%s" % (local, w_repo, w_commit, path))
        oids[location] = oid
        weights.append(d)
        locals_.append(local)
    if v.get("arch") and len(locals_) != 1:
        raise SystemExit("%s: the MLX engine reads one weights file" % spec["model"])
    arch = arch_layers(v, blobs, locals_[0]) if v.get("arch") else []
    data, stats = graph_from_wl(v["wl_dir"], oids)
    print("  %s:%s fp32 graph %.1f MB %s" % (spec["model"], tag, len(data) / 2**20, stats))
    graph = blobs.put(MEDIA["graph"], data, {"org.ollaya.precision": "fp32"})
    tokenizer = upstream(MEDIA["tokenizer"], repo, commit, v["tokenizer"])
    decision_bytes = open(os.path.join(v["wl_dir"], "decision.json"), "rb").read()
    decision = blobs.put(MEDIA["decision"], decision_bytes)
    calibration = blobs.put(MEDIA["calibration"], open(os.path.join(v["wl_dir"], "calibration.json"), "rb").read())
    dj = json.loads(decision_bytes)
    # A fixed-preset model ships its questions as the built-in set (what a Modelfile's QUESTIONS writes).
    questions = [blobs.put(MEDIA["questions"], json.dumps(dj["questions"], indent=2).encode())] \
        if dj.get("preset_only") else []
    ctx = dj.get("max_len") or dj.get("max_length") or v["context_length"]
    license_id = v.get("license") or spec["license"]
    lic = blobs.put(MEDIA["license"], (v.get("license_text") or spec["license_text"]).encode())
    config = blobs.put(MEDIA["config"], json.dumps({
        "model_format": "onnx", "family": spec["family"], "parameter_size": v["parameter_size"],
        "context_length": ctx, "languages": v["languages"], "description": v["description"],
        "source": "huggingface.co/%s@%s" % (repo, commit), "license": license_id,
        "release_date": hf_commit_date(repo, commit),
    }, indent=2).encode())
    return config, [graph] + weights + [tokenizer, decision, calibration] + questions + [lic] + arch


def package_gguf(spec, tag, v, blobs):
    """One tag of a GGUF model, run by llama.cpp: the author's GGUF, unmodified, plus the derived
    decision.json and calibration.json (`llm_common/export_llama.py`)."""
    repo, commit = v["repo"], v["commit"]
    gguf = upstream(MEDIA["gguf"], repo, commit, v["gguf"])
    oid = gguf["digest"].split(":", 1)[1]
    decision_bytes = open(os.path.join(v["export_dir"], "decision.json"), "rb").read()
    dj = json.loads(decision_bytes)
    pin = dj["gguf"]
    if (pin["repo"], pin["revision"], pin["path"], pin["sha256"]) != (repo, commit, v["gguf"], oid):
        raise SystemExit("%s/decision.json was exported from %s@%s:%s (%s), not %s@%s:%s (%s)" % (
            v["export_dir"], pin["repo"], pin["revision"], pin["path"], pin["sha256"], repo, commit, v["gguf"], oid))
    gguf["annotations"] = {"org.ollaya.quantization": pin["quantization"]}
    decision = blobs.put(MEDIA["decision"], decision_bytes)
    calibration = blobs.put(MEDIA["calibration"], open(os.path.join(v["export_dir"], "calibration.json"), "rb").read())
    text = v.get("license_text") or spec["license_text"]
    if v.get("notice"):  # the author's NOTICE, as the pinned commit has it
        with urllib.request.urlopen("%s/%s/resolve/%s/%s" % (HF, repo, commit, v["notice"])) as r:
            text = r.read().decode() + "\n" + text
    lic = blobs.put(MEDIA["license"], text.encode())
    config = blobs.put(MEDIA["config"], json.dumps({
        "model_format": "gguf", "engine": "llama", "layout": dj["layout"], "family": spec["family"],
        "parameter_size": v["parameter_size"], "context_length": dj["llama"]["n_ctx"],
        "languages": v["languages"], "description": v["description"],
        "source": "huggingface.co/%s@%s" % (repo, commit), "license": v.get("license") or spec["license"],
        "release_date": hf_commit_date(repo, commit),
    }, indent=2).encode())
    print("  %s:%s %s %.2f GB (%s)" % (spec["model"], tag, pin["quantization"], gguf["size"] / 1e9, v["gguf"]))
    return config, [gguf, decision, calibration, lic]


def package_model(spec, blobs):
    ns, model = spec["namespace"], spec["model"]
    lic = blobs.put(MEDIA["license"], spec["license_text"].encode())
    for tag, v in spec["tags"].items():
        if v.get("kind") == "gguf":
            config, layers = package_gguf(spec, tag, v, blobs)
            write_manifest(ns, model, tag, config, layers)
            for alias, target in spec.get("aliases", {}).items():
                if target == tag:
                    write_manifest(ns, model, alias, config, layers)
            continue
        if v.get("kind") == "wl":
            config, layers = package_wl(spec, tag, v, blobs)
            write_manifest(ns, model, tag, config, layers)
            for alias, target in spec.get("aliases", {}).items():
                if target == tag:
                    write_manifest(ns, model, alias, config, layers)
            continue
        repo, commit = v["repo"], v["commit"]
        weights = upstream(MEDIA["weights"], repo, commit, v["weights"])
        tokenizer = upstream(MEDIA["tokenizer"], repo, commit, v["tokenizer"])
        oid = weights["digest"].split(":", 1)[1]
        with open(v["checkpoint"], "rb") as f:  # the local copy must be the pinned upstream file
            if hashlib.file_digest(f, "sha256").hexdigest() != oid:
                raise SystemExit("%s does not match %s@%s:%s" % (v["checkpoint"], repo, commit, v["weights"]))
        graphs = []
        for precision, export_dir in v["exports"].items():
            data, stats = graph_bytes(export_dir, v["checkpoint"], v.get("prefix", ""), oid)
            print("  %s:%s %s graph %.1f MB %s" % (model, tag, precision, len(data) / 2**20, stats))
            graphs.append(blobs.put(MEDIA["graph"], data, {"org.ollaya.precision": precision}))
        export0 = next(iter(v["exports"].values()))
        decision = blobs.put(MEDIA["decision"], open(os.path.join(export0, "decision.json"), "rb").read())
        calibration = blobs.put(MEDIA["calibration"], open(os.path.join(export0, "calibration.json"), "rb").read())
        config = blobs.put(MEDIA["config"], json.dumps({
            "model_format": "onnx", "family": spec["family"], "parameter_size": v["parameter_size"],
            "context_length": v["context_length"], "languages": v["languages"],
            "description": v["description"], "source": "huggingface.co/%s@%s" % (repo, commit),
            "license": spec["license"], "release_date": hf_commit_date(repo, commit),
        }, indent=2).encode())
        layers = graphs + [weights, tokenizer, decision, calibration, lic] + arch_layers(v, blobs, v["checkpoint"])
        write_manifest(ns, model, tag, config, layers)
        # Precision-pinned variants: same layers plus a params layer the runtime honours.
        for precision in v["exports"]:
            params = blobs.put(MEDIA["params"], json.dumps({"precision": precision}).encode())
            write_manifest(ns, model, "%s-%s" % (tag, precision), config, layers + [params])

    for tag, r in spec.get("routers", {}).items():
        router = blobs.put(MEDIA["router"], json.dumps(r["router"], indent=2).encode())
        config = blobs.put(MEDIA["config"], json.dumps({
            "model_format": "router", "family": spec["family"], "parameter_size": "",
            "context_length": 0, "languages": r["languages"], "description": r["description"],
            "source": "", "license": spec["license"],
        }, indent=2).encode())
        write_manifest(ns, model, tag, config, [router, lic])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", choices=sorted(CATALOG))
    ap.add_argument("--origin", default=os.environ.get("SITE_ORIGIN", "https://ollaya.dev"),
                    help="public origin that serves registry/ (the website)")
    a = ap.parse_args()
    blobs = Blobs(a.origin)
    for m in a.models:
        print(m)
        package_model(CATALOG[m], blobs)


if __name__ == "__main__":
    main()
