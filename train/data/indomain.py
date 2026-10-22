"""In-domain rows: the *train* splits of the public benchmark's datasets, asked exactly as the benchmark asks
(nimble's own source modules build every record), minus any row that shares a text with a benchmark record.

    NIMBLE_SRC=/path/to/nimble python indomain.py --bench <public-dir> --cache <dir> --out indomain.jsonl

The benchmark evaluates on validation/test splits; nothing here reads those. SummEval has no train split and
MASSIVE de-DE is not English, so both stay out. A model trained with these rows reports its public-benchmark
score as in-domain (train splits seen), not zero-shot.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import urllib.request

import pyarrow.parquet as pq

PER_SOURCE = 20000
MNLI_LABELS = ("entailment", "neutral", "contradiction")
MASSIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
# module name -> (Hugging Face repo, parquet config, adapter from the parquet row to the module's raw row)
SOURCES = {
    "multinli": ("nyu-mll/multi_nli", "default", lambda r: None if r["label"] not in (0, 1, 2) else {
        "pairID": r["pairID"], "promptID": str(r["promptID"]), "genre": r["genre"], "sentence1": r["premise"],
        "sentence2": r["hypothesis"], "gold_label": MNLI_LABELS[r["label"]],
        "annotator_labels": [MNLI_LABELS[r["label"]]]}),
    "boolq": ("google/boolq", "default", dict),
    "paws": ("google-research-datasets/paws", "labeled_final", dict),
    "vitaminc": ("tals/vitaminc", "default", dict),
    "civil_comments": ("google/civil_comments", "default", dict),
    "aegis2": ("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "default", dict),
    "helpsteer2": ("nvidia/HelpSteer2", "default", dict),
    "pubmedqa": ("qiaojin/PubMedQA", "pqa_artificial", dict),
    "squad2": ("rajpurkar/squad_v2", "squad_v2", dict),
}


def fetch(url, path):
    if not os.path.exists(path):
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    return path


def train_rows(repo, config, cache):
    """The train split's rows; only the first parquet shard of a large split (it is sampled anyway)."""
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}/parquet/{config}/train") as f:
        url = json.load(f)[0]
    path = fetch(url, os.path.join(cache, repo.replace("/", "__") + f"__{config}__train0.parquet"))
    return pq.read_table(path).to_pylist()


def texts(value):
    """Every string of 30+ characters in a state, normalized."""
    if isinstance(value, str):
        v = " ".join(value.split()).lower()
        return {v} if len(v) >= 30 else set()
    if isinstance(value, dict):
        return set().union(*(texts(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(texts(v) for v in value)) if value else set()
    return set()


def gold_of(question, rec):
    """Target distribution in our option order (choice criteria order, noul [false, true], score levels)."""
    ref, kind = rec["reference"], question["type"]
    dist = ref.get("distribution")
    if kind == "noul":
        p = dist["true"] if dist else float(ref["target"])
        return [1 - p, p]
    if kind == "score":
        n = len(question["criteria"])
        return [dist[str(i)] for i in range(n)] if dist else [float(i == ref["target"]) for i in range(n)]
    keys = list(question["criteria"])
    return [dist[k] for k in keys] if dist else [float(k == ref["target"]) for k in keys]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, help="the public benchmark's directory (one folder of JSONL per subset)")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-source", type=int, default=PER_SOURCE)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, os.environ["NIMBLE_SRC"])
    import importlib

    os.makedirs(a.cache, exist_ok=True)
    seen = set()
    for path in glob.glob(os.path.join(a.bench, "*", "*.jsonl")):
        with open(path, encoding="utf-8", newline="\n") as f:
            for line in f:
                if line.strip():
                    seen |= texts(json.loads(line)["input"]["state"])
    print(f"{len(seen)} benchmark texts to keep out", flush=True)

    jobs = [(name, *spec) for name, spec in SOURCES.items()] + [("massive", None, None, None)]
    rng = random.Random(a.seed)
    stats = {}
    with open(a.out, "w", encoding="utf-8") as out:
        for name, repo, config, adapt in jobs:
            mod = importlib.import_module(f"nimble.datasets.public_sources.{name}")
            if name == "massive":
                mod.PARTITION = "train"
                raws = mod.rows(fetch(MASSIVE_URL, os.path.join(a.cache, "massive-1.1.tar.gz")), "en-US")
                adapt, subset = dict, "en-US"
            else:
                raws, subset = train_rows(repo, config, a.cache), ""
            if name in ("civil_comments", "helpsteer2"):
                raws = [{**r, "row_index": i} for i, r in enumerate(raws)]
            rng.shuffle(raws)
            n = clash = bad = 0
            for raw in raws:
                if n >= a.per_source:
                    break
                try:
                    raw = adapt(raw)
                    rec = raw and mod.record(raw, subset)
                except (ValueError, KeyError, TypeError):
                    bad += 1
                    continue
                if rec is None:
                    continue
                state = rec["input"]["state"]
                if texts(state) & seen:
                    clash += 1
                    continue
                question = rec["input"]["questions"]["decision"]
                out.write(json.dumps({"id": "indomain:" + rec["id"], "source": "indomain/" + name,
                                      "license": mod.LICENSE, "state": state, "questions": {"q": question},
                                      "gold": {"q": gold_of(question, rec)}}, ensure_ascii=False) + "\n")
                n += 1
            stats[name] = {"rows": n, "benchmark_clash": clash, "unusable": bad, "train_split": len(raws)}
            print(name, stats[name], flush=True)
    with open(a.out + ".stats.json", "w") as f:
        json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
