"""Turn one Ollaya Bench edition's raw outputs into the result files the website reads (results/README.md).

    python -m ollaya_convert.bench_edition --raw OUT_DIR --public-data PUBLIC_DIR --typed-data typed.jsonl \
        --edition 2026.10 --date 2026-10-14 --machine rtx-5090 --models winnow:e4b kev:4b ... --results ../results

With --update RELEASE, the runs go into the existing edition instead (ADR 0009, a rolling edition): only the models
a release changed are measured (`bench_plan`), their run files are added to the edition's runs, where a later run's
row replaces an earlier one of the same model, their scores replace the old ones, and `updates` records the release.
`-fp16` tags are not added to the edition's models: the site shows them under their base tag, unranked.

OUT_DIR is what `bench_run` writes on the bench machine:

    latency-<machine>-cuda.json, latency-<machine>-cpu.json    bench_latency.py, with --memory
    typed/<model>/rows.jsonl                                   bench_typed.py
    public/<model>/<subset>/{rows.jsonl,summary.json}          bench_public.py

(<model> is the tag with ':' as '-'). It writes, under RESULTS:

    runs/<date>-latency-<machine>-<device>.json      the latency files, with the edition named
    runs/<date>-public-benchmark-<machine>-cuda.json Bespoke Labs' public benchmark, as earlier runs
    runs/<date>-typed-decisions-<machine>-cuda.json  typed-decisions
    editions/<edition>.json                          the edition: its runs, models, score and flags

With --label, every run file name ends in `-<label>` (`<date>-latency-<machine>-cuda-<label>.json`), so a second
release measured the same day does not overwrite the first one's files.

The edition score of a model is the mean, over the 13 public subsets and typed-decisions (14 tasks, equal
weight), of the chance-corrected accuracy (accuracy - chance) / (1 - chance), where chance is the mean of
1/K over the task's questions (K options). 0 is guessing, 100 is every answer right; a rejected request is
wrong. A model trained on a task's train split (TRAINED_ON) is scored on the other tasks. Its 95 % interval is a bootstrap (1,000 draws, fixed seed): questions resampled within each public
subset, typed-decisions states resampled whole (their five questions together).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from .bench_public import SUBSETS, summarize as summarize_public
from .bench_typed import option_keys, score_rows, summarize as summarize_typed

DRAWS = 1000
SEED = 20261014

# Models trained on the train split of a task: their score on it is not a held-out measurement.
TRAINED_ON = {
    "laya:typed-decisions": ["typed-decisions"],
    "largha:150m": ["typed-decisions"],
    "jeb:4b": ["typed-decisions"],
    # jeb:9b and jeb:27b are v2.1, whose 90 training sources (the card's table) add civil_comments, HelpSteer2 and
    # MASSIVE intents, parallel across languages, to typed-decisions.
    **{"jeb:" + t: ["typed-decisions", "civil_comments", "helpsteer2", "massive-en-US", "massive-de-DE"]
       for t in ("9b", "27b")},
    # The in-domain coba tags saw the train splits of ten public tasks (DATA_SOURCES.tsv, indomain/*). MASSIVE
    # is parallel across languages, so its German subset is not held out either.
    **{"coba:" + t: ["vitaminc-dev", "massive-en-US", "massive-de-DE", "boolq", "squad2", "paws", "multinli",
                     "civil_comments", "aegis2", "helpsteer2", "pubmedqa"]
       for t in ("400m-ternary", "150m-ternary", "68m-ternary")},
}


def model_dir(model):
    return model.replace(":", "-")


def n_options(qdef):
    return 2 if qdef["type"] == "noul" else len(qdef["criteria"])


def public_items(raw, model, public_data):
    """subset -> [(correct, chance)] in data order; a missing or rejected row is wrong."""
    out = {}
    for subset in SUBSETS:
        rows_path = raw / "public" / model_dir(model) / subset / "rows.jsonl"
        if not rows_path.exists():
            return None
        got = {}
        for line in open(rows_path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                got[r["id"]] = r
        items = []
        for line in open(public_data / subset / "all.jsonl", encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            qdef = d["input"]["questions"]["decision"]
            r = got.get(d["id"])
            correct = 1 if r and "error" not in r and r.get("student", {}).get("correct") else 0
            items.append((correct, 1.0 / n_options(qdef)))
        out[subset] = items
    return out


def typed_items(raw, model, typed_data):
    """[(row, correct, chance)] per question; None when the model has no typed run."""
    path = raw / "typed" / model_dir(model) / "rows.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    if len(rows) < len(typed_data):
        return None
    scored = score_rows(typed_data, rows)
    chances = [1.0 / len(option_keys(q)) for row in typed_data for qid, q in row["questions"].items() if qid in row["gold"]]
    return [(x["row"], x["correct"], c) for x, c in zip(scored, chances)]


def skill(pairs):
    acc = statistics.mean(c for c, _ in pairs)
    chance = statistics.mean(ch for _, ch in pairs)
    return 100 * (acc - chance) / (1 - chance)


def task_skills(pub, typed):
    tasks = {s: skill(items) for s, items in pub.items()}
    tasks["typed-decisions"] = skill([(c, ch) for _, c, ch in typed])
    return tasks


def score_of(tasks, excluded):
    """The mean skill over the tasks the model was not trained on."""
    return statistics.mean(v for k, v in tasks.items() if k not in excluded)


def bootstrap(pub, typed, rng, excluded=()):
    by_row = {}
    for row, c, ch in typed:
        by_row.setdefault(row, []).append((c, ch))
    rows = list(by_row.values())
    scores = []
    for _ in range(DRAWS):
        parts = [skill(rng.choices(items, k=len(items))) for s, items in pub.items() if s not in excluded]
        if "typed-decisions" not in excluded:
            parts.append(skill([q for r in rng.choices(rows, k=len(rows)) for q in r]))
        scores.append(statistics.mean(parts))
    scores.sort()
    return [round(scores[int(0.025 * DRAWS)], 2), round(scores[int(0.975 * DRAWS) - 1], 2)]


def macro_ci(pub, rng):
    """95 % bootstrap interval of the macro accuracy: questions resampled within each subset."""
    vals = []
    for _ in range(DRAWS):
        vals.append(statistics.mean(sum(c for c, _ in rng.choices(items, k=len(items))) / len(items) for items in pub.values()))
    vals.sort()
    return [round(vals[int(0.025 * DRAWS)], 4), round(vals[int(0.975 * DRAWS) - 1], 4)]


def acc_ci(groups, rng):
    """95 % bootstrap interval of a mean of 0/1 items, the groups (lists) resampled whole."""
    vals = []
    for _ in range(DRAWS):
        draw = [x for g in rng.choices(groups, k=len(groups)) for x in g]
        vals.append(sum(draw) / len(draw))
    vals.sort()
    return [round(vals[int(0.025 * DRAWS)], 4), round(vals[int(0.975 * DRAWS) - 1], 4)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--public-data", required=True)
    ap.add_argument("--typed-data", required=True)
    ap.add_argument("--edition", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--machine", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--update", metavar="RELEASE", help="add these runs to the existing edition (ADR 0009)")
    ap.add_argument("--label", default="", help="appended to the run file names (a second update on the same date)")
    ap.add_argument("--note", default="", help="added to the public and typed run files' note (for example, rows "
                    "carried from an earlier build of the same graphs)")
    a = ap.parse_args()
    raw, public_data, results = Path(a.raw), Path(a.public_data), Path(a.results)
    typed_data = [json.loads(line) for line in open(a.typed_data, encoding="utf-8") if line.strip()]
    (results / "runs").mkdir(parents=True, exist_ok=True)
    (results / "editions").mkdir(parents=True, exist_ok=True)
    stem = "%s-%%s-%s" % (a.date, a.machine.replace("-", ""))
    label = "-" + a.label if a.label else ""
    files = []

    # Latency: the files as bench_latency.py wrote them, with the edition named.
    digests, version = {}, None
    for device in ("cuda", "cpu"):
        src = raw / ("latency-%s-%s.json" % (a.machine.replace("-", ""), device))
        if not src.exists():
            continue
        run = json.load(open(src, encoding="utf-8"))
        run["edition"] = a.edition
        version = version or run.get("ollaya")
        for r in run["results"]:
            if r.get("digest"):
                digests.setdefault(r["model"], r["digest"])
        name = "%s-%s%s.json" % (stem % "latency", device, label)
        json.dump(run, open(results / "runs" / name, "w", encoding="utf-8"), indent=1)
        files.append(name)

    rng = random.Random(SEED)
    public_results, typed_results, scores = [], [], []
    for model in a.models:
        pub = public_items(raw, model, public_data)
        if pub is not None:
            s = summarize_public(raw / "public" / model_dir(model))
            per_subset = {k: {"n": v["n"], "errors": v["errors"], "accuracy": round(v["accuracy"], 4),
                              "ece": None if v["ece10"] is None else round(v["ece10"], 4)} for k, v in s["per_subset"].items()}
            public_results.append({
                "server": "ollaya", "model": model, "digest": digests.get(model),
                "records": s["records"], "errors": s["errors"],
                "macro_accuracy": round(s["macro_accuracy"], 4),
                "macro_accuracy_ci": macro_ci(pub, rng),
                "micro_accuracy": round(s["micro_accuracy"], 4),
                "macro_ece": round(s["macro_ece10"], 4), "pooled_ece": round(s["pooled_ece10"], 4),
                "brier": round(s["mean_brier"], 4), "median_ms": round(s["median_ms"], 1), "p95_ms": round(s["p95_ms"], 1),
                "by_type": {k: (None if v is None else round(v, 4)) for k, v in s["by_type"].items()},
                "per_subset": per_subset,
            })
        typed = typed_items(raw, model, typed_data)
        if typed is not None:
            t = summarize_typed(raw / "typed" / model_dir(model), typed_data)
            clusters = {}
            for row, c, _ in typed:
                clusters.setdefault(row, []).append(c)
            typed_results.append({
                "model": model, "digest": digests.get(model), "answered_by": t["answered_by"],
                "rows": t["rows"], "questions": t["questions"], "errors": t["errors"],
                "accuracy": round(t["accuracy"], 4), "accuracy_ci": acc_ci(list(clusters.values()), rng),
                "by_type": {k: (None if v is None else round(v, 4)) for k, v in t["by_type"].items()},
                "by_workflow": {k: round(v, 4) for k, v in t["by_workflow"].items()},
                "ece": None if t["ece"] is None else round(t["ece"], 4),
                "brier": None if t["brier"] is None else round(t["brier"], 4),
                "soft_ce": None if t["soft_ce"] is None else round(t["soft_ce"], 4),
                "median_ms": None if t["median_ms"] is None else round(t["median_ms"], 1),
                "p95_ms": None if t["p95_ms"] is None else round(t["p95_ms"], 1),
                "trained_on_split": "typed-decisions" in TRAINED_ON.get(model, []),
            })
        if pub is not None and typed is not None:
            tasks = task_skills(pub, typed)
            excluded = TRAINED_ON.get(model, [])
            scores.append({"model": model, "score": round(score_of(tasks, excluded), 2),
                           "score_ci": bootstrap(pub, typed, rng, excluded),
                           "tasks": {k: round(v, 2) for k, v in tasks.items()},
                           "trained_on": TRAINED_ON.get(model, [])})
            print("%-24s score %6.2f" % (model, scores[-1]["score"]), flush=True)

    common = {"schema": 1, "date": a.date, "edition": a.edition, "machine": a.machine, "device": "cuda"}
    if public_results:
        name = "%s-cuda%s.json" % (stem % "public-benchmark", label)
        prev = json.load(open(results / "runs" / "2026-09-30-public-benchmark-rtx5090-cuda.json", encoding="utf-8"))
        json.dump({**common, "suite": "public-benchmark", "servers": {"ollaya": version},
                   "protocol": prev["protocol"], "note": ("Ollaya Bench edition %s. %s" % (a.edition, a.note)).strip(),
                   "results": sorted(public_results, key=lambda r: -r["macro_accuracy"])},
                  open(results / "runs" / name, "w", encoding="utf-8"), indent=1)
        files.append(name)
    if typed_results:
        name = "%s-cuda%s.json" % (stem % "typed-decisions", label)
        json.dump({**common, "suite": "typed-decisions", "ollaya": version, **({"note": a.note} if a.note else {}), "protocol": {
            "benchmark": "typed-decisions test split: 400 JSON states from four workflows, five typed questions each "
                         "(2,000 questions: 600 choice, 600 noul, 800 score)",
            "source": "https://huggingface.co/datasets/LocalLLaMA/typed-decisions",
            "runner": "convert/ollaya_convert/bench_typed.py",
            "client": "one request (one state, its five questions) at a time over HTTP to /v1/systemone, timed by the client",
            "accuracy": "argmax of the returned probabilities against the gold label, over all 2,000 questions; a rejected "
                        "request counts as wrong on its five questions; 95 % bootstrap interval over states",
            "ece": "ten-bin expected calibration error of the top probability, pooled over the answered questions"},
            "results": sorted(typed_results, key=lambda r: -r["accuracy"])},
            open(results / "runs" / name, "w", encoding="utf-8"), indent=1)
        files.append(name)

    if a.update:
        update(results / "editions" / ("%s.json" % a.edition), a.update, a.date, files, a.models, scores)
        return
    edition = {
        "schema": 1, "id": a.edition, "title": "Ollaya Bench %s" % a.edition, "date": a.date, "ollaya": version,
        "machine": a.machine, "runs": files, "models": a.models,
        "score": {
            "definition": "Mean over 14 tasks (13 public subsets and typed-decisions, equal weight) of the chance-corrected "
                          "accuracy 100 * (accuracy - chance) / (1 - chance); chance is the mean of 1/K over the task's questions. "
                          "A model trained on a task's train split is scored on the other tasks",
            "interval": "95 %% bootstrap, %d draws, seed %d; questions resampled within each subset, typed-decisions states whole"
                        % (DRAWS, SEED),
            "results": sorted(scores, key=lambda r: -r["score"]),
        },
        "trained_on": TRAINED_ON,
    }
    json.dump(edition, open(results / "editions" / ("%s.json" % a.edition), "w", encoding="utf-8"), indent=1)
    print("wrote", ", ".join(files), "and editions/%s.json" % a.edition)


def merge_update(edition, release, date, files, models, scores):
    """`edition` with one release's runs, models and scores added (a rolling update)."""
    edition["runs"] += [f for f in files if f not in edition["runs"]]
    edition["models"] += [m for m in models if m not in edition["models"] and not m.endswith("-fp16")]
    new = {s["model"] for s in scores}
    edition["score"]["results"] = sorted([r for r in edition["score"]["results"] if r["model"] not in new] + scores,
                                         key=lambda r: -r["score"])
    edition.setdefault("updates", []).append({"release": release, "date": date, "runs": files, "models": models})
    return edition


def update(path, release, date, files, models, scores):
    edition = merge_update(json.load(open(path, encoding="utf-8")), release, date, files, models, scores)
    json.dump(edition, open(path, "w", encoding="utf-8"), indent=1)
    print("wrote", ", ".join(files), "and updated", path.name)


if __name__ == "__main__":
    main()
