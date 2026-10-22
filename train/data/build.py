"""Build the license-clean distillation set for the Laya student (build-time only).

Normalizes every source into rows
  {"id", "source", "license", "state", "questions": {qid: {"type", "instructions", "criteria"}},
   "gold": {qid: [probabilities in option order, noul as [false, true]]} | null}
drops non-permissive jev sources and everything derived from the 11 public-benchmark datasets,
drops rows over 512 ettin tokens, and samples the main set.

  uv run --with datasets --with tokenizers python train/data/build.py --out ~/ollaya-train/data
"""

import argparse
import collections
import json
import os
import random
import re

from datasets import load_dataset
from huggingface_hub import HfApi
from tokenizers import Tokenizer

MAX_TOKENS = 512
CAP_QUESTIONS = 400_000
SOURCE_SHARE = 0.25
WHOLE = {"tasksource-synthetic", "n4ze3m", "fastino", "tasksource-procedural"}  # sampled as one source each

# Bespoke Labs' public benchmark: vitaminc, MASSIVE, boolq, squad_v2, paws, mnli, civil_comments,
# aegis2, helpsteer2, summeval, pubmedqa. Generous: also their derivatives (qnli is SQuAD, wanli is
# seeded from MNLI, xnli and paws-x are translations, every HelpSteer and Jigsaw set, every pubmed set).
HOLDOUT = re.compile(
    r"vitamin|massive|boolq|bool_q|squad|qnli|paws|mnli|multi_?nli|wanli|xnli|civil_?comments|"
    r"jigsaw|unintended_bias|aegis|helpsteer|help_steer|summeval|summ_eval|pubmed",
    re.I,
)
# One recorded license term (jev's `license` column joins several with ", ").
PERMISSIVE = re.compile(
    r"apache-2\.0|apache license 2\.0|mit|mit license|bsd|bsd 2-clause license|"
    r"cc-by-(sa-)?[0-9.]+|cc by(-sa)? [0-9.]+|cc0-1\.0|cc0 1\.0|odc-by|pddl|afl-3\.0|"
    r"cdla-permissive-[0-9.]+"
)


# The student is English only. A row is foreign when another language's function words outnumber
# English's (structured English tables have few of either, so they stay).
EN_WORDS = frozenset(
    "the of and to is that for it with as are be this by not or have from at which but they you he she we "
    "his her their has were been would can there what if does did all about more one my your our its these "
    "those than then when who how why should could into out up any some".split()
)
FOREIGN_WORDS = frozenset(
    "de la el les des und der das ist nicht het een van yang dan ini itu untuk que por para com não une est "
    "pas che sono och är att ett og ikke bir bu los las del ein eine mit sie auf für wir ich je et du le il "
    "na ya ke kwa wa ni".split()
)


def english(row):
    """Cheap check: mostly ASCII letters (English quoting a foreign name passes), and no more
    foreign function words than English ones."""
    parts = [row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], ensure_ascii=False)]
    for q in row["questions"].values():
        if isinstance(q["instructions"], str):
            parts.append(q["instructions"])
        if q["type"] != "noul":
            parts.extend(map(str, q["criteria"]))
    text = " ".join(parts)
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isascii() for c in letters) < 0.85 * len(letters):
        return False
    words = re.findall(r"[^\W\d_]+", text.lower())
    foreign = sum(w in FOREIGN_WORDS for w in words)
    return not (foreign >= 3 and foreign > sum(w in EN_WORDS for w in words))


def permissive(license_str):
    terms = [re.sub(r"\s*\((dpi|original source)\)$", "", t.strip().lower()) for t in license_str.split(", ")]
    return all(PERMISSIVE.fullmatch(t) for t in terms)


def as_state(s):
    """A JSON object or array stays a JSON value; anything else is the string itself."""
    if isinstance(s, str) and s[:1] in "{[":
        try:
            v = json.loads(s)
            if isinstance(v, (dict, list)):
                return v
        except ValueError:
            pass
    return s


def onehot(n, i):
    return [1.0 if j == i else 0.0 for j in range(n)]


def options(q):
    c = q.get("criteria")
    if q["type"] == "choice":
        return [k if v in (None, "", k) else f"{k}: {v if isinstance(v, str) else json.dumps(v)}" for k, v in c.items()]
    if q["type"] == "score":
        return [v if isinstance(v, str) else json.dumps(v) for v in c]
    c = c or {}
    return [str(c.get("false") or "false"), str(c.get("true") or "true")]


def valid(q):
    n = len(options(q))
    if q["type"] == "choice":
        return 2 <= n <= 255
    if q["type"] == "score":
        return 2 <= n <= 10
    return q["type"] == "noul"


def answer_probs(q, a):
    """TypeSafe answer shape -> probabilities in option order."""
    if q["type"] == "noul":
        p = float(a["noul"])
        return [1 - p, p]
    p = a["probabilities"]
    keys = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
    return [float(p[k]) for k in keys]


# ---- sources ---------------------------------------------------------------------------------


def jev(info):
    """info[source] collects its recorded licenses and drop counts."""
    ds = load_dataset("tasksource/tasksource-jev-typed-decisions", "default", split="train")
    for r in ds:
        src, lic = r["source"], r["license"]
        s = info.setdefault(src, {"licenses": set(), "c": collections.Counter()})
        s["licenses"].add(lic)
        s = s["c"]
        s["rows"] += 1
        if HOLDOUT.search(src):
            s["holdout"] += 1
            continue
        if src.startswith("multilingual/"):
            s["non_english"] += 1
            continue
        if not permissive(lic):
            s["license_dropped"] += 1
            continue
        kind, opts = r["kind"], r["options"]
        if kind == "choice":
            if len(set(opts)) != len(opts):
                s["invalid"] += 1
                continue
            crit, gold = {o: None for o in opts}, r["target"]
        elif kind == "score":
            crit, gold = list(opts), r["target"]
        else:
            crit, gold = None, [1 - r["target"][0], r["target"][0]]
        q = {"type": kind, "instructions": r["question"], "criteria": crit}
        yield src, {
            "id": f"jev:{r['id']}",
            "source": f"jev/{src}",
            "license": lic,
            "state": as_state(r["state"]),
            "questions": {"q": q},
            "gold": {"q": [float(x) for x in gold]},
        }


def typesafe_rows(name, config, lic, prefix, sample=None, id_key="id", gold_key="answers"):
    ds = load_dataset(name, config, split="train")
    if sample and len(ds) > sample:
        ds = ds.shuffle(seed=0).select(range(sample))
    for r in ds:
        qs = json.loads(r["questions"])
        ans = json.loads(r[gold_key]) if r.get(gold_key) else {}
        gold = {}
        for qid, q in qs.items():
            q = {"type": q["type"], "instructions": q.get("instructions"), "criteria": q.get("criteria")}
            qs[qid] = q
            a = ans.get(qid)
            try:
                gold[qid] = answer_probs(q, a)
            except (KeyError, TypeError, ValueError):
                pass  # no usable gold for this question; the teacher label still applies
        yield prefix, {
            "id": f"{prefix}:{r[id_key]}",
            "source": prefix,
            "license": lic,
            "state": as_state(r["state"]),
            "questions": qs,
            "gold": gold or None,
        }


def n4ze3m():
    ds = load_dataset("n4ze3m/typed-decisions-synth", "default", split="train")
    for r in ds:
        qs, g = json.loads(r["questions"]), json.loads(r["gold"])
        gold = {}
        for qid, q in qs.items():
            q = {"type": q["type"], "instructions": q.get("instructions"), "criteria": q.get("criteria")}
            qs[qid] = q
            if qid not in g:
                continue
            v = g[qid]
            if q["type"] == "noul":
                gold[qid] = onehot(2, int(bool(v)))
            elif q["type"] == "score":
                gold[qid] = onehot(len(q["criteria"]), int(v))
            elif v in q["criteria"]:
                gold[qid] = onehot(len(q["criteria"]), list(q["criteria"]).index(v))
        yield "n4ze3m", {
            "id": f"n4ze3m:{r['state_id']}",
            "source": "n4ze3m",
            "license": "mit",
            "state": as_state(r["state"]),
            "questions": qs,
            "gold": gold or None,
        }


def fastino():
    files = sorted(f for f in HfApi().list_repo_files("fastino/fast-decisions", repo_type="dataset") if f.endswith(".jsonl"))
    for f in files:
        ds = load_dataset("json", data_files=f"hf://datasets/fastino/fast-decisions/{f}", split="train")
        for i, r in enumerate(ds):
            qs, gold = {}, {}
            # shortcut: multi-label tasks are skipped (no single-answer shape), add them as nouls if needed
            for c in r["output"]["classifications"]:
                if c["multi_label"] or len(c["true_label"]) != 1 or c["true_label"][0] not in c["labels"]:
                    continue
                qs[c["task"]] = {"type": "choice", "instructions": c["task"].replace("_", " "), "criteria": {k: None for k in c["labels"]}}
                gold[c["task"]] = onehot(len(c["labels"]), c["labels"].index(c["true_label"][0]))
            if qs:
                yield "fastino", {
                    "id": f"fastino:{f[:-6]}:{i}",
                    "source": "fastino",
                    "license": "apache-2.0",
                    "state": r["input"],
                    "questions": qs,
                    "gold": gold,
                }


# ---- length filter ---------------------------------------------------------------------------


def texts(row):
    """The parts laya puts in its sequence: each question header, each option, the state."""
    st = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], ensure_ascii=False, separators=(",", ":"))
    out = [st]
    for qid, q in row["questions"].items():
        instr = q["instructions"] if q["instructions"] is not None else qid
        out.append(f"{q['type']} question: {instr if isinstance(instr, str) else json.dumps(instr)}")
        out.extend(options(q))
    return out


def token_lengths(tok, rows, batch=20_000):
    """Sequence length per row: every part's tokens, one [MASK] per option, [CLS] and [SEP]."""
    lens = []
    for b in range(0, len(rows), batch):
        chunk = rows[b : b + batch]
        flat, owner = [], []
        for i, r in enumerate(chunk):
            t = texts(r)
            flat.extend(t)
            owner.extend([i] * len(t))
        enc = tok.encode_batch(flat, add_special_tokens=False)
        out = [2] * len(chunk)
        for i, e in zip(owner, enc):
            out[i] += len(e.ids)
        for i, r in enumerate(chunk):
            out[i] += sum(len(options(q)) for q in r["questions"].values())
        lens.extend(out)
    return lens


def waterfill(counts, budget):
    """Largest per-source cap c with sum(min(n, c)) <= budget."""
    lo, hi = 0, max(counts.values(), default=0)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(min(n, mid) for n in counts.values()) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def selftest():
    assert permissive("cc-by-4.0, CC BY 4.0 (DPI)") and permissive("apache-2.0, apache-2.0 (original source)")
    assert not permissive("cc-by-nc-4.0") and not permissive("apache-2.0, Custom (DPI)") and not permissive("cc")
    assert not permissive("unspecified") and not permissive("mit, other") and permissive("bsd")
    for s in ("glue/mnli", "super_glue/boolq", "squad-v2/answerability", "HelpSteer2/coherence", "multilingual/paws-x/en", "civil_comments/insult_share"):
        assert HOLDOUT.search(s), s
    assert not HOLDOUT.search("anli/a1") and not HOLDOUT.search("lex_glue/ledgar")
    assert waterfill({"a": 10, "b": 3}, 9) == 6
    assert as_state('{"a": 1}') == {"a": 1} and as_state("{not json") == "{not json"
    q = {"q": {"type": "noul", "instructions": "Is it urgent?", "criteria": None}}
    assert english({"state": "The Kalinin K-7 (Russian: Калинин К-7) was a heavy experimental aircraft designed and tested in the Soviet Union in the early 1930s.", "questions": q})
    assert english({"state": "item | qty --- | --- kettle | 3", "questions": q})
    assert not english({"state": "Ich bin auf der Suche nach einem neuen Buch und würde gerne wissen.", "questions": q})
    assert not english({"state": "줄리아니 트럼프 배신 가능성 질문에 좋은 보험 있다", "questions": q})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    selftest()
    os.makedirs(args.out, exist_ok=True)
    tok = Tokenizer.from_pretrained("jhu-clsp/ettin-encoder-150m")
    rng = random.Random(args.seed)

    licenses = {}
    gens = [
        jev(licenses),
        typesafe_rows("tasksource/synthetic-typed-decisions", "default", "apache-2.0", "tasksource-synthetic"),
        n4ze3m(),
        fastino(),
        typesafe_rows("tasksource/procedural-typed-decisions", "all", "apache-2.0", "tasksource-procedural", sample=30_000),
    ]
    by_source = collections.defaultdict(list)
    invalid, non_english = collections.Counter(), collections.Counter()
    for g in gens:
        for key, row in g:
            if not all(valid(q) for q in row["questions"].values()):
                invalid[key] += 1
            elif not english(row):
                non_english[key] += 1
            else:
                by_source[key].append(row)

    stats = {}
    for key, rows in by_source.items():
        lens = token_lengths(tok, rows)
        kept = [r for r, n in zip(rows, lens) if n <= MAX_TOKENS]
        stats[key] = {"rows": len(rows), "fit_512": len(kept), "questions": sum(len(r["questions"]) for r in kept)}
        by_source[key] = kept
        print(key, stats[key], flush=True)

    # Main set: the typed-decision datasets whole, each at most SOURCE_SHARE of CAP_QUESTIONS; the
    # jev sub-sources fill the rest evenly (water-filling over question counts).
    share = int(SOURCE_SHARE * CAP_QUESTIONS)
    qcount = {k: v["questions"] for k, v in stats.items()}
    whole = {k: min(n, share) for k, n in qcount.items() if k in WHOLE}
    jq = {k: n for k, n in qcount.items() if k not in WHOLE}
    cap = min(waterfill(jq, min(CAP_QUESTIONS - sum(whole.values()), sum(jq.values()))), share)
    main_rows = []
    for key, rows in by_source.items():
        rng.shuffle(rows)
        taken, limit = 0, whole.get(key, cap)
        for r in rows:
            if taken + len(r["questions"]) > limit:
                continue
            taken += len(r["questions"])
            main_rows.append(r)
        stats[key]["sampled_questions"] = taken
    rng.shuffle(main_rows)
    with open(os.path.join(args.out, "main.jsonl"), "w") as f:
        for r in main_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # LocalLLaMA/typed-decisions train only, never in the main set.
    td = [r for _, r in typesafe_rows("LocalLLaMA/typed-decisions", "all", "apache-2.0", "td-train", gold_key="gold")]
    with open(os.path.join(args.out, "td-train.jsonl"), "w") as f:
        for r in td:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(os.path.join(args.out, "licenses.tsv"), "w") as f:
        f.write("source\tlicense\tdecision\trows\tholdout\tlicense_dropped\tnon_english\tfit_512\tsampled_questions\n")
        for key in sorted(licenses):
            lic = " | ".join(sorted(licenses[key]["licenses"]))
            c, s = licenses[key]["c"], stats.get(key, {})
            decision = "holdout" if c["holdout"] else "drop" if c["license_dropped"] else "non-english" if c["non_english"] else "keep"
            ne = c["non_english"] + non_english[key]
            f.write(f"jev/{key}\t{lic}\t{decision}\t{c['rows']}\t{c['holdout']}\t{c['license_dropped']}\t{ne}\t{s.get('fit_512', 0)}\t{s.get('sampled_questions', 0)}\n")
        for key in sorted(WHOLE):
            s = stats.get(key, {})
            lic = {"n4ze3m": "mit"}.get(key, "apache-2.0")
            f.write(f"{key}\t{lic}\tkeep\t{s.get('rows', 0) + non_english[key]}\t0\t0\t{non_english[key]}\t{s.get('fit_512', 0)}\t{s.get('sampled_questions', 0)}\n")

    types = collections.Counter(q["type"] for r in main_rows for q in r["questions"].values())
    summary = {
        "cap_per_source": cap,
        "main_rows": len(main_rows),
        "main_questions": sum(types.values()),
        "types": types,
        "invalid_rows": invalid,
        "non_english_rows": sum(non_english.values()) + sum(v["c"]["non_english"] for v in licenses.values()),
        "holdout_hits": {f"jev/{k}": v["c"]["holdout"] for k, v in licenses.items() if v["c"]["holdout"]},
        "td_train_rows": len(td),
        "sources": stats,
    }
    json.dump(summary, open(os.path.join(args.out, "build-stats.json"), "w"), indent=1, default=dict)
    print(json.dumps({k: v for k, v in summary.items() if k != "sources"}, default=dict))


if __name__ == "__main__":
    main()
