"""Quality on typed-decisions (LocalLLaMA/typed-decisions test split, 400 rows x ~5 questions) with its
soft gold distributions: accuracy against the gold argmax label, cross-entropy against the gold
distribution, and ECE (15 bins); plus per-type temperature fitting (grid, min soft cross-entropy).

A scorer maps (state, questions) -> {qid: option logits in wire order} (choice: criteria order; noul:
[false, true]; score: level order). Rows are split in two halves (even / odd) so temperatures fitted on
one half are evaluated on the other.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List

import numpy as np

TYPES = ("choice", "score", "noul")
GRID = [round(float(x), 4) for x in np.exp(np.linspace(np.log(0.2), np.log(40.0), 181))]  # 0.2 .. 40, log-spaced


def gold_vector(qdef, gold):
    t = qdef["type"]
    probs = gold["probabilities"]
    if t == "choice":
        crit = qdef["criteria"]
        keys = list(crit) if isinstance(crit, dict) else list(crit)
    elif t == "noul":
        keys = ["false", "true"]
    else:
        keys = [str(i) for i in range(len(qdef["criteria"]))]
    v = np.array([float(probs.get(k, 0.0)) for k in keys])
    return v / v.sum(), keys.index(str(gold["label"])) if str(gold["label"]) in keys else int(v.argmax())


def collect(scorer: Callable, rows, progress=None):
    """-> list of {"type", "z", "gold", "label", "row"}"""
    out = []
    for ri, ((cid, state, questions), gold) in enumerate(rows):
        z = scorer(state, questions)
        for qid, qdef in questions.items():
            if qid not in z or qid not in gold:
                continue
            g, lab = gold_vector(qdef, gold[qid])
            out.append({"type": qdef["type"], "z": np.asarray(z[qid], dtype=np.float64), "gold": g, "label": lab,
                        "row": ri, "id": cid, "qid": qid})
        if progress and (ri + 1) % progress == 0:
            print("  %d/%d rows" % (ri + 1, len(rows)), flush=True)
    return out


def _probs(z, t):
    zz = z / t
    e = np.exp(zz - zz.max())
    return e / e.sum()


def metrics(items: List[dict], temps: Dict[str, float]):
    res = {}
    for t in TYPES + ("all",):
        sel = [x for x in items if t == "all" or x["type"] == t]
        if not sel:
            continue
        ps = [_probs(x["z"], temps[x["type"]]) for x in sel]
        acc = float(np.mean([p.argmax() == x["label"] for p, x in zip(ps, sel)]))
        ce = float(np.mean([-(x["gold"] * np.log(np.maximum(p, 1e-12))).sum() for p, x in zip(ps, sel)]))
        nll = float(np.mean([-math.log(max(p[x["label"]], 1e-12)) for p, x in zip(ps, sel)]))
        conf = np.array([p.max() for p in ps])
        hit = np.array([p.argmax() == x["label"] for p, x in zip(ps, sel)], dtype=float)
        ece = 0.0
        for b in range(15):
            m = (conf > b / 15) & (conf <= (b + 1) / 15)
            if m.any():
                ece += m.mean() * abs(conf[m].mean() - hit[m].mean())
        res[t] = {"n": len(sel), "acc": round(acc, 4), "soft_ce": round(ce, 4), "nll": round(nll, 4), "ece": round(float(ece), 4)}
    return res


def fit_temperatures(items: List[dict]):
    temps = {}
    for t in TYPES:
        sel = [x for x in items if x["type"] == t]
        if not sel:
            temps[t] = 1.0
            continue
        best = min(GRID, key=lambda T: np.mean([-(x["gold"] * np.log(np.maximum(_probs(x["z"], T), 1e-12))).sum()
                                                for x in sel]))
        temps[t] = best
    return temps


def cross_fit(items: List[dict]):
    """Fit on even rows, evaluate on odd rows, and vice versa; plus the fit on all rows (what ships)."""
    even = [x for x in items if x["row"] % 2 == 0]
    odd = [x for x in items if x["row"] % 2 == 1]
    t_even, t_odd = fit_temperatures(even), fit_temperatures(odd)
    one = {t: 1.0 for t in TYPES}
    return {
        "raw_T1": metrics(items, one),
        "fit_even_eval_odd": {"temperatures": t_even, "metrics": metrics(odd, t_even)},
        "fit_odd_eval_even": {"temperatures": t_odd, "metrics": metrics(even, t_odd)},
        "fit_all": fit_temperatures(items),
    }


def dump(items: List[dict], path: str):
    import json

    with open(path, "w") as f:
        for x in items:
            f.write(json.dumps({"id": x["id"], "qid": x["qid"], "type": x["type"], "row": x["row"],
                                "z": [float(v) for v in x["z"]], "gold": [float(v) for v in x["gold"]],
                                "label": int(x["label"])}) + "\n")


def load(path: str) -> List[dict]:
    import json

    out = []
    with open(path) as f:
        for line in f:
            x = json.loads(line)
            x["z"], x["gold"] = np.array(x["z"]), np.array(x["gold"])
            out.append(x)
    return out
