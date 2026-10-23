"""More NLI, yes/no and toxicity questions for coba's zero-shot mix (COB-553, build-time only).

coba trails laya:en on multinli, boolq and civil_comments. This adds rows of those kinds from permissive
datasets outside the public benchmark (build.HOLDOUT): the rows of jev's NLI and yes/no sources that the
per-source cap left out of main.jsonl, and samples of SNLI, Measuring Hate Speech, Real Toxicity Prompts
and HotpotQA's yes/no questions. Instructions vary between rows, as they do in the benchmark's own sets.
The teacher labels the rows like the main set (label.py); train on main-train.jsonl and boost.jsonl together.

  uv run --with datasets --with tokenizers python train/data/boost.py --main ~/ollaya-train/data/main.jsonl \
      --out ~/ollaya-train/data/boost.jsonl
"""

import argparse
import collections
import json
import random
import re

from datasets import load_dataset
from tokenizers import Tokenizer

from build import MAX_TOKENS, english, jev, onehot, token_lengths, valid

JEV_KINDS = re.compile(r"nli|-yn$|yes-no|health_fact")
NLI3 = {"entailment": None, "neutral": None, "contradiction": None}
NLI_ASK = [
    "What is the relationship between the premise and the hypothesis?",
    "Given the premise, is the hypothesis true, undetermined or false?",
    "Does the premise entail, contradict, or say nothing about the hypothesis?",
]
NLI_YES = ["Does the premise entail the hypothesis?", "If the premise is true, must the hypothesis be true?"]
HATE_ASK = ["Is this comment hate speech?", "Does this text attack people for who they are?", "Is this post hateful?"]
TOXIC_ASK = ["Is this text toxic?", "Is this text rude, disrespectful or offensive?", "Would a moderator flag this as toxic?"]


def snli(rng, n):
    ds = load_dataset("stanfordnlp/snli", split="train").shuffle(seed=0)
    taken = 0
    for i, r in enumerate(ds):
        if taken >= n:
            return
        if r["label"] not in (0, 1, 2):
            continue
        taken += 1
        if rng.random() < 0.5:
            state = {"premise": r["premise"], "hypothesis": r["hypothesis"]}
        else:
            state = f"Premise: {r['premise']}\nHypothesis: {r['hypothesis']}"
        if rng.random() < 0.6:
            q = {"type": "choice", "instructions": rng.choice(NLI_ASK), "criteria": dict(NLI3)}
            gold = onehot(3, r["label"])
        else:
            q = {"type": "noul", "instructions": rng.choice(NLI_YES), "criteria": None}
            gold = onehot(2, int(r["label"] == 0))
        yield _row(f"snli:{i}", "snli", "cc-by-sa-4.0", state, q, gold)


def hate_speech(rng, n):
    """One row per comment; hate_speech_score > 0.5 is hateful, < -1 supportive (the dataset's own cut points)."""
    ds = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train")
    seen = {}
    for r in ds:
        seen.setdefault(r["comment_id"], (r["text"], r["hate_speech_score"]))
    ids = sorted(seen)
    rng.shuffle(ids)
    for cid in ids[:n]:
        text, score = seen[cid]
        if rng.random() < 0.6:
            q = {"type": "noul", "instructions": rng.choice(HATE_ASK), "criteria": None}
            gold = onehot(2, int(score > 0.5))
        else:
            q = {"type": "choice", "instructions": "How does this comment treat the group it is about?",
                 "criteria": {"supportive": None, "neutral": None, "hateful": None}}
            gold = onehot(3, 0 if score < -1 else 2 if score > 0.5 else 1)
        yield _row(f"mhs:{cid}", "measuring-hate-speech", "cc-by-4.0", text, q, gold)


def toxicity(rng, n):
    """Prompt and continuation as one text; half of the sample from texts Perspective scored toxic. No gold:
    Perspective's score is a model's, so the teacher's label stands alone."""
    ds = load_dataset("allenai/real-toxicity-prompts", split="train").shuffle(seed=0)
    toxic, clean = [], []
    for i, r in enumerate(ds):
        t = r["continuation"]["toxicity"]
        (toxic if t is not None and t >= 0.5 else clean).append(i)
    picks = toxic[: n // 2] + clean[: n - min(len(toxic), n // 2)]
    for i in picks:
        r = ds[i]
        q = {"type": "noul", "instructions": rng.choice(TOXIC_ASK), "criteria": None}
        yield _row(f"rtp:{i}", "real-toxicity-prompts", "apache-2.0", r["prompt"]["text"] + r["continuation"]["text"], q, None)


def hotpot_yesno(rng):
    """HotpotQA's yes/no questions over their two supporting paragraphs: a passage and a yes/no question."""
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    for r in ds:
        if r["answer"] not in ("yes", "no"):
            continue
        titles = set(r["supporting_facts"]["title"])
        paras = [f"{t}: {''.join(s)}" for t, s in zip(r["context"]["title"], r["context"]["sentences"]) if t in titles]
        q = {"type": "noul", "instructions": r["question"], "criteria": None}
        yield _row(f"hotpot:{r['id']}", "hotpot-yesno", "cc-by-sa-4.0", "\n\n".join(paras), q, onehot(2, int(r["answer"] == "yes")))


def jev_rest(main_ids):
    for src, r in jev({}):
        if JEV_KINDS.search(src) and r["id"] not in main_ids:
            yield r


def _row(rid, source, lic, state, q, gold):
    return {"id": rid, "source": source, "license": lic, "state": state, "questions": {"q": q},
            "gold": {"q": gold} if gold else None}


def selftest():
    assert JEV_KINDS.search("doc-nli") and JEV_KINDS.search("spartqa-yn") and JEV_KINDS.search("babi_nli/yes-no-questions")
    assert not JEV_KINDS.search("spartqa-mchoice") and not JEV_KINDS.search("math_qa")
    r = _row("x:1", "x", "mit", "s", {"type": "noul", "instructions": "?", "criteria": None}, onehot(2, 1))
    assert r["gold"] == {"q": [0.0, 1.0]} and valid(r["questions"]["q"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True, help="main.jsonl, whose rows are not repeated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--snli", type=int, default=25_000)
    ap.add_argument("--hate", type=int, default=15_000)
    ap.add_argument("--toxicity", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    selftest()
    rng = random.Random(a.seed)
    tok = Tokenizer.from_pretrained("jhu-clsp/ettin-encoder-150m")
    main_ids = {json.loads(line)["id"] for line in open(a.main, encoding="utf-8")}
    gens = [jev_rest(main_ids), snli(rng, a.snli), hate_speech(rng, a.hate), toxicity(rng, a.toxicity), hotpot_yesno(rng)]
    rows = [r for g in gens for r in g if all(valid(q) for q in r["questions"].values()) and english(r)]
    rows = [r for r, n in zip(rows, token_lengths(tok, rows)) if n <= MAX_TOKENS]
    rng.shuffle(rows)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = collections.Counter(r["source"] if not r["source"].startswith("jev/") else "jev" for r in rows)
    print(json.dumps({"rows": len(rows), "by_source": by}))


if __name__ == "__main__":
    main()
