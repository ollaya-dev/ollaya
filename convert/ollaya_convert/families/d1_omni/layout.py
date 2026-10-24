"""`d1-omni-markers-v1`: the request -> token rows layout of d1-omni's text path, written the way the Rust port
is (HF `tokenizers` + `json.dumps`, no upstream code). `parity.py` checks it id-for-id against `prompt.encode`.

One row per question (upstream `prompt.encode`, `max_len` = config `max_length`):

    bos + [STATE] + enc(state)[:room] + question
    question = ([Q] + enc(instructions))[:max(16, budget)]
             + sum([OPT, MASK] + enc(" " + option)[:per] + [OPT_END] for every option) + [DECIDE]
    budget   = max(96, min(24 * k + 32, max_len // 2));  per = max(2, (budget - 3k) // k)
    room     = max(0, max_len - len(question) - 2)

`enc(s)` tokenizes `escape(s)` without special tokens, where `escape` rewrites `<|name|>` to `<¦name¦>` so
caller text can never form a delimiter or a marker. The state is the string itself, or `json.dumps` of a
JSON value (ensure_ascii off); the instructions are `str()` of the value (verbatim for a string). Options are
Laya's texts (`k`, `k: description`, `level i: text`, `false: …` / `true: …`), with the noul read as
[false, true]. Each option is scored at its own MASK; option logits are the row's marker logits in order.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List

_SPECIAL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
NOUL_DEFAULTS = ("no, the statement does not hold", "yes, the statement holds")
QT = {"choice": 0, "score": 1, "noul": 2}


class LayoutError(ValueError):
    """The request is invalid for this model (HTTP 422), as upstream `Question.__post_init__` rejects it."""


def escape(text: str) -> str:
    return _SPECIAL.sub(r"<¦\1¦>", text)


def serialize(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def _criterion(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(", ", ": "), default=str)


def _str(v: Any) -> str:
    """Python `str()` of a JSON value: a string as it is; anything else as its repr (upstream `str(instructions)`)."""
    return v if isinstance(v, str) else str(v)


def render_question(qdef: Any) -> Dict[str, Any]:
    """-> {"type", "instructions", "options"} or LayoutError, as upstream `as_question` + `render_options`."""
    if not isinstance(qdef, dict) or "type" not in qdef or "instructions" not in qdef:
        raise LayoutError("a question is a dict with `type`, `instructions` and, for choice and score, `criteria`")
    t = qdef["type"]
    if t not in QT:
        raise LayoutError("question type must be one of %s, got %r" % (sorted(QT), t))
    crit = qdef.get("criteria")
    ins = _str(qdef["instructions"])
    if t == "choice":
        if not isinstance(crit, dict) or len(crit) < 2:
            raise LayoutError("a choice needs criteria {name: description} with at least two options")
        opts = [k if v is None or v == "" else "%s: %s" % (k, _criterion(v)) for k, v in crit.items()]
    elif t == "score":
        if not isinstance(crit, list) or not 2 <= len(crit) <= 10:
            raise LayoutError("a score needs criteria: a list of 2 to 10 level descriptions, lowest first")
        opts = ["level %d: %s" % (i, _criterion(c)) for i, c in enumerate(crit)]
    else:
        if crit is not None and not isinstance(crit, dict):
            raise LayoutError('noul criteria are optional: {"true": "...", "false": "..."} (or "yes", "no")')
        crit = crit or {}
        false, true = crit.get("false", crit.get("no")), crit.get("true", crit.get("yes"))
        opts = ["false: " + (_criterion(false) if false not in (None, "") else NOUL_DEFAULTS[0]),
                "true: " + (_criterion(true) if true not in (None, "") else NOUL_DEFAULTS[1])]
    return {"type": t, "instructions": ins, "options": opts}


class D1OmniLayout:
    def __init__(self, tok, decision: Dict[str, Any]):
        self.tok = tok
        self.max_len = int(decision["max_len"])
        s = decision["special_tokens"]
        self.bos, self.state_id, self.q_id = s["bos"], s["state"], s["question"]
        self.opt, self.mask, self.opt_end, self.decide = s["option"], s["mask"], s["option_end"], s["decide"]
        b = decision["budget"]
        self.per_option, self.extra, self.min_budget = b["per_option"], b["extra"], b["min_budget"]
        self.min_question, self.min_per = b["min_question"], b["min_per"]

    def enc(self, s: str) -> List[int]:
        return self.tok.encode(escape(s), add_special_tokens=False).ids

    def encode(self, state: Any, questions: Dict[str, Any]):
        """-> rows [{"qid", "type", "k", "ids", "markers", "state_truncated"}]; LayoutError on an invalid question."""
        rqs = {qid: render_question(q) for qid, q in questions.items()}
        state_tokens = self.enc(serialize("" if state is None else state))
        rows = []
        for qid, r in rqs.items():
            opts = r["options"]
            k = len(opts)
            budget = max(self.min_budget, min(k * self.per_option + self.extra, self.max_len // 2))
            per = max(self.min_per, (budget - 3 * k) // k)
            question = ([self.q_id] + self.enc(r["instructions"]))[: max(self.min_question, budget)]
            markers = []
            for text in opts:
                markers.append(len(question) + 1)
                question += [self.opt, self.mask] + self.enc(" " + text)[:per] + [self.opt_end]
            question.append(self.decide)
            room = max(0, self.max_len - len(question) - 2)
            state_ids = [self.state_id] + state_tokens[:room]
            ids = ([self.bos] + state_ids + question)[: self.max_len]
            markers = [m + 1 + len(state_ids) for m in markers]
            if markers[-1] >= self.max_len:
                raise LayoutError("the options do not fit in the context")
            rows.append({"qid": qid, "type": r["type"], "k": k, "ids": ids, "markers": markers,
                         "state_truncated": room < len(state_tokens)})
        return rows


def softmax(z: List[float], t: float) -> List[float]:
    m = max(z)
    e = [math.exp((x - m) / t) for x in z]
    s = sum(e)
    return [x / s for x in e]
