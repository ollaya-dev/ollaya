"""`decima-late-interaction-v1` and `-v2`: the request handling of the author's `decima/systemone.py`, `decima/model.py`
and `decima/render.py` (github.com/amyrmahdy/decima at v2.3.0), written the way the Rust port is (HF `tokenizers` and
`decision.json` only). `check.py` and `goldens.py` compare it id for id with upstream.

Per question, in request order:

    text     = instructions (a string as is, anything else json.dumps(ensure_ascii=False), the question id when
               they are absent or null); a noul question with a "true" or "false" criterion appends
               "\\nTrue if: <true>\\nFalse if: <false>", an empty side written as an em dash (U+2014)
    options  = choice: "label: description" (the label alone when the description renders empty);
               score: each level's text; noul: "yes", "no"
    state    = the state as is when it is a string, else json.dumps(ensure_ascii=False)
    rows     = [cls] + tok(state_prefix + normalize(text + "\\n" + state))[:max_state_tokens - 2] + [sep]
               [cls] + tok(option_text)[:max_option_tokens - 2] + [sep], per option

`option_text` is render.py's `choice_texts`. With `option_fit` (decima.json's choice_fit, layout v2) every option is
`full = option_prefix + normalize(text + " " + option)` when it fits max_option_tokens (or has at most
max_option_tokens - 2 characters), else `option_prefix + head + " … " + normalize(option)`, where head is the longest
run of normalize(text)'s leading whitespace-separated words (found by upstream's binary search) that fits, and
`option_prefix + normalize(option)` when no word does. Without it (v1) every option is `full`, unless cutting them to
max_option_tokens makes two rows identical: then that question's options are rendered as with option_fit.

`normalize` is NFC, then Python's str.strip() (upstream's normalize for lang "en", which is all systemone uses);
the empty instructions leave the state alone. A state row over max_state_tokens is cut and flagged (upstream's
server answers 422 STATE_TRUNCATED; its model.py truncates); an option row is cut silently, as upstream does.

Upstream's checks (systemone.py): 1-256 questions, the state a string, object or array, 2-255 choices, 2-10 score
levels, a known type. Option logits come out in upstream's order: a choice's labels, score levels, noul [yes, no].
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Dict

TYPES = ("choice", "noul", "score")
PLACEHOLDER = re.compile(r"\{(true|false)\}")


class LayoutError(ValueError):
    """The request is invalid for this model (HTTP 422)."""


def text(value: Any) -> str:
    """systemone._text: None as "", a string as is, anything else json.dumps(ensure_ascii=False)."""
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def normalize(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()


class DecimaLayout:
    def __init__(self, tokenizer, decision: Dict[str, Any]):
        self.tok = tokenizer
        self.d = decision
        self.cls = decision["special_tokens"]["cls"]
        self.sep = decision["special_tokens"]["sep"]

    def question(self, qid, q):
        """systemone.to_question: (type, text, options)."""
        lim = self.d["limits"]
        if not isinstance(q, dict):
            raise LayoutError("question %r must be an object" % qid)
        t = q.get("type")
        ins = q.get("instructions")
        body = qid if ins is None else text(ins)
        crit = q.get("criteria")
        if t == "choice":
            if isinstance(crit, list):
                crit = {str(c): None for c in dict.fromkeys(crit)}
            if not isinstance(crit, dict) or not (lim["choices"][0] <= len(crit) <= lim["choices"][1]):
                raise LayoutError("question %r: choice needs %d-%d criteria" % (qid, *lim["choices"]))
            # decima bc02983: the ordinal head was trained on at most 10 levels.
            if q.get("ordinal") is True and not (lim["levels"][0] <= len(crit) <= lim["levels"][1]):
                raise LayoutError("question %r: an ordinal choice takes %d-%d labels" % (qid, *lim["levels"]))
            return t, body, [("%s: %s" % (k, text(v))) if text(v).strip() else str(k) for k, v in crit.items()]
        if t == "score":
            if not isinstance(crit, list) or not (lim["levels"][0] <= len(crit) <= lim["levels"][1]):
                raise LayoutError("question %r: score needs %d-%d levels" % (qid, *lim["levels"]))
            return t, body, [text(v) for v in crit]
        if t == "noul":
            n = self.d["noul"]
            if isinstance(crit, dict):
                tr, fa = text(crit.get("true")).strip(), text(crit.get("false")).strip()
                if tr or fa:
                    vals = {"true": tr or n["missing"], "false": fa or n["missing"]}
                    body += PLACEHOLDER.sub(lambda m: vals[m.group(1)], n["criteria"])
            return t, body, list(n["options"])
        raise LayoutError("question %r: unknown type %r" % (qid, t))

    def ids(self, s, max_len):
        """-> (ids, cut): the tokenizer's <s> ... </s> around the text, cut to max_len tokens."""
        body = self.tok.encode(s, add_special_tokens=False).ids
        return [self.cls] + body[:max_len - 2] + [self.sep], len(body) + 2 > max_len

    def row(self, prefix, s, max_len):
        return self.ids(prefix + normalize(s), max_len)

    def tokens(self, s):
        """len(tok(s)["input_ids"]): the text's tokens and the two special tokens."""
        return len(self.tok.encode(s, add_special_tokens=False).ids) + 2

    def option_texts(self, question, options):
        """render.choice_texts, for lang "en"."""
        prefix, max_len = self.d["option_prefix"], self.d["max_option_tokens"]
        full = [prefix + normalize(("%s %s" % (question, o)).strip()) for o in options]
        if not self.d.get("option_fit", False):
            if len({tuple(self.ids(t, max_len)[0]) for t in full}) == len(full):
                return full
        out = []
        for o, f in zip(options, full):
            if self.tokens(f) <= max_len:
                out.append(f)
                continue
            opt = normalize(o)
            words = normalize(question).split()
            lo, hi = 0, len(words)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if self.tokens("%s%s … %s" % (prefix, " ".join(words[:mid]), opt)) <= max_len:
                    lo = mid
                else:
                    hi = mid - 1
            out.append("%s%s … %s" % (prefix, " ".join(words[:lo]), opt) if lo else prefix + opt)
        return out

    def encode(self, state, questions):
        """-> {"state_tokens", "questions": [{"qid", "type", "state_ids", "option_ids", "state_truncated"}]}."""
        lim = self.d["limits"]
        if not isinstance(questions, dict) or not (lim["questions"][0] <= len(questions) <= lim["questions"][1]):
            raise LayoutError("questions: %d-%d questions required" % tuple(lim["questions"]))
        if state is None or isinstance(state, (bool, int, float)):
            raise LayoutError("state: must be a string, object or array")
        s = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        out = []
        for qid, q in questions.items():
            t, body, options = self.question(qid, q)
            in_state = body if self.d["question_in_state"] else ""
            in_options = body if self.d["question_in_options"] else ""
            state_ids, cut = self.row(self.d["state_prefix"], "%s\n%s" % (in_state, s) if in_state else s,
                                      self.d["max_state_tokens"])
            option_ids = [self.ids(t, self.d["max_option_tokens"])[0] for t in self.option_texts(in_options, options)]
            out.append({"qid": qid, "type": t, "state_ids": state_ids, "option_ids": option_ids,
                        "state_truncated": cut})
        return {"state_tokens": len(self.tok.encode(s, add_special_tokens=False).ids), "questions": out}


def graph_inputs(rows, pad):
    """The graph's inputs for questions' rows (numpy int64): states [q, ls], options [o, lo], option_state [o]."""
    import numpy as np

    ls = max(len(r["state_ids"]) for r in rows)
    lo = max(len(o) for r in rows for o in r["option_ids"])
    n = sum(len(r["option_ids"]) for r in rows)
    state_ids = np.full((len(rows), ls), pad, dtype=np.int64)
    state_mask = np.zeros((len(rows), ls), dtype=np.int64)
    option_ids = np.full((n, lo), pad, dtype=np.int64)
    option_mask = np.zeros((n, lo), dtype=np.int64)
    option_state = np.zeros((n,), dtype=np.int64)
    k = 0
    for i, r in enumerate(rows):
        state_ids[i, :len(r["state_ids"])] = r["state_ids"]
        state_mask[i, :len(r["state_ids"])] = 1
        for o in r["option_ids"]:
            option_ids[k, :len(o)] = o
            option_mask[k, :len(o)] = 1
            option_state[k] = i
            k += 1
    return {"state_ids": state_ids, "state_mask": state_mask, "option_ids": option_ids, "option_mask": option_mask,
            "option_state": option_state}
