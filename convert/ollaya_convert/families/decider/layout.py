"""`decider-slots-v1`: the request -> token rows layout of Mapika/decider, written the way the Rust port
will be (HF `tokenizers` + `json.dumps` only, no `decider` package). `parity.py` checks it id-for-id
against upstream `decider.systemone` + `decider.prompt.build`.

One row per question (upstream `independent=True`, the default); a score question with isolated
levels (upstream `isolated_levels=true`) becomes one yes/no row per level:

    row ids = enc("Context:\\n" + state_text)[:max_ctx_tokens]                   # tokenized once, shared
            + enc("\\n\\nQuestion: " + text + "\\nOptions:" + "\\n(A) o0" + ... + "\\nAnswer: (")    # <= 10 options
            | enc("\\n\\nQuestion: " + text + "\\nOptions:")                                        # > 10 options:
              + sum(enc("\\n(") + [label_id[j]] + enc(") " + o_j)) + enc("\\nAnswer: (")           #   one label token each
    slot    = len(row ids) - 1          # the "(" token; its next-token logits over label ids are the option logits

Option logits per question (what the runner returns; the server applies temperature + softmax):
    choice / noul       z_j = label_logits[row, j], j < K            (noul: j=0 "no" = false, j=1 "yes" = true)
    score (isolated)    z_j = log_sigmoid((label_logits[row_j, 1] - label_logits[row_j, 0]) / T_row)
                        so softmax(z) = p_yes_j / sum_i p_yes_i, upstream's normalised per-level fit; the
                        calibration temperature for score is 1.0 because T_row is already inside z.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List

LETTERS = "ABCDEFGHIJ"
NARROW = 10
ISOLATED = "{q}\nProposed answer: {level}\nDoes the proposed answer fit?"
_LEVEL_NUMBER = re.compile(r"^\s*-?\d+\s*:\s*")


class LayoutError(ValueError):
    """The request is invalid for this model (HTTP 400/422)."""


def _txt(v) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def annotate_indices(x, min_len: int = 8):
    """Arrays of >= min_len elements get their positions written in: dict elements become
    {"_index": i, **element}, anything else {"_index": i, "value": element}. Recursive."""
    if isinstance(x, list):
        if len(x) >= min_len:
            return [({"_index": i, **annotate_indices(v, min_len)} if isinstance(v, dict)
                     else {"_index": i, "value": annotate_indices(v, min_len)}) for i, v in enumerate(x)]
        return [annotate_indices(v, min_len) for v in x]
    if isinstance(x, dict):
        return {k: annotate_indices(v, min_len) for k, v in x.items()}
    return x


def render_state(state) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(annotate_indices(state), ensure_ascii=False)


def render_question(spec: Dict[str, Any], max_choice: int = 255, max_levels: int = 10) -> Dict[str, Any]:
    """-> {"type", "question", "options" (listwise texts), "legend" (score level texts)}"""
    if not isinstance(spec, dict):
        raise LayoutError("question definition must be an object")
    t = spec.get("type", "choice")
    ins = spec.get("instructions", spec.get("question", ""))
    ins = _txt(ins)
    if not ins:
        raise LayoutError("question without instructions")
    crit = spec.get("criteria", spec.get("options"))
    legend = None
    if t == "choice":
        if isinstance(crit, list):
            if not all(isinstance(c, str) for c in crit):
                raise LayoutError("choice labels in a list must be strings")
            crit = {c: None for c in crit}
        if not isinstance(crit, dict) or not 2 <= len(crit) <= max_choice:
            raise LayoutError("choice criteria: a map of 2..%d options" % max_choice)
        opts = [n if crit[n] is None or crit[n] == "" else "%s: %s" % (n, _txt(crit[n])) for n in crit]
    elif t == "score":
        if isinstance(crit, dict):
            try:
                crit = [crit[k] for k in sorted(crit, key=float)]
            except ValueError:
                raise LayoutError("score legend keys must be numbers")
        if not isinstance(crit, list) or not 2 <= len(crit) <= max_levels:
            raise LayoutError("score criteria: an ordered list of 2..%d level descriptions" % max_levels)
        opts = ["%d: %s" % (i, _txt(c)) for i, c in enumerate(crit)]
        legend = [_txt(c) for c in crit]
    elif t in ("noul", "bool"):
        c = crit or {}
        if not isinstance(c, dict):
            raise LayoutError("noul criteria must be an object with optional 'true'/'false'")
        f, tr = c.get("false"), c.get("true")
        opts = ["no" if f is None or f == "" else "no: %s" % _txt(f),
                "yes" if tr is None or tr == "" else "yes: %s" % _txt(tr)]
        t = "noul"
    else:
        raise LayoutError("unknown question type %r" % (t,))
    return {"type": t, "question": ins, "options": opts, "legend": legend}


def strip_level_number(text: str) -> str:
    return _LEVEL_NUMBER.sub("", text, count=1)


class DeciderLayout:
    def __init__(self, tokenizer, decision: Dict[str, Any]):
        """tokenizer: a `tokenizers.Tokenizer`; decision: the model's decision.json."""
        self.tok = tokenizer
        self.max_ctx = decision["max_ctx_tokens"]
        self.isolated = decision["isolated_levels"]
        self.max_choice = decision["max_options"]
        self.max_levels = decision["max_levels"]
        self.label_ids = decision["labels"]["ids"]
        self.row_temperature = decision["isolated_row_temperature"]
        self.open_ids = self.enc("\n(")

    def enc(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def _piece(self, question: str, options: List[str]) -> List[int]:
        head = "\n\nQuestion: %s\nOptions:" % question
        tail = "\nAnswer: ("
        if len(options) <= NARROW:
            return self.enc(head + "".join("\n(%s) %s" % (LETTERS[j], o) for j, o in enumerate(options)) + tail)
        ids = self.enc(head)
        for j, o in enumerate(options):
            ids += self.open_ids + [self.label_ids[j]] + self.enc(") %s" % o)
        return ids + self.enc(tail)

    def encode(self, state, questions: Dict[str, Any]):
        """-> (rows, plan). rows[i] = {"ids", "slot", "k"}; plan[q] = {"qid", "type", "kind", "rows", "k"}."""
        if not isinstance(questions, dict) or not questions:
            raise LayoutError("questions must be a non-empty object")
        ctx = self.enc("Context:\n" + render_state(state))[: self.max_ctx]
        rows, plan = [], []
        for qid, spec in questions.items():
            rq = render_question(spec, self.max_choice, self.max_levels)
            if rq["type"] == "score" and self.isolated:
                idx = []
                for level in rq["legend"]:
                    text = ISOLATED.format(q=rq["question"], level=strip_level_number(level))
                    ids = ctx + self._piece(text, ["no", "yes"])
                    idx.append(len(rows))
                    rows.append({"ids": ids, "slot": len(ids) - 1, "k": 2})
                plan.append({"qid": qid, "type": "score", "kind": "iso", "rows": idx, "k": len(rq["legend"])})
            else:
                ids = ctx + self._piece(rq["question"], rq["options"])
                plan.append({"qid": qid, "type": rq["type"], "kind": "list", "rows": [len(rows)],
                             "k": len(rq["options"])})
                rows.append({"ids": ids, "slot": len(ids) - 1, "k": len(rq["options"])})
        return rows, plan

    def option_logits(self, item, label_logits) -> List[float]:
        """Per-question option logits from the graph output `label_logits` [rows, 255]."""
        if item["kind"] == "list":
            r = item["rows"][0]
            return [float(x) for x in label_logits[r][: item["k"]]]
        out = []
        for r in item["rows"]:
            d = (float(label_logits[r][1]) - float(label_logits[r][0])) / self.row_temperature
            out.append(-math.log1p(math.exp(-d)) if d > -30 else d - math.log1p(math.exp(d)))
        return out


def softmax(z, t: float = 1.0):
    m = max(z)
    e = [math.exp((x - m) / t) for x in z]
    s = sum(e)
    return [x / s for x in e]
