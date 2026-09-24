"""`kev-pointer-v1`: the request -> token rows layout of jaredpalmer/kev, written the way the Rust port
will be (HF `tokenizers` only, no `kev` package). `parity.py` checks it id-for-id against upstream
`kev.api.to_record` + `kev.model.encode` + `kev.model.rows_of`.

One causal row per question (Qwen3.5 is hybrid, so upstream always uses its "row form"):

    row = [<|fim_prefix|>] + user(render(state))[:max_state - 1]                    # the state, shared
        + [<|fim_middle|>] + user(render(instructions))                             # the question
        + for each option: [<|box_start|>] + user(option_text) + [<|box_end|>]      # option spans
        + [<|fim_suffix|>]                                                          # the decide token
    decide_pos = len(row) - 1;  opt_pos[j] = index of option j's <|box_end|>
    user(text) = enc(text with every "<|name|>" rewritten to "<¦name¦>")        # delimiters are unforgeable

The graph returns raw pointer-head scores; option logits are those scores for every type (noul options
are [no, yes] = [false, true]; score options are the level descriptions in order). The server divides
by the checkpoint's temperature and softmaxes.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
MAX_OPTIONS = 255


class LayoutError(ValueError):
    """The request is invalid for this model (HTTP 422)."""


def render(v, indent: int = 0) -> str:
    """kev.api.render: str | object | array -> text. Scalars use Python str() (True, 1.0, 1e-05)."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join("%s- %s" % (pad, render(x, indent + 1).lstrip()) for x in v)
    return "\n".join("%s%s:\n%s" % (pad, k, render(x, indent + 1)) if isinstance(x, (dict, list))
                     else "%s%s: %s" % (pad, k, render(x)) for k, x in v.items())


def option_text(name: str, desc) -> str:
    return name if desc is None or desc == "" else "%s: %s" % (name, render(desc))


def to_record(state, questions: Dict[str, Any]):
    """kev.api.to_record plus the SystemOneRequest validation: -> (state_text, [(instr, options)], meta)."""
    if not isinstance(questions, dict) or not questions:
        raise LayoutError("questions must contain at least one question")
    qs, meta = [], []
    for qid, q in questions.items():
        if not isinstance(q, dict):
            raise LayoutError("question %r must be an object" % qid)
        t = q.get("type")
        crit = q.get("criteria")
        if t == "noul":
            if crit is not None and not isinstance(crit, dict):
                raise LayoutError("noul criteria must be an object")
            c = crit or {}
            opts = [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
        elif t == "choice":
            if not isinstance(crit, dict) or not 1 <= len(crit) <= MAX_OPTIONS:
                raise LayoutError("choice criteria must be an object of 1..%d options" % MAX_OPTIONS)
            opts = [option_text(k, v) for k, v in crit.items()]
        elif t == "score":
            if not isinstance(crit, list) or not 1 <= len(crit) <= MAX_OPTIONS:
                raise LayoutError("score criteria must be a list of 1..%d levels" % MAX_OPTIONS)
            opts = [render(x) for x in crit]
        else:
            raise LayoutError("unknown question type %r" % (t,))
        qs.append((render(q.get("instructions")), opts))
        meta.append({"qid": qid, "type": t, "k": len(opts)})
    return render(state), qs, meta


class KevLayout:
    def __init__(self, tokenizer, decision: Dict[str, Any]):
        self.tok = tokenizer
        self.max_state = decision["max_state_tokens"]
        self.max_row = decision["max_row_tokens"]
        sp = decision["special_tokens"]
        self.state_id, self.q_id, self.o_id, self.c_id, self.d_id = (sp[k] for k in ("state", "question", "option_open", "option_close", "decide"))

    def user(self, text: str) -> List[int]:
        return self.tok.encode(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).ids

    def encode(self, state, questions):
        """-> (rows, meta): rows[q] = {"ids", "decide", "opts"} (one row per question, request order)."""
        state_text, qs, meta = to_record(state, questions)
        S = [self.state_id] + self.user(state_text)[: self.max_state - 1]
        rows = []
        for instr, opts in qs:
            br = [self.q_id] + self.user(instr)
            ends = []
            for o in opts:
                br += [self.o_id] + self.user(o) + [self.c_id]
                ends.append(len(S) + len(br) - 1)
            br.append(self.d_id)
            if len(br) > self.max_row - len(S):
                raise LayoutError("branch too long: %d tokens with a %d-token state (row limit %d)"
                                  % (len(br), len(S), self.max_row))
            ids = S + br
            rows.append({"ids": ids, "decide": len(ids) - 1, "opts": ends})
        return rows, meta
