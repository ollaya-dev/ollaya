"""Reference implementation of `jev-style-v1`: chaoliangUNSW/Jev-Style-2B-Decision-v3's render v2
(`macjev-render-v2-long-options`, layout `sb`) and verdict readout, ported from the model's own
self-contained runtime `jev_style_decision_gguf.py` (the "Shared core", Apache-2.0).

Input layout (token segments are encoded separately so the ` ->` slots are exact):

    state prefix   State:\n<state>\n\n
    short form     Question [<type>]: <question>\nOptions:\n
    (q+opts+slots   - <option 1> ->\n ... - <option K> ->\n
     <= 2048 tok)
    overflow form  Question [<type>]: <question>\nOptions:\n
    (otherwise)    Option 1: <option 1>\n ... Option K: <option K>\n            (the catalogue)
                   Judge each numbered option in the complete catalogue above:\n
                   Option 1 ->\n ... Option K ->\n                               (the rubric)

Score of option k = logit(" yes") - logit(" no") at its ` ->` slot; probabilities are
softmax(scores / T) in canonical option order, one global temperature. Everything is tokenised with
no BOS/EOS and special tokens split (never control tokens).
"""
from __future__ import annotations

import json

LAYOUT = "jev-style-v1"
TEMPLATE_VERSION = "macjev-render-v2-long-options"
BLOCK = 2048
CONTEXT_LIMIT = 25_600
UPSTREAM = {
    "runtime": "https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF",
    "source": "jev_style_decision_gguf.py",
}


class QuestionError(ValueError):
    """A question the layout rejects (HTTP 400)."""


class InputBudgetError(ValueError):
    """The rendered input exceeds the token budget (HTTP 422). Nothing was truncated."""


def option_names(question):
    """Canonical option identifiers, in the order the probabilities are returned."""
    if not isinstance(question, dict):
        raise QuestionError("question must be a dict with type/instructions/criteria")
    t, crit = question.get("type"), question.get("criteria")
    ins = question.get("instructions")
    if not isinstance(ins, str) or not ins.strip():
        raise QuestionError("question text (instructions) must be a non-empty string")
    if t == "choice":
        if not isinstance(crit, dict) or not crit:
            raise QuestionError("choice needs a non-empty object {option name: description or null}")
        return [str(k) for k in crit]
    if t == "score":
        if not isinstance(crit, list) or not 2 <= len(crit) <= 10:
            raise QuestionError("score needs a list of 2..10 level descriptions")
        return [str(i) for i in range(len(crit))]
    if t == "noul":
        if crit is not None and not isinstance(crit, dict):
            raise QuestionError("noul criteria must be null or {'false': ..., 'true': ...}")
        return ["false", "true"]
    raise QuestionError("unknown question type %r (expected choice, score or noul)" % (t,))


def serialize_state(state) -> str:
    """A string state passes through; any other JSON value is `json.dumps`ed (ensure_ascii=False)."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def _criterion(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(question):
    """Option bodies in canonical order (without the `- ` / ` ->` wrapping)."""
    t, crit = question["type"], question.get("criteria")
    if t == "choice":
        return [k if v is None or v == "" else "%s: %s" % (k, _criterion(v)) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, _criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    false_c, true_c = crit.get("false"), crit.get("true")
    return ["false: " + (_criterion(false_c) if false_c not in (None, "") else "no, the statement does not hold"),
            "true: " + (_criterion(true_c) if true_c not in (None, "") else "yes, the statement holds")]


def _blocks(start, stop):
    return [(s, min(s + BLOCK, stop)) for s in range(start, stop, BLOCK)]


class Rendered:
    __slots__ = ("ids", "prefix_len", "slots", "blocks", "names", "qtype", "catalogue_overflow")

    def __init__(self, ids, prefix_len, slots, blocks, names, qtype, catalogue_overflow):
        self.ids, self.prefix_len, self.slots, self.blocks = ids, prefix_len, slots, blocks
        self.names, self.qtype, self.catalogue_overflow = names, qtype, catalogue_overflow


def prefix_ids(state, enc):
    """The state prefix `State:\\n<state>\\n\\n`, tokenised once and shared across a request."""
    return enc("State:\n") + enc(serialize_state(state)) + enc("\n\n")


def render(state, question, enc, arrow, prefix=None, max_len=CONTEXT_LIMIT):
    """Render one question. `enc(text)` tokenises with no BOS and special tokens split; `arrow` is the
    ` ->` token id. Returns a Rendered (ids, prefix_len, verdict slots, attention blocks)."""
    names = option_names(question)
    opts = [enc(o) for o in render_options(question)]
    k = len(opts)
    if k < 1:
        raise QuestionError("at least one option is required")
    prefix = prefix_ids(state, enc) if prefix is None else list(prefix)
    head = enc("Question [%s]: %s\nOptions:\n" % (question["type"], question["instructions"]))
    dash, newline = enc("- "), enc("\n")

    short, rel = list(head), []
    for o in opts:
        short += dash + list(o) + [arrow]
        rel.append(len(short) - 1)
        short += newline
    overflow = len(short) > BLOCK
    if not overflow:
        ids = prefix + short
        slots = [len(prefix) + s for s in rel]
        blocks = _blocks(0, len(prefix)) + [(len(prefix), len(ids))]
    else:
        catalogue = list(head)
        for pos, o in enumerate(opts):
            catalogue += enc("Option %d: " % (pos + 1)) + list(o) + newline
        doc_end = len(prefix) + len(catalogue)
        rubric, rel = enc("Judge each numbered option in the complete catalogue above:\n"), []
        for pos in range(k):
            rubric += enc("Option %d" % (pos + 1)) + [arrow]
            rel.append(len(rubric) - 1)
            rubric += newline
        ids = prefix + catalogue + rubric
        slots = [doc_end + s for s in rel]
        blocks = _blocks(0, len(prefix)) + _blocks(len(prefix), doc_end) + _blocks(doc_end, len(ids))
    if len(ids) > max_len:
        raise InputBudgetError("the complete input needs %d tokens; the limit is %d. Nothing was "
                               "truncated." % (len(ids), max_len))
    return Rendered(ids, len(prefix), slots, blocks, names, question["type"], overflow)
