"""`arbiter-fixed-v1`: the request -> token rows layout of hiteshluke/arbiter-4b, written the way the
Rust port will be (HF `tokenizers` only, no external `arbiter` package). `parity.py` checks it id-for-id
against the inline reference in `ref.py`.

One causal row per question. Each row is Gemma 3's tokenization of a fixed text prompt:

    State: {render(state)}

    Question: {render(instructions)}

    Options:
    {options block, one per line}

    Answer:

The row's last position is where the pointer head is read. There are no delimiter tokens: the layout is
just the tokenized prompt, right-padded for batching. The graph returns 24-slot scores; the valid slots
for each row depend on the question type (see `slot_layout` in `decision.json`):

    noul     -> slots 0..1   (T, F)
    choice   -> slots 2..17  (A..P), the first k slots used, 1 <= k <= 16
    score    -> slots 18..23 (0..5), always 6 levels

The server softmaxes over the valid slots and divides by the checkpoint's temperature.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

CHOICE_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
MAX_CHOICE = len(CHOICE_LETTERS)
SCORE_LEVELS = 6

# Slot layout. The two F's below are at different positions in slot order: slot 1 is the F in T/F,
# slot 7 is the F in A..P.
VERBALIZERS: List[str] = ["T", "F"] + CHOICE_LETTERS + [str(i) for i in range(SCORE_LEVELS)]
NUM_SLOTS = len(VERBALIZERS)                            # 24
NOUL_SLOTS = [0, 1]                                     # [true, false]
CHOICE_SLOTS = list(range(2, 2 + MAX_CHOICE))           # 2..17
SCORE_SLOTS = list(range(2 + MAX_CHOICE, NUM_SLOTS))    # 18..23


class LayoutError(ValueError):
    """The request is invalid for this model (HTTP 422)."""


def render(v, indent: int = 0) -> str:
    """arbiter text renderer: None -> '', scalars -> str(v), lists -> '- item' lines, dicts -> 'k: v'
    lines, 2-space nesting. Kept local to this file to keep the family self-contained."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join("%s- %s" % (pad, render(x, indent + 1).lstrip()) for x in v)
    return "\n".join(
        "%s%s:\n%s" % (pad, k, render(x, indent + 1)) if isinstance(x, (dict, list))
        else "%s%s: %s" % (pad, k, render(x))
        for k, x in v.items()
    )


def option_text(name: str, desc) -> str:
    return name if desc is None or desc == "" else "%s: %s" % (name, render(desc))


def _noul_options_block() -> str:
    return "T. Yes / True\nF. No / False"


def _choice_options_block(options: List[str]) -> str:
    return "\n".join("%s. %s" % (CHOICE_LETTERS[i], o) for i, o in enumerate(options))


def _score_options_block() -> str:
    return "\n".join(str(i) for i in range(SCORE_LEVELS))


def _slots_for(qtype: str, k: int) -> List[int]:
    if qtype == "noul":
        return list(NOUL_SLOTS)
    if qtype == "choice":
        return CHOICE_SLOTS[:k]
    if qtype == "score":
        return list(SCORE_SLOTS)
    raise LayoutError("unknown question type %r" % (qtype,))


def to_record(state, questions: Dict[str, Any]) -> Tuple[str, List[Tuple[str, List[str], str]], List[Dict[str, Any]]]:
    """-> (state_text, [(instructions, options, options_block)], meta). Validates each question the way
    the server will (HTTP 422 on failure)."""
    if not isinstance(questions, dict) or not questions:
        raise LayoutError("questions must contain at least one question")
    qs: List[Tuple[str, List[str], str]] = []
    meta: List[Dict[str, Any]] = []
    for qid, q in questions.items():
        if not isinstance(q, dict):
            raise LayoutError("question %r must be an object" % qid)
        t = q.get("type")
        crit = q.get("criteria")
        if t == "noul":
            if crit is not None and not isinstance(crit, dict):
                raise LayoutError("noul criteria must be an object")
            opts = ["Yes / True", "No / False"]
            block = _noul_options_block()
            k = 2
        elif t == "choice":
            if not isinstance(crit, dict) or not 1 <= len(crit) <= MAX_CHOICE:
                raise LayoutError("choice criteria must be an object of 1..%d options" % MAX_CHOICE)
            opts = [option_text(k_, v) for k_, v in crit.items()]
            block = _choice_options_block(opts)
            k = len(opts)
        elif t == "score":
            if not isinstance(crit, list) or len(crit) != SCORE_LEVELS:
                raise LayoutError("score criteria must be a list of exactly %d levels" % SCORE_LEVELS)
            opts = [render(x) for x in crit]
            block = _score_options_block()
            k = SCORE_LEVELS
        else:
            raise LayoutError("unknown question type %r" % (t,))
        qs.append((render(q.get("instructions")), opts, block))
        meta.append({"qid": qid, "type": t, "k": k, "slots": _slots_for(t, k)})
    return render(state), qs, meta


def format_prompt(state_text: str, instructions: str, options_block: str) -> str:
    """The exact training prompt. Keep this verbatim: any whitespace change shifts the last-position
    hidden state the pointer head reads."""
    return (
        "State: %s\n\n"
        "Question: %s\n\n"
        "Options:\n%s\n\n"
        "Answer:"
    ) % (state_text, instructions, options_block)


class ArbiterLayout:
    """Encode a request into one causal row per question, with the last-position index for the head."""

    def __init__(self, tokenizer, decision: Dict[str, Any]):
        self.tok = tokenizer
        self.max_state = decision.get("max_state_tokens", 8192)
        self.max_row = decision.get("max_row_tokens", 8192)

    def _encode(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode(self, state, questions):
        """-> (rows, meta). rows[q] = {'ids': [...], 'last_pos': int, 'slots': [valid slot indices]}.
        One row per question, in request order."""
        state_text, qs, meta = to_record(state, questions)
        rows: List[Dict[str, Any]] = []
        for (instr, _opts, block), m in zip(qs, meta):
            prompt = format_prompt(state_text, instr, block)
            ids = self._encode(prompt)
            if len(ids) > self.max_row:
                raise LayoutError(
                    "row too long: %d tokens (limit %d)" % (len(ids), self.max_row)
                )
            rows.append({"ids": ids, "last_pos": len(ids) - 1, "slots": m["slots"]})
        return rows, meta
