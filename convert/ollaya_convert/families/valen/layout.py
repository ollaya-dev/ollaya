"""`valen-decision-1.5-v1`: the request -> token layout of Valen-Decision-1.5 (upstream `Compiler._messages`,
the Qwen chat template and `Compiler._compile_shared` in the model repository's `compiler.py`, with `schema.py`'s
checks), written the way the Rust port is (HF `tokenizers` and `decision.json` only). `check.py` and `goldens.py`
compare it id for id with upstream.

One sequence per request holds the state once and every question (upstream's `shared_state` execution):

    ids = tok(user_open + strip(image + state) + user_close)                       the shared prefix
          + tok(questions_open)
          + for question i: tok(question(n=i+1, type) + instructions + candidates) <- instruction span
                            + for each candidate: tok([key + key_sep] + description)   <- candidate span
                                                  + tok(candidate_end)
          + tok(readouts_open)
          + for question i: tok(readout(n=i+1)) <- its last token is the decision position
                            + tok(readout_end)

* `image` is `image_open + image_token * n + image_close` when the request has an image (n visual tokens, the
  image's patch count / 4), else empty. `strip` is the chat template's `|trim` (Python `str.strip`).
* Every piece after the prefix is tokenized on its own. A span is the tokens whose characters overlap the
  instructions (or the description): upstream reads the tokenizer's offsets, so a token that also holds the
  character before or after belongs to the span.
* noul reads two fixed candidates, `true` then `false`; its criteria, when given, are appended to the
  instructions. A choice's candidates are `key: description` in the caller's order, a score's levels are the
  descriptions alone.
* Nothing is truncated: a sequence longer than `max_tokens` is rejected, as upstream does. So is any text that
  holds one of the tokenizer's special tokens.

`blocks` gives each token its attention block under upstream's `bbox_blocks` mask (`block_attention.py`): -1 the
prefix, 0 the shared glue (`questions_open`, `readouts_open`), 1 every decision question (they share one block).
`positions` are the interleaved M-RoPE position ids (upstream `get_rope_index`).
"""
from __future__ import annotations

import json
from typing import Any, Dict

LAYOUT = "valen-decision-1.5-v1"
PREFIX, COMMON, PADDING = -1, 0, -2   # upstream block_attention.py
QUESTION_BLOCK = 1


class LayoutError(ValueError):
    """The request is invalid for this model (HTTP 422)."""


def state_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True, allow_nan=False)


def prompt_for():
    """The layout's fixed strings (compiler.py and schema.py at the pinned revisions); check.py proves them
    against upstream."""
    return {
        "user_open": "<|im_start|>user\n",
        "user_close": "<|im_end|>\n",
        "image_open": "<|vision_start|>",
        "image_token": "<|image_pad|>",
        "image_close": "<|vision_end|>",
        "questions_open": "<|im_start|>user\n",
        "question": "Question {n}\nTask: {type}\nQuestion: ",
        "candidates": "\nCandidates:\n",
        "key_sep": ": ",
        "candidate_end": "\n",
        "readouts_open": "<|im_end|>\n<|im_start|>assistant\n",
        "readout": "Question {n} Decision:",
        "readout_end": "\n",
        "noul": {"true": "True / 是：满足问题中的条件。", "false": "False / 否：不满足问题中的条件。",
                 "true_criterion": "\nTrue criterion: ", "false_criterion": "\nFalse criterion: "},
    }


class ValenLayout:
    def __init__(self, tokenizer, decision: Dict[str, Any]):
        self.tok = tokenizer
        self.p = decision["prompt"]
        self.reserved = decision["reserved_tokens"]
        self.max_tokens = decision["max_tokens"]
        self.min_levels = decision["min_levels"]
        self.max_options = decision["max_options"]
        self.image_id = decision["tokens"]["image"]

    def enc(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def span(self, text, start, end):
        """(ids, [first, last + 1]) of the tokens whose characters overlap text[start:end]."""
        e = self.tok.encode(text, add_special_tokens=False)
        inside = [i for i, (a, b) in enumerate(e.offsets) if a < end and b > start]
        if not inside:
            raise LayoutError("a question's instructions or a candidate's description holds no token")
        return e.ids, [inside[0], inside[-1] + 1]

    def clean(self, what, text):
        if any(t in text for t in self.reserved):
            raise LayoutError("%s holds a reserved tokenizer control token" % what)

    def candidates(self, qid, q):
        """upstream `fold_noul_criteria` then `schema.candidates`: (type, instructions, [(key, description)])."""
        if not isinstance(q, dict):
            raise LayoutError("question %r must be an object" % qid)
        kind, instructions, criteria = q.get("type"), q.get("instructions"), q.get("criteria")
        n = self.p["noul"]
        if kind == "noul" and criteria:
            if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
                raise LayoutError("question %r: noul criteria must define true and false" % qid)
            if not isinstance(instructions, str):
                raise LayoutError("question %r: instructions must be text" % qid)
            instructions = (instructions + n["true_criterion"] + str(criteria["true"])
                            + n["false_criterion"] + str(criteria["false"]))
        if not isinstance(instructions, str) or not instructions.strip():
            raise LayoutError("question %r: instructions must be non-empty text" % qid)
        if kind == "noul":
            pairs = [("true", n["true"]), ("false", n["false"])]
        elif kind == "choice":
            if not isinstance(criteria, dict) or not 1 <= len(criteria) <= self.max_options:
                raise LayoutError("question %r: a choice takes 1-%d named criteria" % (qid, self.max_options))
            pairs = list(criteria.items())
        elif kind == "score":
            if not isinstance(criteria, list) or not self.min_levels <= len(criteria) <= self.max_options:
                raise LayoutError("question %r: a score takes a list of %d-%d level descriptions"
                                  % (qid, self.min_levels, self.max_options))
            pairs = [(str(i), text) for i, text in enumerate(criteria)]
        else:
            raise LayoutError("question %r: type must be noul, choice or score" % qid)
        if any(not isinstance(k, str) or not k or not isinstance(v, str) or not v.strip() for k, v in pairs):
            raise LayoutError("question %r: candidate names and descriptions must be non-empty strings" % qid)
        return kind, instructions, pairs

    def encode(self, state, questions, image_grid=None):
        """-> {"ids", "prefix", "blocks", "positions", "questions": [{"qid", "type", "keys", "instruction",
        "candidates", "decision"}]}. `image_grid`: the image's merged (rows, columns) grid, None without an image.
        Spans are [start, end) token positions in "ids"."""
        if not isinstance(questions, dict) or not questions:
            raise LayoutError("questions must be a non-empty object")
        p = self.p
        selected = [(qid, *self.candidates(qid, q)) for qid, q in questions.items()]
        text = state_text(state)
        self.clean("the state", text)
        image = ""
        if image_grid is not None:
            image = p["image_open"] + p["image_token"] * (image_grid[0] * image_grid[1]) + p["image_close"]
        ids = self.enc(p["user_open"] + (image + text).strip() + p["user_close"])
        prefix = len(ids)
        blocks = [PREFIX] * prefix

        def glue(piece, block):
            new = self.enc(piece)
            ids.extend(new)
            blocks.extend([block] * len(new))

        def spanned(lead, body, tail):
            start = len(ids)
            new, (a, b) = self.span(lead + body + tail, len(lead), len(lead) + len(body))
            ids.extend(new)
            blocks.extend([QUESTION_BLOCK] * len(new))
            return [start + a, start + b]

        glue(p["questions_open"], COMMON)
        out = []
        for i, (qid, kind, instructions, pairs) in enumerate(selected):
            for value in [instructions] + [v for pair in pairs for v in pair]:
                self.clean("question %r" % qid, value)
            lead = p["question"].replace("{n}", str(i + 1)).replace("{type}", kind)
            instruction = spanned(lead, instructions, p["candidates"])
            spans = []
            for key, description in pairs:
                spans.append(spanned("" if kind == "score" else key + p["key_sep"], description, ""))
                glue(p["candidate_end"], QUESTION_BLOCK)
            out.append({"qid": qid, "type": kind, "keys": [k for k, _ in pairs], "instruction": instruction,
                        "candidates": spans, "decision": -1})
        glue(p["readouts_open"], COMMON)
        for i, q in enumerate(out):
            glue(p["readout"].replace("{n}", str(i + 1)), QUESTION_BLOCK)
            q["decision"] = len(ids) - 1
            glue(p["readout_end"], QUESTION_BLOCK)
        if len(ids) > self.max_tokens:
            raise LayoutError("the request takes %d tokens; this model reads at most %d and nothing is truncated"
                              % (len(ids), self.max_tokens))
        return {"ids": ids, "prefix": prefix, "blocks": blocks, "positions": self.positions(ids, image_grid),
                "questions": out, "image_id": self.image_id}

    def positions(self, ids, image_grid):
        """Interleaved M-RoPE positions [3][tokens]: text counts up on all three axes; the image's merged cell
        (i, j), starting at position s, is (s, s + i, s + j), and the text after it continues at
        s + max(rows, columns)."""
        t, h, w = [], [], []
        cur, k = 0, 0
        while k < len(ids):
            if ids[k] == self.image_id and image_grid is not None:
                rows, cols = image_grid
                for i in range(rows):
                    for j in range(cols):
                        t.append(cur), h.append(cur + i), w.append(cur + j)
                cur += max(rows, cols)
                k += rows * cols
                continue
            t.append(cur), h.append(cur), w.append(cur)
            cur += 1
            k += 1
        return [t, h, w]


def graph_inputs(row, pad, chunk, hidden, image_embeds=None):
    """The decoder graph's inputs for one encoded request (numpy). `image_embeds`: [n, hidden] from the vision
    graph, None without an image. The sequence is right-padded to a multiple of `chunk` with at least one padding
    position; a text-only request writes one zero embedding there (no token attends to padding)."""
    import numpy as np

    n = len(row["ids"])
    seq = -(-(n + 1) // chunk) * chunk
    ids = np.full((1, seq), pad, dtype=np.int64)
    ids[0, :n] = row["ids"]
    blocks = np.full((1, seq), PADDING, dtype=np.int64)
    blocks[0, :n] = row["blocks"]
    positions = np.zeros((3, 1, seq), dtype=np.int64)
    positions[:, 0, :n] = row["positions"]
    if image_embeds is None:
        image_embeds = np.zeros((1, hidden), dtype=np.float32)
        image_pos = np.array([seq - 1], dtype=np.int64)
    else:
        image_pos = np.flatnonzero(ids[0] == row["image_id"]).astype(np.int64)
    qs = row["questions"]
    return {
        "input_ids": ids,
        "position_ids": positions,
        "image_embeds": np.asarray(image_embeds, dtype=np.float32),
        "image_pos": image_pos,
        "blocks": blocks,
        "instruction_spans": np.array([q["instruction"] for q in qs], dtype=np.int64),
        "decision_pos": np.array([q["decision"] for q in qs], dtype=np.int64),
        "candidate_spans": np.array([s for q in qs for s in q["candidates"]], dtype=np.int64),
        "candidate_question": np.array([i for i, q in enumerate(qs) for _ in q["candidates"]], dtype=np.int64),
    }
