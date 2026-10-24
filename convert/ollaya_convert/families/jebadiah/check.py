"""Check the `jebadiah-v1` port (ref.py) against the author's own renderer, text for text.

    uv run python -m ollaya_convert.families.jebadiah.check <jebadiah GGUF repo snapshot>

For every case of the shared set (Laya's edge cases, the decoder edge cases, typed-decisions rows)
and every question on its own: the author's `jebadiah_prompt.Renderer.render(state, q).prompt`
(AINode's renderer and the repo's chat template through transformers, thinking off) must equal
`PRE + user_message + POST`, the candidate labels must match, and a question the author rejects
must be rejected by the port. Prompts over the author's 2,048-token budget (cut upstream, or for v2.1
an error when no state is left to cut; rejected by Ollaya) and the port's own extensions (a list of
choice labels) are counted separately.
"""
from __future__ import annotations

import argparse
import sys

from ..llm_common import cases
from . import ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", help="a frontier-infra/jebadiah-*-GGUF snapshot (scripts/, tokenizer, chat template)")
    ap.add_argument("--td", type=int, default=200)
    a = ap.parse_args()
    sys.path.insert(0, a.snapshot + "/scripts")
    from jebadiah_prompt import Renderer  # noqa: E402
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.snapshot)
    renderer = Renderer(tok, ref.MAX_PROMPT_TOKENS)
    n_labels = renderer.max_options
    extended = [L for L, _ in renderer._extended]
    same = rejected = cut = extension = 0
    bad = []
    for cid, state, qs in cases.edge_cases() + cases.typed_decisions(a.td):
        for qid, q in qs.items():
            try:
                up = renderer.render(state, q)
                up_err = None
            except Exception as e:  # DecideError, ValueError
                up, up_err = None, e
            try:
                _, _, keys, user, wire, letters = ref.compile_request(state, {qid: q}, n_labels, extended)[0]
                ours = ref.PRE + user + ref.POST
                our_err = None
            except ref.JebError as e:
                ours, our_err = None, e
            if up_err is not None and our_err is not None:
                rejected += 1
            elif up_err is not None and "exceed max sequence length" in str(up_err):
                cut += 1  # v2.1 raises where v2 cut the state away; over the budget, which Ollaya rejects, either way
            elif up_err is not None:
                crit = q.get("criteria") if isinstance(q, dict) else None
                if q.get("type") == "choice" and isinstance(crit, list):
                    extension += 1
                else:
                    bad.append((cid, qid, "author rejects: %s" % up_err))
            elif our_err is not None:
                bad.append((cid, qid, "port rejects: %s" % our_err))
            elif up.truncated:
                cut += 1
            elif up.prompt != ours or len(up.keys) != len(keys) or (letters or up.letters) != up.letters:
                bad.append((cid, qid, "prompt differs"))
            else:
                same += 1
    print("identical prompts %d | both reject %d | cut upstream (Ollaya rejects) %d | list-form choices (Ollaya "
          "extension) %d | mismatches %d" % (same, rejected, cut, extension, len(bad)))
    for b in bad[:20]:
        print("  ", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
