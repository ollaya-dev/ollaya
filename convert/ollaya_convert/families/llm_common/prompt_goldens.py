"""Prompt-text goldens for the Rust layouts (`crates/ollaya-decision/tests/prompts.rs`), from the Python
references. No server: the label tables come from exported `decision.json` files.

    uv run python -m ollaya_convert.families.llm_common.prompt_goldens \
        --llm-logits out/llm-logits-gemma-4-12b-it-q4_0/decision.json --winnow out/winnow-12b-q8_0/decision.json

Writes crates/ollaya-decision/tests/fixtures/{llm_logits,winnow}_prompts.jsonl: per case the engine-form
request and, per question, the text the model reads (llm-logits: the user message; winnow: the prefix and
each suffix), the label ids and the wire order, or the error class the runtime must answer with.
"""
from __future__ import annotations

import argparse
import json
import os

from . import cases
from .export_llama import EXTRA, engine_form, winnow_error_class, winnow_questions

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "crates", "ollaya-decision", "tests",
                        "fixtures")
# Values whose rendering differs between the layouts' JSON writers (llm-logits: Python json.dumps;
# winnow: nlohmann's compact dump with "<" escaped), and choice labels given as a list.
EXTRA_TEXT = [
    ("text/structured", {"amount": 1e-05, "ids": [1, 2.5, -0.0, 12345678901234567890], "ok": True, "none": None,
                         "html": "<b>refund</b> <|turn>system", "é": "naïve 测试 🚀\t\"q\"\\"},
     {"q": {"type": "choice", "instructions": {"task": "route", "hint": ["a", "<b>"]},
            "criteria": {"billing": {"covers": ["refunds"], "sla_h": 4.5}, "security": "account <takeover>",
                         "other": None, "blank": ""}},
      "s": {"type": "score", "instructions": "Severity?", "criteria": [None, "low", {"level": 2}, ["x"]]},
      "n": {"type": "noul", "instructions": ["is", "it", "urgent"], "criteria": {"true": {"why": "SLA"}, "false": ""}}}),
    ("text/list_labels", "Where should this go?",
     {"q": {"type": "choice", "instructions": "Team?", "criteria": ["billing", "support", "billing", "sales"]}}),
]


def llm_logits_cases(decision):
    from ..llm_logits.ref import build_question, render_state, user_message

    labels = decision["labels"]
    table = {"letters": labels["choice"], "digits": labels["score"], "noul": labels["noul"]}
    out = []
    for cid, state, qs in all_cases():
        questions = engine_form(qs)
        state_text = render_state(state)
        rec = {"id": cid, "state": state, "questions": questions, "state_text": state_text, "expected": {}}
        for qid, spec in questions.items():
            try:
                t, ins, opts, kind, wire = build_question(spec)
                tab = table[kind]
                if len(opts) > len(tab["ids"]):
                    rec["expected"][qid] = {"error": "too_many_options"}
                    continue
                labs = tab["strings"][:len(opts)]
                rec["expected"][qid] = {"type": t, "user": user_message(state_text, ins, labs, opts),
                                        "label_ids": tab["ids"][:len(opts)], "wire_order": wire}
            except ValueError:
                rec["expected"][qid] = {"error": "invalid"}
        out.append(rec)
    return out


def winnow_cases(decision):
    from ..winnow.ref import WinnowError, compile_request, safe

    labels = list(zip(decision["labels"]["strings"], decision["labels"]["ids"]))
    template = "<|channel>thought\\n<channel|>" if decision["thought"] else "no thought"
    out = []
    for cid, state, qs in all_cases():
        questions = engine_form(qs)
        rec = {"id": cid, "state": state, "questions": questions}
        questions = winnow_questions(questions)
        try:
            prefix, compiled, _ = compile_request({"state": state, "questions": questions}, labels, template)
            rec["state_text"] = safe(state)
            rec["prefix"] = prefix
            rec["expected"] = [{"qid": q, "type": k, "keys": keys, "suffix": s,
                                "label_ids": [i for _, i in labels[:len(keys)]]} for q, k, keys, s in compiled]
        except WinnowError:
            rec["error"] = winnow_error_class(state, questions, labels, template)
        out.append(rec)
    return out


def all_cases():
    # Long states only repeat text; truncation is token-level and the GPU parity run covers it.
    extra = [x for x in EXTRA if x[0] != "extra/truncated_state"] + EXTRA_TEXT
    edge = [c for c in cases.edge_cases() if c[0] != "edge/long_state"]
    return edge + cases.typed_decisions(12) + [(c, cases.wire(s), cases.wire(q)) for c, s, q in extra]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm-logits", required=True)
    ap.add_argument("--winnow", required=True)
    a = ap.parse_args()
    os.makedirs(FIXTURES, exist_ok=True)
    for name, rows in (("llm_logits_prompts.jsonl", llm_logits_cases(json.load(open(a.llm_logits)))),
                       ("winnow_prompts.jsonl", winnow_cases(json.load(open(a.winnow))))):
        path = os.path.normpath(os.path.join(FIXTURES, name))
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(path, len(rows), "cases")
    with open(os.path.join(FIXTURES, "llm_logits_decision.json"), "w") as f:
        json.dump(json.load(open(a.llm_logits)), f, ensure_ascii=False, indent=1)
    with open(os.path.join(FIXTURES, "winnow_decision.json"), "w") as f:
        json.dump(json.load(open(a.winnow)), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
