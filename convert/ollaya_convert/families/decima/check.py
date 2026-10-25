"""The layout port against upstream, id for id, on the shared request set.

    uv run python -m ollaya_convert.families.decima.check [--model decima-small] [--td 400]

Upstream: serve.py's verdict, then decima.systemone.to_question, DecimaConfig.state_of / choices_of (render.py) and
the checkpoint's tokenizer through transformers, truncated as Decima._encode truncates (`ref.encode`). The port:
`layout.DecimaLayout` on the same tokenizer.json loaded by HF `tokenizers` (what the runtime loads), configured by
`decision_for` below. Every question's state row and option rows must be identical, so must the state's truncation
flag (upstream's 422 STATE_TRUNCATED), and rejections must match. Requests Ollaya's shared question parser rejects
first are counted on their own.

The shared set is `llm_common.cases` (the edge cases and typed-decisions rows); `extra_cases` adds the inputs
Decima's mapping treats specially and `long_cases` (states that fill decima-agent's 8,192-token window, and one over
it), and `fit_cases` the questions too long for the option window (render.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import tokenizers

from ..clef.layout import ollaya_rule
from ..llm_common import cases
from . import ref
from .layout import DecimaLayout, LayoutError

# systemone.to_question's noul rendering, `f"{text}\nTrue if: {tr or M}\nFalse if: {fa or M}"` with M an em dash
# (U+2014), and the options of its `verify` question.
NOUL = {"options": ["yes", "no"], "criteria": "\nTrue if: {true}\nFalse if: {false}", "missing": "\u2014"}


def decision_for(d):
    """The layout's fields of decision.json, from the upstream config and modules. `option_fit` (decima.json's
    choice_fit) is written only for a model that has it, so the decision.json of the models without it is unchanged."""
    from decima import systemone

    cfg, tok = d.cfg, d.tok
    return {
        "state_prefix": cfg.state_prefix,
        "option_prefix": cfg.choice_prefix,
        "question_in_state": cfg.question_in_state,
        "question_in_options": cfg.question_in_choices,
        "max_state_tokens": cfg.max_state_tokens,
        "max_option_tokens": cfg.max_choice_tokens,
        "special_tokens": {"cls": tok.cls_token_id, "sep": tok.sep_token_id, "pad": tok.pad_token_id},
        "noul": NOUL,
        "limits": {"questions": [1, systemone.MAX_QUESTIONS], "choices": [2, systemone.MAX_CHOICES],
                   "levels": [systemone.MIN_LEVELS, systemone.MAX_LEVELS]},
        "temperature": cfg.temperature,
        **({"option_fit": True} if cfg.choice_fit else {}),
    }


def wire(x):
    return json.loads(json.dumps(x, ensure_ascii=False))


def extra_cases():
    """Inputs Decima's mapping treats specially, as a JSON client sends them."""
    long_option = "a remarkably long option description " * 12
    qs = {
        "noul_false_only": {"type": "noul", "instructions": "Is the customer angry?", "criteria": {"false": "calm"}},
        "noul_true_object": {"type": "noul", "instructions": "Is money involved?",
                             "criteria": {"true": {"any_of": ["refund", "charge"]}}},
        "noul_null_instructions": {"type": "noul", "instructions": None},
        "choice_numbers": {"type": "choice", "instructions": {"task": "route", "lang": "é"},
                           "criteria": {"a": 1, "b": 2.5, "c": True, "d": None, "e": " ", "f": ["x", "y"]}},
        "choice_long_option": {"type": "choice", "instructions": "Which one?",
                               "criteria": {"long": long_option, "short": "short"}},
        "score_objects": {"type": "score", "instructions": "", "criteria": [{"what": "low"}, "medium", ["high"]]},
        "choice_duplicates": {"type": "choice", "instructions": "Pick", "criteria": ["a", "b", "a", "c"]},
    }
    nfd = "Café über  näive Å ﬁ  "
    return [
        ("decima/specials", wire({"ticket": "Refund my double charge", "n": 3}), wire(qs)),
        ("decima/nfd_state", nfd, wire({"q": {"type": "choice", "instructions": " Résumé? ",
                                              "criteria": {"yes": "ök", "no": None}}})),
        ("decima/list_state", wire([1, "two", {"three": 3.0}]), wire({"q": qs["noul_false_only"]})),
        ("decima/number_state", 42, wire({"q": qs["noul_false_only"]})),
        ("decima/one_choice", "hi", wire({"q": {"type": "choice", "instructions": "x", "criteria": ["only", "only"]}})),
        ("decima/eleven_levels", "hi", wire({"q": {"type": "score", "instructions": "x",
                                                   "criteria": [str(i) for i in range(11)]}})),
        ("decima/256_choices", "hi", wire({"q": {"type": "choice", "instructions": "x",
                                                 "criteria": ["c%d" % i for i in range(256)]}})),
        ("decima/255_choices", "hi", wire({"q": {"type": "choice", "instructions": "x",
                                                 "criteria": ["c%d" % i for i in range(255)]}})),
        ("decima/empty_instructions_state", "", wire({"q": {"type": "noul", "instructions": ""}})),
    ] + ordinal_cases() + long_cases()


def ordinal_cases():
    """Choices read by the ordinal head (systemone's "ordinal": true): decima-guard's guard-web preset, labels with
    descriptions, the limit of 10 labels, and the flag off."""
    web = json.load(open(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "crates", "ollaya-api", "src",
                                      "presets", "decima-guard-web.json"), encoding="utf-8"))
    severity = {"type": "choice", "ordinal": True, "instructions": "How severe is the incident?",
                "criteria": {"none": "", "low": "a cosmetic issue", "high": "customers are blocked",
                             "critical": {"data_loss": True, "outage": "full"}}}
    return [
        ("decima/ordinal/web_sqli", "GET /search?q=' OR 1=1 -- HTTP/1.1\nHost: shop.example", wire(web)),
        ("decima/ordinal/web_benign", "GET /products?page=2&sort=price HTTP/1.1", wire(web)),
        ("decima/ordinal/web_traversal", wire({"method": "GET", "path": "/static/../../etc/passwd"}), wire(web)),
        ("decima/ordinal/severity", "Checkout fails for every customer since the 14:00 deploy.",
         wire({"severity": severity, "plain": {**severity, "ordinal": False}})),
        ("decima/ordinal/ten_labels", "a mild complaint",
         wire({"q": {"type": "choice", "ordinal": True, "instructions": "Level?", "criteria": [str(i) for i in range(10)]}})),
        ("decima/ordinal/eleven_labels", "a mild complaint",
         wire({"q": {"type": "choice", "ordinal": True, "instructions": "Level?", "criteria": [str(i) for i in range(11)]}})),
    ]


def agent_log(steps):
    """A coding agent's transcript of `steps` test runs, about 95 mmBERT tokens each."""
    return "".join("[step %d] $ cargo test -p billing --release\n   Compiling billing v0.4.%d (/work/billing)\n"
                   "running 48 tests\ntest refund::partial_refund_keeps_the_order_open ... ok\n"
                   "test refund::refund_over_total_is_rejected ... %s\nassistant: The refund check compares with >= "
                   "now; I will read billing/refund.py again before I edit it.\n"
                   % (i, i % 10, "FAILED" if i % 3 else "ok") for i in range(steps))


def long_cases():
    """States for decima-agent's 8,192-token window: state rows (question included) of about 3k, 6k and 8.1k
    mmBERT tokens, which agent keeps whole, and 8.3k, which upstream answers with STATE_TRUNCATED. Every other tag
    truncates all four (512 or 2,048 tokens)."""
    qs = wire({"passed": {"type": "noul", "instructions": "Did every test in the last run pass?"},
               "next": {"type": "choice", "instructions": "What should the agent do next?",
                        "criteria": {"edit": "change the code", "rerun": "run the tests again", "ask": None}}})
    return [("decima/long_%s" % name, agent_log(steps), qs)
            for name, steps in (("3k", 32), ("6k", 63), ("8100", 85), ("8300_truncated", 87))]


# A question long enough to push every option out of the 64-token option window on its own.
REVIEW = ("You are the merge gate of a payments service. Read the pull request below and decide whether it can be "
          "merged without a second human reviewer. Consider security, data loss, backwards compatibility of the public "
          "API, database migrations, the amount of test coverage that comes with the change, and whether the author "
          "touched code that moves money. When in doubt, ask for a second reviewer.")
DIFF = ("diff --git a/billing/refund.py b/billing/refund.py\n-    if amount > order.total:\n+    if amount >= order.total:\n"
        "         raise RefundError('refund exceeds the order total')\n")


def fit_cases():
    """Questions whose instructions and option do not fit max_choice_tokens together: render.py shortens the question's
    copy beside each option (option_fit), or, without it, keeps the right-hand cut unless two options' rows collapse.
    Each request holds one question, so a collapse in one never changes another's rendering."""
    route = ("Route this support ticket to exactly one team. The ticket was written by a paying customer through the "
             "web form, may mix several issues, and may be in any language; pick the team that has to act first, not "
             "the one the customer names, and ignore greetings, signatures and quoted earlier replies.")
    calm = ("the customer stays calm and polite through the whole conversation, thanks the agent for every answer, "
            "never raises their voice, never threatens to cancel the subscription or to leave for a competitor, and "
            "accepts the proposed solution without asking for a refund, a discount, a call with a manager or any other "
            "compensation")
    ws = ("Decide\twhether  the\nfollowing\u00a0request\u2003needs\u3000a human\u001fto approve it \u2028before "
          "it\u0085runs; Re\u0301sume\u0301 the plan, zero\u200bwidth stays inside a word, and read every line of "
          "the request carefully, including the parts that only repeat what was said before, the signature, the "
          "quoted earlier replies and the legal footer that the mail client adds to every message, then answer.")
    q = {
        "noul_true_false": {"type": "noul", "instructions": REVIEW,
                            "criteria": {"true": "the diff only touches tests, documentation or comments",
                                         "false": "it changes refunds, charges, authentication or migrations"}},
        "noul_long": {"type": "noul", "instructions": REVIEW},
        "choice_long": {"type": "choice", "instructions": route,
                        "criteria": {"billing": "charges, refunds, invoices", "technical": "bugs, outages, errors",
                                     "account": "login, password, profile", "other": None}},
        "choice_mid": {"type": "choice", "instructions": "Which team should take this ticket first?",
                       "criteria": {"billing": "charges, refunds, invoices, duplicate payments, failed card "
                                               "payments, VAT questions and anything else about money " * 2,
                                    "technical": "bugs, outages, errors, slow pages, broken integrations " * 2}},
        "score_long": {"type": "score", "instructions": route.replace("Route this support ticket to exactly one "
                                                                     "team", "Rate how upset the customer is"),
                       "criteria": [None, "calm", "annoyed", "angry, threatening to leave"]},
        "score_partial_collapse": {"type": "score",
                                   "instructions": "How upset is the customer who wrote this ticket to support today?",
                                   "criteria": [calm, calm + " twice", "furious"]},
        "choice_no_spaces": {"type": "choice",
                             "instructions": "请根据工单内容判断应该由哪个团队首先处理这个问题客户可能同时提到多个问题请选择必须"
                                             "最先采取行动的团队而不是客户点名的团队忽略问候语签名和引用的旧回复" * 2,
                             "criteria": {"billing": "账单", "technical": "技术", "other": None}},
        "noul_whitespace": {"type": "noul", "instructions": ws},
        "choice_long_option_text": {"type": "choice", "instructions": REVIEW,
                                    "criteria": {"merge": "merge it now " * 30, "wait": "ask for a second reviewer"}},
        "noul_few_chars": {"type": "noul", "instructions": "\U00013000" * 40 + "?"},
    }
    return [("decima/fit/" + k, wire({"pr": DIFF, "author": "first-time contributor"}) if i % 2 else DIFF,
             wire({k: v})) for i, (k, v) in enumerate(q.items())]


def all_cases(td):
    return cases.edge_cases() + cases.typed_decisions(td) + extra_cases() + fit_cases()


def shared_rule(questions):
    """Ollaya's shared question parser rejects this request before any layout sees it."""
    return not isinstance(questions, dict) or not questions or any(ollaya_rule(k, v) for k, v in questions.items())


def tokenizer(path):
    tok = tokenizers.Tokenizer.from_file(path)
    tok.no_truncation()
    tok.no_padding()
    return tok


def compare(d, lay, state, questions):
    """-> "identical", "rejected" (both reject) or the first difference."""
    code, msg = ref.verdict(d, state, questions)
    try:
        up, ue = ref.encode(d, state, questions), None
    except ref.RequestError as e:
        up, ue = None, e
    try:
        ours, oe = lay.encode(state, questions), None
    except LayoutError as e:
        ours, oe = None, e
    upstream_rejects = ue is not None or code not in (None, "STATE_TRUNCATED")
    if upstream_rejects or oe is not None:
        if upstream_rejects and oe is not None:
            return "rejected"
        return "rejection differs: upstream %s, port %s" % (ue or code, oe)
    if (code == "STATE_TRUNCATED") != any(q["state_truncated"] for q in ours["questions"]):
        return "STATE_TRUNCATED differs: upstream %s" % code
    for u, o in zip(up, ours["questions"]):
        if (u["qid"], u["type"]) != (o["qid"], o["type"]) or u["state_truncated"] != o["state_truncated"]:
            return "%s: question or truncation differs" % u["qid"]
        if u["state_ids"] != o["state_ids"]:
            a, b = u["state_ids"], o["state_ids"]
            at = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            return "%s: state ids %d vs %d, first difference at %d" % (u["qid"], len(a), len(b), at)
        if u["option_ids"] != o["option_ids"]:
            return "%s: option ids differ" % u["qid"]
    return "identical"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=ref.DEFAULT, choices=sorted(ref.MODELS))
    ap.add_argument("--td", type=int, default=400, help="typed-decisions rows (0: all 400)")
    a = ap.parse_args()
    d = ref.load("cpu", a.model)
    lay = DecimaLayout(tokenizer(os.path.join(ref.checkpoint(a.model), "encoder", "tokenizer.json")), decision_for(d))
    counts = {"identical": 0, "rejected": 0, "shared": 0, "truncated": 0}
    bad = []
    for cid, state, questions in all_cases(a.td):
        if shared_rule(questions):
            counts["shared"] += 1
            continue
        r = compare(d, lay, state, questions)
        if r in ("identical", "rejected"):
            counts[r] += 1
            if r == "identical" and ref.verdict(d, state, questions)[0] == "STATE_TRUNCATED":
                counts["truncated"] += 1
        else:
            bad.append((cid, r))
    print("identical requests %(identical)d (%(truncated)d with a truncated state) | both reject %(rejected)d | "
          "Ollaya's shared rules reject %(shared)d" % counts + " | mismatches %d" % len(bad))
    for b in bad[:15]:
        print("  ", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
