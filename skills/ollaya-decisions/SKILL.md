---
name: ollaya-decisions
description: Make typed, calibrated decisions about text or JSON with local decision models served by Ollaya: classify (choice), rate (score) or check a yes/no statement (noul) in milliseconds, with probabilities you can threshold. Use it to triage tickets and emails, route requests, moderate posts, screen prompts for jailbreaks or injections, or any step where the agent needs a quick judgement it can act on, instead of reasoning it out in text. Works through the Ollaya MCP server (the `decide` tool), the `ollaya` CLI, or the local HTTP API.
license: Apache-2.0
metadata:
  homepage: https://ollaya.dev
---

# Typed decisions with Ollaya

Ollaya runs *decision models* on this machine. A decision model reads a **state** (a message, an
email, a ticket, any JSON) and a set of **typed questions**, and returns a calibrated answer to
each question in one forward pass, in about 10 ms on a GPU and a few hundred ms on a CPU. It never
generates text.

Reach for it when the answer is one of a known set of outcomes and you will act on it: route,
label, block, escalate, pick a template. Keep reasoning in text for open-ended work.

## When a typed decision is the right tool

| Situation | Use |
|---|---|
| Label, route or filter many items (tickets, emails, messages, rows) | `decide` on each item |
| A yes/no gate before an action (is this spam, a jailbreak, a refund request) | a `noul` question |
| A rating you will compare with a threshold (urgency, severity, frustration) | a `score` question |
| One of N categories, teams or intents | a `choice` question |
| Summaries, explanations, answers that need new text | not this skill |

A decision costs no tokens and no API call, so it is fine to run one per item, per step.

## How to call it

Check which of these is available, in this order.

1. **MCP** (tool `decide`, server `ollaya`): pass `state`, and `questions` or `preset`, plus
   optionally `model`. Other tools: `list_models`, `show_model`, `pull_model`. Resources
   `ollaya://presets/<name>` show each preset's questions.
2. **CLI** (`ollaya` on PATH):
   ```sh
   ollaya run laya --preset triage --format json "I was charged twice and want a refund."
   ollaya run laya --questions questions.json --format json "$TEXT"
   echo '{"subject": "…", "body": "…"}' | ollaya run laya --preset email --format json
   ```
   `--format json` prints the full response. The CLI starts the server if it isn't running and
   pulls the model on first use.
3. **HTTP** (`http://127.0.0.1:11435`), TypeSafe-compatible:
   ```sh
   curl -s http://127.0.0.1:11435/v1/systemone -H 'Content-Type: application/json' \
     -d '{"model": "laya", "state": "…", "questions": {…}}'
   ```

If none is available, tell the user how to install Ollaya:
`curl -fsSL https://ollaya.dev/install.sh | sh`.

## Picking a model

| Model | Strength | Speed (5 questions) |
|---|---|---|
| `laya` (default) | English and 100+ languages, routed automatically; calibrated | ~10 ms GPU, ~0.2–0.4 s CPU |
| `decider` | The most accurate; slower | ~0.2 s GPU, ~1 s CPU |
| `nli` | Zero-shot, good at yes/no with clear statements | ~20 ms GPU |
| `gliclass` | Zero-shot, many options in one pass | ~15 ms GPU |
| `kev` | Qwen3.5-0.8B decoder with a pointer head; calibrated | ~0.2 s GPU, ~2 s CPU |
| `qwen3guard` | Safety guard; answers only its built-in questions (send no `questions`) | ~40 ms GPU, ~2 s CPU |
| `von` | ModernBERT-large, every option scored at its own marker; states up to 8k tokens; calibrated | ~25 ms GPU, ~0.8 s CPU |

Start with `laya`. Move to `decider` when accuracy matters more than latency, or when `laya`'s
confidence is often low on your data.

## Presets

Built-in question sets, usable as `preset` (MCP), `--preset` (CLI):

- `triage`: intent (refund, technical_help, billing_question, information, cancellation, other),
  is_urgent, frustration (0–3), refund_requested, churn_risk. State: a customer `message`.
- `email`: category, is_spam, is_phishing, urgency, needs_reply. State: an email `body`.
- `guard`: jailbreak, prompt_injection, sensitive_data, harm_severity, topic. State: a `prompt`.
- `moderation`: toxic, harassment, threat, spam, severity. State: a `post`.
- `router`: difficulty, domain, needs_tools, is_sensitive. State: a `request`.

A preset's instructions refer to a field (`message`, `body`, `prompt`, `post`, `request`). Pass the
state as an object with that field, or as a plain string.

## Writing your own questions

Questions are an object keyed by an id you choose:

```json
{
  "team": {
    "type": "choice",
    "instructions": "Which team should handle `ticket`?",
    "criteria": {
      "billing": "invoices, charges, refunds, plans",
      "engineering": "bugs, outages, errors, integrations",
      "sales": "pricing, demos, upgrades",
      "other": "anything else"
    }
  },
  "urgency": {
    "type": "score",
    "instructions": "How urgent is `ticket`?",
    "criteria": ["can wait", "this week", "today", "right now: an outage or a deadline"]
  },
  "wants_refund": {
    "type": "noul",
    "instructions": "Does the customer ask for their money back?"
  }
}
```

Rules that make answers better:

- **choice**: give every option a short description in `criteria`, make the options mutually
  exclusive, and include a catch-all (`other`). Up to a few dozen options work.
- **score**: 3–5 levels, lowest first, each described concretely. The answer is the expected level
  (it can fall between levels), with a `legend`.
- **noul**: phrase one statement that is true or false ("Does the customer ask for a refund?").
  Avoid double negatives and two questions in one.
- Name the state's field in `instructions` (`` `ticket` ``) when the state is an object.
- Keep instructions short and literal. The model reads them; it does not follow long prompts.

## Reading answers and acting on them

| `type` | Fields |
|---|---|
| `choice` | `choice`, `confidence` (0–1), `probabilities` per option |
| `score` | `score` (expected level), `confidence`, `legend`, `probabilities` per level |
| `noul` | `noul`: the probability that the statement is true |

Act on the answer only when it is clear, and escalate otherwise:

- `choice`: act when `confidence` ≥ 0.6; below that, treat the top two options as candidates or
  ask a human.
- `noul`: treat ≥ 0.8 as yes and ≤ 0.2 as no; in between, escalate or look closer.
- `score`: compare `score` with your threshold and look at `confidence` before acting on it.

Tune thresholds on a handful of real examples when the stakes are high. Say in your output which
model answered and the probability behind each action, so a person can audit it.

## Example: triage a batch of tickets

1. Pick questions: the `triage` preset, or your own `team`/`urgency` set.
2. For each ticket, call `decide` with `{"model": "laya", "state": {"message": ticket_text}, "preset": "triage"}`.
3. Route by `intent`, flag `is_urgent` ≥ 0.8 or `frustration` ≥ 2, and send `churn_risk` ≥ 0.8 to
   a retention queue. Put every ticket with a low-confidence intent in a "needs review" list.
4. Report a table: ticket, action, and the probabilities behind it.
