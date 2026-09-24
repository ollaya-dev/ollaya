"""`qwen3guard-gen-v1`: Qwen3Guard-Gen as a fixed-preset decision model (reference, PyTorch fp32).

Qwen3Guard-Gen's task prompt (policy, categories, output format) lives in its chat template; custom
instructions are not part of it. It therefore answers one fixed question set, not arbitrary typed
questions:

    rows   prompt moderation:  chat_template([{"role": "user", "content": state_text}])
           (the template ends in "<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n")
      r0 = prompt + "Safety:"                              next token in {" Safe", " Unsafe", " Cont"(roversial)}
      r1 = prompt + "Safety: Unsafe\\nCategories:"          next token in the 10 category first-tokens

    preset questions (option logits are first-token logits; the model writes each label as text, so a
    multi-token label is represented by its first token, which is unambiguous in both sets):
      safety    choice  [safe, controversial, unsafe]           r0 logits
      unsafe    noul    [false, true] = [logsumexp(safe, controversial), unsafe]   (loose)
      unsafe_strict noul [false, true] = [safe, logsumexp(controversial, unsafe)]
      category  choice  [none, violent, non_violent_illegal, sexual, pii, suicide_self_harm, unethical,
                         politically_sensitive, copyright, jailbreak]           r1 logits, i.e. the first
                         category the model would list *given* it judged the text unsafe

The state is the text to moderate (strings verbatim; other JSON as json.dumps(ensure_ascii=False)).
"""
from __future__ import annotations

import json

import numpy as np
import torch

REPO = "Qwen/Qwen3Guard-Gen-0.6B"
REVISION = "fada3b2f655b89601929198343c94cd2f64d93cc"
SAFETY = [("safe", " Safe"), ("controversial", " Controversial"), ("unsafe", " Unsafe")]
CATEGORIES = [("none", " None"), ("violent", " Violent"), ("non_violent_illegal", " Non-violent Illegal Acts"),
              ("sexual", " Sexual Content or Sexual Acts"), ("pii", " PII"), ("suicide_self_harm", " Suicide & Self-Harm"),
              ("unethical", " Unethical Acts"), ("politically_sensitive", " Politically Sensitive Topics"),
              ("copyright", " Copyright Violation"), ("jailbreak", " Jailbreak")]
PRESET = {
    "safety": {"type": "choice", "instructions": "Qwen3Guard safety level of the user text",
               "criteria": {k: None for k, _ in SAFETY}},
    "unsafe": {"type": "noul", "instructions": "Qwen3Guard: the user text is unsafe"},
    "unsafe_strict": {"type": "noul", "instructions": "Qwen3Guard: the user text is unsafe or controversial"},
    "category": {"type": "choice", "instructions": "Qwen3Guard: first unsafe category, given unsafe",
                 "criteria": {k: None for k, _ in CATEGORIES}},
}


def first_tokens(tok, labels):
    ids = [tok.encode(text, add_special_tokens=False)[0] for _, text in labels]
    assert len(set(ids)) == len(ids), "first tokens collide"
    return ids


def state_text(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def rows(tok, state):
    prompt = tok.apply_chat_template([{"role": "user", "content": state_text(state)}], tokenize=False)
    r0 = tok.encode(prompt + "Safety:", add_special_tokens=False)
    r1 = tok.encode(prompt + "Safety: Unsafe\nCategories:", add_special_tokens=False)
    return [r0, r1]


def load(root, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(root)
    m = AutoModelForCausalLM.from_pretrained(root, dtype=torch.float32).to(device).eval()
    return tok, m


@torch.no_grad()
def last_logits(m, row_ids, cand):
    dev = next(m.parameters()).device
    out = []
    for ids in row_ids:
        lg = m(input_ids=torch.tensor([ids], device=dev)).logits[0, -1].float()
        out.append(lg[cand].cpu().numpy())
    return out


def lse(*x):
    x = np.array(x, dtype=np.float64)
    return float(x.max() + np.log(np.exp(x - x.max()).sum()))


def option_logits(z_safety, z_cat):
    s, c, u = (float(x) for x in z_safety)
    return {"safety": [s, c, u], "unsafe": [lse(s, c), u], "unsafe_strict": [s, lse(c, u)],
            "category": [float(x) for x in z_cat]}


def decide(tok, m, state):
    r = rows(tok, state)
    zs = last_logits(m, [r[0]], first_tokens(tok, SAFETY))[0]
    zc = last_logits(m, [r[1]], first_tokens(tok, CATEGORIES))[0]
    return option_logits(zs, zc)
