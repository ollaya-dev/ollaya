"""Check an NLI ONNX export against the PyTorch fp32 reference and the transformers zero-shot pipeline.

    uv run python -m ollaya_convert.families.nli.parity deberta-v3-large out/deberta-v3-large-zeroshot-v2.0-wl

Checks per question:
  1. tokenizer   `tokenizers.Tokenizer.from_file(<dir>/tokenizer.json)` (the Rust core), padding off and
                 truncation max_length=512 / only_first, reproduces every row from (premise, hypothesis);
                 and so does the manual rule [CLS] premise[:budget] [SEP] hypothesis [SEP].
  2. export      ONNX (padded batch of all rows, CPU EP) vs the fp32 model per unpadded row (TF32 off).
  3. upstream    reference probabilities vs `pipeline("zero-shot-classification")` on the same module.
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import onnxruntime as ort
import torch
from tokenizers import Tokenizer

from ... import cases
from . import ref
from .export import INPUT_NAMES


def wire(x):
    return json.loads(json.dumps(x, ensure_ascii=False))


def manual_pair(rt, cls, sep, premise, hypothesis):
    p = rt.encode(premise, add_special_tokens=False).ids
    h = rt.encode(hypothesis, add_special_tokens=False).ids
    budget = ref.MAX_LEN - 3 - len(h)
    return [cls] + p[:max(budget, 0)] + [sep] + h + [sep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(ref.MODELS))
    ap.add_argument("model_dir")
    ap.add_argument("--td-limit", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--provider", default="CPUExecutionProvider")
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m = ref.load(a.device, a.model)
    rt = Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json"))
    print("tokenizer.json truncation=%s padding=%s" % (rt.truncation, rt.padding))
    rt.no_padding()
    rt.no_truncation()
    rt_trunc = Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json"))
    rt_trunc.no_padding()
    rt_trunc.enable_truncation(max_length=ref.MAX_LEN, strategy="only_first")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(os.path.join(a.model_dir, "model.onnx"), so, providers=[a.provider])

    stats, worst, rejected, up_err = defaultdict(list), [], [], []
    n_q = n_rows = agree = tok_ok = man_ok = up_n = up_agree = 0
    t_ort = 0.0
    for cid, state, questions in cases.all_cases(a.td_limit):
        state, questions = wire(state), wire(questions)
        enc = ref.encode(m, state, questions)
        for qid, err in enc["rejected"].items():
            rejected.append("%s/%s: %s" % (cid, qid, err))
        if not enc["items"]:
            continue
        for it in enc["items"]:
            for r in it["rows"]:
                tok_ok += int(rt_trunc.encode(enc["premise"], r["hypothesis"], add_special_tokens=True).ids == r["ids"])
                man_ok += int(manual_pair(rt, m.tok.cls_token_id, m.tok.sep_token_id, enc["premise"], r["hypothesis"]) == r["ids"])
                n_rows += 1
        ref_scores = ref.forward(m, enc)
        batch = ref.collate(enc, m.tok.pad_token_id)
        t0 = time.perf_counter()
        (got,) = sess.run(None, {n: batch[n] for n in INPUT_NAMES})
        t_ort += time.perf_counter() - t0
        for (it, s_ref), (_, s_got) in zip(ref.split(enc, ref_scores), ref.split(enc, got)):
            stats["logit/" + it["qtype"]].append(float(np.abs(s_ref - s_got).max()))
            p_ref = ref.softmax(ref.option_logits(it, s_ref))
            p_got = ref.softmax(ref.option_logits(it, s_got))
            d = float(np.abs(p_ref - p_got).max())
            same = int(p_ref.argmax() == p_got.argmax())
            n_q += 1
            agree += same
            stats["prob/" + it["qtype"]].append(d)
            worst.append((d, cid, it["qid"], same))
            try:
                p_up = ref.upstream_probabilities(m, enc["premise"], it)
            except Exception as e:  # an upstream limitation, not a parity failure: report it
                up_err.append("%s/%s: %s" % (cid, it["qid"], e))
                p_up = None
            if p_up is not None:
                up_n += 1
                up_agree += int(p_up.argmax() == p_ref.argmax())
                stats["upstream_prob"].append(float(np.abs(p_up - p_ref).max()))

    print("questions: %d   rows: %d   ort time: %.1fs (%s)" % (n_q, n_rows, t_ort, a.provider))
    print("rejected (answered 400): %d %s" % (len(rejected), rejected))
    print("tokenizer.json (Rust core, only_first@512) reproduces rows: %d/%d; manual rule: %d/%d"
          % (tok_ok, n_rows, man_ok, n_rows))
    print("argmax agreement  onnx vs fp32 reference: %.4f" % (agree / n_q))
    print("reference vs zero-shot pipeline: %d questions, argmax agreement %.4f (errors: %s)"
          % (up_n, up_agree / max(up_n, 1), up_err))
    for key in sorted(stats):
        v = np.array(stats[key])
        print("  %-16s max %.2e   p99 %.2e   mean %.2e" % (key, v.max(), np.quantile(v, 0.99), v.mean()))
    print("worst probability deltas (onnx vs reference):")
    for d, cid, qid, same in sorted(worst, reverse=True)[:8]:
        print("  %.2e  %s  %s%s" % (d, cid, qid, "" if same else "  ARGMAX DIFFERS"))


if __name__ == "__main__":
    main()
