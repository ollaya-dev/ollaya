"""Check the schema-scorer ONNX export against upstream `LocalSystemOne` (fp32) on the case set.

    uv run python -m ollaya_convert.families.schema_scorer.parity out/jev-schema-scorer-deberta-v3-large-wl

Checks: the Rust tokenizer core reproduces every pair (no truncation, padding off); ONNX (one padded batch
per request, CPU EP) vs upstream score_pairs (fp32, TF32 off, its own batching); our probabilities vs
upstream `system_one` answers.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--td-limit", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--provider", default="CPUExecutionProvider")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    m = ref.load(a.device)
    rt = Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json"))
    print("tokenizer.json truncation=%s padding=%s" % (rt.truncation, rt.padding))
    rt.no_truncation()
    rt.no_padding()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(os.path.join(a.model_dir, "model.onnx"), so, providers=[a.provider])

    stats, worst, rejected, up_err = defaultdict(list), [], defaultdict(int), []
    n_q = n_rows = agree = tok_ok = up_agree = up_n = 0
    t_ort = 0.0
    for cid, state, questions in cases.all_cases(a.td_limit):
        state, questions = json.loads(json.dumps(state)), json.loads(json.dumps(questions))
        enc = ref.encode(m, state, questions)
        for err in enc["rejected"].values():
            rejected[err] += 1
        if not enc["items"]:
            continue
        for it in enc["items"]:
            for r in it["rows"]:
                tok_ok += int(rt.encode(r["text_a"], r["text_b"], add_special_tokens=True).ids == r["ids"])
                n_rows += 1
        s_ref = ref.forward(m, enc)
        batch = ref.collate(enc, m.tok.pad_token_id)
        t0 = time.perf_counter()
        (got,) = sess.run(None, {n: batch[n] for n in INPUT_NAMES})
        t_ort += time.perf_counter() - t0
        try:
            up = ref.upstream_answers(m, state, {it["qid"]: ref.to_upstream(questions[it["qid"]]) for it in enc["items"]})
        except Exception as e:  # an upstream limitation, not a parity failure: report it
            up_err.append("%s: %s" % (cid, e))
            up = {}
        for (it, sr), (_, sg) in zip(ref.split(enc, s_ref), ref.split(enc, got)):
            stats["logit/" + it["qtype"]].append(float(np.abs(sr - sg).max()))
            p_ref, p_got = ref.softmax(ref.option_logits(it, sr)), ref.softmax(ref.option_logits(it, sg))
            d = float(np.abs(p_ref - p_got).max())
            same = int(p_ref.argmax() == p_got.argmax())
            n_q += 1
            agree += same
            stats["prob/" + it["qtype"]].append(d)
            worst.append((d, cid, it["qid"], same))
            if it["qid"] not in up:
                continue
            up_n += 1
            u = up[it["qid"]]
            p_up = (np.array([1 - u["noul"], u["noul"]]) if it["qtype"] == "noul" else
                    np.array(list(u["probabilities"].values()) if isinstance(u["probabilities"], dict) else u["probabilities"]))
            up_agree += int(p_up.argmax() == p_ref.argmax())
            stats["upstream_prob"].append(float(np.abs(p_up - p_ref).max()))

    print("questions: %d   rows: %d   ort time: %.1fs (%s)" % (n_q, n_rows, t_ort, a.provider))
    print("rejected by upstream (answered 400): %s" % dict(rejected))
    print("tokenizer.json (Rust core) reproduces rows: %d/%d" % (tok_ok, n_rows))
    print("argmax agreement  onnx vs fp32 reference: %.4f" % (agree / n_q))
    print("reference vs upstream system_one: %d questions, argmax agreement %.4f (errors: %s)"
          % (up_n, up_agree / max(up_n, 1), up_err))
    for key in sorted(stats):
        v = np.array(stats[key])
        print("  %-16s max %.2e   p99 %.2e   mean %.2e" % (key, v.max(), np.quantile(v, 0.99), v.mean()))
    print("worst probability deltas (onnx vs reference):")
    for d, cid, qid, same in sorted(worst, reverse=True)[:8]:
        print("  %.2e  %s  %s%s" % (d, cid, qid, "" if same else "  ARGMAX DIFFERS"))


if __name__ == "__main__":
    main()
