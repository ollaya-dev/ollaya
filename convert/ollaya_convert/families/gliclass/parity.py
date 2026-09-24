"""Check a GLiClass ONNX export against the PyTorch fp32 reference and the upstream pipeline call.

    uv run --with gliclass==0.1.20 python -m ollaya_convert.families.gliclass.parity large out/gliclass-instruct-large-wl

Checks per question:
  1. tokenizer   `tokenizers.Tokenizer.from_file(<dir>/tokenizer.json)` (the Rust core) with padding off
                 and truncation set to max_length=1024 reproduces the row ids from the row text.
  2. export      ONNX (padded batch, CPU EP) vs upstream GLiClassModel.forward per unpadded row (fp32,
                 TF32 off): label logits, option probabilities, argmax.
  3. upstream    the reference's probabilities vs `ZeroShotClassificationPipeline.__call__` itself.
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
from .export import INPUT_NAMES, MIN_MARKERS


def wire(x):
    return json.loads(json.dumps(x, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=sorted(ref.VARIANTS))
    ap.add_argument("model_dir")
    ap.add_argument("--td-limit", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--provider", default="CPUExecutionProvider")
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m = ref.load(a.device, a.variant)
    rt = Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json"))
    print("tokenizer.json truncation=%s padding=%s" % (rt.truncation, rt.padding))
    rt.no_padding()
    rt.enable_truncation(max_length=ref.MAX_LEN)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(os.path.join(a.model_dir, "model.onnx"), so, providers=[a.provider])

    stats, worst, rejected, up_err = defaultdict(list), [], [], []
    n_q = agree = tok_ok = up_agree = up_n = 0
    t_ort = 0.0
    for cid, state, questions in cases.all_cases(a.td_limit):
        state, questions = wire(state), wire(questions)
        enc = ref.encode(m, state, questions)
        for qid, err in enc["rejected"].items():
            rejected.append("%s/%s: %s" % (cid, qid, err))
        if not enc["items"]:
            continue
        for it in enc["items"]:
            tok_ok += int(rt.encode(it["text"], add_special_tokens=True).ids == it["ids"])
        ref_logits = ref.forward(m, enc)
        batch = ref.collate(enc, m.tok.pad_token_id, MIN_MARKERS)
        t0 = time.perf_counter()
        (got,) = sess.run(None, {n: batch[n] for n in INPUT_NAMES})
        t_ort += time.perf_counter() - t0
        masked = got[~batch["marker_mask"]]
        if masked.size and not np.all(masked == -1e4):
            raise AssertionError("masked slots must be -1e4")
        for i, (it, lr) in enumerate(zip(enc["items"], ref_logits)):
            lg = got[i, : len(it["markers"])]
            stats["logit/" + it["qtype"]].append(float(np.abs(lr - lg).max()))
            p_ref = ref.softmax(ref.option_logits(it, lr))
            p_got = ref.softmax(ref.option_logits(it, lg))
            d = float(np.abs(p_ref - p_got).max())
            same = int(p_ref.argmax() == p_got.argmax())
            n_q += 1
            agree += same
            stats["prob/" + it["qtype"]].append(d)
            worst.append((d, cid, it["qid"], same))
            try:
                p_up = ref.upstream_scores(m, it, enc["state_text"])
            except Exception as e:  # an upstream limitation, not a parity failure: report it
                up_err.append("%s/%s: %s" % (cid, it["qid"], e))
                continue
            up_n += 1
            up_agree += int(p_up.argmax() == p_ref.argmax())
            stats["upstream_prob"].append(float(np.abs(p_up - p_ref).max()))

    print("questions: %d   ort time: %.1fs (%s)" % (n_q, t_ort, a.provider))
    print("rejected (answered 400): %d %s" % (len(rejected), rejected))
    print("tokenizer.json (Rust core) reproduces row ids: %d/%d" % (tok_ok, n_q))
    print("argmax agreement  onnx vs fp32 reference: %.4f" % (agree / n_q))
    print("reference vs gliclass pipeline call: %d questions, argmax agreement %.4f (errors: %s)"
          % (up_n, up_agree / max(up_n, 1), up_err))
    for key in sorted(stats):
        v = np.array(stats[key])
        print("  %-16s max %.2e   p99 %.2e   mean %.2e" % (key, v.max(), np.quantile(v, 0.99), v.mean()))
    print("worst probability deltas (onnx vs reference):")
    for d, cid, qid, same in sorted(worst, reverse=True)[:8]:
        print("  %.2e  %s  %s%s" % (d, cid, qid, "" if same else "  ARGMAX DIFFERS"))


if __name__ == "__main__":
    main()
