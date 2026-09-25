"""Check the Von ONNX export against the PyTorch reference (and against von itself) on the case set.

    uv run --with von-sdk==1.1.1 python -m ollaya_convert.families.von.parity out/von --td-limit 100

Three checks per question:
  1. tokenizer   `tokenizers.Tokenizer.from_file(<dir>/tokenizer.json)` -- the Rust core the runtime
                 links -- with truncation and padding switched off, reproduces every row's ids from
                 its packed text (add_special_tokens=True).
  2. export      ONNX (onnxruntime, padded batch of all rows) vs the upstream network run one unpadded
                 row at a time in float64 (`ref.Exact`, as the goldens; `--precision fp32` for upstream's
                 own fp32 forward, TF32 off): logits, calibrated probabilities, argmax.
  3. upstream    the reference's calibrated probabilities vs `OptionMarkerBackend.evaluate` (von's own
                 answer, rounded by von to 4 dp) for every question upstream accepts unchanged.
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
    """The request as it arrives over HTTP (JSON has only string keys)."""
    return json.loads(json.dumps(x, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--td-limit", type=int, default=100, help="typed-decisions rows (0 = all 400)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--provider", default="CPUExecutionProvider")
    ap.add_argument("--no-upstream", action="store_true")
    ap.add_argument("--precision", choices=["fp64", "fp32"], default="fp64",
                    help="reference: the network in float64 (default, as the goldens) or upstream's fp32 forward")
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    backend = ref.load(a.device)
    exact = ref.Exact(backend, a.device) if a.precision == "fp64" else None
    tok = backend._get_model().tokenizer
    rust_tok = Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json"))
    # Upstream's tokenizer.json bakes in truncation=512 / padding; the runtime must switch both off.
    print("tokenizer.json truncation=%s padding=%s -> disabled" % (rust_tok.truncation, rust_tok.padding))
    rust_tok.no_truncation()
    rust_tok.no_padding()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(os.path.join(a.model_dir, "model.onnx"), so, providers=[a.provider])

    stats = defaultdict(list)
    worst = []
    n_q = n_rows = agree = tok_ok = 0
    up_n = up_agree = 0
    t_ort = 0.0
    for cid, state, questions in cases.all_cases(a.td_limit):
        state, questions = wire(state), wire(questions)
        enc = ref.encode(tok, state, questions)
        rows = [r for it in enc["items"] for r in it["rows"]]
        for r in rows:
            tok_ok += int(rust_tok.encode(r["text"], add_special_tokens=True).ids == r["ids"])
        n_rows += len(rows)
        ref_rows = ref.forward_rows(backend, enc, exact)
        batch = ref.collate(enc, tok.pad_token_id, MIN_MARKERS)
        t0 = time.perf_counter()
        (got,) = sess.run(None, {n: batch[n] for n in INPUT_NAMES})
        t_ort += time.perf_counter() - t0

        i = 0
        for it in enc["items"]:
            nr = len(it["rows"])
            r_ref = ref_rows[i:i + nr]
            r_got = [got[i + j, : len(it["rows"][j]["markers"])] for j in range(nr)]
            for j in range(nr):
                stats["row_logit/" + it["qtype"]].append(float(np.abs(r_ref[j] - r_got[j]).max()))
            masked = got[i:i + nr][~batch["marker_mask"][i:i + nr]]
            if masked.size and not np.all(masked == -1e4):
                raise AssertionError("masked slots must be -1e4")
            i += nr
            p_ref = ref.probabilities(backend, it, r_ref, enc["state_tokens"])
            p_got = ref.probabilities(backend, it, r_got, enc["state_tokens"])
            d = float(np.abs(p_ref - p_got).max())
            same = int(p_ref.argmax() == p_got.argmax())
            n_q += 1
            agree += same
            stats["prob/" + it["qtype"]].append(d)
            worst.append((d, cid, it["qid"], same))
            if it["upstream_exact"] and not a.no_upstream:
                p_up = ref.upstream_probabilities(backend, state, it["qid"], questions[it["qid"]])
                up_n += 1
                up_agree += int(p_up.argmax() == p_ref.argmax() or np.abs(p_up - p_ref).max() < 1e-4)
                stats["upstream_prob"].append(float(np.abs(p_up - p_ref).max()))

    print("questions: %d   rows: %d   ort time: %.1fs (%s)" % (n_q, n_rows, t_ort, a.provider))
    print("tokenizer.json (Rust core) reproduces row ids: %d/%d" % (tok_ok, n_rows))
    print("argmax agreement  onnx vs %s reference: %.4f" % (a.precision, agree / n_q))
    if up_n:
        print("reference vs von evaluate(): %d questions, argmax agreement %.4f" % (up_n, up_agree / up_n))
    for key in sorted(stats):
        v = np.array(stats[key])
        print("  %-18s max %.2e   p99 %.2e   mean %.2e" % (key, v.max(), np.quantile(v, 0.99), v.mean()))
    print("worst probability deltas (onnx vs reference):")
    for d, cid, qid, same in sorted(worst, reverse=True)[:8]:
        print("  %.2e  %s  %s%s" % (d, cid, qid, "" if same else "  ARGMAX DIFFERS"))


if __name__ == "__main__":
    main()
