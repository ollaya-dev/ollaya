"""Check an ONNX export against the PyTorch reference on the full case set.

    uv run python -m ollaya_convert.parity en out/laya-en/model.onnx

Reference is the checkpoint in fp32 (TF32 off). The production path laya-app serves today --
bf16 autocast on GPU -- is measured against the same reference, so the numbers show how far the
export sits from fp32 relative to the drift users already live with.
"""
import argparse
import time
from collections import defaultdict

import numpy as np
import onnxruntime as ort
import torch
from laya.common import QTYPE_NAMES, temp_bucket

from . import cases, laya_ref
from .export import onnx_feed


def probs(agent, logits: np.ndarray, qtype: int, k: int) -> np.ndarray:
    t = agent.temperature_by_options.get(temp_bucket(qtype, k), agent.temperature[qtype])
    z = logits[:k] / t
    p = np.exp(z - z.max())
    return p / p.sum()


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def session(path: str, providers):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=providers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", choices=sorted(laya_ref.CHECKPOINTS))
    ap.add_argument("onnx")
    ap.add_argument("--root", default=laya_ref.DEFAULT_ROOT)
    ap.add_argument("--td-limit", type=int, default=0, help="typed-decisions rows (0 = all 400)")
    ap.add_argument("--provider", default="CPUExecutionProvider")
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    agent = laya_ref.load(a.checkpoint, root=a.root, device=dev)
    sess = session(a.onnx, [a.provider])

    stats = defaultdict(list)
    worst = []
    n_q = agree = agree_bf16 = 0
    t_ort = 0.0
    for cid, state, questions in cases.all_cases(a.td_limit):
        enc = laya_ref.encode(agent, state, questions)
        b = enc["batch"]
        ref, ref_act = laya_ref.forward(agent, b)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            bf, _ = laya_ref.forward(agent, b)
        feed = onnx_feed(b)
        t0 = time.perf_counter()
        got, got_act = sess.run(None, feed)
        t_ort += time.perf_counter() - t0

        for r, item in enumerate(enc["items"]):
            k, qt = len(item["markers"]), item["qtype"]
            p_ref, p_got, p_bf = (probs(agent, x[r], qt, k) for x in (ref, got, bf))
            d_logit = float(np.abs(ref[r, :k] - got[r, :k]).max())
            d_prob = float(np.abs(p_ref - p_got).max())
            same = int(p_ref.argmax() == p_got.argmax())
            n_q += 1
            agree += same
            agree_bf16 += int(p_ref.argmax() == p_bf.argmax())
            t = QTYPE_NAMES[qt]
            stats["logit/" + t].append(d_logit)
            stats["prob/" + t].append(d_prob)
            stats["act_logit"].append(float(np.abs(ref_act[r] - got_act[r]).max()))
            stats["act_prob"].append(abs(softmax(ref_act[r])[0] - softmax(got_act[r])[0]))
            worst.append((d_prob, cid, item["qid"], same))

    print("questions: %d   ort time: %.1fs" % (n_q, t_ort))
    print("argmax agreement  onnx vs fp32: %.4f   (bf16 autocast vs fp32: %.4f)"
          % (agree / n_q, agree_bf16 / n_q))
    for key in sorted(stats):
        v = np.array(stats[key])
        print("  %-14s max %.2e   p99 %.2e   mean %.2e" % (key, v.max(), np.quantile(v, 0.99), v.mean()))
    print("worst probability deltas:")
    for d, cid, qid, same in sorted(worst, reverse=True)[:8]:
        print("  %.2e  %s  %s%s" % (d, cid, qid, "" if same else "  ARGMAX DIFFERS"))


if __name__ == "__main__":
    main()
