"""PyTorch baseline for `crates/ollaya-runner/examples/bench.rs`, on the same requests.

    uv run python -m ollaya_convert.bench_torch en out/goldens-all/laya-en.jsonl [--fp32]

Default is what laya-app serves today: `Agent.system_one` on CUDA (bf16 autocast on Ampere+).
`--fp32` disables autocast and TF32 to match the fp32 ONNX graph's precision.
"""
import argparse
import json
import time

import numpy as np
import torch

from . import laya_ref


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure(agent, requests, label):
    for state, qs in requests[:20]:
        agent.system_one(state, qs)
    ms = []
    for _ in range(2):
        for state, qs in requests:
            sync()
            t = time.perf_counter()
            agent.system_one(state, qs)
            sync()
            ms.append((time.perf_counter() - t) * 1000)
    ms = np.array(ms)
    q = sum(len(qs) for _, qs in requests) / len(requests)
    print("%-22s n=%-4d avg %.1f q/request | p50 %7.1f ms  p95 %7.1f ms  p99 %7.1f ms"
          % (label, len(ms), q, *np.percentile(ms, [50, 95, 99])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", choices=sorted(laya_ref.CHECKPOINTS))
    ap.add_argument("goldens")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    a = ap.parse_args()

    agent = laya_ref.load(a.checkpoint, device=a.device)
    if a.fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        agent.dtype = torch.float32  # system_one autocasts to agent.dtype; fp32 autocast is a no-op
    full = []
    with open(a.goldens) as f:
        for line in f:
            rec = json.loads(line)
            full.append((rec["state"], rec["questions"]))
    single = [(s, dict(list(qs.items())[:1])) for s, qs in full]
    print("precision:", "fp32" if a.fp32 else "production (autocast %s)" % agent.dtype)
    measure(agent, single, "1 question")
    measure(agent, full, "full request")


if __name__ == "__main__":
    main()
