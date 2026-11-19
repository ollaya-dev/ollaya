"""Run token rows through an exported decoder graph with ONNX Runtime: sort by length, pad each batch to
a multiple of 64 (right padding), keep padded tokens per batch under a budget."""
from __future__ import annotations

import time
from typing import Callable, Dict, List

import numpy as np

SEQ_MULTIPLE = 64


def session(path: str, threads: int = 0, provider: str = "CPUExecutionProvider"):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(path, so, providers=[provider])


def run_rows(sess, rows: List[List[int]], extra: Callable[[List[int], int], Dict[str, np.ndarray]], pad_id: int,
             budget: int = 8192, width: int = 0):
    """rows: token id lists. extra(batch_row_indices, padded_len) -> the other graph inputs for that batch.
    Returns (outputs [len(rows), ...] in input order, seconds). A graph in the fork form (`prefix_ids`) runs the
    rows of the request in one run, split as the runner splits them (`qwen35.fork_rows`)."""
    if any(i.name == "prefix_ids" for i in sess.get_inputs()):
        from .qwen35 import fork_rows

        prefixes, branches, job = fork_rows(rows, pad_id)
        L = len(prefixes[0]) + len(branches[0])
        feed = {"prefix_ids": np.array(prefixes, dtype=np.int64), "branch_ids": np.array(branches, dtype=np.int64),
                "branch_job": np.array(job, dtype=np.int64), **extra(list(range(len(rows))), L)}
        t0 = time.perf_counter()
        res = sess.run(None, feed)[0]
        return list(res), time.perf_counter() - t0
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
    out = [None] * len(rows)
    t = 0.0
    i = 0
    while i < len(order):
        j = i + 1
        L = -(-len(rows[order[i]]) // SEQ_MULTIPLE) * SEQ_MULTIPLE
        while j < len(order):
            L2 = -(-len(rows[order[j]]) // SEQ_MULTIPLE) * SEQ_MULTIPLE
            if L2 * (j + 1 - i) > budget:
                break
            L = L2
            j += 1
        idx = order[i:j]
        ids = np.full((len(idx), L), pad_id, dtype=np.int64)
        for b, r in enumerate(idx):
            ids[b, :len(rows[r])] = rows[r]
        feed = {"input_ids": ids, **extra(idx, L)}
        t0 = time.perf_counter()
        res = sess.run(None, feed)[0]
        t += time.perf_counter() - t0
        for b, r in enumerate(idx):
            out[r] = res[b]
        i = j
    return out, t
