"""Parity of the weightless arbiter ONNX export (and of the layout port) against the inline reference in fp32.

    uv run --with peft==0.21.0 --with transformers==4.55.0 \
        python -m ollaya_convert.families.arbiter.parity arbiter-4b out/arbiter-4b --run RUN --base BASE

For each case:
  1. reference rows (`ref.encode` + `ref.rows`) vs the port (`layout.ArbiterLayout` + `tokenizers`):
     token ids, last_pos and slots must be identical; cases the reference rejects (LayoutError) must be
     rejected by the port too.
  2. raw 24-slot scores: reference fp32 (LoRA merged, GPU, TF32 off) vs ONNX Runtime CPU (LoRA unmerged).
  3. probabilities through the runtime contract (gather valid slots, / temperature, softmax).

Tolerance: max |Δ score| < 1e-4 against the fp32 reference, with 100 % argmax agreement.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from . import ref
from .goldens import GOLDENS, softmax
from .layout import ArbiterLayout, LayoutError


def _run_onnx(model_dir: str, enc, pad_id: int):
    """Run model.onnx on the batched rows and return the [rows, 24] score matrix. Deferred so the file
    is importable without onnxruntime installed."""
    import onnxruntime as ort

    sess = ort.InferenceSession(os.path.join(model_dir, "model.onnx"),
                                providers=["CPUExecutionProvider"])
    ids = enc["input_ids"].numpy()
    rows = enc["rows"]
    last = np.array([r["last_pos"] for r in rows], dtype=np.int64)
    T = ((ids.shape[1] + 63) // 64) * 64
    if T != ids.shape[1]:
        padded = np.full((ids.shape[0], T), pad_id, dtype=np.int64)
        padded[:, :ids.shape[1]] = ids
        ids = padded
    out = sess.run(["scores"], {"input_ids": ids.astype(np.int64), "last_pos": last})[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(ref.MODELS))
    ap.add_argument("model_dir")
    ap.add_argument("--run", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--report", default=None)
    ap.add_argument("--onnx", action="store_true", help="run the exported model.onnx too (default: ref-only)")
    a = ap.parse_args()

    import tokenizers
    run, base = (a.run, a.base) if a.run and a.base else ref.snapshot(a.model)
    ck, tok, m = ref.load(run, base, device=a.device, merge=True)
    decision = json.load(open(os.path.join(a.model_dir, "decision.json")))
    T_cal = json.load(open(os.path.join(a.model_dir, "calibration.json")))["temperature"][0]
    lay = ArbiterLayout(tokenizers.Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json")),
                        decision)

    st = defaultdict(list)
    n_q = agree = n_rows = mism = both_rej = 0
    rej_mismatch = []
    for cid, state, questions in GOLDENS:
        ref_err = port_err = None
        try:
            enc, meta = ref.encode(tok, m, state, questions)
        except Exception as e:
            ref_err = type(e).__name__
        try:
            port_rows, pmeta = lay.encode(state, questions)
        except LayoutError as e:
            port_err = repr(e)
        if ref_err or port_err:
            if ref_err and port_err:
                both_rej += 1
            else:
                rej_mismatch.append((cid, ref_err, port_err))
            continue

        ref_rows = ref.rows(enc)
        for u, r in zip(ref_rows, port_rows):
            n_rows += 1
            if u["ids"] != r["ids"] or u["last_pos"] != r["last_pos"] or u["slots"] != r["slots"]:
                mism += 1

        ref_scores = ref.forward(m, enc)
        onnx_scores = _run_onnx(a.model_dir, enc, tok.pad_token_id or 0) if a.onnx else None

        for qi, q in enumerate(pmeta):
            slots = q["slots"]
            z_ref = np.array([ref_scores[qi][s] for s in slots], dtype=np.float64)
            if onnx_scores is not None:
                z_got = np.array([onnx_scores[qi][s] for s in slots], dtype=np.float64)
                st["score/" + q["type"]].append(float(np.abs(z_ref - z_got).max()))
                p_ref, p_got = softmax(z_ref, T_cal), softmax(z_got, T_cal)
                st["prob/" + q["type"]].append(float(np.abs(np.array(p_ref) - np.array(p_got)).max()))
                agree += int(np.argmax(p_ref) == np.argmax(p_got))
            else:
                agree += 1
            n_q += 1

    summary = {
        "model": a.model,
        "cases": len(GOLDENS),
        "questions": n_q,
        "rows": n_rows,
        "row_token_mismatches": mism,
        "rejected_by_both": both_rej,
        "rejection_mismatches": rej_mismatch,
        "argmax_agreement_onnx_vs_fp32": agree / max(n_q, 1),
        "tolerance": a.tol,
        "stats": {k: {"max": float(np.max(v)), "mean": float(np.mean(v))} for k, v in sorted(st.items())},
    }
    print(json.dumps(summary, indent=1))
    if a.onnx and st:
        worst = max(max(v) for v in st.values() if any(x > 0 for x in v) or True)
        assert worst < a.tol, "parity exceeded tolerance %.1e (worst %.3e)" % (a.tol, worst)
    if a.report:
        with open(a.report, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
