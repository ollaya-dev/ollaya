"""The PyTorch reference for Von: load `wfzyx/von` through `von-sdk` and encode requests the way it does.

    uv run --with von-sdk==1.1.1 python -m ollaya_convert.families.von.ref   # smoke test

Upstream (Apache-2.0):
  * von-sdk 1.1.1 on PyPI; its inference code is identical to github.com/wfzyx/von @ 657f42f for this
    checkpoint (HEAD only adds an optional `noul_zero_shot_prior`, which the checkpoint does not ship).
  * weights: huggingface.co/wfzyx/von @ d8bb5e0745d8 (`option_marker.pt` holds the full state dict:
    ModernBERT-large encoder + OptionMarkerScorer; `model.safetensors` only seeds the architecture).

Delegated to upstream code, so it cannot drift:
  * state rendering        von.backends.option_marker_backend._format_state
  * sequence text          von.models.option_marker.OptionMarkerModel.pack_sequence
  * the network            OptionMarkerModel.forward (encoder + gather + OptionMarkerScorer)
  * calibration map load   OptionMarkerBackend._get_model (reads marker_calibration.json)
Re-stated here, cited against option_marker_backend.py (von-sdk 1.1.1):
  * option descriptions per question type      evaluate_choice / evaluate_noul / evaluate_score
  * zero-shot noul debiasing                    evaluate_noul, `logits[0] - 0.7 * bias`
  * input-conditioned temperature               _effective_temperature

Ollaya deviations. Each one only touches inputs upstream rejects or mis-scores; on everything else the
encoding is byte-identical to upstream (checked by `parity.py` against `OptionMarkerBackend.evaluate`):
  1. A literal mask token ("[MASK]") in the state, instructions or option text is replaced by " ".
     Upstream takes *every* [MASK] id as an option marker, so such text shifts every option's score.
  2. Sequences longer than `MAX_LEN` tokens get their state truncated (see `fit_row`); upstream has no
     limit and would run a 20k-token sequence.
  3. Inputs von's pydantic schema rejects with a 422 are accepted: non-string instructions render as
     `json.dumps` (the Laya rule the shared Rust parser already applies), list-valued choice criteria
     become `{label: None}`, and non-string criterion values render with Python `str()`.
  4. noul criteria keys are matched case-insensitively ("True" == "true"; last spelling wins), as the
     shared Rust question parser does. Upstream reads only the exact keys "true" / "false".
"""
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO = "wfzyx/von"
REVISION = "d8bb5e0745d8ee1fb65d536d6d4892d54d5a93fd"
UPSTREAM = "von-sdk==1.1.1"
MAX_LEN = 8192  # von's advertised context (OptionMarkerModel sets max_position_embeddings=8192)

# evaluate_noul: defaults when the caller gives no description, and the context-free debias coefficient.
NOUL_DEFAULT_TRUE = "Yes, condition holds true."
NOUL_DEFAULT_FALSE = "No, condition is false."
NOUL_PRIOR_COEF = 0.7

QTYPES = {"choice": 0, "score": 1, "noul": 2}


def load(device: str = "cpu"):
    """OptionMarkerBackend with the model loaded as `von serve` loads it (fp32, eval), pinned.

    Upstream downloads the unpinned `main`, which moved to von 1.2 on 2026-09-24 (a different
    attention layout that von-sdk 1.1.1 does not implement). So the backend is pointed at the pinned
    snapshot through its own local-checkpoint path (`checkpoint_dir` with option_marker.pt,
    config/tokenizer files and marker_calibration.json): same code, same weights, no drift.
    """
    from huggingface_hub import snapshot_download
    from von.backends.option_marker_backend import OptionMarkerBackend

    snap = snapshot_download(REPO, revision=REVISION,
                             allow_patterns=["*.json", "model.safetensors", "option_marker.pt"])
    backend = OptionMarkerBackend(checkpoint_dir=snap, device=device)
    backend._get_model()
    assert backend._calib_map is not None, "marker_calibration.json not picked up"
    return backend


# ----------------------------------------------------------------------------------------- questions


def _text(v: Any) -> str:
    """A criterion value as text. Upstream only accepts strings; anything else uses Python `str()`."""
    return v if isinstance(v, str) else str(v)


def upstream_accepts(qdef: Dict[str, Any]) -> bool:
    """Would von's own schema (von.types) accept this question unchanged?"""
    from pydantic import ValidationError
    from von.types import Choice, Noul, Score

    cls = {"choice": Choice, "noul": Noul, "score": Score}.get(qdef.get("type"))
    if cls is None:
        return False
    try:
        cls(**qdef)
    except (ValidationError, TypeError):
        return False
    if qdef["type"] == "noul":
        # deviation 4: upstream reads only lower-case keys
        crit = qdef.get("criteria") or {}
        if any(k.lower() in ("true", "false") and k not in ("true", "false") for k in crit):
            return False
    return True


def question_parts(qdef: Dict[str, Any]) -> Dict[str, Any]:
    """Labels (answer keys), option descriptions (what the model reads) and the noul debias flag."""
    t = qdef["type"]
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins)  # deviation 3 (Laya's rule)
    crit = qdef.get("criteria")
    if t == "choice":
        if isinstance(crit, list):  # deviation 3
            crit = {c: None for c in crit}
        labels = list(crit.keys())
        # evaluate_choice: `descriptions.append(desc.strip() if desc else opt.strip())`
        descs = [_text(d).strip() if d else o.strip() for o, d in crit.items()]
        return {"type": t, "instructions": ins, "labels": labels, "descriptions": descs, "zero_shot": False}
    if t == "noul":
        crit = {str(k).lower(): v for k, v in (crit or {}).items()}  # deviation 4
        pos, neg = crit.get("true"), crit.get("false")
        # evaluate_noul: `has_explicit = bool(pos_desc or neg_desc)`, then defaults; order is [true, false]
        has_explicit = bool(pos or neg)
        pos = _text(pos) if pos else NOUL_DEFAULT_TRUE
        neg = _text(neg) if neg else NOUL_DEFAULT_FALSE
        return {"type": t, "instructions": ins, "labels": ["true", "false"], "descriptions": [pos, neg],
                "zero_shot": not has_explicit}
    if t == "score":
        descs = []
        for item in crit:
            # evaluate_score
            if isinstance(item, dict):
                what = item.get("what", "")
                examples = item.get("examples", [])
                ex_str = f" Examples: {', '.join(examples)}" if examples else ""
                desc = f"{what}{ex_str}".strip()
            else:
                desc = str(item).strip()
            descs.append(desc)
        return {"type": t, "instructions": ins, "labels": [str(i) for i in range(len(descs))],
                "descriptions": descs, "zero_shot": False}
    raise ValueError("unknown question type %r" % t)


# ------------------------------------------------------------------------------------------ sequences


class _Packer:
    """Just enough of OptionMarkerModel for its (unbound) `pack_sequence`."""

    def __init__(self, tok):
        self.tokenizer = tok


def pack(tok, state_text: str, instructions: str, descriptions: List[str]) -> str:
    from von.models.option_marker import OptionMarkerModel

    return OptionMarkerModel.pack_sequence(_Packer(tok), state_text, instructions, descriptions)


def sanitize(tok, text: str) -> str:
    """Deviation 1: no literal mask token may reach the tokenizer outside the option markers."""
    return text.replace(tok.mask_token, " ")


def encode_row(tok, state_text: str, instructions: str, descriptions: List[str]) -> List[int]:
    # option_marker_backend: `inputs = tok(packed_text, return_tensors="pt")` (special tokens on, no truncation)
    return tok(pack(tok, state_text, instructions, descriptions))["input_ids"]


def fit_row(tok, state_text: str, instructions: str, descriptions: List[str], max_len: int = MAX_LEN):
    """Deviation 2: encode; if longer than max_len, cut the state at a token boundary and re-encode.

    Returns (text, ids, markers, state_chars_kept). Deterministic and exact whenever no cut is needed.
    """
    text = pack(tok, state_text, instructions, descriptions)
    ids = encode_row(tok, state_text, instructions, descriptions)
    kept = len(state_text)
    if len(ids) > max_len:
        enc = tok(state_text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        n = len(enc["input_ids"])
        while len(ids) > max_len:
            if n <= 0:
                raise ValueError("instructions and options alone exceed max_len=%d tokens" % max_len)
            n -= len(ids) - max_len
            kept = offsets[n - 1][1] if n > 0 else 0
            text = pack(tok, state_text[:kept], instructions, descriptions)
            ids = encode_row(tok, state_text[:kept], instructions, descriptions)
    markers = [i for i, t in enumerate(ids) if t == tok.mask_token_id]
    if len(markers) != len(descriptions):
        raise AssertionError("marker count %d != options %d" % (len(markers), len(descriptions)))
    return text, ids, markers, kept


def encode(tok, state: Any, questions: Dict[str, Dict[str, Any]], max_len: int = MAX_LEN) -> Dict[str, Any]:
    """Every encoder row the request needs, grouped per question.

    Row 0 of a question is the real sequence. A noul question without criteria adds row 1: the same
    sequence with an empty state, used for von's context-free debiasing.
    """
    from von.backends.option_marker_backend import _format_state

    raw_state_text = _format_state(state)
    state_text = sanitize(tok, raw_state_text)
    # _effective_temperature: `len(tokenizer.encode(state_text, add_special_tokens=False))` on the raw text
    state_tokens = max(len(tok.encode(raw_state_text, add_special_tokens=False)), 1)
    items = []
    for qid, qdef in questions.items():
        p = question_parts(qdef)
        ins = sanitize(tok, p["instructions"])
        descs = [sanitize(tok, d) for d in p["descriptions"]]
        text, ids, markers, kept = fit_row(tok, state_text, ins, descs, max_len)
        rows = [{"text": text, "ids": ids, "markers": markers}]
        if p["zero_shot"]:
            ntext, nids, nmarkers, _ = fit_row(tok, "", ins, descs, max_len)
            rows.append({"text": ntext, "ids": nids, "markers": nmarkers})
        mask_text = any(tok.mask_token in s for s in [raw_state_text, p["instructions"]] + p["descriptions"])
        items.append({"qid": qid, "qtype": p["type"], "labels": p["labels"], "rows": rows,
                      "zero_shot": p["zero_shot"], "truncated": kept < len(state_text),
                      "upstream_exact": upstream_accepts(qdef) and not mask_text and kept == len(state_text)})
    return {"state_text": raw_state_text, "state_tokens": state_tokens, "items": items}


def collate(enc: Dict[str, Any], pad_id: int, min_markers: int = 1):
    """Contract M tensors for all rows of a request, in item order, row order."""
    rows = [(r, QTYPES[it["qtype"]]) for it in enc["items"] for r in it["rows"]]
    s = max(len(r["ids"]) for r, _ in rows)
    k = max(min_markers, max(len(r["markers"]) for r, _ in rows))
    n = len(rows)
    input_ids = np.full((n, s), pad_id, dtype=np.int64)
    attention_mask = np.zeros((n, s), dtype=np.int64)
    marker_pos = np.zeros((n, k), dtype=np.int64)
    marker_mask = np.zeros((n, k), dtype=bool)
    qtype = np.zeros((n,), dtype=np.int64)
    for i, (r, qt) in enumerate(rows):
        input_ids[i, : len(r["ids"])] = r["ids"]
        attention_mask[i, : len(r["ids"])] = 1
        marker_pos[i, : len(r["markers"])] = r["markers"]
        marker_mask[i, : len(r["markers"])] = True
        qtype[i] = qt
    return {"input_ids": input_ids, "attention_mask": attention_mask, "marker_pos": marker_pos,
            "marker_mask": marker_mask, "qtype": qtype}


# ------------------------------------------------------------------------------------------- network


@torch.no_grad()
def forward_rows(backend, enc: Dict[str, Any]) -> List[np.ndarray]:
    """Upstream forward, one unpadded row at a time (batch 1, as von runs it). fp32 row logits."""
    model = backend._get_model()
    out = []
    for it in enc["items"]:
        for r in it["rows"]:
            ids = torch.tensor([r["ids"]], device=backend.device)
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), mask_positions=[r["markers"]])[0]
            out.append(logits.float().cpu().numpy())
    return out


# ---------------------------------------------------------------------------------- runtime semantics


def option_logits(item: Dict[str, Any], row_logits: List[np.ndarray], prior: Optional[Dict[str, float]] = None):
    """Per-question option logits in Ollaya order: K for choice/score, [false, true] for noul."""
    if item["qtype"] != "noul":
        return np.asarray(row_logits[0], dtype=np.float64)
    l_true, l_false = (float(x) for x in row_logits[0])
    if item["zero_shot"]:
        n_true, n_false = (float(x) for x in row_logits[1])
        bias = n_true - n_false
        corr = prior["a"] * bias + prior["b"] if prior else NOUL_PRIOR_COEF * bias
        l_true -= corr
    return np.array([l_false, l_true], dtype=np.float64)


def temperature(calib: Optional[Dict[str, float]], default_temp: float, logits: np.ndarray, state_tokens: int) -> float:
    """_effective_temperature: T = clamp(bias + entropy*H + log_tokens*log10(tokens)/4 + n_options*K/8)."""
    if not calib:
        return default_temp
    z = logits - logits.max()
    p = np.exp(z) / np.exp(z).sum()
    n = max(p.size, 1)
    ent = float(-(p * np.log(np.clip(p, 1e-12, None))).sum() / np.log(n)) if n > 1 else 0.0
    feats = {"bias": 1.0, "entropy": ent, "log_tokens": np.log10(state_tokens) / 4.0, "n_options": logits.size / 8.0}
    raw = sum(calib.get(k, 0.0) * v for k, v in feats.items())
    return min(calib["hi"], max(calib["lo"], raw))


def probabilities(backend, item, row_logits, state_tokens: int) -> np.ndarray:
    """Calibrated probabilities in Ollaya option order, as von reports them (unrounded)."""
    lg = option_logits(item, row_logits, getattr(backend, "_noul_prior", None))
    t = temperature(backend._calib_map, backend._default_temp, lg, state_tokens)
    z = lg / max(t, 1e-4)
    z = z - z.max()
    return np.exp(z) / np.exp(z).sum()


def upstream_probabilities(backend, state: Any, qid: str, qdef: Dict[str, Any]) -> np.ndarray:
    """von's own answer for one question, as probabilities in Ollaya option order (rounded to 4 dp by von)."""
    ans = backend.evaluate(state, {qid: qdef}).answers[qid]
    if qdef["type"] == "noul":
        return np.array([1.0 - ans.noul, ans.noul])
    return np.array(list(ans.probabilities.values()))


def main():
    os.environ.setdefault("USE_TF", "0")
    backend = load("cpu")
    tok = backend._get_model().tokenizer
    state = {"subject": "Invoice #4411", "body": "We were billed twice for March, please refund."}
    qs = {"dept": {"type": "choice", "instructions": "Which team handles this?",
                   "criteria": {"billing": "payments", "technical": "bugs", "other": ""}},
          "refund": {"type": "noul", "instructions": "The user asks for a refund."}}
    enc = encode(tok, state, qs)
    rl = forward_rows(backend, enc)
    i = 0
    for it in enc["items"]:
        rows = rl[i:i + len(it["rows"])]
        i += len(it["rows"])
        print(it["qid"], probabilities(backend, it, rows, enc["state_tokens"]),
              upstream_probabilities(backend, state, it["qid"], qs[it["qid"]]))


if __name__ == "__main__":
    main()
