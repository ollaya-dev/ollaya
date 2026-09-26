"""Export a GGUF decision model for Ollaya's llama runner: the runtime `decision.json`, `calibration.json`
and goldens (prompt token ids and option logits through the fixed evaluation plan).

    uv run python -m ollaya_convert.families.llm_common.export_llama winnow \
        --server <cuda build>/llama-server --gguf .../Winnow-12B-Q8_0.gguf --slug 12b-q8_0 \
        --repo EldanRing/Winnow-12B --revision <sha> --file gguf/Winnow-12B-Q8_0.gguf [--device CUDA0|cpu]
    uv run python -m ollaya_convert.families.llm_common.export_llama llm-logits \
        --server ... --gguf .../gemma-4-12B-it-Q4_0.gguf --slug gemma-4-12b-it-q4_0 \
        --repo ggml-org/gemma-4-12B-it-GGUF --revision <sha> --file gemma-4-12B-it-Q4_0.gguf \
        --calibration out/llm-logits-gemma-4-12b-it-q4_0/calibration.json

Writes out/<layout>-<slug>/ (llm-logits-<slug> or winnow-<slug>):
  decision.json          layout, GGUF pin, template pieces / flags, label tables, llama-server settings
  calibration.json
  goldens-<device>.jsonl per case: state/questions (engine form), state_tokens/state_truncated, and per
                         question the prompt ids, split point P, candidates and option logits (wire order);
                         or the error class the runtime must answer with
  goldens-<device>.meta.json  the llama-server build, device and arguments

The reference is the family's Python port of the author's prompt (`winnow/ref.py`, `llm_logits/ref.py`)
on the pinned llama-server build Ollaya ships, started with the runtime's exact arguments
(`plan.server_args`) and driven through the runtime's fixed evaluation plan (`plan.py`). The Rust
runtime must reproduce the token ids exactly and the option logits on the same build and device
(`cargo run --release -p ollaya-server --example parity_llama`). Other devices replay these prompts:
`replay.py`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import urllib.request

from . import cases
from .llama_server import LlamaServer
from .plan import FixedPlan, server_args, split_point

LLAMA_BUILD = "b11146"
MAX_STATE_TOKENS = 6144
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "out")
# Extra decoder cases: a state longer than max_state_tokens (truncation) and two questions whose
# suffixes start alike (the plan's barrier).
EXTRA = [
    ("extra/truncated_state", " ".join("word%d" % (i % 997) for i in range(9000)),
     {"urgent": {"type": "noul", "instructions": "Is this urgent?"},
      "topic": {"type": "choice", "instructions": "Topic?", "criteria": {"billing": None, "other": None}}}),
    ("extra/twin_questions", {"message": "My card was charged twice for one order."},
     {"a": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
      "b": {"type": "noul", "instructions": "Is the customer asking for a refund of money?"},
      "c": {"type": "noul", "instructions": "Is the customer asking for a refund?"}}),
]


def engine_form(questions):
    """What the daemon hands the engine (`ollaya_api::decide::Question::to_engine`): missing instructions
    read as the question id; noul criteria keep only the exact keys "true" and "false"."""
    out = {}
    for qid, q in questions.items():
        e = {"type": q["type"], "instructions": q.get("instructions") if q.get("instructions") is not None else qid}
        crit = q.get("criteria")
        if (q["type"] == "score" and not isinstance(crit, list)) or \
                (q["type"] == "choice" and not isinstance(crit, (dict, list))):
            continue  # the API rejects these before any engine sees them
        if q["type"] == "noul":
            if isinstance(crit, dict):
                c = {k: crit[k] for k in ("true", "false") if k in crit}
                e["criteria"] = c
        elif crit is not None:
            e["criteria"] = crit
        out[qid] = e
    return out


def winnow_questions(questions):
    """Ollaya's one addition to winnow-v1's compile(): choice criteria given as a list of labels (an
    Ollaya API form) read as labels without descriptions, `dict.fromkeys(labels)`."""
    return {qid: ({**q, "criteria": dict.fromkeys(q["criteria"])}
                  if q["type"] == "choice" and isinstance(q.get("criteria"), list) else q)
            for qid, q in questions.items()}


def gguf_metadata(path, keys):
    """A few scalar GGUF metadata values (arrays are skipped)."""
    fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def read(f, t):
        n = struct.calcsize("<" + t)
        return struct.unpack("<" + t, f.read(n))[0]

    def rstr(f):
        return f.read(read(f, "Q")).decode("utf-8", "replace")

    def value(f, t):
        if t in fmt:
            return read(f, fmt[t])
        if t == 8:
            return rstr(f)
        et, n = read(f, "I"), read(f, "Q")
        return [value(f, et) for _ in range(n)]

    out = {}
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF", path
        read(f, "I")
        read(f, "Q")
        for _ in range(read(f, "Q")):
            k = rstr(f)
            v = value(f, read(f, "I"))
            if any(k == x or k.endswith("." + x) for x in keys):
                out[k] = v
    return out


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def post_to(url):
    def post(path, body):
        req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.loads(r.read())
    return post


class LlmLogits:
    def __init__(self, srv, a):
        from ..llm_logits.ref import LAYOUT, SYSTEM, LlmLogitsLayout, render_state
        self.render_state = render_state
        self.lay = LlmLogitsLayout(srv, assistant_prefix=a.assistant_prefix, max_state_tokens=MAX_STATE_TOKENS)
        self.srv = srv
        pre, mid, post = self.lay.template
        self.decision = {
            "family": "llm-logits", "layout": LAYOUT, "system": SYSTEM,
            "template": {"pre": pre, "mid": mid, "post": post},
            "chat_template_kwargs": self.lay.kwargs, "assistant_prefix": self.lay.prefix,
            "labels": self.lay.labels.to_json(), "add_bos": self.lay.add_bos,
            "max_state_tokens": MAX_STATE_TOKENS,
        }

    def encode(self, state, questions):
        """-> (state_tokens, truncated, [(qid, ids, P, candidates, wire_order)])"""
        text = self.render_state(state)
        n = len(self.srv.tokenize(text))
        items = self.lay.encode(state, questions)
        ref = self.lay.encode(state, {"ref": {"type": "noul", "instructions": "\u0001"}})[0]["ids"]
        rows = [(it["qid"], it["ids"], split_point(it["ids"], ref), it["label_ids"], it["wire_order"]) for it in items]
        return n, n > MAX_STATE_TOKENS, rows


class Winnow:
    def __init__(self, srv, a):
        from ..winnow.ref import LAYOUT, THOUGHT, labels_for
        from ..winnow import ref
        self.ref = ref
        self.srv = srv
        tok = lambda t: srv.tokenize(t, add_special=False, parse_special=False)
        self.labels = labels_for(tok, lambda i: srv.pieces([i])[0])
        template = srv.props().get("chat_template", "")
        self.template = template
        thought = not template or THOUGHT in template or "<|channel>thought\\n<channel|>" in template
        self.decision = {
            "family": "winnow", "layout": LAYOUT,
            "labels": {"strings": [s for s, _ in self.labels], "ids": [i for _, i in self.labels]},
            "thought": thought, "max_state_tokens": MAX_STATE_TOKENS,
            "upstream": {"inference": "https://github.com/EldanRing/winnow-inference",
                         "commit": a.upstream_commit, "source": "native/protocol.h"},
        }

    def encode(self, state, questions):
        """compile() plus Ollaya's state cut: safe(state) is cut to MAX_STATE_TOKENS tokens."""
        prefix, qs, _ = self.ref.compile_request({"state": state, "questions": winnow_questions(questions)},
                                                  self.labels, self.template)
        text = self.ref.safe(state)
        ids = self.srv.tokenize(text, add_special=False, parse_special=True)
        n = len(ids)
        if n > MAX_STATE_TOKENS:
            text = self.srv.detokenize(ids[:MAX_STATE_TOKENS])
            prefix = prefix.split("State:\n", 1)[0] + "State:\n" + text + "\n"
        pre = self.srv.tokenize(prefix, add_special=True, parse_special=True)
        rows = []
        for qid, kind, keys, suffix in qs:
            ids = pre + self.srv.tokenize(suffix, add_special=False, parse_special=True)
            cands = [i for _, i in self.labels[:len(keys)]]
            rows.append((qid, ids, len(pre), cands, list(range(len(keys)))))
        return n, n > MAX_STATE_TOKENS, rows


def winnow_error_class(state, questions, labels, template):
    """The runtime's error class for a request compile() rejects: the first failing question decides.
    More options than the label table holds is 422 TOO_MANY_OPTIONS in Ollaya, anything else 400."""
    from ..winnow.ref import WinnowError, compile_request

    for qid, q in questions.items():
        try:
            compile_request({"state": state, "questions": {qid: q}}, labels, template)
        except WinnowError as e:
            c = q.get("criteria") if isinstance(q, dict) else None
            k = len(dict.fromkeys(c)) if isinstance(c, (list, dict)) else 2
            return "too_many_options" if "2-64" in str(e) and k > len(labels) else "invalid"
    return "invalid"


def error_class(layout, lay, state, questions, e):
    if layout == "winnow":
        return winnow_error_class(state, winnow_questions(questions), lay.labels, lay.template)
    return "too_many_options" if "exceed this model's" in str(e) else "invalid"


def device_class(device):
    return "cuda" if device.startswith("CUDA") else "metal" if device.startswith("MTL") else "cpu"


def server_version(binary):
    import subprocess

    out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=60)
    return " ".join(l.strip() for l in (out.stdout + out.stderr).splitlines() if l.strip().startswith(("version", "built")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("layout", choices=["llm-logits", "winnow"])
    ap.add_argument("--server", required=True, help="the pinned llama-server build the runtime ships")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--calibration", default=None, help="llm-logits: fitted calibration.json (demo output)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="winnow: the author's fitted decision temperature (default 1.0, Winnow's own default)")
    ap.add_argument("--temperature-source", default="")
    ap.add_argument("--upstream-commit", default="6c2b3c04e248a319f2cb43832628eba03e55fe38",
                    help="winnow: the winnow-inference commit the model card pins")
    ap.add_argument("--n-ctx", type=int, default=8192)
    ap.add_argument("--assistant-prefix", default="")
    ap.add_argument("--td", type=int, default=40, help="typed-decisions rows in the goldens")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--device", default="CUDA0", help="llama.cpp device (CUDA0, MTL0), or cpu")
    a = ap.parse_args()

    out = os.path.abspath(os.path.join(OUT, "%s-%s" % (a.layout, a.slug)))
    os.makedirs(out, exist_ok=True)
    meta = gguf_metadata(a.gguf, ["general.architecture", "sliding_window", "file_type"])
    swa = any(k.endswith(".attention.sliding_window") for k in meta)
    dev = None if a.device == "cpu" else a.device
    argv = server_args(a.n_ctx, swa, dev)
    srv = LlamaServer.start(a.server, a.gguf, port=a.port, argv=argv,
                            log=os.path.join(out, "llama-server-%s.log" % device_class(a.device)), timeout=1800)
    try:
        props = srv.props()
        lay = (LlmLogits if a.layout == "llm-logits" else Winnow)(srv, a)
        sha = sha256_file(a.gguf)
        decision = {"engine": "llama", **lay.decision}
        decision["gguf"] = {
            "repo": a.repo, "revision": a.revision, "path": a.file, "sha256": sha,
            "size": os.path.getsize(a.gguf), "quantization": props.get("model_ftype", ""),
            "architecture": meta.get("general.architecture", ""),
            "url": "https://huggingface.co/%s/resolve/%s/%s" % (a.repo, a.revision, a.file),
        }
        decision["llama"] = {"n_ctx": a.n_ctx, "swa_full": swa, "plan": "prefix", "build": LLAMA_BUILD}
        with open(os.path.join(out, "decision.json"), "w") as f:
            json.dump(decision, f, indent=1, ensure_ascii=False)
        if a.calibration:
            cal = json.load(open(a.calibration))
            cal = {"temperature": cal["temperature"], "temperature_by_options": cal.get("temperature_by_options", {}),
                   "temperature_range": [0.2, 40.0], "source": cal.get("source", "")}
        elif a.temperature is not None:
            cal = {"temperature": [a.temperature] * 3, "temperature_by_options": {},
                   "source": a.temperature_source or "the author's fitted decision temperature"}
        else:
            cal = {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {},
                   "source": a.temperature_source or "uncalibrated (temperature 1.0)"}
        with open(os.path.join(out, "calibration.json"), "w") as f:
            json.dump(cal, f, indent=2)

        plan = FixedPlan(post_to(srv.url))
        n_rows = 0
        name = "goldens-%s" % device_class(a.device)
        with open(os.path.join(out, name + ".meta.json"), "w") as f:
            json.dump({"server": server_version(a.server), "device": a.device,
                       "args": argv, "gguf_sha256": sha, "reference": "ollaya_convert.families.%s.ref + "
                       "llm_common.plan (the fixed evaluation plan)" % a.layout.replace("-", "_")}, f, indent=1)
        with open(os.path.join(out, name + ".jsonl"), "w") as f:
            for cid, state, qs in cases.edge_cases() + cases.typed_decisions(a.td) + [
                    (c, cases.wire(s), cases.wire(q)) for c, s, q in EXTRA]:
                questions = engine_form(qs)
                rec = {"id": cid, "state": state, "questions": questions}
                try:
                    n, cut, rows = lay.encode(state, questions)
                except Exception as e:  # the runtime answers these with 400/422
                    rec["error"] = error_class(a.layout, lay, state, questions, e)
                    rec["message"] = str(e)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    continue
                rec.update({"state_tokens": n, "state_truncated": cut, "rows": []})
                for qid, ids, P, cands, wire in rows:
                    if len(cands) == 1:
                        z = [0.0]
                    else:
                        lp = plan.ask(ids, P, cands)
                        z = [lp[j] for j in wire]
                    rec["rows"].append({"qid": qid, "ids": ids, "P": P, "candidates": cands, "wire_order": wire,
                                        "option_logits": z})
                    n_rows += 1
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("%s/%s.jsonl: %d questions; plan steps %s" % (out, name, n_rows,
              {k: sum(1 for t in plan.trace if t["step"] == k) for k in ("prefix", "barrier", "question", "cold")}))
    finally:
        srv.stop()


if __name__ == "__main__":
    main()
