"""Export d1 metadata and same-device parity goldens using the author's pinned prompt.py.

Download prompt.py at NATIVE_REVISION, then run:
 python -m ollaya_convert.families.d1.goldens --url http://localhost:11455 \
   --reference /path/to/prompt.py --gguf /path/to/d1-3B-Q8_0.gguf
The stock server must use b11146 and the exact decision.json context and batch settings.
No weights are modified or rehosted. Temperatures are neutral, as the author ships them.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from ..llm_common.llama_server import LlamaServer

NATIVE_REVISION = "051bcc464b01b9f92942b364d9586b0ef5912432"
REFERENCE_SHA256 = "a20b5f52e41a6d29e694bb8c861a5a4803ce3eb69364edad79511ccf8edb5c31"


def load_reference(path):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError(
            "reference is not prompt.py from the pinned LiquidAI/d1-3B revision"
        )
    spec = importlib.util.spec_from_file_location("d1_author_prompt", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Tokenizer:
    bos_token = "<|startoftext|>"

    def __init__(self, server):
        self.server = server
        self.tokenizations = {}

    def encode(self, text, add_special_tokens=False):
        ids = self.server.tokenize(
            text, add_special=add_special_tokens, parse_special=False
        )
        self.tokenizations[text] = ids
        return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--out", default=str(Path(__file__).resolve().parents[3] / "out" / "d1-3b-q8_0")
    )
    args = parser.parse_args()
    srv = LlamaServer(args.url)
    props = srv.props()
    if (
        props["build_info"] != "b11146-7fe450e19"
        or props["default_generation_settings"]["n_ctx"] != 32768
    ):
        raise ValueError(
            "reference must use the pinned b11146 server and shipped 32768 context"
        )
    author = load_reference(args.reference)
    tok = Tokenizer(srv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    weights = Path(args.gguf)
    with weights.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    assert sha == "2f0942d5a5f64cf69c3356d2be439b71644b1f0eb3a580912a5a0179eeaba77d"
    revision = "bb1e436ea78eb96a3f1acb6da865f70c2fbeb563"
    d = {
        "engine": "llama",
        "layout": "d1-v1",
        "gguf": {
            "repo": "LiquidAI/d1-3B-GGUF",
            "revision": revision,
            "path": weights.name,
            "sha256": sha,
            "size": weights.stat().st_size,
            "quantization": "Q8_0",
            "architecture": "lfm2",
        },
        "llama": {"n_ctx": 32768, "swa_full": False, "plan": "cold"},
    }
    # Use the shipped context for exact validation.
    (out / "decision.json").write_text(json.dumps(d, indent=2) + "\n")
    (out / "calibration.json").write_text(
        json.dumps({"temperature": [1.0, 1.0, 1.0]}, indent=2) + "\n"
    )
    if not (out / "model.gguf").exists():
        (out / "model.gguf").symlink_to(weights)
    cases = []
    fixtures = []
    seen_questions = set()
    base = {
        "n": {"type": "noul", "instructions": "Is this positive?"},
        "c": {
            "type": "choice",
            "instructions": "Sentiment?",
            "criteria": {
                "positive": "positive sentiment",
                "negative": "negative sentiment",
                "neutral": None,
            },
        },
        "s": {
            "type": "score",
            "instructions": "Intensity?",
            "criteria": ["low", "medium", "high"],
        },
    }
    for i, state in enumerate(
        [
            "Great work!",
            {"message": "Café 😃", "count": 1.0},
            None,
            "",
            ["a", 1, True],
            "bad result",
        ]
    ):
        cases.append((f"state-{i}", state, base))
    for n in [1, 2, 26, 27, 40]:
        cases.append(
            (
                f"choice-{n}",
                "Pick",
                {
                    "q": {
                        "type": "choice",
                        "instructions": "Choose",
                        "criteria": {f"option_{i}": None for i in range(n)},
                    }
                },
            )
        )
    cases.append(
        (
            "native-letters",
            "abc",
            {
                "q": {
                    "type": "choice",
                    "instructions": "Choose",
                    "criteria": {"a": None, "b": None, "é": None},
                }
            },
        )
    )
    cases.append(
        (
            "noul-descriptions",
            "a",
            {
                "q": {
                    "type": "noul",
                    "instructions": "Accept?",
                    "criteria": {"true": "approved", "false": "rejected"},
                }
            },
        )
    )
    cases.append(
        (
            "noul-missing",
            "a",
            {
                "q": {
                    "type": "noul",
                    "instructions": "Accept?",
                    "criteria": {"true": "approved"},
                }
            },
        )
    )
    cases.append(
        (
            "twin",
            "a",
            {
                "a": base["n"],
                "b": {**base["n"], "instructions": "Is this negative?"},
                "c": base["n"],
            },
        )
    )
    with (out / f"goldens-{args.device}.jsonl").open("w") as f:
        for name, state, qs in cases:
            rows = []
            st = (
                author.state_block(state) + "\nQUESTION:\n" if state is not None else ""
            )
            for qid, raw in qs.items():
                q = author.as_question(raw)
                groups = author.readout_ids(tok, q)
                if raw["type"] == "noul":
                    groups = groups[::-1]
                prefix = author.prefix_text(tok, state, bos=tok.bos_token)
                suffix = author.suffix_text(tok, q)
                ids = srv.tokenize(prefix + suffix, parse_special=True)
                cands = [i for g in groups for i in g]
                unique = list(dict.fromkeys(cands))
                z, _ = srv.candidate_logprobs(ids, unique, cache_prompt=False)
                lookup = dict(zip(unique, map(float, z)))
                pooled = [max(lookup[i] for i in g) for g in groups]
                key = json.dumps(raw, ensure_ascii=False)
                if key not in seen_questions:
                    fixtures.append(
                        {
                            "qid": qid,
                            "question": raw,
                            "suffix": suffix,
                            "groups": groups,
                        }
                    )
                    seen_questions.add(key)
                rows.append(
                    {
                        "qid": qid,
                        "ids": ids,
                        "P": 0,
                        "candidates": cands,
                        "groups": groups,
                        "option_logits": pooled,
                    }
                )
            f.write(
                json.dumps(
                    {
                        "id": name,
                        "state": state,
                        "questions": qs,
                        "state_tokens": len(srv.tokenize(st)),
                        "state_truncated": False,
                        "rows": rows,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()
            print(name, flush=True)

    (out / "protocol-fixtures.json").write_text(
        json.dumps(
            {"tokenizations": tok.tokenizations, "fixtures": fixtures},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
