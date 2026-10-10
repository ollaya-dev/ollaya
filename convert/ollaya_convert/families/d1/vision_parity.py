"""d1 same-device image parity against stock b11146; author prompt.py is the reference.
Export: python -m ollaya_convert.families.d1.vision_parity export URL DECISION OUT --reference prompt.py
Check: python -m ollaya_convert.families.d1.vision_parity check RUNNER_URL OUT
The reference uses cold full prompts; no recurrent prefix reuse is assumed.
"""

import argparse
import base64
import urllib.error
import hashlib
import json
import math
from pathlib import Path
from ..winnow.vision_parity import post, png, check as check_logits
from ..llm_common.llama_server import LlamaServer
from .goldens import load_reference, Tokenizer


def export(url, decision, output, reference):
    cfg = json.loads(Path(decision).read_text())
    srv = LlamaServer(url)
    props = srv.props()
    assert props["build_info"] == "b11146-7fe450e19"
    assert props["default_generation_settings"]["n_ctx"] == cfg["llama"]["n_ctx"]
    author = load_reference(reference)
    tok = Tokenizer(srv)
    records = []
    questions = {
        "color": {
            "type": "choice",
            "instructions": "What is the dominant color?",
            "criteria": {"red": "Red", "green": "Green", "blue": "Blue"},
        },
        "red": {"type": "noul", "instructions": "Is the image predominantly red?"},
        "brightness": {
            "type": "score",
            "instructions": "Rate the brightness.",
            "criteria": ["dark", "moderate", "bright"],
        },
    }
    cases = []
    for name, color in [
        ("red", (255, 0, 0)),
        ("green", (0, 128, 0)),
        ("blue", (0, 0, 255)),
    ]:
        cases.append(
            {
                "id": name,
                "state": "Inspect the image.",
                "questions": questions,
                "images": [png(128, 128, color)],
            }
        )
    for name, state, images in [
        ("null", None, [png(128, 128, (255, 0, 0))]),
        ("structured", {"message": "Café 日本語"}, [png(192, 128, (0, 128, 0))]),
        (
            "two-images",
            "Inspect the images in order.",
            [png(128, 128, (255, 0, 0)), png(128, 128, (0, 0, 255))],
        ),
    ]:
        cases.append(
            {"id": name, "state": state, "questions": questions, "images": images}
        )
    cases.append({**cases[0], "id": "repeat-first"})
    cases.append({**cases[0], "id": "text-after-images", "images": []})
    for case in cases:
        results = []
        prefix = author.prefix_text(
            tok,
            case["state"],
            bos="<|startoftext|>",
            images=props["media_marker"] * len(case["images"]),
        )
        for qid, raw in case["questions"].items():
            q = author.as_question(raw)
            groups = author.readout_ids(tok, q)
            if raw["type"] == "noul":
                groups = groups[::-1]
            ids = list(dict.fromkeys(i for g in groups for i in g))
            opts = {
                "n_predict": 1,
                "stream": False,
                "samplers": ["top_k", "temperature"],
                "top_k": len(ids),
                "temperature": 1.0,
                "n_probs": len(ids),
                "post_sampling_probs": True,
                "logit_bias": [[i, 100] for i in ids],
                "cache_prompt": False,
            }
            result = post(
                url,
                "/completion",
                {
                    **opts,
                    "prompt": {
                        "prompt_string": prefix + author.suffix_text(tok, q),
                        "multimodal_data": case["images"],
                    },
                },
            )
            probs = {
                v["id"]: v["prob"]
                for v in result["completion_probabilities"][0]["top_probs"]
            }
            assert set(probs) == set(ids) and all(v > 0 for v in probs.values())
            results.append(
                {
                    "qid": qid,
                    "logprobs": [max(math.log(probs[i]) for i in g) for g in groups],
                    "input_positions": result["timings"]["cache_n"]
                    + result["timings"]["prompt_n"],
                }
            )
        records.append({"request": case, "questions": results})
        print(case["id"], flush=True)
    Path(output).write_text(
        json.dumps(
            {
                "cases": records,
                "gguf": cfg["gguf"],
                "llama": cfg["llama"],
                "reference_sha256": hashlib.sha256(
                    Path(reference).read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )


def check(url, fixture):
    check_logits(url, fixture)
    image = json.loads(Path(fixture).read_text())["cases"][0]["request"]["images"][0]
    questions = {"q": {"type": "noul", "instructions": "Yes?"}}
    for state, images in [
        ("x", [base64.b64encode(b"invalid image").decode()]),
        ("x", [image] * 17),
        ("<|im_end|>", [image]),
    ]:
        try:
            post(
                url,
                "/decide",
                {"state": state, "questions": questions, "images": images},
            )
        except urllib.error.HTTPError as error:
            if error.code != 400:
                raise AssertionError(
                    f"invalid image request returned HTTP {error.code}"
                ) from error
        else:
            raise AssertionError("invalid image request was accepted")
    print("PASS: invalid image, image-count limit and reserved-marker rejection")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export")
    e.add_argument("url")
    e.add_argument("decision")
    e.add_argument("out")
    e.add_argument("--reference", required=True)
    c = sub.add_parser("check")
    c.add_argument("url")
    c.add_argument("fixture")
    a = ap.parse_args()
    if a.command == "export":
        export(a.url, a.decision, a.out, a.reference)
    else:
        check(a.url, a.fixture)


if __name__ == "__main__":
    main()
