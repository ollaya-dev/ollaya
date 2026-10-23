"""Stage every model of the lineup as a Hugging Face repository directory, with its card filled in.

    cd convert && uv run python ../train/stage.py <name> <models-dir> <out-dir>

<models-dir> holds ckpt/<run> (train.py's checkpoints) and web/<run> (webgraph.py's onnx/model.onnx with the
tokenizer, decision and calibration files) for each run in LINEUP: tag -> run, its scores and parity. Repos are
ollaya-dev/<name>-<tag>; upload each with `huggingface-cli upload ollaya-dev/<name>-<tag> <out-dir>/<tag>`.
"""
import os
import subprocess
import sys

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# tag: (run, backbone size, ternary?, typed, public, public kind, parity, web graph MB)
LINEUP = {
    "400m": ("fp400-T", "400m", False, 0.606, 0.605, "zero-shot", "4.9e-6", None),
    "400m-ternary": ("t400-id-T3w", "400m", True, 0.548, 0.716, "in-domain", "8.2e-6", 147),
    "150m-ternary": ("t150-id-T3w-e2", "150m", True, 0.547, 0.699, "in-domain", "3.4e-6", 66),
    "68m-ternary": ("t68-id-T3w", "68m", True, 0.448, 0.664, "in-domain", "1.4e-6", 35),
}

REFERENCE = [
    ("`pplx-decider:flash` (the teacher, 4B decoder)", "bf16", "0.726", "0.824 zero-shot"),
    ("`decider:0.8b` (0.8B decoder)", "bf16", "0.46–0.51", "0.669 zero-shot"),
    ("`laya:en` (Laya's English model)", "fp32", "–", "0.532"),
]

TERNARY_VARIANT = (
    "Ternary: every weight of the encoder layers and the head is −1, 0 or +1 times a scale per output row, and the "
    "token embeddings have 4 bits, so in a browser tab the whole model is a {mb} MB download (`onnx/model.onnx` and "
    "the tokenizer); `model.safetensors` holds the same weights as float32, for `laya` and Ollaya. It trained "
    "**in-domain**: its data includes the *train* splits of ten of the public benchmark's datasets (see Results)."
)
FP_VARIANT = (
    "The most accurate size, for [Ollaya](https://ollaya.dev) on a CPU or GPU. It trained **zero-shot** with respect "
    "to both benchmarks below: neither its data nor its teacher's labels include any of their datasets."
)
TERNARY_RECIPE = (
    "Ternary weights: quantization-aware training from the fp model of the same size (absmean per row, as BitNet "
    "b1.58, with a straight-through estimator), distilled from it (logits, attention relations and every layer's "
    "hidden states), with the fp weights blended into the quantized ones over the first 30% of the steps."
)
WEB_PARITY = ", and so does the browser (checked against Ollaya's runtime on the same questions)"


def results(name, tag):
    rows = []
    for t, (_, _, tern, typed, public, kind, _, _) in LINEUP.items():
        label = f"`{name}:{t}`" + (" (this model)" if t == tag else "")
        cell = lambda s: f"**{s}**" if t == tag else s
        rows.append(f"| {cell(label)} | {'ternary' if tern else 'fp32'} | {cell(f'{typed:.3f}')} | "
                    f"{cell(f'{public:.3f} {kind}')} |")
    rows += [f"| {m} | {w} | {ty} | {pu} |" for m, w, ty, pu in REFERENCE]
    return "\n".join(rows)


def main(name, models, out):
    os.makedirs(out, exist_ok=True)
    for tag, (run, size, tern, _, _, _, parity, mb) in LINEUP.items():
        title = f"{name.capitalize()} {size}" + (" ternary" if tern else "")
        res = os.path.join(out, f".{tag}.results.md")
        with open(res, "w") as f:
            f.write(results(name, tag))
        args = [sys.executable, f"{R}/train/hub.py", f"{models}/ckpt/{run}", f"{out}/{tag}", "--card", f"{R}/train/card.md"]
        if tern:
            args += ["--web", f"{models}/web/{run}"]
        v = {
            "name": name, "tag": tag, "size": size, "title": title, "repo": f"ollaya-dev/{name}-{tag}",
            "tags": "- ternary\n- bitnet" if tern else "", "parity": parity,
            "variant": TERNARY_VARIANT.format(mb=mb) if tern else FP_VARIANT,
            "recipe": TERNARY_RECIPE if tern else "", "web_size": f"{mb} MB" if tern else "",
            "web_parity": WEB_PARITY if tern else "", "web": f"@{R}/train/card-web.md" if tern else "",
            "results": f"@{res}",
        }
        for k, val in v.items():
            args += ["--var", f"{k}={val}"]
        subprocess.run(args, check=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
