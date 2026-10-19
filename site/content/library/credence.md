> Needs Ollaya 0.7.0 or newer for the text tags and 0.10.0 or newer for the vision tags (`ollaya --version`).

Credence v1 by [Txoka](https://huggingface.co/Txoka/Credence-v1-Gemma4-E4B)
refines EldanRing's Winnow-E4B using MiCA. Both variants run their Q8_0 GGUF files
through Ollaya's existing Winnow prompt and llama.cpp engine.

| Tag | Public accuracy | Public ECE ↓ | Typed accuracy | Typed ECE ↓ |
|---|---:|---:|---:|---:|
| Original `winnow:e4b` | 73.46% | 7.09% | 72.30% | 2.51% |
| `credence:e4b` | 74.12% | 5.48% | 72.35% | 4.83% |
| `credence:e4b-calibrated` | 73.72% | 3.74% | 72.20% | 4.63% |

Local measurements through Ollaya: 13 public subsets/3,880 rows and 400 typed
states/2,000 questions. Public ECE is pooled ten-bin ECE. These are single-run
results, not significance claims. Neither variant improves every metric over
original Winnow. Full probability metrics and per-subset results are retained in
the repository's `docs/measurements/credence/` directory. Over HTTP on an RTX 5090 it
answers five questions in {{result credence:e4b gpu.p50_ms 0}} ms (`winnow:e4b`: {{result winnow:e4b gpu.p50_ms 0}} ms).

```shell
ollaya run credence:e4b --preset triage "My order never arrived. Please refund it."
ollaya run credence:e4b-calibrated --preset triage "My order never arrived. Please refund it."
```

The accuracy tag was trained with 5% procedural synthetic questions; the calibrated
tag is a separate 0%-synthetic checkpoint. Both start from calibrated Winnow.
Their readout scaling is folded into the weights and softcap. The first uses
external temperature 1; the second uses validation-fitted 1.0408574437121012.
No temperature was fitted on benchmark data.

Weights are 8.72 GB each, downloaded from a commit-pinned public Hugging Face
repository and verified by SHA256. Text tags support up to 64 candidate options with the existing Winnow context
limits. Separate vision tags add the projector described below. Apache-2.0; preserve
Winnow/Gemma4 notices and individual training-data attribution.

Both published files passed clean, unauthenticated downloads and SHA256 verification.
Both packages match stock llama-server of the pinned build on the CPU, an RTX 4070 and an
RTX 4090: 505/505 decisions each. Exact evidence and
reproduction instructions are in the family documentation.

## Vision variants

`credence:e4b-vision` and `credence:e4b-calibrated-vision` are the same checkpoints with Winnow-E4B's unchanged
projector (990 MB), pulled from EldanRing's repository, so a store that already has `winnow:e4b-vision` keeps one
copy. Winnow kept its vision weights frozen and Credence trained on text only. Both tags match stock llama-server
on images, on the CPU and on an RTX 4090: 65/65 decisions each; no visual accuracy or image calibration is claimed.

```shell
ollaya run credence:e4b-vision --image shelf.png --questions '{"blocked":{"type":"noul","instructions":"Is the aisle blocked?"}}' "A photo from the warehouse camera."
```
