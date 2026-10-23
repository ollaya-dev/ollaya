# coba: distilled decision encoders, in full precision and ternary

Technical report, draft of 2026-10-24. Ollaya.

## Abstract

coba is Ollaya's own family of English decision models. A decision model reads a state (a message, a ticket, a
JSON object) and typed questions (yes/no, choice, score) and returns a probability for every option of every
question in one forward pass, without generating text. coba runs Laya's decision network, which Ollaya already ran,
on Ettin encoders (ModernBERT architecture). It learns from the full probability distributions of
`pplx-decider:flash`, a 4B decoder, mixed with gold labels, on permissively licensed data only. Five tags ship:
`coba:1b` and `coba:400m` in full precision, trained zero-shot with respect to every public task of Ollaya Bench, and
`coba:400m-ternary`, `coba:150m-ternary` and `coba:68m-ternary`, whose weights are ternary per output row and which
also trained on the train splits of ten public benchmark datasets (in-domain).[^1] On Ollaya Bench 2026.10,
`coba:1b` scores 36.33 and `coba:400m` 35.83, against 73.65 for the teacher.[^2] On the held-out
typed-decisions test split, accuracy is 0.608 for `coba:1b`, 0.591 for `coba:400m`, and 0.559, 0.535 and 0.4825 for
the three ternary tags.[^3] The ternary tags' browser graphs keep the weights at 2 bits and the token embeddings
at 4 bits, 32 MB to 144 MB per graph.[^4] Five questions take 14.7 ms to 23.6 ms (p50) on an RTX 5090.[^5]
Ollaya's runtime matches the PyTorch reference on CPU, CUDA and Metal with every decision the same and
probabilities within 1.4e-05.[^6]

## 1. Decision models and the coba tags

A decision request is a state plus questions of three types:[^7]

- `noul`: yes or no, answered as the probability of yes;
- `choice`: one of up to 255 options, each described in plain words;
- `score`: a level on a scale of 2 to 10 described levels.

The five tags, their encoders and their training runs are in Section 2.2.[^1] Ollaya also runs an fp16 graph of
each tag (`coba:1b-fp16` and so on), which the parity runs check against the same goldens with a looser gate
(Section 8.6).[^8]

## 2. Architecture

### 2.1 Laya's decision network

coba uses the network of `laya` 0.3.7 unchanged.[^9] One sequence of up to 512 tokens holds the question, a
`[MASK]` marker before every option, and the state. A two-layer transformer head on top of the encoder scores each
option at its marker, and one temperature per question type calibrates the result.[^9] Besides the encoder and the
head, the network has a question-type embedding and a scorer.[^10] A small act head, trained with weight 0.1,
predicts whether the top option matches the target's top option; the saved configuration sets `act_costs` to
`{"escalate": 0.5}`.[^11]

Ollaya already ran this network (`laya-markers-v1`, with parity on CPU, CUDA and Metal), so a checkpoint in Laya's
format needs no new runtime code, only a manifest.[^12] A checkpoint is `rl_agent_config.json`,
`model.safetensors`, `tokenizer/` and `encoder/config.json`; `laya.load` reads it and Ollaya's `convert/` exports it
like any other Laya model.[^13]

### 2.2 Ettin encoders

The encoders are Ettin (`jhu-clsp/ettin-encoder-*`, MIT), which use the ModernBERT architecture.[^14]

| Tag | Encoder | Parameters, with head | Weights | Training run | Public tasks |
|---|---|---:|---|---|---|
| `coba:1b` | `jhu-clsp/ettin-encoder-1b` | ~1.05B | fp32 | `fp1b-T` | zero-shot |
| `coba:400m` | `jhu-clsp/ettin-encoder-400m` | ~420M | fp32 | `fp400-T` | zero-shot |
| `coba:400m-ternary` | `jhu-clsp/ettin-encoder-400m` | ~420M | ternary | `t400-id-T3w-e2` | in-domain |
| `coba:150m-ternary` | `jhu-clsp/ettin-encoder-150m` | ~163M | ternary | `t150-id-T3w-e2` | in-domain |
| `coba:68m-ternary` | `jhu-clsp/ettin-encoder-68m` | ~74M | ternary | `t68-id-T3w-e2` | in-domain |

[^15][^1]

In `ettin-encoder-150m`, the token embeddings are 38.7M of the roughly 150M parameters, and the two-layer decision
head adds about 10M parameters at d = 768.[^16] The encoder's configuration marks which layers use global
(full) attention; the training script takes the last of them for relation distillation (Section 6.3).[^17]

| Encoder | Layers | Hidden size | Attention heads | Intermediate size |
|---|---:|---:|---:|---:|
| `ettin-encoder-68m` | 19 | 512 | 8 | 768 |
| `ettin-encoder-150m` | 22 | 768 | 12 | 1,152 |
| `ettin-encoder-400m` | 28 | 1,024 | 16 | 2,624 |
| `ettin-encoder-1b` | 28 | 1,792 | 28 | 3,840 |

Every size uses global attention in every third layer and local attention over a 128-token window in the others,
with a vocabulary of 50,368 tokens.[^105]

## 3. Data

### 3.1 Sources

The main set draws on five typed-decision datasets.[^18]

| Source | License as recorded | Sub-sources kept | Questions sampled |
|---|---|---:|---:|
| `tasksource/tasksource-jev-typed-decisions` | per sub-source | 246 of 751 | 316,218 |
| `tasksource/procedural-typed-decisions` (30,000 rows sampled) | apache-2.0 | whole | 50,163 |
| `n4ze3m/typed-decisions-synth` | mit | whole | 18,725 |
| `tasksource/synthetic-typed-decisions` | apache-2.0 | whole | 12,256 |
| `fastino/fast-decisions` | apache-2.0 | whole | 2,600 |
| Total | | | 399,962 |

[^19]

`train/data/build.py` applies these filters, and `train/data/licenses.tsv` records the decision for every
sub-source:

1. **Held out.** A sub-source whose name matches any of the eleven public benchmark datasets, or a derivative of
   one (QNLI is SQuAD, WANLI is seeded from MNLI, XNLI and PAWS-X are translations, every HelpSteer and Jigsaw set,
   every PubMed set), is dropped whole.[^20] In the jev set this removed 44 sub-sources, 38,718
   rows.[^21]
2. **License.** Every license term recorded for a sub-source must be permissive: Apache-2.0, MIT, BSD, CC BY,
   CC BY-SA, CC0, ODC-By, PDDL, AFL-3.0 or CDLA-Permissive. Non-commercial, custom and unspecified licenses
   fail.[^22] This dropped 301 jev sub-sources, 422,018 rows.[^21]
3. **English.** Sub-sources under `multilingual/` are dropped, and so is any row where under 85% of the letters are
   ASCII or where foreign function words (at least three) outnumber English ones.[^23] 160 jev sub-sources were
   dropped as non-English.[^21]
4. **Shape.** A choice needs 2 to 255 options and a score 2 to 10 levels.[^24]
5. **Length.** A row must fit 512 tokens of the `ettin-encoder-150m` tokenizer, counting every question header,
   every option, one `[MASK]` per option, `[CLS]`, `[SEP]` and the state.[^25]

Each source's license ships with the weights: every Hugging Face repository carries `DATA_SOURCES.tsv`, with the
kept sources, their licenses and the number of questions sampled from each (plus the in-domain rows for the ternary
tags).[^26]

### 3.2 Sampling

The main set is capped at 400,000 questions. Each of the four datasets sampled whole may fill at most 25% of the cap;
the jev sub-sources share the rest with one equal per-source cap, found by water-filling.[^27] The result is
399,962 questions.[^19] They sit in 343,955 rows (states): 287,281 choice, 69,630 noul and 43,051 score
questions.[^106] The model card and the library page describe the training data as "about 350,000" questions;[^28]
that figure matches the row count, not the question count.

### 3.3 Teacher labels

`train/data/label.py` sends each row's state and questions (at most 64 questions per request) to an Ollaya daemon's
`/v1/systemone` and stores the teacher's full probability distribution for every question, in option order. A row the
teacher rejects is stored with its error instead and gets no teacher label.[^29] The script's default model is
`pplx-decider:flash-fp16`.[^30]

The shipped labels come from `pplx-decider:flash-fp16`, the teacher's fp16 graph, served by Ollaya 0.16.0 on CUDA.
Three daemons on `hope-wsl`'s RTX 5090 labeled `main.jsonl` from its first row. One daemon on `white`'s RTX 3090
labeled the same rows in reverse order (`main.rev.jsonl`), so the two met in the middle. The in-domain rows were
labeled the same way.[^107] The teacher rejected no row: the label files hold no `error` record, and every row of
`main.jsonl` (343,955) and every in-domain id (178,116) has a teacher label.[^108] Some rows were labeled twice:
1,239 main rows, where the two machines met, and 3,455 lines of the in-domain label files repeat an id. `train.py`
keeps the first label it reads for an id, so `hope-wsl`'s files take precedence.[^108][^109]

### 3.4 What was held out

- **The public benchmark.** No row from its eleven datasets or their derivatives is in the main set (Section 3.1,
  filter 1).
- **typed-decisions.** `build.py` writes the train split of `LocalLLaMA/typed-decisions` to its own file,
  `td-train.jsonl`, and never into the main set; it was kept apart for an optional fine-tuned variant.[^31] The
  test split (400 states, 2,000 questions) is what Ollaya measures. The typed-decisions run records
  `trained_on_split: false` for all five tags, and no coba tag's edition `trained_on` list includes
  typed-decisions.[^32]

### 3.5 In-domain rows (ternary tags only)

`train/data/indomain.py` builds rows from the *train* splits of ten of the public benchmark's datasets, at most
20,000 per dataset, asked exactly as the benchmark asks them: Bespoke Labs' `nimble` source modules build every
record. For a large split it reads only the first parquet shard. It never reads a validation or test split. SummEval
has no train split, and MASSIVE de-DE is not English, so both stay out; MASSIVE contributes en-US only.[^33]
Any row that shares a normalized string of 30 or more characters with any benchmark record is dropped: 195 rows,
mostly BoolQ passages that recur across its splits.[^34]

| Dataset | License | Rows |
|---|---|---:|
| aegis2 | CC BY 4.0 | 20,000 |
| boolq | CC BY-SA 3.0 | 9,263 |
| civil_comments | CC0 1.0 | 20,000 |
| helpsteer2 | CC BY 4.0 | 20,000 |
| massive (en-US) | CC BY 4.0 | 11,512 |
| multinli | mixed per genre | 20,000 |
| paws | Google PAWS license | 20,000 |
| pubmedqa (artificial) | MIT | 20,000 |
| squad2 | CC BY-SA 4.0 | 20,000 |
| vitaminc | CC BY-SA 3.0 | 20,000 |
| Total | | 180,775 |

[^35]

Only the ternary tags trained with these rows; the fp tags' data and teacher labels include none of the benchmark's
datasets.[^36]

The in-domain rows carry teacher labels too: every ternary run reads `teacher-indomain.jsonl` and
`teacher-indomain-white.jsonl` beside the main label files.[^110] Each ternary tag starts from an fp run of its own
size and distills from it. That run trained on the same main and in-domain rows with the same labels: `fp400-id-T`,
`fp150-id-T` or `fp68-id-T`. It is not the zero-shot `fp400-T` that ships as `coba:400m`.[^110][^111]

`indomain.jsonl`'s 180,775 rows carry 178,116 distinct ids: 2,058 ids repeat, accounting for 2,659 extra rows. For
21 of those ids the repeated rows differ. Teacher labels are matched by id (Section 3.3), so one row of each such
pair may train on the other's teacher distribution.[^112]

### 3.6 Validation split

Accuracy during training (`val`) is measured on a held-out slice of the training data, and the calibration
temperatures are fitted on it (Section 4.4).[^37]

The validation file is `main-val.jsonl`: 3,475 of `main.jsonl`'s 343,955 rows (1.0%, 4,074 questions), spread over
the whole file and kept in its order. `main-train.jsonl` holds the other 340,480 rows (395,888 questions).[^112] How
the split was cut was not recorded. Every run, fp and ternary, validates on this same file, which has no in-domain
rows.[^113]

The in-domain training file `main-id-train.jsonl` is `main-train.jsonl` followed by `indomain.jsonl`, joined with
`cat`: 521,255 rows, 576,663 questions.[^112][^114] Teacher labels stay in their own files. `train.py` reads them
from `--teacher` and matches them to rows by id (Section 3.3).[^109]

## 4. Distillation from `pplx-decider:flash`

`pplx-decider:flash` is a 4B decoder under Apache-2.0, and it is first in Ollaya Bench 2026.10.[^38] The student
learns its full option distributions, mixed with the gold answer where the data has one.

### 4.1 Targets

For each question with options in a fixed order (a choice's criteria order, a score's levels, a yes/no as
[false, true]), `train.py` forms the target

    y = alpha * gold + (1 - alpha) * teacher      when the question has both,
    y = gold  or  y = teacher                     when it has one,

then normalizes y to sum to 1. alpha defaults to 0.5. A question with neither label is skipped.[^39] No
temperature is applied to the teacher's probabilities when the targets are built.[^39]

### 4.2 Loss

For a question with K options, student probabilities p and target y:

    L = -sum_i y_i log p_i  +  w_rps * [type = score] * RPS(p, y)
    RPS(p, y) = sum_i (P_i - Y_i)^2 / (K - 1),    P, Y the cumulative sums of p, y

averaged over the batch, with w_rps = 1.0 by default, plus 0.1 times the act head's cross-entropy.[^40]

### 4.3 Optimization

- A choice's options (and its target) are permuted in half of the training examples, so position never decides.[^41]
- Batches group states of similar length, in random order.[^42]
- AdamW with betas (0.9, 0.98) and weight decay 0.01; learning rates 5e-5 for the encoder and 2e-4 for the head;
  linear warm-up over 3% of the steps, then cosine decay to 0; batch size 32; gradient clipping at 1.0; bf16 autocast;
  512 tokens. These are the script's defaults. Gradient checkpointing is available for 400m and larger on 24 GB.[^43]
- The fp runs train for one epoch: a second epoch on the 400m raised `val` by 0.015 but lowered typed-decisions by
  0.029 and left the public score flat.[^44]

The two shipped fp runs used these defaults, and their saved configurations record them:

- one epoch at batch size 32, 12,372 steps;
- learning rates 5e-5 (encoder) and 2e-4 (head);
- weight decay 0.01 and alpha 0.5;
- gradient checkpointing on.

`fp1b-T` trained on `hope-wsl`'s RTX 5090 in 2.61 hours. `fp400-T` trained on `choso-wsl`'s RTX 4090 in 1.56
hours.[^115][^116]

### 4.4 Calibration temperatures

After training, `train.py` fits one temperature per question type (choice, score, noul) on the validation split. It
searches 0.5 to 5.0 in steps of 0.01 for the value that minimizes the negative log-likelihood of the label (the gold
argmax, or the target's argmax when there is no gold). laya clamps the temperatures to [0.5, 5]. They are saved in
`rl_agent_config.json` as `temperature`.[^45]

| Tag | choice | score | noul |
|---|---:|---:|---:|
| `coba:1b` | 0.89 | 0.80 | 0.69 |
| `coba:400m` | 0.91 | 0.91 | 0.55 |
| `coba:400m-ternary` | 1.03 | 0.81 | 0.76 |
| `coba:150m-ternary` | 1.02 | 0.90 | 0.86 |
| `coba:68m-ternary` | 0.96 | 0.83 | 0.79 |

[^117]

### 4.5 Development measurements

During development, `train/eval.sh` scored checkpoints with the PyTorch reference served by `train/serve.py`.[^37]
These numbers are not the release measurement, and they differ from it: Section 8 uses Ollaya's runtime over HTTP,
which rejects a state it would have to truncate where the reference truncates it.[^46]

| Run | Weights | Data | val | typed | public |
|---|---|---|---:|---:|---:|
| fp150-gold | fp | gold only | 0.749 | 0.527 | 0.564 zero-shot |
| fp150-T | fp | gold + teacher | 0.765 | 0.541 | 0.594 zero-shot |
| fp150-id-T | fp | + in-domain | 0.765 | 0.544 | 0.705 in-domain |
| t150-T3w | ternary | gold + teacher | 0.718 | 0.502 | 0.478 zero-shot |
| t150-id-T3w | ternary | + in-domain | 0.736 | 0.531 | 0.688 in-domain |
| t150-id-T3w-e2 | ternary | + in-domain, 2 epochs | 0.764 | 0.547 | 0.699 in-domain |

[^37]

At 150m, the teacher's soft labels add 0.016 `val` and 0.03 zero-shot public over gold labels alone. From 400m
to 1b, `val` gains 0.005, typed 0.015 and zero-shot public 0.019.[^47] `t150-id-T3w-e2` is the shipped
`coba:150m-ternary`; Ollaya measures it at 0.535 typed and 0.6222 public (in-domain).[^3][^48]

## 5. Zero-shot and in-domain tags

The edition's score is defined as follows:[^49]

> Mean over 14 tasks (13 public subsets and typed-decisions, equal weight) of the chance-corrected accuracy
> 100 * (accuracy - chance) / (1 - chance); chance is the mean of 1/K over the task's questions. A model trained on a
> task's train split is scored on the other tasks

Each result in the edition carries a `trained_on` list:[^50]

- `coba:1b` and `coba:400m`: empty. Both are scored on all 14 tasks, zero-shot.
- `coba:400m-ternary`, `coba:150m-ternary` and `coba:68m-ternary`: `vitaminc-dev`, `massive-en-US`,
  `massive-de-DE`, `boolq`, `squad2`, `paws`, `multinli`, `civil_comments`, `aegis2`, `helpsteer2`, `pubmedqa`.
  Their edition score is therefore the mean of the three remaining tasks: `summeval-relevance`,
  `summeval-consistency` and `typed-decisions`.

Recomputing the mean of those three tasks from the edition's task values gives 4.607, 4.387 and 1.637, which matches
the published 4.6, 4.39 and 1.63 to within the rounding of the task values.[^51] The same check over all 14
tasks reproduces the published scores of `coba:1b` and `coba:400m`.[^51]

Two details:

- `indomain.py` trains on MASSIVE en-US only, but the edition also lists `massive-de-DE` as trained, so it leaves that
  task out of the ternary tags' score as well.[^33][^50]
- The edition's top-level `trained_on` map lists only `laya:typed-decisions` and the `jeb` tags; the coba entries are
  in each result's own `trained_on` list.[^52]

Rule for this report and anything quoting it: every public benchmark number of a ternary tag is labeled *in-domain*,
and its edition score is the three-task score above, which is not comparable with a 14-task score.

## 6. Ternary quantization-aware training (T3w)

The ternary runs are named `T3w` (`t400-id-T3w-e2` and so on). The name appears in the run names and in
`docs/browser-model.md`, which describes recipe T3w as quantization-aware training from the fp model, logit,
relation and every-layer hidden-state distillation from it, a quantization warm-up over 30% of the steps, and 4-bit
embeddings.[^53] `train/ternary.py` does not use the name. The model card states the quantizer as absmean per
row.[^54] In the pilot matrix of `ternary-research.md`, T3 is the variant with a learned scale.[^55]

The shipped runs used absmean, not the learned scale. Every ternary run passes `--quant absmean`, and the published
configurations record `training.quant: absmean`.[^110] The run names follow the development pilots, not the research
matrix:

- `T0`: quantization-aware training alone;
- `T1`: adds logit distillation;
- `T2`: logit and relation distillation;
- `T2w`: `T2` plus a quantization warm-up;
- `T3w`: `T2w` plus every-layer hidden-state distillation.

The `w` stands for the warm-up.[^118]

### 6.1 The absmean quantizer

`ternary.py` implements BitNet b1.58's absmean ternarization, one scale per output row:[^56]

```python
def absmean_ternary(w: torch.Tensor) -> torch.Tensor:
    g = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
    return (w / g).round().clamp(-1, 1) * g
```

For a weight matrix W with N output rows and K inputs: g_i = max(1e-8, (1/K) sum_j |W_ij|) and
Q_ij = g_i * clamp(round(W_ij / g_i), -1, 1), so every row takes the values {-g_i, 0, +g_i}. During training the
forward pass sees W + mix * (Q(W) - W) with the difference detached, a straight-through estimator: the gradient
reaches the latent float weight.[^57] `--quant learned` selects ParetoQ's form with a learnable per-row scale
instead, initialized where it equals absmean.[^58]

### 6.2 What is quantized

- **Ternary:** every `nn.Linear` weight and every `nn.MultiheadAttention` input projection inside the encoder layers
  and inside the decision head.[^59]
- **4 bits:** the token embeddings, per row: s = max|row| / 7, codes clamp(round(w / s), -7, 7) (the general form is
  s = max|row| / (2^(b-1) - 1)).[^60] The recipe uses 4 bits, with 8 as the fallback.[^16]
- **Full precision:** the norms, the question-type embedding, the scorer and the act head.[^10][^119]

### 6.3 Distillation from the fp model

A ternary run starts from an fp checkpoint of the same backbone (`--init`) and distills from it (`--kd-from`), which
is loaded frozen beside the student.[^61] The `pplx-decider` distributions stay in the target, so the student has
two teachers.[^62] The loss is Section 4.2's loss on the mixed targets plus three terms:[^63]

- **Logits:** tau^2 * KL(teacher || student) over a question's options, both softened by tau; `--kd-tau` defaults to
  2.0.[^64]
- **Attention relations** (MiniLM-v2, as in BitDistill): for each of Q, K and V of the last global-attention layer,
  the KL between the teacher's and the student's softmax(X X^T / sqrt(d_head)), per head, over the real tokens,
  averaged over heads and over the three relations. It uses the first 16 rows of each batch (`--rel-rows`), for
  memory.[^65]
- **Hidden states** (as in TernaryBERT): the MSE between the layer-normalized outputs of every encoder layer of
  student and teacher, over the real tokens, averaged over layers.[^66]

The total is L + kd * L_kd + rel * L_rel + hid * L_hid. The script's usage example passes `--kd 1 --rel 1 --wd 0`.[^67]

The three shipped ternary runs use:

- loss weights kd 1, rel 10 and hid 1, with `--kd-tau` at its default 2.0;
- learning rates 5e-5 (encoder) and 1e-4 (head), weight decay 0;
- 4-bit embeddings (`--emb-bits 4`);
- batch size 32 and two epochs, 36,042 steps.

`t400-id-T3w-e2` starts from and distills from `fp400-id-T`, `t150-id-T3w-e2` from `fp150-id-T`, and
`t68-id-T3w-e2` from `fp68-id-T`.[^110]

### 6.4 Schedule

- **Quantization warm-up:** the forward pass sees w + mix * (q(w) - w) with mix = min(1, step / (quant_warmup *
  total)), so it starts at the fp weights and is fully quantized after the warm-up. T3w uses 30% of the steps.[^68]
- **Two epochs:** unlike the fp runs, the ternary runs gained from a second epoch. At 150m it added 0.028 `val`, 0.016
  typed and 0.011 public (in-domain); at 68m 0.029, 0.045 and 0.013 (public in-domain).[^69] All three shipped ternary
  tags are two-epoch runs (`-e2`).[^1]
- The research notes planned a learning-rate grid of {2e-5, 5e-5, 1e-4}, cosine to 0, and weight decay 0, following
  ParetoQ.[^70]
- The share of zero weights is logged at every evaluation, since a runaway value signals a collapsing scale.[^71]

In the finished checkpoints, 31.38% of `coba:400m-ternary`'s ternary weights are zero, 31.39% of
`coba:150m-ternary`'s and 32.94% of `coba:68m-ternary`'s. Per tensor, the share runs from 27% to 28% (the decision
head's second feed-forward layer) up to 35% to 39% (the last encoder layer's QKV projection).[^119]

During development (Section 4.5's reference measurements), T3w kept 96% of the fp model's `val` and 98% of its typed
and in-domain public scores at 150m, and 96% to 98% at 400m.[^72]

### 6.5 Finalizing

After the last step, every quantized weight is replaced by its exact quantized value (with -0.0 turned into +0.0)
and the parametrizations are removed; the temperatures are then fitted on the finalized model.[^73] The
checkpoint stores these exact values as float32, so Ollaya runs them as ordinary float32 weights and gets the same
answers.[^73][^74] In Ollaya's registry the ternary 400m is therefore the same size as the fp 400m: 1617 MB
each.[^75]

## 7. The browser graph

### 7.1 Conversion

`train/webgraph.py` turns the ONNX export of a ternary checkpoint (`ollaya_convert.export`'s `model.onnx`) into the
browser's graph:[^76]

- Every `MatMul` whose weight is an initializer used once, two-dimensional float32, and ternary per output row
  becomes `com.microsoft` `MatMulNBits` with `bits=2`. The block size is the largest of 256, 128, 64, 32 and 16 that
  divides K. The ternary codes {-1, 0, +1} are stored as 1, 2, 3 (zero point 2), four per byte with the first code in
  the lowest bits, and the row's g is every block's scale. The script asserts that the codes times the scales give
  back the float32 weights exactly.[^76]
- Every `Gather` on a two-dimensional float32 initializer along axis 0, whose rows are exactly int4 times a per-row
  scale (the token embeddings), becomes `GatherBlockQuantized` with `bits=4`, zero point 8, `gather_axis=0`,
  `quantize_axis=1`, and the same block-size rule over the hidden size.[^77]
- Everything else stays float32. The output is one self-contained file.[^76]

Both operators dequantize to exactly the float32 weights, so the graph computes what the float32 graph does up to the
order of the sums. The self-test checks a converted MatMul and Gather against the float32 graph with rtol 1e-5 and
atol 1e-4.[^78] Unpacking the weights to float32 instead cost 2.5 GB of resident memory for the 150m,
because onnxruntime-web holds several copies while it loads.[^79]

### 7.2 Size and speed

| Ternary tag | 2-bit graph | Download (graph and tokenizer) | Resident memory | Per request, p50 |
|---|---:|---:|---:|---:|
| `coba:68m-ternary` | 32 MB | 35 MB | 0.55 GB | 0.33 s |
| `coba:150m-ternary` | 63 MB | 66 MB | 0.68 GB | 0.91 s |
| `coba:400m-ternary` | 144 MB | 147 MB | 1.03 GB | 2.44 s |

[^4][^80]

Graph, memory and speed were measured on Node 26 with 4 WASM threads on an M4 Pro, over 83 requests of about five
questions with states up to 512 tokens.[^4] For the 150m, the 2-bit graph against the float32 graph and
weights: 63 MB (one file, plus a 3.6 MB tokenizer) against 57 MB packed, 0.5 GB resident against 2.5 GB, 0.3 s load
against 1.2 s, and 1.9 s against 1.6 s for five requests.[^79]

The graph sizes are those of the shipped files. In the three Hugging Face repositories, `onnx/model.onnx` is
31.6 MB, 63.0 MB and 143.8 MB, and the tokenizer is 3.6 MB.[^120] Which 400m run the memory and speed were measured
on was not recorded. At 400m, `browser-model.md`'s browser parity list names only the one-epoch `t400-id-T3w`. The two
runs have the same graph shapes.[^84]

### 7.3 The /try runtime

`crates/ollaya-web` compiles `ollaya-decision` (question rules, the Laya layout, calibration, answers) and the
`tokenizers` crate to WebAssembly. `web/` runs the graph with onnxruntime-web (WASM, and WebGPU when available) and
exposes `decide({state, questions})`, the same request and answer as `/api/decide`.[^81] A page loads a model
straight from its Hugging Face repository with `load(fromHub('<org>/<repo>'))` (`npm install @ollaya/web`); the model
downloads once and runs on the device.[^82] The `/try` page loads `ollaya-dev/coba-150m-ternary`, pinned to
commit `3609cf41de228721ac41b0bae70a6c26e7c19f17`, so the browser caches its files and a returning visitor loads
without downloading.[^83]

Browser parity was checked with `web/parity.mjs` against Ollaya's native runtime, on the 391 of 483 parity questions
the API accepts: every decision the same and probabilities within 2e-6, for the 2-bit graphs of `t150-id-T3w-e2`,
`t68-id-T3w-e2` and `t400-id-T3w`, among others.[^84]

Browser parity for `t400-id-T3w-e2`, the run that ships as `coba:400m-ternary`, was not recorded. The release parity
in Section 8.6 covers only its native runtime.

## 8. Results

The accuracy and latency numbers below were measured through Ollaya over HTTP on the `rtx-5090` machine on
2026-10-23; parity is Section 8.6. The run files record Ollaya 0.16.0; the edition lists these runs under release
0.17.0.[^85]

### 8.1 Ollaya Bench 2026.10

The edition scores 46 models. Position is the place in the edition's results list, which is in descending score
order; rank is the leaderboard's: 1 plus the number of models whose 95% interval lies entirely above the model's, so
models whose intervals overlap share a rank.[^2][^104]

| Position | Rank | Model | Score | 95% interval | Tasks scored |
|---:|---:|---|---:|---|---:|
| 1 | 1 | `pplx-decider:flash` (teacher) | 73.65 | 71.96 to 75.21 | 14 |
| 23 | 22 | `decider:0.8b` | 47.82 | 45.72 to 49.98 | 14 |
| 27 | 24 | `laya:multilingual` | 36.74 | 34.66 to 38.94 | 14 |
| 28 | 25 | `coba:1b` (zero-shot) | 36.33 | 34.25 to 38.64 | 14 |
| 29 | 25 | `coba:400m` (zero-shot) | 35.83 | 33.71 to 37.93 | 14 |
| 30 | 30 | `decima:base` | 31.04 | 28.88 to 33.22 | 14 |
| 32 | 30 | `laya:en` | 30.12 | 28.16 to 31.99 | 14 |
| 44 | 42 | `coba:150m-ternary` (in-domain, 3 tasks) | 4.6 | 1.69 to 7.89 | 3 |
| 45 | 42 | `coba:400m-ternary` (in-domain, 3 tasks) | 4.39 | 1.35 to 7.52 | 3 |
| 46 | 44 | `coba:68m-ternary` (in-domain, 3 tasks) | 1.63 | -1.32 to 4.86 | 3 |

[^2]

The teacher leads `coba:1b` by 37.32 points and `coba:400m` by 37.82.[^86] The intervals of `coba:1b`, `coba:400m`
and `laya:multilingual` overlap. The ternary tags' positions are not comparable with the others: their scores average
three tasks, two of which (SummEval) are below chance for every coba tag (Section 8.2).

On the typed-decisions task alone, which no coba tag trained on, the 41 models of the edition not trained on
typed-decisions, ordered by that task's score, place `coba:1b` 19th (42.56), `coba:400m` 21st (40.07), `coba:400m-ternary` 22nd (35.38),
`coba:150m-ternary` 24th (31.87) and `coba:68m-ternary` 29th (24.18); the teacher is first (59.85).[^87]

### 8.2 Per task

Chance-corrected accuracy per task (0 is chance, 100 is every answer right). The last column says what each row is
for the three ternary tags: *in-domain* rows are tasks whose train split they trained on, which the edition leaves out
of their score.

| Task | teacher | `coba:1b` | `coba:400m` | `400m-ternary` | `150m-ternary` | `68m-ternary` | Ternary tags |
|---|---:|---:|---:|---:|---:|---:|---|
| vitaminc-dev | 86.23 | 62.69 | 58.68 | 84.22 | 74.21 | 69.45 | in-domain |
| massive-en-US | 90.92 | 76.1 | 78.82 | 89.41 | 87.29 | 88.81 | in-domain |
| massive-de-DE | 91.23 | 48.27 | 54.02 | 38.29 | 30.42 | 27.09 | in-domain (edition) |
| boolq | 76.0 | 51.33 | 49.33 | 62.0 | 46.0 | 39.33 | in-domain |
| squad2 | 85.28 | 33.78 | 37.12 | 67.89 | 54.52 | 43.81 | in-domain |
| paws | 87.2 | 20.0 | 24.8 | 87.2 | 85.6 | 78.4 | in-domain |
| multinli | 87.96 | 61.37 | 64.38 | 85.45 | 73.91 | 72.91 | in-domain |
| civil_comments | 85.33 | 76.67 | 76.67 | 85.33 | 83.33 | 85.33 | in-domain |
| aegis2 | 68.8 | 19.2 | 8.8 | 66.4 | 65.6 | 53.6 | in-domain |
| helpsteer2 | 23.19 | -6.93 | -7.93 | -4.92 | -4.42 | -4.42 | in-domain |
| pubmedqa | 66.4 | 43.0 | 39.4 | 48.4 | 42.4 | 41.2 | in-domain |
| summeval-relevance | 42.71 | -16.15 | -14.06 | -14.58 | -15.62 | -15.1 | held out |
| summeval-consistency | 80.03 | -3.3 | -8.51 | -7.64 | -2.43 | -4.17 | held out |
| typed-decisions | 59.85 | 42.56 | 40.07 | 35.38 | 31.87 | 24.18 | held out |

[^88]

`massive-de-DE` is marked in-domain by the edition, although the in-domain rows hold MASSIVE en-US only (Section 5).

### 8.3 Typed decisions

The typed-decisions test split: 400 JSON states from four workflows, five typed questions each (2,000 questions: 600
choice, 600 noul, 800 score). Accuracy is the argmax against the gold label over all 2,000 questions, with a 95%
bootstrap interval over states; ECE is the ten-bin expected calibration error of the top probability.[^89]
No tag trained on this split.[^32]

| Tag | Accuracy | 95% interval | choice | noul | score | ECE | Brier |
|---|---:|---|---:|---:|---:|---:|---:|
| `coba:1b` | 0.608 | 0.581 to 0.6355 | 0.6217 | 0.6367 | 0.5763 | 0.0719 | 0.5191 |
| `coba:400m` | 0.591 | 0.5635 to 0.6195 | 0.5933 | 0.6933 | 0.5125 | 0.0939 | 0.5436 |
| `coba:400m-ternary` | 0.559 | 0.532 to 0.586 | 0.5983 | 0.5683 | 0.5225 | 0.1175 | 0.5683 |
| `coba:150m-ternary` | 0.535 | 0.5075 to 0.5645 | 0.4517 | 0.6633 | 0.5012 | 0.1137 | 0.5951 |
| `coba:68m-ternary` | 0.4825 | 0.46 to 0.505 | 0.42 | 0.65 | 0.4037 | 0.092 | 0.6195 |

[^3]

Every tag has 7 errors; a rejected request counts as wrong on its five questions.[^89]

### 8.4 Public benchmark

Bespoke Labs' public decision benchmark: 13 subsets of 11 human-labeled public datasets, 3,880 questions, scored with
Bespoke's own runner. Macro accuracy is the mean of the 13 subsets; pooled ECE is over all 3,880 questions.[^90]
The ternary tags' numbers are in-domain.

| Tag | Macro accuracy | 95% interval | choice | noul | score | Pooled ECE | Macro ECE |
|---|---|---|---:|---:|---:|---:|---:|
| `coba:1b` | 0.561 zero-shot | 0.5476 to 0.5743 | 0.6799 | 0.701 | 0.1297 | 0.0903 | 0.1594 |
| `coba:400m` | 0.5606 zero-shot | 0.5465 to 0.5745 | 0.6898 | 0.6967 | 0.1187 | 0.0892 | 0.1476 |
| `coba:400m-ternary` | 0.6537 in-domain | 0.6422 to 0.6647 | 0.7542 in-domain | 0.8688 in-domain | 0.1276 in-domain | 0.0466 in-domain | 0.0847 in-domain |
| `coba:150m-ternary` | 0.6222 in-domain | 0.6098 to 0.6345 | 0.6986 in-domain | 0.835 in-domain | 0.1401 in-domain | 0.0526 in-domain | 0.0956 in-domain |
| `coba:68m-ternary` | 0.6033 in-domain | 0.5901 to 0.6166 | 0.6859 in-domain | 0.8005 in-domain | 0.1368 in-domain | 0.0765 in-domain | 0.1109 in-domain |

[^48]

Every tag has the same 453 errors out of 3,880 requests, and a rejected request counts as wrong. They fall in
`summeval-relevance` (170 of 240), `helpsteer2` (144 of 249), `summeval-consistency` (101 of 144), `squad2` (19),
`pubmedqa` (11) and `aegis2` (8).[^91] `/v1/systemone`, which the benchmark runner calls, rejects a state
it would have to truncate to fit the model's context with `422 STATE_TRUNCATED`.[^92]

Every tag's 7 typed-decisions errors are `STATE_TRUNCATED`: the raw rows record each error's response
body.[^121] For the public benchmark's 453, the runner records only `TypeSafe HTTP 422`, so their error code was not
recorded. They fall in the same subsets with the same counts for all five tags. `laya:en` also fails 453 public
requests and 7 typed ones, and all 7 typed failures are `STATE_TRUNCATED`.[^91][^121][^122]

### 8.5 Latency

Five questions per request (the `triage` preset, 30 states), one request at a time over HTTP to `/api/decide`, timed
by the client; 5 warm-up and 20 timed requests on CUDA, 2 and 10 on the CPU. The machine is `rtx-5090`: an NVIDIA
GeForce RTX 5090 (32607 MiB, driver 596.21) and an AMD Ryzen Threadripper 3960X 24-Core Processor (48 threads),
under Linux 6.6.87.2-microsoft-standard-WSL2. CUDA runs the F16 graph, the CPU the F32 graph.[^5]

| Tag | GPU p50 | GPU p90 | CPU p50 | CPU p90 | Size | VRAM | CPU RAM peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| `coba:1b` | 20.9 ms | 25.8 ms | 1529.8 ms | 1575.5 ms | 4239 MB | 3703 MB | 4392 MB |
| `coba:400m` | 23.6 ms | 24.6 ms | 918.7 ms | 929.6 ms | 1617 MB | 1653 MB | 1890 MB |
| `coba:400m-ternary` | 20.0 ms | 23.7 ms | 886.6 ms | 890.0 ms | 1617 MB | 1653 MB | 1973 MB |
| `coba:150m-ternary` | 16.7 ms | 19.2 ms | 495.6 ms | 505.6 ms | 634 MB | 1139 MB | 822 MB |
| `coba:68m-ternary` | 14.7 ms | 17.2 ms | 323.0 ms | 326.6 ms | 293 MB | 885 MB | 389 MB |

[^5]

In Ollaya the ternary weights run as float32 (Section 6.5), so the ternary 400m has the fp 400m's size and about its
CPU time.

### 8.6 Parity

Ollaya's runtime against goldens from the authors' code (laya 0.3.7 in PyTorch fp32 on the CPU, TF32 off): 477 cases,
2,383 questions per tag. The fp32 gate requires identical encodings and decisions, option logits within 1e-3, and wire
answers as the authors' code. The fp16 gate requires identical encodings and at least 99.5% of decisions the
same.[^93] CPU and CUDA ran on `rtx-5090`, Metal on `mac-m4-pro` through MLX.[^94]

| Tag | CPU | CUDA | Metal (MLX) | CUDA fp16 | fp16 decisions same |
|---|---:|---:|---:|---:|---:|
| `coba:1b` | 3.8e-06 | 3.1e-06 | 4.9e-06 | 0.012 | 2379 / 2383 |
| `coba:400m` | 1.1e-05 | 1.1e-05 | 1.4e-05 | 0.015 | 2379 / 2383 |
| `coba:400m-ternary` | 2.7e-06 | 1.6e-06 | 3.4e-06 | 0.0069 | 2374 / 2383 |
| `coba:150m-ternary` | 2.4e-06 | 1.7e-06 | 3.6e-06 | 0.015 | 2377 / 2383 |
| `coba:68m-ternary` | 1.8e-06 | 1.8e-06 | 2.8e-06 | 0.011 | 2376 / 2383 |

[^6]

Values are the maximum absolute probability difference against the goldens (`max_prob_diff`). Every fp32 run has
2,383 of 2,383 decisions the same and zero encoding mismatches, and every run, fp32 and fp16, passed its gate. The
gate's 1e-3 tolerance is on option logits, which the run files do not report; they report probabilities.[^6]

## 9. Limitations

- **English only.** The data was filtered to English; other languages are not refused but are not
  measured.[^95] The zero-shot tags score 48.27 and 54.02 on `massive-de-DE` against 76.1 and 78.82 on
  `massive-en-US`.[^88]
- **512 tokens** for the state and the questions together: messages, tickets and small JSON states, not long
  documents or agent traces.[^96] `/v1/systemone` rejects a longer state with `STATE_TRUNCATED`, and
  `/api/decide` answers from its first tokens.[^96] On the public benchmark every tag errors on the same 453 of
  3,880 requests, most of them in SummEval and HelpSteer2 (Section 8.4).
- **Score questions** are the weakest kind for every model of this size.[^97] Public-benchmark accuracy on
  score questions is 0.1187 to 0.1401 for every tag (in-domain for the ternary tags), and every coba tag is below
  chance on `helpsteer2` (in-domain for the ternary tags), `summeval-relevance` and
  `summeval-consistency`.[^48][^88]
- **Calibration** holds on average, not per domain; check it on your own data before acting on the probabilities
  automatically.[^98]
- **The ternary tags are in-domain.** Their public-benchmark numbers are not held-out measurements, and their edition
  score covers three tasks. No zero-shot ternary tag ships; the zero-shot ternary 150m development run scored 0.478 on
  the public benchmark against 0.594 for the zero-shot fp 150m, `fp150-T` (reference measurements,
  Section 4.5).[^37]
- **fp against ternary is confounded.** The fp tags trained zero-shot and the ternary tags in-domain, so the gap
  between `coba:400m` and `coba:400m-ternary` on typed decisions (0.591 against 0.559) mixes the cost of ternary
  weights with the effect of the in-domain rows.[^3][^36]
- **Ternary saves bytes only in the browser.** In Ollaya the ternary weights are stored and run as float32 (Section
  8.5). In a browser, the ternary 400m takes 2.44 s per request on an M4 Pro (Section 7.2).
- **Small test set.** typed-decisions has 400 states; the intervals of `coba:1b` and `coba:400m` overlap
  (0.581 to 0.6355 against 0.5635 to 0.6195).[^3]

## 10. Reproduction

Data, labels and training are build-time Python in `train/`; nothing at runtime depends on it. Released numbers come
from Ollaya, not from `train/eval.sh`.[^99]

```sh
# 1. The license-clean main set, td-train.jsonl and licenses.tsv
uv run --with datasets --with tokenizers python train/data/build.py --out ~/ollaya-train/data

# 2. In-domain rows (ternary tags only), from the public benchmark's train splits
NIMBLE_SRC=/path/to/nimble python indomain.py --bench <public-dir> --cache <dir> --out indomain.jsonl

# 3. Teacher labels through an Ollaya daemon (resumable)
python3 train/data/label.py --in data/main.jsonl --out data/teacher.jsonl \
    --url http://127.0.0.1:11500 --model pplx-decider:flash-fp16

# 4. The fp student, then a ternary student from it
uv run train.py --backbone jhu-clsp/ettin-encoder-150m --data data/train.jsonl \
    --teacher data/teacher.jsonl --val data/val.jsonl --out runs/fp150
uv run train.py ... --init runs/fp150 --quant absmean --kd-from runs/fp150 --kd 1 --rel 1 --wd 0 --out runs/t150
```

[^100][^101]

The shipped runs' command lines come from the scripts that started them. None was rebuilt from a configuration, and
their flags agree with each run's saved `training` block.[^123]

```sh
# The in-domain training file; main-val.jsonl was cut from main.jsonl beforehand (how was not recorded)
cat data/main-train.jsonl data/indomain.jsonl > data/main-id-train.jsonl
M=data/teacher-main.jsonl,data/teacher-main-white.jsonl
I=$M,data/teacher-indomain.jsonl,data/teacher-indomain-white.jsonl

# coba:1b and coba:400m (zero-shot); <s> is 1b or 400m
train.py --backbone jhu-clsp/ettin-encoder-<s> --data data/main-train.jsonl --teacher $M --val data/main-val.jsonl \
    --out runs/fp<s>-T --bs 32 --epochs 1 --eval-every 2000 --workers 8 --grad-ckpt --name fp<s>-T

# The ternary tags' fp starting points; <s> is 400m, 150m or 68m (--grad-ckpt on the 400m only)
train.py --backbone jhu-clsp/ettin-encoder-<s> --data data/main-id-train.jsonl --teacher $I --val data/main-val.jsonl \
    --out runs/fp<s>-id-T --bs 32 --epochs 1 --eval-every 2000 --workers 8 [--grad-ckpt] --name fp<s>-id-T

# coba:<s>-ternary
train.py --backbone jhu-clsp/ettin-encoder-<s> --data data/main-id-train.jsonl --teacher $I --val data/main-val.jsonl \
    --init runs/fp<s>-id-T --quant absmean --emb-bits 4 --wd 0 --bs 32 --epochs 2 --lr-enc 5e-5 --lr-head 1e-4 \
    --eval-every 4000 --workers 8 --kd-from runs/fp<s>-id-T --kd 1 --rel 10 --quant-warmup 0.3 --hid 1 \
    --grad-ckpt --out runs/t<s>-id-T3w-e2 --name t<s>-id-T3w-e2
```

The machines ran the same `train.py` and `ternary.py` as this repository. The only differences are one docstring
line and `ternary.py`'s packing commands, which have since been removed and are not used in training.[^124]

```sh
# 5. Self-tests of the quantizer and the browser graph
python ternary.py
python webgraph.py

# 6. Development scores on both held-out suites (PyTorch reference via serve.py)
train/eval.sh <checkpoint> <bench-dir> [port]

# 7. The browser's 2-bit graph from ollaya_convert.export's output
python webgraph.py <export dir> <out.onnx>

# 8. Stage the Hugging Face repositories with their cards, then upload each
cd convert && uv run python ../train/stage.py <name> <models-dir> <out-dir> [tag ...]
huggingface-cli upload ollaya-dev/<name>-<tag> <out-dir>/<tag>
```

[^102]

The release measurements follow `docs/model-lifecycle.md`: export, goldens, `parity_run` on the CPU, CUDA and Metal,
package, then `bench_plan` and `bench_run` on the RTX 5090.[^103]

## Notes

Notes that cite a *training record* point to files on the three training machines (`~/ollaya-train/` and
`~/ollaya-own-work/` on `hope-wsl`, `choso-wsl` and `white`), which are not in this repository. Each shipped
checkpoint's `rl_agent_config.json` is also published in its Hugging Face repository.

[^1]: [`results/editions/2026.10.json`](../results/editions/2026.10.json) `models` (lines 69-73) and `updates[2].models`; run names and zero-shot or in-domain kind from [`train/stage.py`](../train/stage.py) lines 20-26 (`LINEUP`); `-e2` is the two-epoch suffix, [`browser-model.md`](browser-model.md) lines 109, 112.
[^2]: [`results/editions/2026.10.json`](../results/editions/2026.10.json) `score.results[i].model`, `.score`, `.score_ci`, `.trained_on`; rank is i + 1 (the list is in descending score order, checked). Teacher `score.results[?model=="pplx-decider:flash"]` (line 80), `coba:1b` (line 763), `coba:400m` (line 788), `coba:150m-ternary` (line 1165), `coba:400m-ternary` (line 1202), `coba:68m-ternary` (line 1239). Scores as published, two decimals.
[^3]: [`results/runs/2026-10-23-typed-decisions-rtx5090-cuda.json`](../results/runs/2026-10-23-typed-decisions-rtx5090-cuda.json) `results[?model=="<tag>"]`: `accuracy`, `accuracy_ci`, `by_type.choice`, `by_type.noul`, `by_type.score`, `ece`, `brier`, `errors`.
[^4]: [`browser-model.md`](browser-model.md) lines 158-165 (graph, resident memory, per-request p50).
[^5]: [`results/runs/2026-10-23-latency-rtx5090-cuda.json`](../results/runs/2026-10-23-latency-rtx5090-cuda.json) and [`results/runs/2026-10-23-latency-rtx5090-cpu.json`](../results/runs/2026-10-23-latency-rtx5090-cpu.json): `results[?model=="<tag>"].p50_ms`, `.p90_ms`, `.size_mb`, `.precision`; `.vram_mb` from the CUDA file, `.ram_peak_mb` from the CPU file; `machine_detail`; `protocol.preset`, `.questions`, `.states`, `.warm`, `.n`, `.client`.
[^6]: [`results/runs/2026-10-23-parity-rtx5090-cpu-coba.json`](../results/runs/2026-10-23-parity-rtx5090-cpu-coba.json), [`results/runs/2026-10-23-parity-rtx5090-cuda-coba.json`](../results/runs/2026-10-23-parity-rtx5090-cuda-coba.json), [`results/runs/2026-10-23-parity-macm4pro-metal-coba.json`](../results/runs/2026-10-23-parity-macm4pro-metal-coba.json): `results[?model=="<tag>"].max_prob_diff`, `.decisions_same`, `.questions`, `.encoding_mismatches`, `.pass`; fp16 rows are `results[?model=="<tag>-fp16"]` in the CUDA file. The largest fp32 value is `coba:400m` on Metal, 1.4e-05.
[^7]: [`train/card.md`](../train/card.md) lines 19-25.
[^8]: [`results/runs/2026-10-23-parity-rtx5090-cuda-coba.json`](../results/runs/2026-10-23-parity-rtx5090-cuda-coba.json) `results[*].model` and `.gate` (lines 28-41, 56-69, 84-97, 112-125, 140-153).
[^9]: [`train/card.md`](../train/card.md) lines 74-76; [`train/pyproject.toml`](../train/pyproject.toml) line 10 (the `laya` 0.3.7 pin); [`train/train.py`](../train/train.py) lines 407-416 (`head_layers: 2`, `max_len`, `temperature`).
[^10]: [`train/train.py`](../train/train.py) lines 22-23; [`ternary-research.md`](ternary-research.md) lines 133-134.
[^11]: [`train/train.py`](../train/train.py) lines 131-133 (act target and weight 0.1), line 412 (`act_costs`).
[^12]: [`browser-model.md`](browser-model.md) lines 29-32.
[^13]: [`train/train.py`](../train/train.py) lines 13-14.
[^14]: [`browser-model.md`](browser-model.md) lines 36-43; [`train/card.md`](../train/card.md) lines 5, 75-76; [`train/hub.py`](../train/hub.py) lines 18-35 (Ettin's MIT notice).
[^15]: [`browser-model.md`](browser-model.md) lines 38-43, "Params (with head)", approximate. The library page, [`site/content/library/coba.md`](../site/content/library/coba.md) lines 7-11, gives 1.1B, 421M, 421M, 164M and 75M.
[^16]: [`ternary-research.md`](ternary-research.md) lines 131-134.
[^17]: [`train/train.py`](../train/train.py) lines 178-183, 318-321.
[^18]: [`train/data/build.py`](../train/data/build.py) lines 317-323; [`browser-model.md`](browser-model.md) lines 53-55.
[^19]: [`train/data/licenses.tsv`](../train/data/licenses.tsv): `sampled_questions` (column 9) of lines 753-756 for the four whole datasets; the jev row is the sum of `sampled_questions` over the 751 `jev/*` lines and the count of `decision == keep` among them; the total is the sum over all 755 data lines (sums computed for this report). The 30,000-row sample: [`train/data/build.py`](../train/data/build.py) line 322.
[^20]: [`train/data/build.py`](../train/data/build.py) lines 28-35, 138-140.
[^21]: [`train/data/licenses.tsv`](../train/data/licenses.tsv), computed over the `jev/*` lines: `decision` counts (246 keep, 301 drop, 44 holdout, 160 non-english); sums of `holdout` (38,718) and `license_dropped` (422,018).
[^22]: [`train/data/build.py`](../train/data/build.py) lines 36-41, 76-78, 144-146, 291-293 (self-test cases); [`train/card.md`](../train/card.md) lines 77-79.
[^23]: [`train/data/build.py`](../train/data/build.py) lines 44-73, 141-143.
[^24]: [`train/data/build.py`](../train/data/build.py) lines 107-113.
[^25]: [`train/data/build.py`](../train/data/build.py) lines 23, 247-275, 313, 336-340.
[^26]: [`train/hub.py`](../train/hub.py) lines 73-79.
[^27]: [`train/data/build.py`](../train/data/build.py) lines 24-26, 278-287, 343-359.
[^28]: [`train/card.md`](../train/card.md) line 77; [`site/content/library/coba.md`](../site/content/library/coba.md) line 1.
[^29]: [`train/data/label.py`](../train/data/label.py) lines 1-8, 20, 55-63, 126-131; [`train/train.py`](../train/train.py) lines 57-58.
[^30]: [`train/data/label.py`](../train/data/label.py) line 77; [`browser-model.md`](browser-model.md) line 59; [`train/card.md`](../train/card.md) lines 80-81.
[^31]: [`train/data/build.py`](../train/data/build.py) lines 365-369; [`browser-model.md`](browser-model.md) lines 56-58.
[^32]: [`results/runs/2026-10-23-typed-decisions-rtx5090-cuda.json`](../results/runs/2026-10-23-typed-decisions-rtx5090-cuda.json) `results[*].trained_on_split` and `protocol.benchmark`; [`results/editions/2026.10.json`](../results/editions/2026.10.json) `score.results[?model=="coba:*"].trained_on`.
[^33]: [`train/data/indomain.py`](../train/data/indomain.py) lines 1-8, 22, 26-39, 50-55, 110-113.
[^34]: [`train/data/indomain.py`](../train/data/indomain.py) lines 58-66, 96-102, 131-134; [`browser-model.md`](browser-model.md) line 64.
[^35]: [`train/hub.py`](../train/hub.py) lines 83-94 (`INDOMAIN`); the multinli license there reads "Mixed per genre: OANC public domain; fiction and other sources CC BY-SA 3.0 or similar". The total is computed; [`train/stage.py`](../train/stage.py) line 86 rounds it to 180,000.
[^36]: [`train/stage.py`](../train/stage.py) lines 20-26 (public kind), 35-45; [`train/card.md`](../train/card.md) lines 63-68.
[^37]: [`browser-model.md`](browser-model.md) lines 89-94 (protocol) and 96-112 (table rows fp150-gold, fp150-T, fp150-id-T, t150-T3w, t150-id-T3w, t150-id-T3w-e2).
[^38]: [`train/stage.py`](../train/stage.py) line 30 (4B decoder); [`train/card.md`](../train/card.md) line 81 (Apache-2.0); [`results/editions/2026.10.json`](../results/editions/2026.10.json) `score.results[0].model`.
[^39]: [`train/train.py`](../train/train.py) lines 10-11, 46-82 (mix at line 78, normalization at lines 79-80), 281.
[^40]: [`train/train.py`](../train/train.py) lines 121-133, 287.
[^41]: [`train/train.py`](../train/train.py) lines 96-99.
[^42]: [`train/train.py`](../train/train.py) lines 108-118.
[^43]: [`train/train.py`](../train/train.py) lines 278, 283-289, 292, 304, 336-342, 378.
[^44]: [`browser-model.md`](browser-model.md) lines 89, 121-122.
[^45]: [`train/train.py`](../train/train.py) lines 222-238, 402-405, 415.
[^46]: [`decisions/0007-ollaya-bench.md`](decisions/0007-ollaya-bench.md) lines 46-47; [`api.md`](api.md) line 199.
[^47]: [`browser-model.md`](browser-model.md) lines 117, 124.
[^48]: [`results/runs/2026-10-23-public-benchmark-rtx5090-cuda.json`](../results/runs/2026-10-23-public-benchmark-rtx5090-cuda.json) `results[?model=="<tag>"]`: `macro_accuracy`, `macro_accuracy_ci`, `by_type.choice`, `by_type.noul`, `by_type.score`, `pooled_ece`, `macro_ece`.
[^49]: [`results/editions/2026.10.json`](../results/editions/2026.10.json) `score.definition`.
[^50]: [`results/editions/2026.10.json`](../results/editions/2026.10.json) `score.results[?model=="<tag>"].trained_on` (lines 785, 810, 1187-1199, 1224-1236, 1261-1273).
[^51]: Computed for this report from [`results/editions/2026.10.json`](../results/editions/2026.10.json) `score.results[?model=="<tag>"].tasks`, averaging the tasks not in `trained_on`; for `coba:1b` and `coba:400m` the 14-task means are 36.328 and 35.828.
[^52]: [`results/editions/2026.10.json`](../results/editions/2026.10.json) top-level `trained_on` (lines 1277-1290).
[^53]: [`browser-model.md`](browser-model.md) lines 126-128; [`train/stage.py`](../train/stage.py) lines 23-25.
[^54]: [`train/stage.py`](../train/stage.py) lines 46-49 (`TERNARY_RECIPE`).
[^55]: [`ternary-research.md`](ternary-research.md) lines 148-157.
[^56]: [`train/ternary.py`](../train/ternary.py) lines 9, 29-31.
[^57]: [`train/ternary.py`](../train/ternary.py) lines 5-7, 38-61 (absmean forward at lines 56-58).
[^58]: [`train/ternary.py`](../train/ternary.py) lines 10-12, 42-54; [`train/train.py`](../train/train.py) line 270.
[^59]: [`train/ternary.py`](../train/ternary.py) lines 79-96.
[^60]: [`train/ternary.py`](../train/ternary.py) lines 14-15, 64-76, 95-96; [`train/train.py`](../train/train.py) line 271.
[^61]: [`train/train.py`](../train/train.py) lines 269, 272, 307-308, 316-317.
[^62]: [`ternary-research.md`](ternary-research.md) lines 122-127.
[^63]: [`train/train.py`](../train/train.py) lines 364-375.
[^64]: [`train/train.py`](../train/train.py) lines 136-140, 274, 366-368.
[^65]: [`train/train.py`](../train/train.py) lines 143-158, 276, 318-321, 369-372; [`ternary-research.md`](ternary-research.md) lines 124-126.
[^66]: [`train/train.py`](../train/train.py) lines 161-169, 277, 322-323, 373-375.
[^67]: [`train/train.py`](../train/train.py) line 5.
[^68]: [`train/ternary.py`](../train/ternary.py) lines 99-103; [`train/train.py`](../train/train.py) lines 279-280, 357-358; [`train/stage.py`](../train/stage.py) lines 48-49; [`browser-model.md`](browser-model.md) line 127.
[^69]: [`browser-model.md`](browser-model.md) lines 129-132.
[^70]: [`ternary-research.md`](ternary-research.md) lines 128-130.
[^71]: [`train/train.py`](../train/train.py) lines 394-395; [`train/ternary.py`](../train/ternary.py) lines 120-128; [`ternary-research.md`](ternary-research.md) line 138.
[^72]: [`browser-model.md`](browser-model.md) lines 126-128.
[^73]: [`train/ternary.py`](../train/ternary.py) lines 106-117; [`train/train.py`](../train/train.py) lines 21-22, 252-256, 402-405.
[^74]: [`browser-model.md`](browser-model.md) lines 79-82.
[^75]: [`results/runs/2026-10-23-latency-rtx5090-cuda.json`](../results/runs/2026-10-23-latency-rtx5090-cuda.json) `results[?model=="coba:400m"].size_mb` and `results[?model=="coba:400m-ternary"].size_mb`.
[^76]: [`train/webgraph.py`](../train/webgraph.py) lines 1-12, 26-51, 65-99.
[^77]: [`train/webgraph.py`](../train/webgraph.py) lines 54-62, 78-83; [`train/ternary.py`](../train/ternary.py) lines 138-150.
[^78]: [`train/webgraph.py`](../train/webgraph.py) lines 7-10, 102-131 (tolerance at line 129).
[^79]: [`browser-model.md`](browser-model.md) lines 140-150.
[^80]: [`train/stage.py`](../train/stage.py) lines 23-25 (web graph MB) and 35-39 ("`onnx/model.onnx` and the tokenizer"); [`site/src/data/try.ts`](../site/src/data/try.ts) line 15 (66 MB).
[^81]: [`browser-model.md`](browser-model.md) lines 136-138.
[^82]: [`browser-model.md`](browser-model.md) line 167; [`train/card-web.md`](../train/card-web.md) lines 1-6.
[^83]: [`site/src/data/try.ts`](../site/src/data/try.ts) lines 8-16.
[^84]: [`browser-model.md`](browser-model.md) lines 152-156.
[^85]: `ollaya` and `note` in [`results/runs/2026-10-23-latency-rtx5090-cuda.json`](../results/runs/2026-10-23-latency-rtx5090-cuda.json) and the other 2026-10-23 run files; [`results/editions/2026.10.json`](../results/editions/2026.10.json) `updates[2].release` and `updates[2].runs`.
[^86]: Computed from [`results/editions/2026.10.json`](../results/editions/2026.10.json) `score.results[?model=="pplx-decider:flash"].score` minus each tag's `.score`.
[^87]: Computed from [`results/editions/2026.10.json`](../results/editions/2026.10.json) `score.results[*].tasks.typed-decisions`, over the 41 results whose `trained_on` does not include `typed-decisions`.
[^88]: [`results/editions/2026.10.json`](../results/editions/2026.10.json) `score.results[?model=="<tag>"].tasks.<task>`; ternary column from `score.results[?model=="<tag>"].trained_on`.
[^89]: [`results/runs/2026-10-23-typed-decisions-rtx5090-cuda.json`](../results/runs/2026-10-23-typed-decisions-rtx5090-cuda.json) `protocol.benchmark`, `protocol.accuracy`, `protocol.ece`, `results[*].errors`.
[^90]: [`results/runs/2026-10-23-public-benchmark-rtx5090-cuda.json`](../results/runs/2026-10-23-public-benchmark-rtx5090-cuda.json) `protocol.benchmark`, `protocol.runner`, `protocol.accuracy`, `protocol.ece`.
[^91]: [`results/runs/2026-10-23-public-benchmark-rtx5090-cuda.json`](../results/runs/2026-10-23-public-benchmark-rtx5090-cuda.json) `results[*].errors`, `results[*].per_subset.<subset>.errors` and `.n`, `protocol.accuracy`.
[^92]: [`api.md`](api.md) line 199; [`results/runs/2026-10-23-public-benchmark-rtx5090-cuda.json`](../results/runs/2026-10-23-public-benchmark-rtx5090-cuda.json) `protocol.client`.
[^93]: Parity run files, `protocol.reference`, `protocol.gate`, `results[*].goldens` (for example [`results/runs/2026-10-23-parity-rtx5090-cpu-coba.json`](../results/runs/2026-10-23-parity-rtx5090-cpu-coba.json) lines 8-11, 25).
[^94]: Parity run files, `machine`, `device`, `ollaya`; `results[*].engine` (`mlx`) in [`results/runs/2026-10-23-parity-macm4pro-metal-coba.json`](../results/runs/2026-10-23-parity-macm4pro-metal-coba.json).
[^95]: [`train/card.md`](../train/card.md) line 87; [`train/data/build.py`](../train/data/build.py) lines 44-73.
[^96]: [`train/card.md`](../train/card.md) lines 88-89; [`site/content/library/coba.md`](../site/content/library/coba.md) line 56.
[^97]: [`train/card.md`](../train/card.md) lines 90-91.
[^98]: [`train/card.md`](../train/card.md) lines 92-93.
[^99]: [`train/serve.py`](../train/serve.py) lines 1-7; [`browser-model.md`](browser-model.md) lines 70, 84-85; [`train/pyproject.toml`](../train/pyproject.toml) line 4.
[^100]: [`train/data/build.py`](../train/data/build.py) line 9; [`train/data/indomain.py`](../train/data/indomain.py) line 4; [`train/data/label.py`](../train/data/label.py) lines 6-7.
[^101]: [`train/train.py`](../train/train.py) lines 3-5.
[^102]: [`train/ternary.py`](../train/ternary.py) line 3; [`train/webgraph.py`](../train/webgraph.py) lines 3-4; [`train/eval.sh`](../train/eval.sh) lines 2-5; [`train/stage.py`](../train/stage.py) lines 3-7.
[^103]: [`browser-model.md`](browser-model.md) lines 188-190.
[^104]: Computed from [`results/editions/2026.10.json`](../results/editions/2026.10.json) `score.results[*].score_ci` with the leaderboard's rule, [`site/src/bench/data.ts`](../site/src/bench/data.ts) line 185.
[^105]: `config.json` of [`jhu-clsp/ettin-encoder-68m`](https://huggingface.co/jhu-clsp/ettin-encoder-68m/blob/main/config.json), [`-150m`](https://huggingface.co/jhu-clsp/ettin-encoder-150m/blob/main/config.json), [`-400m`](https://huggingface.co/jhu-clsp/ettin-encoder-400m/blob/main/config.json) and [`-1b`](https://huggingface.co/jhu-clsp/ettin-encoder-1b/blob/main/config.json): `num_hidden_layers`, `hidden_size`, `num_attention_heads`, `intermediate_size`, `global_attn_every_n_layers`, `local_attention`, `vocab_size`. Each coba repository's `encoder/config.json` has the same values.
[^106]: Training record `hope-wsl:~/ollaya-train/data/build-stats.json` (`build.py`'s summary): `main_rows`, `main_questions`, `types`.
[^107]: Training records: `hope-wsl:~/ollaya-train/serve.log`, `serve11501.log` and `serve11502.log`, and `white:~/ollaya-train/serve.log`, each `loaded model model=pplx-decider:flash-fp16 device=cuda:0 precision=fp16`, Ollaya 0.16.0. `main.rev.jsonl` is `main.jsonl` in reverse order; `hope-wsl`'s main labels cover rows 0 to 207,320 of `main.jsonl` and `white`'s rows 206,080 to the last; `white` also holds `indomain.rev.jsonl` (checked for this report).
[^108]: Computed for this report from the training records `hope-wsl:~/ollaya-train/data/teacher-main.jsonl`, `teacher-main-white.jsonl`, `teacher-indomain.jsonl`, `teacher-indomain-white.jsonl`, `main.jsonl` and `indomain.jsonl`; every progress line of `label.log` and `label-indomain.log` on both machines reports 0 rejected.
[^109]: [`train/train.py`](../train/train.py) lines 46-58 (`teacher.setdefault`, over the `--teacher` files in order).
[^110]: Training records `choso-wsl:~/ollaya-train/run-t400ide2.sh`, `white:~/ollaya-train/run-t150e2.sh` and `white:~/ollaya-train/run-t68e2.sh` (`--teacher`, `--init`, `--kd-from`, `--quant`, `--emb-bits`, `--kd`, `--rel`, `--hid`, `--lr-enc`, `--lr-head`, `--wd`, `--epochs`); the published configurations, `training.teacher`, `.kd_from`, `.quant`, `.emb_bits`, `.kd`, `.kd_tau`, `.rel`, `.hid`, `.lr_encoder`, `.lr_head`, `.wd`, `.batch`, `.epochs`, `.steps`, in [`coba-400m-ternary`](https://huggingface.co/ollaya-dev/coba-400m-ternary/blob/main/rl_agent_config.json), [`coba-150m-ternary`](https://huggingface.co/ollaya-dev/coba-150m-ternary/blob/main/rl_agent_config.json), [`coba-68m-ternary`](https://huggingface.co/ollaya-dev/coba-68m-ternary/blob/main/rl_agent_config.json).
[^111]: Training records `hope-wsl:~/ollaya-train/queue-hope.sh` (`fp400-id-T`), `choso-wsl:~/ollaya-train/queue4.sh` (`fp150-id-T`) and `white:~/ollaya-train/queue-white.sh` (`fp68-id-T`): `--data ../data/main-id-train.jsonl` with the four teacher files.
[^112]: Computed for this report from the training records `hope-wsl:~/ollaya-train/data/main.jsonl`, `main-train.jsonl`, `main-val.jsonl`, `main-id-train.jsonl` and `indomain.jsonl`: row and question counts, ids and their order.
[^113]: Every command line of Section 10 passes `--val ../data/main-val.jsonl`, and so do the development runs' scripts in the training records.
[^114]: Training records `choso-wsl:~/ollaya-train/run-id-gold.sh`, `queue2.sh` and `queue3.sh`: `cat ../data/main-train.jsonl ../data/indomain.jsonl > ../data/main-id-train.jsonl`.
[^115]: `training` in the published configurations of [`coba-1b`](https://huggingface.co/ollaya-dev/coba-1b/blob/main/rl_agent_config.json) and [`coba-400m`](https://huggingface.co/ollaya-dev/coba-400m/blob/main/rl_agent_config.json): `epochs`, `batch`, `steps`, `lr_encoder`, `lr_head`, `wd`, `alpha`, `hours`; `--grad-ckpt` from the training records `hope-wsl:~/ollaya-train/queue-1b2.sh` and `choso-wsl:~/ollaya-train/run-fp400z.sh`.
[^116]: Training records: the run directories `hope-wsl:~/ollaya-train/runs/fp1b-T` and `choso-wsl:~/ollaya-train/runs/fp400-T`, and `nvidia-smi` on each machine (`hope-wsl` an NVIDIA GeForce RTX 5090, `choso-wsl` an RTX 4090, `white` an RTX 3090).
[^117]: `temperature` in each `ollaya-dev/coba-*` repository's `rl_agent_config.json` on Hugging Face, in laya's type order (choice, score, noul: `laya.common.QTYPES`, which [`train/train.py`](../train/train.py) line 223 follows); the ternary repositories' `calibration.json` holds the same values.
[^118]: Training records `choso-wsl:~/ollaya-train/run-pilots.sh` (`t150-T0`, `t150-T1` with `--kd 1`, `t150-T2` with `--kd 1 --rel 10`), `queue2.sh` (`t150-T2w`, "run T2 with quantization warm-up") and `queue3.sh` (`t150-T3w`, "T2 + warm-up + every-layer hidden distillation").
[^119]: Computed for this report from the `model.safetensors` of `ollaya-dev/coba-400m-ternary`, `coba-150m-ternary` and `coba-68m-ternary` on Hugging Face: the share of exact zeros over every two-dimensional tensor whose rows take only the values {-g, 0, +g} (120, 96 and 84 tensors); the token embeddings are int4 rows; the act head, the scorer and the type embedding are neither. The last `zeros` entry of each run's `train.log.jsonl` in the training records gives the same shares (0.3138, 0.3139, 0.3294).
[^120]: The Hugging Face file listings of `ollaya-dev/coba-68m-ternary`, `coba-150m-ternary` and `coba-400m-ternary`, read 2026-10-24: `onnx/model.onnx` 31,595,842, 63,030,220 and 143,785,773 bytes; `tokenizer.json` 3,583,228 bytes.
[^121]: Training records `hope-wsl:~/ollaya-own-work/bench/raw/typed/coba-*/rows.jsonl` and `raw/public/coba-*/<subset>/rows.jsonl`, the `error` field (counted for this report).
[^122]: Training records `hope-wsl:~/ollaya-bench/e2610/out/typed/laya-en/rows.jsonl` and `out/public/laya-en/*/rows.jsonl`, the `error` field (counted for this report); [`results/runs/2026-10-14-public-benchmark-rtx5090-cuda.json`](../results/runs/2026-10-14-public-benchmark-rtx5090-cuda.json) and [`results/runs/2026-10-14-typed-decisions-rtx5090-cuda.json`](../results/runs/2026-10-14-typed-decisions-rtx5090-cuda.json), `results[?model=="laya:en"].errors`.
[^123]: Training records: `hope-wsl:~/ollaya-train/queue-1b2.sh` (`fp1b-T`, its `run` function at `--bs 32`; `--eval-every $((64000 / bs))`), `hope-wsl:~/ollaya-train/queue-hope.sh` (`fp400-id-T`), `choso-wsl:~/ollaya-train/run-fp400z.sh` (`fp400-T`), `choso-wsl:~/ollaya-train/queue4.sh` (`fp150-id-T`), `choso-wsl:~/ollaya-train/run-t400ide2.sh` (`t400-id-T3w-e2`), `choso-wsl:~/ollaya-train/run-id-gold.sh` (the `cat`), `white:~/ollaya-train/queue-white.sh` (`fp68-id-T`), `white:~/ollaya-train/run-t150e2.sh` and `white:~/ollaya-train/run-t68e2.sh`. Each runs `.venv/bin/python train.py` in `~/ollaya-train/code` with `../data` and `../runs` paths, shortened here, and then `./eval.sh`. The saved `training` blocks: notes 110 and 115.
[^124]: Training records `~/ollaya-train/code/train.py` and `ternary.py`, the same file on all three machines, compared for this report with [`train/train.py`](../train/train.py) and [`train/ternary.py`](../train/ternary.py).
