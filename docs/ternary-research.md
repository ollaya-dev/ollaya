# Ternary weights: what the literature says, and the recipe it implies

Research notes for the ternary version of our own decision model (`docs/browser-model.md`), 2026-10-22.
The model is a ModernBERT-architecture encoder (ettin, English) with Laya's decision head, trained by
distillation; what we want from ternary weights is a small download, not faster arithmetic (yet).

## Decoder-side state of the art (2025–2026)

**BitNet b1.58** (Ma et al., [arXiv:2402.17764](https://arxiv.org/abs/2402.17764)): absmean ternarization,
`W_q = g · RoundClip(W / g, -1, 1)`, `g = mean(|W|)`, 8-bit per-token activations, straight-through
estimator. Trained from scratch, it matches fp16 at 3B and up; [BitNet b1.58 2B4T](https://arxiv.org/abs/2504.12285)
(MIT) is the open 2B model trained this way on 4T tokens.

**BitNet Distillation / BitDistill** (Microsoft, [arXiv:2510.13998](https://arxiv.org/abs/2510.13998)): the closest
published setup to ours. It turns a fine-tuned full-precision model into a ternary one *for one task*:

1. **SubLN**: an extra normalization before the attention output projection and before the FFN down
   projection.
2. **Continued pre-training**: about 10B tokens of general text with ternary weights, as a warm-up.
3. **Distillation-based fine-tuning**: the task loss plus a logit KL from the fp fine-tuned model (τ = 5,
   λ = 10 for classification) and MiniLM attention-relation distillation (Q-Q, K-K, V-V relations) on a
   **single late layer** (γ = 1e5).

Results on MNLI, QNLI and SST-2 at 0.6B–4B: plain ternary fine-tuning loses 14–16 points; BitDistill matches
fp16 within about 0.1 (Qwen3-0.6B MNLI: fp16 88.01, ternary SFT 74.09, BitDistill 88.17). Its ablation on
0.6B MNLI:

| Stages | MNLI |
|---|---|
| ternary SFT | 74.09 |
| + SubLN | 76.30 |
| + SubLN + continued pre-training | 86.73 |
| + SubLN + distillation (no continued pre-training) | 88.04 |
| all three | 88.17 |

Distillation alone recovers almost everything on classification. Logits only gives 87.32, attention only
87.67, both 88.17, so both terms help. A larger teacher helps the student further.

**ParetoQ** (Meta, NeurIPS 2025, [arXiv:2502.02631](https://arxiv.org/abs/2502.02631)): one framework for 1 to 4 bits.

- Ternary, 2-bit and 3-bit lie on the same size–accuracy frontier, and all beat 4-bit and binary.
- Below 3 bits the representations change a lot, so ternary needs real training rather than a light touch.
- **Start from a pretrained fp model**, not from scratch. The best split is about 90% fp training and 10%
  quantization-aware training.
- Scales are learnable, per channel, initialised at `max(|W|)` ("SEQ", an LSQ variant).
- QAT uses AdamW, no weight decay, and a learning rate of about 2e-5 with cosine decay.
- Embeddings at 4 bits sit on the Pareto front.
- Ternary costs 2–3 points against fp at 125M–600M on zero-shot reasoning.
- Caveat: the paper's SEQ formula with k = 3 does not produce three levels, so the exact ternary grid has to
  come from its code.

**Tequila** (ICLR 2026, [arXiv:2509.23809](https://arxiv.org/abs/2509.23809)): finds the main loss of absmean
ternary training in **deadzone trapping**. Weights near the zero threshold get only noisy straight-through
gradients and stay stuck at zero.

- Its fix adds the trapped weights back as an input-independent **per-channel bias**, `λ · Σ_{i∈deadzone} w_i`
  with λ = 1e-3. During training that bias carries a direct gradient; at inference it is precomputed.
- 1B, 10B tokens: average 0.519, against 0.398 for ParetoQ and 0.386 for TernaryLLM at the same budget.
- Group size 128 for the scales.
- For us the bias costs nothing at runtime: ModernBERT's config can turn on Linear biases, so Ollaya's graph
  and the browser simply carry one more vector per layer.

**Extra RMSNorm** ([arXiv:2505.08823](https://arxiv.org/abs/2505.08823)): a norm before every quantized linear
layer, plus quantizing gradually layer by layer, matches more elaborate distillation pipelines when
fine-tuning to 1.58 bits.

**Ternary Mamba** ([arXiv:2606.18114](https://arxiv.org/abs/2606.18114)) warns that learnable scales can cause a
"zero-ratio collapse", where the share of zero weights runs away. Track the zero ratio during training.

**Storage**: real ternary models are 30–50% zeros; a presence bitmap plus signs reaches about 1.49 bits per
weight ([arXiv:2609.16338](https://arxiv.org/abs/2609.16338)). Our 2-bit packing is simpler. Gzip over HTTP
recovers part of this, so it is not worth a custom format yet.

**Products**: PrismML's Ternary Bonsai (1.7B–27B, April 2026) are ternary conversions of existing models with
group-wise fp16 scales (about 1.71 bits per weight), not trained from scratch.

## Encoder-side literature

No published ternary result exists for ModernBERT or ettin; the evidence comes from BERT and from BitNet.

- **TernaryBERT** (EMNLP 2020, [arXiv:2009.12812](https://arxiv.org/abs/2009.12812)):
  - **What is ternary:** Q, K, V, O and both FFN matrices of every layer, plus the word embedding.
    Biases, LayerNorms and the task layer stay fp.
  - **Quantizer:** TWN, `Δ = 0.7·mean|w|`, with an L1-optimal scale. Loss-aware ternarization (LAT) gives
    the same result. One scale per matrix, one per row for the embedding.
  - **Training:** the student starts from the *fine-tuned* fp BERT and distills from it: MSE on every hidden
    state and attention map, plus soft cross-entropy on the logits. Learning rate 2e-5.
  - **Result:** GLUE 77.3 against 78.2 fp with fp activations, 14.9× smaller.
  - **Ablations:** with labels only, MNLI drops 2.5 and CoLA 10. Starting from the pre-trained instead of the
    fine-tuned model costs 9.7 on CoLA.
- **XTC** (Microsoft 2022, [arXiv:2206.01859](https://arxiv.org/abs/2206.01859)): earlier ternary BERTs were
  undertrained.
  - A longer budget with learning-rate decay brings ternary BERT-base to GLUE 84.10, against 83.95 for its fp
    teacher.
  - One stage of distillation (logits, attention, hidden states) is as good as several.
  - A learning rate of 5e-4 can collapse. Shallower students (5–6 layers) lose more.
- **BiT** (Meta 2022, [arXiv:2205.13016](https://arxiv.org/abs/2205.13016)) and **BiBERT**
  ([arXiv:2203.06390](https://arxiv.org/abs/2203.06390)), both binary:
  - Distilling raw attention scores hurts; BiT gained 2.3 points by dropping them.
  - Learnable scales and logit plus hidden-state distillation help.
  - BiBERT distills the normalized Q/K/V similarity matrices, which is MiniLM's relation distillation.
- **"When are 1.58 bits enough?"** ([arXiv:2411.05882](https://arxiv.org/abs/2411.05882)) and **"BitNet b1.58
  Reloaded"** ([arXiv:2407.09527](https://arxiv.org/abs/2407.09527)):
  - Trained *from scratch*, a ternary encoder needs about twice the width to match fp.
  - Mean and median scales come out about even.
- **BitNet text embeddings** ([arXiv:2606.25674](https://arxiv.org/abs/2606.25674)), BitDistill's recipe on
  0.27B–0.6B embedding models:
  - Within 0.35–0.6 MMTEB points of the fp teacher.
  - Without SubLN −2.1, without relation distillation −1.1, without continued pre-training −2.8; naive
    ternary −8.7.
- **Embeddings:** Q-BERT ([arXiv:1909.05840](https://arxiv.org/abs/1909.05840)) loses almost nothing at 8 bits
  and about 1 point at 4 bits with a single scale for the whole table. Per-row scales should do better.
  ParetoQ puts 4-bit embeddings on the Pareto front.
- **Activations:** weight-only ternary is enough for a small download. 8-bit activations cost TernaryBERT
  another 0.4 GLUE points and only buy kernel speed.

## Recipe for our model

Ranked by expected impact. Everything in steps 1 to 6 is implemented in `train/train.py` and `train/ternary.py`.

1. **fp first, then ternary from it.** Never ternarize after training, and never train ternary from scratch
   (TernaryBERT, ParetoQ, Continual QAT). The fp student is the best checkpoint we can train.
2. **Two teachers.**
   - The pplx-decider distributions stay in the target.
   - The fp student, which has the same architecture, adds a logit KL (`--kd`, τ 2) and MiniLM Q-K-V relation
     KL on the last global-attention layer (`--rel`), as in BitDistill.
   - No raw attention-score MSE (BiT).
3. **One long stage on all teacher-labelled data.**
   - Learning rate grid {2e-5, 5e-5, 1e-4}, cosine to 0, weight decay 0 (ParetoQ).
   - More teacher-scored inputs is our form of data augmentation.
4. **Token embeddings at 4 bits per row** (`--emb-bits 4`), with 8 bits as the fallback. In ettin-150m they
   are 38.7M of the ~150M parameters, so they decide the download size as much as the layers do.
5. **Ternary per output row everywhere else**, including the 2-layer decision head (about 10M parameters at
   d = 768, not negligible). Norms, the type embedding and the scorer stay fp.
6. **Quantizer:**
   - Absmean as the baseline.
   - ParetoQ's learned per-row scale as the first ablation, initialised where it equals absmean.
   - Track the share of zero weights: a runaway value means a collapsing scale (Ternary Mamba).
7. **Only if a gap remains after step 6:**
   - Tequila's deadzone bias: runtime-compatible, since ModernBERT's config turns on Linear biases.
   - SubLN: new graph ops, so Ollaya's export would need two extra norms per layer.
   - Continued ternary MLM pre-training on general text: worth 0.1 for BitDistill's classifier, 2.8 for
     BitEmbed's embeddings.
8. **Size against bytes.** Compare ternary 400m (~95 MB packed) with ternary 150m (~41 MB) and with an 8-bit
   150m of similar size. Width helps ternary most.
9. **Measure** accuracy, ECE and KL to the teacher on both held-out suites, not accuracy alone.

### Pilot matrix (ettin-150m, once the teacher labels are in)

| Run | What | Answers |
|---|---|---|
| fp | gold + pplx soft labels | the ceiling |
| T0 | ternary, labels only | the naive gap (literature: −2 to −15) |
| T1 | + logit KD from fp | — |
| T2 | + relation KD | — |
| T3 | T2 with a learned scale | — |
| T4 | T2 with 8-bit embeddings | the cost of 4-bit embeddings |

Each ternary run is one epoch, about 40 minutes on an RTX 4090. The winning settings then go to ettin-400m.
