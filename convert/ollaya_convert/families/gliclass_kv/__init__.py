"""knowledgator/gliclass-v4-{2b,9b}: GLiClass's decoder-KV architecture (layout `gliclass-kv-v1`): a Qwen3.5 decoder
over `prompt + text <<SEP>> label1<<LABEL>> ... <<SEP>>`, a two-layer DeBERTa scorer over the label section with a
gated recurrent cell (up to 3 steps, halting), an MLP per (text, label) pair and a learned calibrator. Text only; the
checkpoint's image path is not ported. See docs/families/gliclass-kv.md."""
