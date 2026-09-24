"""The tokenizer exactly as a model repo's `tokenizer.json` defines it.

transformers 5 no longer loads `tokenizer.json` verbatim for some model classes. For DeBERTa-v2/v3
it rebuilds the normalizer in Python (`Replace(\\s{2,}|[\\n\\r\\t] -> " ")`, NFC, right strip)
instead of the file's `Strip` + sentencepiece `Precompiled` charsmap, so e.g. a full-width comma
becomes [UNK] instead of ",". Checkpoints trained under transformers 4.x (MoritzLaurer
zeroshot-v2.0: 4.37.2, GLiClass-instruct: 4.57.3) were trained with the file's behaviour, and the Rust
runtime loads that same file with the `tokenizers` crate. So the references tokenize with the file:
`TokenizersBackend(tokenizer_file=...)` (transformers' `PreTrainedTokenizerFast`), carrying over only the
special-token names, max length and input names from `AutoTokenizer`.
"""
from huggingface_hub import hf_hub_download


def load(repo: str, revision: str):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    auto = AutoTokenizer.from_pretrained(repo, revision=revision)
    tok = PreTrainedTokenizerFast(
        tokenizer_file=hf_hub_download(repo, "tokenizer.json", revision=revision),
        model_max_length=auto.model_max_length, padding_side=auto.padding_side,
        truncation_side=auto.truncation_side, model_input_names=auto.model_input_names,
        **{k: v for k, v in auto.special_tokens_map.items() if isinstance(v, str)},
    )
    # The file may bake in truncation/padding (GLiClass: 4096 / BatchLongest); callers pass their own.
    tok.backend_tokenizer.no_truncation()
    tok.backend_tokenizer.no_padding()
    return tok, auto
