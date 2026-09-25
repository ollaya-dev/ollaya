"""The `arch` layer (`application/vnd.ollaya.arch`): what the MLX engine builds its network from.

    python3 -m ollaya_convert.arch laya --config encoder/config.json --agent-config rl_agent_config.json \\
        --weights model.safetensors --out arch.json
    python3 -m ollaya_convert.arch sequence-classification --config config.json --weights model.safetensors --out arch.json
    python3 -m ollaya_convert.arch von --config config.json --weights option_marker.pt --out arch.json
    python3 -m ollaya_convert.arch gliclass --config config.json --weights model.safetensors --out arch.json

The MLX engine (`crates/ollaya-runner/src/mlx/`) runs our own code per backbone and head on the
author's unmodified weights file. This layer tells it which network that file holds: the encoder's
hyperparameters from the author's `config.json`, the head's from the author's model code (named
below), and the tensor names. It holds no weights. A model without it runs on ONNX Runtime only.

Every tensor the Rust loader reads is checked against the weights file here, so a wrong prefix
fails at build time. PyTorch zip checkpoints (von) also get each tensor's byte offset: every
storage is an uncompressed zip member, and the pickle is read here with a restricted unpickler
that resolves only tensor-rebuild records (as `families/weightless_ext.py` does), so the runtime
never reads a pickle.

Standard library only: this runs anywhere, without the conversion environment.
"""
import argparse
import collections
import json
import pickle
import struct
import zipfile

SCHEMA = 1
DTYPE_SIZES = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8}
TORCH_STORAGES = {"FloatStorage": "F32", "HalfStorage": "F16", "BFloat16Storage": "BF16", "LongStorage": "I64"}


# ------------------------------------------------------------------------------ weights indexes


def safetensors_index(path):
    """{name: (dtype, shape)} from a safetensors header."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return {k: (v["dtype"], tuple(v["shape"])) for k, v in header.items()}


_Storage = collections.namedtuple("_Storage", "dtype key numel")
_Tensor = collections.namedtuple("_Tensor", "storage offset size stride")


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
    return _Tensor(storage, storage_offset, tuple(size), tuple(stride))


class _MetadataUnpickler(pickle.Unpickler):
    """Resolves only what a state_dict pickle needs; any other global is refused."""

    def find_class(self, module, name):
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor_v2
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if module == "torch" and name in TORCH_STORAGES:
            return name
        raise pickle.UnpicklingError("refusing to resolve %s.%s" % (module, name))

    def persistent_load(self, pid):
        kind, storage_type, key, _location, numel = pid
        if kind != "storage" or storage_type not in TORCH_STORAGES:
            raise pickle.UnpicklingError("unsupported storage %r" % (pid,))
        return _Storage(TORCH_STORAGES[storage_type], key, numel)


def torch_zip_index(path):
    """{name: (dtype, shape, absolute offset)} for every contiguous tensor of a torch.save zip."""
    zf = zipfile.ZipFile(path)
    pkl = next(n for n in zf.namelist() if n.endswith("/data.pkl"))
    root = pkl[: -len("data.pkl")]
    obj = _MetadataUnpickler(zf.open(pkl)).load()
    starts = {}
    with open(path, "rb") as f:
        for info in zf.infolist():
            if not info.filename.startswith(root + "data/"):
                continue
            if info.compress_type != zipfile.ZIP_STORED:
                raise SystemExit("%s is compressed; cannot reference it" % info.filename)
            f.seek(info.header_offset)
            hdr = f.read(30)
            n_name, n_extra = struct.unpack("<HH", hdr[26:30])
            starts[info.filename[len(root) + 5:]] = info.header_offset + 30 + n_name + n_extra
    index = {}
    for name, t in obj.items():
        if not isinstance(t, _Tensor):
            continue
        expect, acc = [], 1
        for d in reversed(t.size):
            expect.append(acc)
            acc *= d
        if tuple(reversed(expect)) != tuple(t.stride):
            continue  # not contiguous: cannot be read in place
        item = DTYPE_SIZES[t.storage.dtype]
        index[name] = (t.storage.dtype, t.size, starts[t.storage.key] + t.offset * item)
    return index


# ------------------------------------------------------------------------------------- backbone


def modernbert(config, prefix):
    """Hugging Face ModernBertConfig, normalised to transformers 5 (layer types, RoPE per type).

    Older configs (transformers 4, e.g. Ettin in GLiClass) give `global_attn_every_n_layers`,
    `global_rope_theta` and `local_rope_theta`; transformers 5 derives the same values from them
    (configuration_modernbert.py), with defaults 3, 160000 and 10000."""
    c = config
    n = c["num_hidden_layers"]
    types = c.get("layer_types") or [
        "sliding_attention" if i % c.get("global_attn_every_n_layers", 3) else "full_attention" for i in range(n)]
    rope = c.get("rope_parameters") or {}
    theta = {
        "full_attention": float((rope.get("full_attention") or {}).get("rope_theta", c.get("global_rope_theta", 160000.0))),
        "sliding_attention": float((rope.get("sliding_attention") or {}).get("rope_theta", c.get("local_rope_theta", 10000.0))),
    }
    for kind, params in rope.items():
        if (params or {}).get("rope_type", "default") != "default":
            raise SystemExit("unsupported rope_type for %s: %r" % (kind, params))
    if c.get("model_type") != "modernbert":
        raise SystemExit("not a ModernBERT config: %r" % c.get("model_type"))
    return {
        "type": "modernbert",
        "prefix": prefix,
        "vocab_size": c["vocab_size"],
        "hidden_size": c["hidden_size"],
        "intermediate_size": c["intermediate_size"],
        "num_hidden_layers": n,
        "num_attention_heads": c["num_attention_heads"],
        "norm_eps": c.get("norm_eps", 1e-5),
        "norm_bias": c.get("norm_bias", False),
        "attention_bias": c.get("attention_bias", False),
        "mlp_bias": c.get("mlp_bias", False),
        "hidden_activation": c.get("hidden_activation", "gelu"),
        "local_attention": c.get("local_attention", 128),
        "layer_types": types,
        "rope_theta": theta,
    }


def modernbert_tensors(b):
    """Every tensor `crates/ollaya-runner/src/mlx/modernbert.rs` reads."""
    p = b["prefix"]
    names = [p + "embeddings.tok_embeddings.weight", p + "embeddings.norm.weight", p + "final_norm.weight"]
    for i in range(b["num_hidden_layers"]):
        lp = "%slayers.%d." % (p, i)
        if i:
            names.append(lp + "attn_norm.weight")
        names += [lp + "attn.Wqkv.weight", lp + "attn.Wo.weight", lp + "mlp_norm.weight",
                  lp + "mlp.Wi.weight", lp + "mlp.Wo.weight"]
    return names


# ----------------------------------------------------------------------------------------- heads


def laya(config, agent):
    """laya 0.3.7 `laya.common.DecisionModel`: `nn.TransformerEncoderLayer(d, max(1, d // 64),
    4 * d, dropout, batch_first=True, norm_first=True)` x `head_layers` (PyTorch defaults: ReLU,
    eps 1e-5), `type_emb`, `scorer` (LayerNorm, Linear, GELU, Linear) and `act_head`."""
    d = config["hidden_size"]
    head = {
        "type": "laya",
        "type_embeddings": "type_emb.weight",
        "layers_prefix": "head.layers.",
        "layers": agent.get("head_layers", 2),
        "num_heads": max(1, d // 64),
        "activation": "relu",
        "norm_first": True,
        "layer_norm_eps": 1e-5,
        "scorer_prefix": "scorer.",
        "scorer_norm_eps": 1e-5,
        "act_prefix": "act_head.",
    }
    names = [head["type_embeddings"], "scorer.0.weight", "scorer.0.bias", "scorer.1.weight", "scorer.1.bias",
             "scorer.3.weight", "scorer.3.bias", "act_head.0.weight", "act_head.0.bias", "act_head.2.weight",
             "act_head.2.bias"]
    for i in range(head["layers"]):
        lp = "head.layers.%d." % i
        names += [lp + n for n in ("self_attn.in_proj_weight", "self_attn.in_proj_bias", "self_attn.out_proj.weight",
                                   "self_attn.out_proj.bias", "linear1.weight", "linear1.bias", "linear2.weight",
                                   "linear2.bias", "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias")]
    return modernbert(config, "encoder."), head, names


def sequence_classification(config):
    """transformers `ModernBertForSequenceClassification`: pooling, `head` (dense, activation,
    norm), `classifier`."""
    head = {
        "type": "sequence-classification",
        "head_prefix": "head.",
        "classifier_prefix": "classifier.",
        "pooling": config.get("classifier_pooling", "cls"),
        "activation": config.get("classifier_activation", "gelu"),
        "classifier_bias": config.get("classifier_bias", False),
        "num_labels": len(config["id2label"]),
    }
    names = ["head.dense.weight", "head.norm.weight", "classifier.weight", "classifier.bias"]
    if head["classifier_bias"]:
        names.append("head.dense.bias")
    return modernbert(config, "model."), head, names


def von(config):
    """von-sdk 1.1.1 `von.models.option_marker.OptionMarkerScorer`: LayerNorm (eps 1e-5), Linear
    to hidden / 2, GELU, LayerNorm, Linear to 1, at every option marker."""
    head = {"type": "option-marker", "prefix": "scorer.", "norm_eps": 1e-5}
    names = ["scorer." + n for n in ("input_norm.weight", "input_norm.bias", "dense.weight", "dense.bias",
                                     "norm.weight", "norm.bias", "out_proj.weight", "out_proj.bias")]
    return modernbert(config, "encoder."), head, names


def gliclass(config):
    """gliclass 0.1.20 `GLiClassUniEncoder` (uni-encoder, prompt first): segment embeddings added to
    the word embeddings, label-span mean pooling (`class_token_pooling` average), `FeaturesProjector`
    x 2, `MLPScorer`."""
    if config.get("architecture_type") != "uni-encoder" or not config.get("use_segment_embeddings"):
        raise SystemExit("only the uni-encoder with segment embeddings is supported")
    if config.get("normalize_features") or config.get("use_lstm") or config.get("extract_text_features"):
        raise SystemExit("unsupported GLiClass options")
    if config.get("class_token_pooling") != "average":
        raise SystemExit("unsupported class_token_pooling %r" % config.get("class_token_pooling"))
    head = {
        "type": "gliclass-uni",
        "text_token_id": config["text_token_index"],
        "segment_embeddings": "model.segment_embeddings.weight",
        "pooling": config.get("pooling_strategy", "first"),
        "text_projector": "model.text_projector.",
        "classes_projector": "model.classes_projector.",
        "projector_activation": config.get("projector_hidden_act", "gelu"),
        "scorer_type": config.get("scorer_type", "mlp"),
        "scorer_prefix": "model.scorer.",
    }
    names = [head["segment_embeddings"]]
    for p in ("model.text_projector.", "model.classes_projector."):
        names += [p + n for n in ("linear_1.weight", "linear_1.bias", "linear_2.weight", "linear_2.bias")]
    names += ["model.scorer.mlp.%d.%s" % (i, n) for i in (0, 2, 4) for n in ("weight", "bias")]
    encoder = dict(config["encoder_config"], vocab_size=config.get("vocab_size", config["encoder_config"]["vocab_size"]))
    return modernbert(encoder, "model.encoder_model."), head, names


# ----------------------------------------------------------------------------------------- build


def build(family, config, weights_path, agent_config=None):
    if family == "laya":
        backbone, head, names = laya(config, agent_config or {})
    elif family == "sequence-classification":
        backbone, head, names = sequence_classification(config)
    elif family == "von":
        backbone, head, names = von(config)
    elif family == "gliclass":
        backbone, head, names = gliclass(config)
    else:
        raise SystemExit("unknown family %r" % family)
    names += modernbert_tensors(backbone)

    torch_zip = zipfile.is_zipfile(weights_path)
    index = torch_zip_index(weights_path) if torch_zip else safetensors_index(weights_path)
    missing = [n for n in names if n not in index]
    if missing:
        raise SystemExit("%s lacks %d tensors, e.g. %s" % (weights_path, len(missing), missing[:3]))
    bad = [n for n in names if index[n][0] not in ("F32", "F16", "BF16")]
    if bad:
        raise SystemExit("unsupported dtypes: %s" % bad[:3])
    b = backbone
    emb = index[b["prefix"] + "embeddings.tok_embeddings.weight"][1]
    if tuple(emb) != (b["vocab_size"], b["hidden_size"]):
        raise SystemExit("embedding shape %s does not match the config" % (emb,))

    weights = {"format": "torch-zip" if torch_zip else "safetensors",
               "file": weights_path.replace("\\", "/").rsplit("/", 1)[-1]}
    if torch_zip:
        weights["tensors"] = {n: {"dtype": index[n][0], "shape": list(index[n][1]), "offset": index[n][2]}
                              for n in sorted(names)}
    return {"schema": SCHEMA, "backbone": backbone, "head": head, "weights": weights}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("family", choices=["laya", "sequence-classification", "von", "gliclass"])
    ap.add_argument("--config", required=True, help="the author's config.json")
    ap.add_argument("--agent-config", help="laya: rl_agent_config.json")
    ap.add_argument("--weights", required=True, help="the author's weights file (safetensors or torch zip)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    config = json.load(open(a.config))
    agent = json.load(open(a.agent_config)) if a.agent_config else None
    arch = build(a.family, config, a.weights, agent)
    with open(a.out, "w") as f:
        json.dump(arch, f, indent=1)
        f.write("\n")
    print("wrote %s: %s + %s head, %s weights, %d tensor offsets" % (
        a.out, arch["backbone"]["type"], arch["head"]["type"], arch["weights"]["format"],
        len(arch["weights"].get("tensors", {}))))


if __name__ == "__main__":
    main()
