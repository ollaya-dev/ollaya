"""Train a Laya-network decision model by distillation, optionally with ternary weights.

    uv run train.py --backbone jhu-clsp/ettin-encoder-150m --data data/train.jsonl \
        --teacher data/teacher.jsonl --val data/val.jsonl --out runs/fp150
    uv run train.py ... --init runs/fp150 --quant absmean --kd-from runs/fp150 --kd 1 --rel 1 --wd 0 --out runs/t150

Rows (JSONL, one state each):
    {"id", "source", "state", "questions": {qid: {"type", "instructions", "criteria"}},
     "gold": {qid: [p, ...]} | null}
Teacher rows (JSONL): {"id", "teacher": {qid: [p, ...]}}. Probabilities are in option order: a choice's
criteria order, a score's levels, noul as [false, true] (the order `laya.common.render_options` uses).

The output directory is a Laya checkpoint (`rl_agent_config.json`, `model.safetensors`, `tokenizer/`,
`encoder/config.json`): `laya.load(out)` reads it and Ollaya's convert/ exports it like any other.

Ternary (`--quant`, recipe and evidence in docs/ternary-research.md): every weight matrix of the encoder layers
and the decision head is trained ternary per output row (absmean, or ParetoQ's form with a learned scale), and
the token embeddings optionally at 4 or 8 bits (`--emb-bits`). Start from the fp model (`--init`) and distill
from it (`--kd-from`): its option distributions (`--kd`, temperature `--kd-tau`), MiniLM-style the Q-Q, K-K
and V-V relations of one late global-attention layer (`--rel`), and every layer's output (`--hid`); the
weights can blend from fp to quantized over the first steps (`--quant-warmup`). The saved weights are exactly quantized, so
Ollaya runs them as ordinary weights and `webgraph.py` keeps them at 2 bits for the browser. Norms, the type embedding and the
scorer stay full precision.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from laya.agent import Agent
from laya.common import QTYPES, DecisionModel, build_sequence, collate_items, render_options
from safetensors.torch import save_file
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

from ternary import finalize_, quantize_, set_mix, zero_ratio


def load_rows(path, teacher_path=None, alpha=0.5):
    teacher = {}
    for p in (teacher_path or "").split(","):
        if not p:
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:  # a line cut off when a labeller was stopped
                    continue
                if "teacher" in r:  # rows the teacher rejected carry "error" instead
                    teacher.setdefault(r["id"], r["teacher"])
    out, skipped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            gold, tch = r.get("gold") or {}, teacher.get(r["id"], {})
            for qid, qdef in r["questions"].items():
                try:
                    Agent._check_question(qid, qdef)
                except ValueError:
                    skipped += 1
                    continue
                q = Agent._to_internal(qdef)
                k = len(render_options(q))
                g, t = gold.get(qid), tch.get(qid)
                g = g if g is not None and len(g) == k else None
                t = t if t is not None and len(t) == k else None
                if g is None and t is None:
                    skipped += 1
                    continue
                target = [alpha * a + (1 - alpha) * b for a, b in zip(g, t)] if g and t else (g or t)
                s = sum(target)
                out.append({"state": r["state"], "q": q, "target": [x / s for x in target],
                            "gold": g, "source": r.get("source", ""), "len": len(json.dumps(r["state"]))})
    return out, skipped


class Items(torch.utils.data.Dataset):
    def __init__(self, rows, tok, max_len, head_max_len, shuffle_choices):
        self.rows, self.tok, self.max_len, self.head_max_len = rows, tok, max_len, head_max_len
        self.shuffle_choices = shuffle_choices

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        q, target = r["q"], r["target"]
        order = list(range(len(target)))
        # Options must not win by position: permute a choice's options (and its target) half the time.
        if self.shuffle_choices and q["t"] == "choice" and random.random() < 0.5:
            random.shuffle(order)
        ids, markers = build_sequence(self.tok, r["state"], q, self.max_len, self.head_max_len, option_order=order)
        if len(markers) != len(order):
            return []
        gold = [r["gold"][j] for j in order] if r["gold"] else None
        return [{"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]], "target": [target[j] for j in order],
                 "label": max(range(len(gold)), key=gold.__getitem__) if gold else -1, "source": r["source"]}]


def length_batches(rows, bs, seed):
    """Batches of similar state length (less padding), in random order."""
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    batches = []
    for c in range(0, len(idx), bs * 64):
        chunk = sorted(idx[c:c + bs * 64], key=lambda i: rows[i]["len"])
        batches += [chunk[b:b + bs] for b in range(0, len(chunk), bs)]
    rng.shuffle(batches)
    return batches


def losses(logits, act_logits, batch, w_rps):
    mask = batch["marker_mask"]
    target = batch["target"]
    logp = F.log_softmax(logits, -1)
    ce = -(target * logp.masked_fill(~mask, 0)).sum(-1)
    p = logp.exp()
    is_score = batch["qtype"] == QTYPES["score"]
    k = mask.sum(-1).clamp(min=2).float()
    rps = (((p.cumsum(-1) - target.cumsum(-1)) ** 2) * mask).sum(-1) / (k - 1)
    loss = ce + w_rps * rps * is_score
    # The act head says whether to trust the answer: class 0 when the top option is the target's.
    act_t = (p.argmax(-1) != target.argmax(-1)).long()
    return loss.mean() + 0.1 * F.cross_entropy(act_logits.float(), act_t), ce.mean()


def kd_kl(s, t, mask, tau):
    """tau^2 * KL(teacher || student) over a question's options."""
    s = (s / tau).masked_fill(~mask, -1e4)
    pt = (t / tau).masked_fill(~mask, -1e4).softmax(-1)
    return (pt * (pt.clamp_min(1e-12).log() - s.log_softmax(-1))).sum(-1).mean() * tau * tau


def relation_kl(s, t, mask, heads):
    """MiniLM-v2 relation distillation: KL between the teacher's and student's softmax(X X^T / sqrt(d)) for
    X in (Q, K, V), per head, over the real tokens. s, t: the layer's Wqkv outputs [B, L, 3d]."""
    B, L, _ = s.shape
    keys = (~mask)[:, None, None, :]
    rows = mask[:, None, :].float()
    total = 0.0
    for i in range(3):
        xs = s.view(B, L, 3, heads, -1)[:, :, i].transpose(1, 2).float()
        xt = t.view(B, L, 3, heads, -1)[:, :, i].transpose(1, 2).float()
        scale = xs.shape[-1] ** -0.5
        ls = (xs @ xs.transpose(-1, -2) * scale).masked_fill(keys, -1e4).log_softmax(-1)
        pt = (xt @ xt.transpose(-1, -2) * scale).masked_fill(keys, -1e4).softmax(-1)
        kl = (pt * (pt.clamp_min(1e-12).log() - ls)).sum(-1)
        total = total + (kl * rows).sum() / (rows.sum() * heads)
    return total / 3


def hidden_mse(s_layers, t_layers, mask):
    """TernaryBERT-style layer-to-layer distillation: MSE between the student's and teacher's normalized
    layer outputs over the real tokens, averaged over layers."""
    m = mask[..., None].float()
    total = 0.0
    for s, t in zip(s_layers, t_layers):
        d = F.layer_norm(s.float(), s.shape[-1:]) - F.layer_norm(t.float(), t.shape[-1:])
        total = total + (d.pow(2) * m).sum() / (m.sum() * s.shape[-1])
    return total / len(s_layers)


class Capture:
    def __init__(self, module):
        self.out = None
        module.register_forward_hook(lambda m, i, o: setattr(self, "out", o))


def last_global_layer(config):
    types = getattr(config, "layer_types", None)
    if types:
        return max(i for i, t in enumerate(types) if t == "full_attention")
    n = config.global_attn_every_n_layers
    return max(i for i in range(config.num_hidden_layers) if i % n == 0)


def load_model(backbone, ckpt=None):
    model = DecisionModel(AutoModel.from_pretrained(backbone, attn_implementation="sdpa"), 2, 2)
    if ckpt:
        model.load_state_dict(load_file(os.path.join(ckpt, "model.safetensors")), strict=True)
    model.encoder.config.reference_compile = False
    return model


def to_dev(batch, dev):
    return {k: v.to(dev, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def forward(model, b):
    return model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])


@torch.no_grad()
def evaluate(model, loader, dev, amp):
    """Raw logits per question, with accuracy against gold (or the target when no gold)."""
    model.eval()
    out = []
    for b in loader:
        if b is None:
            continue
        b = to_dev(b, dev)
        with torch.autocast(dev.type, dtype=amp, enabled=dev.type == "cuda"):
            logits, _ = forward(model, b)
        for r in range(logits.size(0)):
            k = int(b["marker_mask"][r].sum())
            lab = int(b["label"][r])
            out.append((int(b["qtype"][r]), logits[r, :k].float().cpu(), b["target"][r, :k].cpu(),
                        lab if lab >= 0 else int(b["target"][r, :k].argmax())))
    model.train()
    return out


def fit_temperatures(evals):
    """One temperature per type (choice, score, noul), minimizing NLL of the label; laya clamps to [0.5, 5]."""
    temps = []
    grid = torch.arange(0.5, 5.0001, 0.01)
    for t in range(3):
        rows = [(z, y) for qt, z, _, y in evals if qt == t]
        if not rows:
            temps.append(1.0)
            continue
        k = max(len(z) for z, _ in rows)
        z = torch.full((len(rows), k), -1e4)
        for i, (zi, _) in enumerate(rows):
            z[i, :len(zi)] = zi
        y = torch.tensor([yi for _, yi in rows])
        nll = torch.stack([F.cross_entropy(z / T, y, reduction="sum") for T in grid])
        temps.append(round(float(grid[int(nll.argmin())]), 2))
    return temps


def report(evals, temps):
    by = {}
    for qt, z, _, y in evals:
        p = F.softmax(z / temps[qt], -1)
        by.setdefault(qt, []).append(float(int(p.argmax()) == y))
    names = {v: k for k, v in QTYPES.items()}
    acc = {names[t]: round(sum(v) / len(v), 4) for t, v in by.items()}
    acc["all"] = round(sum(sum(v) for v in by.values()) / max(1, sum(len(v) for v in by.values())), 4)
    return acc


def save(model, tok, cfg, out):
    os.makedirs(os.path.join(out, "encoder"), exist_ok=True)
    state = {k: v.detach().float().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(out, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(out, "encoder"))
    tok.save_pretrained(os.path.join(out, "tokenizer"))
    with open(os.path.join(out, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="jhu-clsp/ettin-encoder-150m")
    ap.add_argument("--data", required=True)
    ap.add_argument("--teacher", help="teacher label files, comma-separated (the first label of an id wins)")
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", help="a checkpoint this script wrote, to continue from (e.g. fp -> ternary)")
    ap.add_argument("--quant", choices=["none", "absmean", "learned"], default="none")
    ap.add_argument("--emb-bits", type=int, choices=[4, 8, 32], default=32)
    ap.add_argument("--kd-from", help="an fp checkpoint of the same backbone to distill from (logits and relations)")
    ap.add_argument("--kd", type=float, default=0.0, help="weight of the logit KL to --kd-from")
    ap.add_argument("--kd-tau", type=float, default=2.0)
    ap.add_argument("--rel", type=float, default=0.0, help="weight of the Q/K/V relation KL to --kd-from")
    ap.add_argument("--rel-rows", type=int, default=16, help="batch rows used for the relation loss (memory)")
    ap.add_argument("--hid", type=float, default=0.0, help="weight of the every-layer hidden-state MSE to --kd-from")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--quant-warmup", type=float, default=0.0,
                    help="share of the steps over which the weights blend linearly from fp to quantized")
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of gold vs teacher when a question has both")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr-enc", type=float, default=5e-5)
    ap.add_argument("--lr-head", type=float, default=2e-4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--w-rps", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--head-max-len", type=int, default=192)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--grad-ckpt", action="store_true", help="recompute the encoder's activations (for 400m+ on 24 GB)")
    ap.add_argument("--name", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        # Under WSL the driver spills past VRAM into system memory instead of failing, so the caching allocator
        # grows into it and training runs several times slower; a cap makes it reuse its cache first.
        torch.cuda.set_per_process_memory_fraction(0.9)
    amp = torch.bfloat16
    os.makedirs(a.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(os.path.join(a.init, "tokenizer") if a.init else a.backbone)
    model = load_model(a.backbone, a.init)
    if a.grad_ckpt:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if a.quant != "none":
        quantize_(model, a.quant, a.emb_bits)
    # from_pretrained leaves the encoder in eval mode, where gradient checkpointing is off
    model.to(dev).train()
    teacher = cap_s = cap_t = None
    if a.kd_from:
        teacher = load_model(a.backbone, a.kd_from).to(dev).eval().requires_grad_(False)
        layer = last_global_layer(model.encoder.config)
        heads = model.encoder.config.num_attention_heads
        cap_s = Capture(model.encoder.layers[layer].attn.Wqkv)
        cap_t = Capture(teacher.encoder.layers[layer].attn.Wqkv)
        hid_s = [Capture(m) for m in model.encoder.layers] if a.hid else []
        hid_t = [Capture(m) for m in teacher.encoder.layers] if a.hid else []

    rows, skipped = load_rows(a.data, a.teacher, a.alpha)
    val_rows, _ = load_rows(a.val, None, a.alpha)
    print("train questions %d (skipped %d), val %d" % (len(rows), skipped, len(val_rows)), flush=True)

    collate = lambda b: collate_items(b, tok.pad_token_id)  # noqa: E731
    train_ds = Items(rows, tok, a.max_len, a.head_max_len, True)
    val_loader = torch.utils.data.DataLoader(Items(val_rows, tok, a.max_len, a.head_max_len, False),
                                             batch_size=64, collate_fn=collate, num_workers=a.workers)

    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr_enc}, {"params": head_params, "lr": a.lr_head}],
                            weight_decay=a.wd, betas=(0.9, 0.98))
    steps_per_epoch = math.ceil(len(rows) / a.bs)
    total = int(steps_per_epoch * a.epochs)
    warm = max(1, int(total * a.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    log = open(os.path.join(a.out, "train.log.jsonl"), "a")
    step, t0, run = 0, time.time(), {}
    last = None
    epoch = 0
    while step < total:
        loader = torch.utils.data.DataLoader(train_ds, batch_sampler=length_batches(rows, a.bs, a.seed + epoch),
                                             collate_fn=collate, num_workers=a.workers, persistent_workers=False)
        for b in loader:
            if step >= total:
                break
            if b is None:
                continue
            b = to_dev(b, dev)
            if a.quant != "none" and a.quant_warmup:
                set_mix(model, min(1.0, step / (a.quant_warmup * total)))
            with torch.autocast(dev.type, dtype=amp, enabled=dev.type == "cuda"):
                logits, act = forward(model, b)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits, _ = forward(teacher, b)
            loss, ce = losses(logits.float(), act, b, a.w_rps)
            parts = {"loss": loss, "ce": ce}
            if teacher is not None and a.kd:
                parts["kd"] = kd_kl(logits.float(), t_logits.float(), b["marker_mask"], a.kd_tau)
                loss = loss + a.kd * parts["kd"]
            if teacher is not None and a.rel:
                n = a.rel_rows
                parts["rel"] = relation_kl(cap_s.out[:n], cap_t.out[:n], b["attention_mask"][:n].bool(), heads)
                loss = loss + a.rel * parts["rel"]
            if teacher is not None and a.hid:
                parts["hid"] = hidden_mse([c.out for c in hid_s], [c.out for c in hid_t], b["attention_mask"].bool())
                loss = loss + a.hid * parts["hid"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            for k, v in parts.items():
                run[k] = run.get(k, 0.0) + v.item()
            if step % 100 == 0:
                rec = {"step": step, "of": total, **{k: round(v / 100, 5) for k, v in run.items()},
                       "lr": sched.get_last_lr()[0], "min": round((time.time() - t0) / 60, 1)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                run = {}
            if step % a.eval_every == 0 or step == total:
                acc = report(evaluate(model, val_loader, dev, amp), [1.0, 1.0, 1.0])
                rec = {"step": step, "val": acc}
                if a.quant != "none":
                    rec["zeros"] = round(zero_ratio(model), 4)
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                last = acc
        epoch += 1

    if a.quant != "none":
        finalize_(model)
    evals = evaluate(model, val_loader, dev, amp)
    temps = fit_temperatures(evals)
    final = report(evals, temps)
    cfg = {
        "encoder": "encoder",
        "head_layers": 2,
        "max_len": a.max_len,
        "head_max_len": a.head_max_len,
        "act_costs": {"escalate": 0.5},
        "amp_dtype": "fp16",
        "model_name": a.name or os.path.basename(os.path.normpath(a.out)),
        "temperature": temps,
        "fine_tuned": True,
        "training": {"backbone": a.backbone, "quant": a.quant, "emb_bits": a.emb_bits,
                     "kd_from": a.kd_from and os.path.basename(os.path.normpath(a.kd_from)), "kd": a.kd,
                     "kd_tau": a.kd_tau, "rel": a.rel, "hid": a.hid, "quant_warmup": a.quant_warmup, "wd": a.wd, "data": os.path.basename(a.data),
                     "teacher": [os.path.basename(t) for t in a.teacher.split(",")] if a.teacher else None, "alpha": a.alpha,
                     "epochs": a.epochs, "batch": a.bs, "lr_encoder": a.lr_enc, "lr_head": a.lr_head,
                     "steps": step, "hours": round((time.time() - t0) / 3600, 3), "val": final, "last_eval": last},
    }
    save(model, tok, cfg, a.out)
    print(json.dumps({"saved": a.out, "temperature": temps, "val": final}), flush=True)


if __name__ == "__main__":
    main()
