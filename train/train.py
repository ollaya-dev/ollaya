"""Train a Laya-network decision model by distillation, optionally with ternary weights.

    uv run train.py --backbone jhu-clsp/ettin-encoder-150m --data data/train.jsonl \
        --teacher data/teacher.jsonl --val data/val.jsonl --out runs/fp150
    uv run train.py ... --init runs/fp150 --ternary --out runs/t150      # ternary QAT from the fp model

Rows (JSONL, one state each):
    {"id", "source", "state", "questions": {qid: {"type", "instructions", "criteria"}},
     "gold": {qid: [p, ...]} | null}
Teacher rows (JSONL): {"id", "teacher": {qid: [p, ...]}}. Probabilities are in option order: a choice's
criteria order, a score's levels, noul as [false, true] (the order `laya.common.render_options` uses).

The output directory is a Laya checkpoint (`rl_agent_config.json`, `model.safetensors`, `tokenizer/`,
`encoder/config.json`): `laya.load(out)` reads it and Ollaya's convert/ exports it like any other.

Ternary: every Linear weight of the encoder layers and the decision head (and the head's attention
in-projection) is trained through absmean ternarization per output row (BitNet b1.58, straight-through
estimator). The saved weights are exactly {-g_r, 0, +g_r} per row, so a 2-bit packing restores them
bit for bit (`ternary.py`). Embeddings, norms, the type embedding and the small scorer stay full precision.
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
from torch.nn.utils import parametrize
from transformers import AutoModel, AutoTokenizer

from ternary import Ternary, ternarize_, ternary_targets


def load_rows(path, teacher_path=None, alpha=0.5):
    teacher = {}
    if teacher_path:
        with open(teacher_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                teacher[r["id"]] = r["teacher"]
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
    for t in range(3):
        rows = [(z, y) for qt, z, _, y in evals if qt == t]
        if not rows:
            temps.append(1.0)
            continue
        grid = [0.5 + 0.01 * i for i in range(451)]
        nll = [sum(-F.log_softmax(z / T, -1)[y].item() for z, y in rows) for T in grid]
        temps.append(grid[min(range(len(grid)), key=nll.__getitem__)])
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
    ap.add_argument("--teacher")
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", help="a checkpoint this script wrote, to continue from (e.g. fp -> ternary)")
    ap.add_argument("--ternary", action="store_true")
    ap.add_argument("--int8-embeddings", action="store_true", help="with --ternary: also round the token embeddings to int8 per row")
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
    ap.add_argument("--name", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.bfloat16
    os.makedirs(a.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(os.path.join(a.init, "tokenizer") if a.init else a.backbone)
    if a.init:
        enc = AutoModel.from_pretrained(a.backbone, attn_implementation="sdpa")
        model = DecisionModel(enc, 2, 2)
        from safetensors.torch import load_file
        model.load_state_dict(load_file(os.path.join(a.init, "model.safetensors")), strict=True)
    else:
        model = DecisionModel(AutoModel.from_pretrained(a.backbone, attn_implementation="sdpa"), 2, 2)
    model.encoder.config.reference_compile = False
    if a.ternary:
        for m, name in ternary_targets(model):
            parametrize.register_parametrization(m, name, Ternary())
    model.to(dev)

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
                            weight_decay=0.01, betas=(0.9, 0.98))
    steps_per_epoch = math.ceil(len(rows) / a.bs)
    total = int(steps_per_epoch * a.epochs)
    warm = max(1, int(total * a.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    log = open(os.path.join(a.out, "train.log.jsonl"), "a")
    step, t0, run_loss, run_ce = 0, time.time(), 0.0, 0.0
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
            with torch.autocast(dev.type, dtype=amp, enabled=dev.type == "cuda"):
                logits, act = forward(model, b)
            loss, ce = losses(logits.float(), act, b, a.w_rps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run_loss += loss.item()
            run_ce += ce.item()
            if step % 100 == 0:
                rec = {"step": step, "of": total, "loss": round(run_loss / 100, 4), "ce": round(run_ce / 100, 4),
                       "lr": sched.get_last_lr()[0], "min": round((time.time() - t0) / 60, 1)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                run_loss = run_ce = 0.0
            if step % a.eval_every == 0 or step == total:
                acc = report(evaluate(model, val_loader, dev, amp), [1.0, 1.0, 1.0])
                rec = {"step": step, "val": acc}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                last = acc
        epoch += 1

    if a.ternary:
        for m, name in ternary_targets(model):
            parametrize.remove_parametrizations(m, name, leave_parametrized=False)
        ternarize_(model, a.int8_embeddings)
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
        "training": {"backbone": a.backbone, "ternary": a.ternary, "int8_embeddings": a.int8_embeddings, "data": os.path.basename(a.data),
                     "teacher": os.path.basename(a.teacher) if a.teacher else None, "alpha": a.alpha,
                     "epochs": a.epochs, "batch": a.bs, "lr_encoder": a.lr_enc, "lr_head": a.lr_head,
                     "steps": step, "hours": round((time.time() - t0) / 3600, 3), "val": final, "last_eval": last},
    }
    save(model, tok, cfg, a.out)
    print(json.dumps({"saved": a.out, "temperature": temps, "val": final}), flush=True)


if __name__ == "__main__":
    main()
