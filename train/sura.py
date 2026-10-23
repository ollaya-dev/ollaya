"""Train sura: a Qwen3.5 decoder in pplx-decider's joint format, distilled from pplx-decider:flash (COB-552).

    uv run sura.py --base Qwen/Qwen3.5-0.8B-Base --teacher-dir <pplx-decider-flash-v1.2 snapshot> \
        --data data/main-train.jsonl --teacher data/teacher-main.jsonl,data/teacher-main-white.jsonl \
        --val data/main-val.jsonl --out runs/sura-0.8b

Rows and teacher rows are train.py's. The student is the teacher's layout on a smaller base: a `Qwen3_5Model` whose
full-attention layers see the whole sequence, and a `Linear(hidden -> 255)` readout at the marker after each
question, one sequence per state with all its questions (protocol `joint-question-inline-readout-v1`). The prompt,
tokenizer and chat template are the teacher's (`pplx_decider` from its snapshot; the vocabulary is Qwen3.5's), so the
output directory is a pplx-decider checkpoint: upstream's `Decider` loads it, and Ollaya's pplx family exports it.

It needs the teacher's own requirements next to train/'s (its `requirements-fast.txt`: torchvision 0.29, pillow and
flash-linear-attention 0.5.2, whose kernels make the Gated DeltaNet layers train on a GPU).

The readout starts from the base's tied embeddings of the answer-code tokens: the base's next-token logits of the
codes. Loss per question, as train.py: cross-entropy to the blend of gold and teacher (`--alpha`), plus the ranked
probability score on score questions. Choice options are permuted half the time, so no option wins by position.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import random
import shutil
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

QTYPES = {"choice": 0, "score": 1, "noul": 2}
# The teacher's files the checkpoint keeps as they are: the protocol and the processor (tokenizer, chat template).
TEACHER_FILES = ["joint_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "processor_config.json"]


def pplx(snapshot):
    src = os.path.join(snapshot, "source", "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from pplx_decider import model, protocol

    return model, protocol


def load_rows(path, teacher_path, alpha, protocol, codes, limit=None):
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
                if "teacher" in r:
                    teacher.setdefault(r["id"], r["teacher"])
    out, skipped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            gold, tch = r.get("gold") or {}, teacher.get(r["id"], {})
            qs, target, label, qtype = {}, {}, {}, {}
            for qid, q in r["questions"].items():
                try:  # upstream's schema check, one question at a time (it reads only the codes)
                    protocol.Encoder.validate(SimpleNamespace(codes=codes), {"state": "", "questions": {qid: q}})
                except (protocol.RequestError, AttributeError, TypeError):
                    skipped += 1
                    continue
                k = len(protocol.options(q)[0])
                g, t = gold.get(qid), tch.get(qid)
                g = g if g is not None and len(g) == k else None
                t = t if t is not None and len(t) == k else None
                if g is None and t is None:
                    skipped += 1
                    continue
                mix = [alpha * a + (1 - alpha) * b for a, b in zip(g, t)] if g and t else (g or t)
                s = sum(mix)
                qs[qid], target[qid], qtype[qid] = q, [x / s for x in mix], QTYPES[q["type"]]
                ref = g or target[qid]
                label[qid] = max(range(k), key=ref.__getitem__)
            if qs:
                est = (len(json.dumps(r["state"])) + len(json.dumps(qs))) // 3 + 64 * len(qs) + 80
                out.append({"state": r["state"], "questions": qs, "target": target, "label": label, "qtype": qtype,
                            "est": est})
            if limit and len(out) >= limit:
                break
    return out, skipped


class Requests(torch.utils.data.Dataset):
    """One row: the request's sequences (usually one), each with its questions' targets, labels and types."""

    def __init__(self, rows, snapshot, codes, max_len, shuffle):
        self.rows, self.snapshot, self.codes, self.max_len, self.shuffle = rows, snapshot, codes, max_len, shuffle
        self.encoder = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        _, protocol = pplx(self.snapshot)
        if self.encoder is None:  # per worker
            from transformers import AutoProcessor

            self.encoder = protocol.Encoder(AutoProcessor.from_pretrained(self.snapshot), self.codes)
        r = self.rows[i]
        qs, target, label = {}, {}, {}
        for qid, q in r["questions"].items():
            t, y = r["target"][qid], r["label"][qid]
            if self.shuffle and q["type"] == "choice" and random.random() < 0.5:
                order = list(range(len(t)))
                random.shuffle(order)
                items = list(q["criteria"].items())
                q, t, y = {**q, "criteria": dict(items[j] for j in order)}, [t[j] for j in order], order.index(y)
            qs[qid], target[qid], label[qid] = q, t, y
        try:
            chunks = self.encoder.plan({"state": r["state"], "questions": qs}, self.max_len)
        except protocol.LengthError:
            return []
        return [(c, [target[k] for k in c.keys], [label[k] for k in c.keys], [r["qtype"][k] for k in c.keys])
                for c in chunks]


def collate(items, pad_id, left_pad):
    flat = [x for it in items for x in it]
    if not flat:
        return None
    chunks = [c for c, *_ in flat]
    b = left_pad(chunks, pad_id)
    width = b["input_ids"].shape[1]
    b["rows"] = torch.cat([torch.full((len(c.keys),), i) for i, c in enumerate(chunks)])
    b["cols"] = torch.cat([c.markers + width - c.length for c in chunks])
    targets = [t for _, ts, _, _ in flat for t in ts]
    b["target"] = torch.zeros(len(targets), 255)
    for i, t in enumerate(targets):
        b["target"][i, :len(t)] = torch.tensor(t)
    b["k"] = torch.tensor([len(t) for t in targets])
    b["label"] = torch.tensor([y for _, _, ys, _ in flat for y in ys])
    b["qtype"] = torch.tensor([q for _, _, _, qt in flat for q in qt])
    return b


def length_batches(rows, budget, seed):
    """Batches of similar length within `budget` padded tokens (by estimate), in random order."""
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    batches = []
    for c in range(0, len(idx), 4096):
        group = []
        for i in sorted(idx[c:c + 4096], key=lambda i: rows[i]["est"]):
            if group and (len(group) + 1) * rows[i]["est"] > budget:
                batches.append(group)
                group = []
            group.append(i)
        if group:
            batches.append(group)
    rng.shuffle(batches)
    return batches


def forward(backbone, readout, b):
    hidden = backbone(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False).last_hidden_state
    z = readout(hidden[b["rows"], b["cols"]]).float()
    valid = torch.arange(z.shape[1], device=z.device)[None] < b["k"][:, None]
    return z.masked_fill(~valid, -1e4), valid


def losses(z, valid, b, w_rps):
    logp = F.log_softmax(z, -1)
    t = b["target"]
    ce = -(t * logp).sum(-1)
    rps = (((logp.exp().cumsum(-1) - t.cumsum(-1)) ** 2) * valid).sum(-1) / (b["k"] - 1).clamp(min=1)
    return (ce + w_rps * rps * (b["qtype"] == QTYPES["score"])).mean(), ce.mean()


def to_dev(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


@torch.no_grad()
def evaluate(backbone, readout, loader, dev):
    backbone.eval()
    out = []
    for b in loader:
        if b is None:
            continue
        b = to_dev(b, dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            z, _ = forward(backbone, readout, b)
        for i in range(z.shape[0]):
            out.append((int(b["qtype"][i]), z[i, :int(b["k"][i])].cpu(), int(b["label"][i])))
    backbone.train()
    return out


def report(evals):
    by = {}
    for qt, z, y in evals:
        by.setdefault(qt, []).append(float(int(z.argmax()) == y))
    names = {v: k for k, v in QTYPES.items()}
    acc = {names[t]: round(sum(v) / len(v), 4) for t, v in sorted(by.items())}
    acc["all"] = round(sum(sum(v) for v in by.values()) / max(1, sum(len(v) for v in by.values())), 4)
    return acc


def fit_temperature(evals):
    """One temperature for every question (the pplx layout has one), minimizing the labels' NLL."""
    grid = torch.arange(0.5, 5.0001, 0.01)
    nll = torch.zeros(len(grid))
    for _, z, y in evals:
        nll += torch.stack([F.cross_entropy(z[None] / T, torch.tensor([y])) for T in grid])
    return round(float(grid[int(nll.argmin())]), 4)


def save(backbone, readout, snapshot, base, base_dir, temperature, out):
    os.makedirs(out, exist_ok=True)
    backbone.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    save_file({"weight": readout.weight.detach().to(torch.bfloat16).cpu().contiguous()},
              os.path.join(out, "readout.safetensors"))
    for name in TEACHER_FILES:
        shutil.copy(os.path.join(snapshot, name), os.path.join(out, name))
    with open(os.path.join(snapshot, "decision_config.json")) as f:
        cfg = json.load(f)
    cfg.update(base_model=base, revision=os.path.basename(os.path.normpath(base_dir)), temperature=temperature)
    with open(os.path.join(out, "decision_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--teacher-dir", required=True, help="the pplx-decider-flash-v1.2 snapshot (protocol, processor)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--teacher", help="teacher label files, comma-separated (the first label of an id wins)")
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of gold vs teacher when a question has both")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--tokens", type=int, default=16384, help="padded tokens per batch")
    ap.add_argument("--max-len", type=int, default=4096, help="tokens per sequence; longer requests are bisected")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--w-rps", type=float, default=1.0)
    ap.add_argument("--limit", type=int, help="the first N training rows (a pilot)")
    ap.add_argument("--val-limit", type=int)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    model_mod, protocol = pplx(a.teacher_dir)
    with open(os.path.join(a.teacher_dir, "decision_config.json")) as f:
        tcfg = json.load(f)
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    rows, skipped = load_rows(a.data, a.teacher, a.alpha, protocol, tcfg["codes"], a.limit)
    val, _ = load_rows(a.val, None, 1.0, protocol, tcfg["codes"], a.val_limit)
    print("train states %d (questions skipped %d), val %d" % (len(rows), skipped, len(val)), flush=True)

    pad_id = AutoTokenizer.from_pretrained(a.teacher_dir).pad_token_id
    col = functools.partial(collate, pad_id=pad_id, left_pad=protocol.left_pad)
    loader = lambda rs, batches, shuffle: torch.utils.data.DataLoader(  # noqa: E731
        Requests(rs, a.teacher_dir, tcfg["codes"], a.max_len, shuffle), batch_sampler=batches, collate_fn=col,
        num_workers=a.workers, persistent_workers=False, pin_memory=dev.type == "cuda")
    val_loader = loader(val, length_batches(val, a.tokens, 0), False)

    base_dir = snapshot_download(a.base)
    backbone = Qwen3_5Model.from_pretrained(base_dir, dtype=torch.float32, attn_implementation="sdpa")
    backbone.visual.requires_grad_(False)  # text only; kept so the checkpoint has upstream's layout
    readout = torch.nn.Linear(backbone.config.text_config.hidden_size, protocol.MAX_OPTIONS, bias=False)
    with torch.no_grad():
        readout.weight.copy_(backbone.language_model.embed_tokens.weight[tcfg["token_ids"]])
    model_mod.enable_noncausal_full_attention(backbone.language_model)
    if a.grad_ckpt:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone.to(dev).train()
    readout.to(dev)

    params = [p for p in backbone.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim >= 2]
    opt = torch.optim.AdamW([{"params": decay, "lr": a.lr, "weight_decay": a.wd},
                             {"params": [p for p in params if p.ndim < 2], "lr": a.lr, "weight_decay": 0.0},
                             {"params": readout.parameters(), "lr": a.lr_head, "weight_decay": 0.0}],
                            fused=dev.type == "cuda")
    per_epoch = len(length_batches(rows, a.tokens, 0))
    total = max(1, int(per_epoch * a.epochs))
    warm = max(1, int(total * a.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    step, t0, seen, epoch = 0, time.time(), 0, 0
    print(json.dumps({"step": 0, "val": report(evaluate(backbone, readout, val_loader, dev))}), flush=True)
    while step < total:
        for b in loader(rows, length_batches(rows, a.tokens, a.seed + epoch), True):
            if b is None:
                continue
            b = to_dev(b, dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                z, valid = forward(backbone, readout, b)
            loss, ce = losses(z, valid, b, a.w_rps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params + list(readout.parameters()), 1.0)
            opt.step()
            sched.step()
            step += 1
            seen += int(b["attention_mask"].sum())
            if step % 50 == 0:
                m = (time.time() - t0) / 60
                print(json.dumps({"step": step, "of": total, "loss": round(loss.item(), 5), "ce": round(ce.item(), 5),
                                  "lr": sched.get_last_lr()[0], "tok_s": round(seen / (m * 60)), "min": round(m, 1)}),
                      flush=True)
            if step % a.eval_every == 0:
                print(json.dumps({"step": step, "val": report(evaluate(backbone, readout, val_loader, dev))}),
                      flush=True)
            if step >= total:
                break
        epoch += 1

    evals = evaluate(backbone, readout, val_loader, dev)
    temperature = fit_temperature(evals)
    print(json.dumps({"step": step, "val": report(evals), "temperature": temperature}), flush=True)
    save(backbone, readout, a.teacher_dir, a.base, base_dir, temperature, a.out)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
