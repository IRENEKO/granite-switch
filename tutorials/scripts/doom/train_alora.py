# SPDX-License-Identifier: Apache-2.0
"""Train one demo aLoRA with PEFT: loss on the single output token only.

Adapted from ``peft/examples/alora_finetuning/alora_finetuning.py``. Differences:
the prompt is assembled exactly like the demo's inference prompt
(:class:`policy.PromptBuilder`, base-model form), and the loss is cross-entropy
over the allowed output tokens at the last position. Every example has one
target token.

Behavior adapter, from collect.py rows (label = ``expert``)::

    python train_alora.py --adapter hunter --data data/round0/hunter.jsonl \
        data/round1/hunter.jsonl --base /path/granite-4.1-3b --out runs/r1/hunter

Router, from router_data.py rows (label = ``label``)::

    python train_alora.py --adapter router --data data/router/train.jsonl \
        --eval-data data/router/heldout.jsonl --epochs 4 --out runs/r1/router

Writes the PEFT adapter to ``--out``, plus ``metrics.json`` and
``heldout_preds.jsonl`` (text + PEFT argmax), which ``build_model.py verify``
compares against the composed checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from policy import (
    ROUTER,
    ROUTER_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    PromptBuilder,
    action_token_ids,
    alora_invocation_ids,
    route_token_ids,
)

DEFAULT_TARGETS = "q_proj,k_proj,v_proj,o_proj"


def load_rows(paths: list[Path], label_key: str) -> list[dict]:
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                rows.append(
                    {
                        "text": r["text"],
                        "label": r[label_key],
                        "ep": (str(p), r.get("ep", 0)),
                    }
                )
    return rows


def split_by_episode(rows: list[dict], val_frac: float, seed: int) -> tuple[list, list]:
    """Hold out whole episodes: consecutive tics are near-duplicates."""
    eps = sorted({r["ep"] for r in rows})
    random.Random(seed).shuffle(eps)
    n_val = max(1, int(len(eps) * val_frac)) if len(eps) > 1 else 0
    val_eps = set(eps[:n_val])
    return [r for r in rows if r["ep"] not in val_eps], [
        r for r in rows if r["ep"] in val_eps
    ]


def batches(items: list, size: int, shuffle: bool, seed: int):
    idx = list(range(len(items)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), size):
        yield [items[j] for j in idx[i : i + size]]


def collate(batch: list[tuple[list[int], int]], pad_id: int, device):
    """Left-pad so every prompt ends at the last position; explicit position ids
    keep RoPE identical to the unpadded prompt the engine sees at inference."""
    import torch

    n = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (ids, _) in enumerate(batch):
        input_ids[i, n - len(ids) :] = torch.tensor(ids)
        mask[i, n - len(ids) :] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    targets = torch.tensor([t for _, t in batch])
    return input_ids.to(device), mask.to(device), pos.to(device), targets.to(device)


def last_logits(model, input_ids, mask, pos, allowed):
    out = model(
        input_ids=input_ids, attention_mask=mask, position_ids=pos, logits_to_keep=1
    )
    return out.logits[:, -1, :][:, allowed].float()


def evaluate(
    model, items, bs, pad_id, device, allowed, classes
) -> tuple[dict, list[int]]:
    import torch

    model.eval()
    preds, correct = [], 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in batches(items, bs, False, 0):
            ids, m, p, t = collate(b, pad_id, device)
            pr = last_logits(model, ids, m, p, allowed).argmax(-1)
            correct += int((pr == t).sum())
            preds.extend(pr.tolist())
    model.train()
    gold = [t for _, t in items]
    maj = Counter(gold).most_common(1)[0]
    per = {}
    for ci, c in enumerate(classes):
        idx = [i for i, g in enumerate(gold) if g == ci]
        if idx:
            per[c] = {
                "n": len(idx),
                "acc": round(sum(preds[i] == ci for i in idx) / len(idx), 4),
            }
    return {
        "n": len(items),
        "acc": round(correct / max(1, len(items)), 4),
        "majority_class": classes[maj[0]],
        "majority_baseline": round(maj[1] / max(1, len(items)), 4),
        "per_class": per,
    }, preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--adapter", required=True, help="hunter | survivor | scavenger | router"
    )
    ap.add_argument("--data", type=Path, nargs="+", required=True)
    ap.add_argument(
        "--eval-data",
        type=Path,
        nargs="*",
        help="Explicit held-out set (default: split by episode)",
    )
    ap.add_argument("--base", default="ibm-granite/granite-4.1-3b")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument(
        "--max-examples", type=int, default=150_000, help="Subsample training rows"
    )
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.base)
    is_router = args.adapter == ROUTER
    label_key = "label" if is_router else "expert"
    if is_router:
        label_ids = route_token_ids(tok)
        pb = PromptBuilder(tok, ROUTER_SYSTEM_PROMPT)
    else:
        label_ids = action_token_ids(tok)
        pb = PromptBuilder(tok, SYSTEM_PROMPT)
    classes = list(label_ids)
    allowed = torch.tensor(list(label_ids.values()), device=device)

    rows = load_rows(args.data, label_key)
    if args.eval_data:
        train_rows, val_rows = rows, load_rows(args.eval_data, label_key)
    else:
        train_rows, val_rows = split_by_episode(rows, args.val_frac, args.seed)
    if len(train_rows) > args.max_examples:
        train_rows = random.Random(args.seed).sample(train_rows, args.max_examples)

    def encode(rs):
        return [(pb.ids(r["text"], None), classes.index(r["label"])) for r in rs]

    train, val = encode(train_rows), encode(val_rows)
    lens = [len(i) for i, _ in train]
    print(
        f"{args.adapter}: {len(train)} train / {len(val)} held-out examples, prompt "
        f"{min(lens)}-{max(lens)} tokens; label mix {Counter(t for _, t in train).most_common(5)}",
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).to(
        device
    )
    cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=args.targets.split(","),
        alora_invocation_tokens=alora_invocation_ids(tok),
        bias="none",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    steps_per_epoch = math.ceil(len(train) / args.batch)
    total = max(1, int(steps_per_epoch * args.epochs))
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0
    )
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / warm)
        * 0.5
        * (1 + math.cos(math.pi * min(1.0, s / total))),
    )

    step, t0, run_loss = 0, time.time(), 0.0
    history = []
    model.train()
    epoch = 0
    while step < total:
        for b in batches(train, args.batch, True, args.seed + epoch):
            ids, m, p, t = collate(b, pad_id, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(
                    last_logits(model, ids, m, p, allowed), t
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            run_loss += loss.item()
            if step % 50 == 0:
                print(
                    f"step {step}/{total} loss {run_loss / 50:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s",
                    flush=True,
                )
                run_loss = 0.0
            if step % args.eval_every == 0 and val:
                ev, _ = evaluate(
                    model, val[:4000], args.batch, pad_id, device, allowed, classes
                )
                history.append(
                    {"step": step, **{k: ev[k] for k in ("acc", "majority_baseline")}}
                )
                print(
                    f"  held-out acc {ev['acc']:.4f} (majority baseline {ev['majority_baseline']:.4f})",
                    flush=True,
                )
            if step >= total:
                break
        epoch += 1

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    ev, preds = evaluate(model, val, args.batch, pad_id, device, allowed, classes)
    with open(args.out / "heldout_preds.jsonl", "w") as f:
        for r, pr in zip(val_rows, preds):
            f.write(
                json.dumps(
                    {"text": r["text"], "label": r["label"], "peft": classes[pr]}
                )
                + "\n"
            )
    metrics = {
        "adapter": args.adapter,
        "args": {k: str(v) for k, v in vars(args).items()},
        "train_examples": len(train),
        "steps": step,
        "seconds": round(time.time() - t0, 1),
        "heldout": ev,
        "history": history,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(
        f"\n{args.adapter}: held-out acc {ev['acc']:.4f} vs majority baseline {ev['majority_baseline']:.4f} "
        f"({ev['majority_class']}); saved -> {args.out}"
    )


if __name__ == "__main__":
    main()
