# SPDX-License-Identifier: Apache-2.0
"""Train one demo adapter with PEFT: loss on the single output token only.

Adapted from ``peft/examples/alora_finetuning/alora_finetuning.py``. The prompt
is assembled exactly like the demo's inference prompt (:class:`policy.PromptBuilder`,
base-model form): system prompt, the row's windowed 5 Hz history (rebuilt from
the match's history stream), the current state, the assistant header. The loss
is cross-entropy at the last position over the adapter's output vocabulary.
Labels are the teacher's whole distribution when it has one (``soft``, from the
RL teacher; this is KL to the teacher up to a constant), else one-hot.

Adapters and their labels (rows from ``collect.py``):

* ``fighter`` / ``cautious`` / ``collector``: the teacher's move, every tic.
* ``arms``: the teacher's weapon slot, planner tics only.
* ``critic``: ``low`` / ``mid`` / ``high``, the outcome of the next second.
* ``probe``: where the last enemy in the history was (history-only question;
  used for the aLoRA-vs-LoRA comparison, not composed into the demo).
* ``router``: rows from ``router_data.py`` (instruction, label); no history.

``--kind lora`` trains a plain LoRA on the same data and prompts: the baseline
the aLoRA is compared against. ``--kind sr`` trains a Shadow Residual adapter
(github.ibm.com/generative-computing/shadow-residual, its ``src`` on
``PYTHONPATH``): a second, adapter stream through every layer that reads the
frozen base stream's K/V, with LoRA on Q, O and the MLP (never K/V) plus a
per-layer low-rank cross-stream from the base. The adapter never writes K/V, so
with a one-token answer only the last position's adapter stream matters, and
training it always-active is exactly inference with the control token on the
last prompt token. Settings follow IBM's shipped aLoRAs: rank 32 on every
linear layer (for SR, every linear layer but K and V).

::

    python train_alora.py --adapter fighter --data data/d0 data/d1 \
        --base /path/granite-4.1-3b --out runs/d1/fighter
    python train_alora.py --adapter router --data data/router/train.jsonl \
        --eval-data data/router/heldout.jsonl --epochs 4 --out runs/router/router

Writes the PEFT adapter to ``--out``, ``metrics.json`` and
``heldout_preds.jsonl`` (history ids, state and the PEFT argmax per held-out
row), which ``build_model.py verify`` compares against the composed checkpoint.
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

from history import PROBE_WORDS, History
from policy import (
    _EOR,
    ARMS,
    CRITIC,
    DANGER_LEVELS,
    OUTPUTS,
    ROUTER,
    ROUTER_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    PromptBuilder,
    alora_invocation_ids,
    output_token_ids,
    route_token_ids,
)

PROBE = "probe"
DEFAULT_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
SR_CROSS = "cross_stream"
VOCAB = {**OUTPUTS, PROBE: PROBE_WORDS}


# ── Data ───────────────────────────────────────────────────────────────────────
class Stream:
    """One match's history entries, tokenized once; ``ids(n)`` is the windowed
    history after the first ``n`` entries (the same rule as History.append)."""

    def __init__(self, entries: list[str], tok):
        h = History(tok)
        self.starts, self.entry_ids = [0], []
        for e in entries:
            h.append(e)
            self.starts.append(len(self.entry_ids) + 1 - len(h.entries))
            self.entry_ids.append(h.entry_ids[-1])

    def ids(self, n: int) -> list[int]:
        return [i for e in self.entry_ids[self.starts[n] : n] for i in e]


def row_target(r: dict, adapter: str) -> dict[str, float] | None:
    """The adapter's label distribution for one collect.py row, or None to skip."""
    if adapter == ARMS:
        if r.get("weapon") is None:
            return None
        return r.get("weapon_soft") or {r["weapon"]: 1.0}
    if adapter == CRITIC:
        return {r["critic"]: 1.0}
    if adapter == PROBE:
        return {r["probe"]: 1.0} if r.get("probe") else None
    return r.get("soft") or {r["expert"]: 1.0}


def load_game_rows(dirs: list[Path], adapter: str, styles: list[str], tok, every: int):
    """Rows as (stream, hist_n, state, target, episode key); history streams are
    tokenized once per match."""
    out, streams = [], {}
    for d in dirs:
        for style in styles:
            hp, rp = d / f"{style}_history.jsonl", d / f"{style}.jsonl"
            if not rp.exists():
                continue
            for line in open(hp):
                h = json.loads(line)
                streams[(str(d), style, h["ep"])] = Stream(h["entries"], tok)
            for line in open(rp):
                r = json.loads(line)
                if r["t"] % every:
                    continue
                target = row_target(r, adapter)
                if target is None:
                    continue
                key = (str(d), style, r["ep"])
                out.append((streams[key], r["hist_n"], r["state"], target, key))
    return out


def load_router_rows(paths: list[Path]):
    out = []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            out.append(
                (None, 0, r["text"], {r["label"]: 1.0}, (str(p), r.get("ep", 0)))
            )
    return out


def split_by_episode(rows: list, val_frac: float, seed: int) -> tuple[list, list]:
    """Hold out whole matches: consecutive tics are near-duplicates."""
    eps = sorted({r[4] for r in rows})
    random.Random(seed).shuffle(eps)
    n_val = max(1, int(len(eps) * val_frac)) if len(eps) > 1 else 0
    val_eps = set(eps[:n_val])
    return [r for r in rows if r[4] not in val_eps], [
        r for r in rows if r[4] in val_eps
    ]


def batches(items: list, size: int, shuffle: bool, seed: int):
    idx = list(range(len(items)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), size):
        yield [items[j] for j in idx[i : i + size]]


# ── Model helpers ──────────────────────────────────────────────────────────────
def collate(prompts: list[list[int]], targets: list[list[float]], pad_id: int, device):
    """Left-pad so every prompt ends at the last position; explicit position ids
    keep RoPE identical to the unpadded prompt the engine sees at inference."""
    import torch

    n = max(len(p) for p in prompts)
    input_ids = torch.full((len(prompts), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(prompts), n), dtype=torch.long)
    for i, p in enumerate(prompts):
        input_ids[i, n - len(p) :] = torch.tensor(p)
        mask[i, n - len(p) :] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    tgt = torch.tensor(targets, dtype=torch.float32)
    return input_ids.to(device), mask.to(device), pos.to(device), tgt.to(device)


def last_logits(model, input_ids, mask, pos, allowed):
    out = model(
        input_ids=input_ids, attention_mask=mask, position_ids=pos, logits_to_keep=1
    )
    return out.logits[:, -1, :][:, allowed].float()


def auc(scores: list[float], positives: list[bool]) -> float | None:
    """Rank AUC (probability a random positive outscores a random negative)."""
    pos = [s for s, y in zip(scores, positives) if y]
    neg = [s for s, y in zip(scores, positives) if not y]
    if not pos or not neg:
        return None
    ranked = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum, i = 0.0, 0
    while i < len(ranked):  # average ranks over ties
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        r = (i + j + 1) / 2
        rank_sum += r * sum(y for _, y in ranked[i:j])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapter", required=True, choices=sorted(VOCAB))
    ap.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="collect.py dirs (router: jsonl)",
    )
    ap.add_argument("--eval-data", type=Path, nargs="*", help="Explicit held-out set")
    ap.add_argument(
        "--styles",
        nargs="+",
        help="Row files to read (default: the adapter's own, "
        "or every style for arms/critic/probe)",
    )
    ap.add_argument("--kind", choices=("alora", "lora", "sr"), default="alora")
    ap.add_argument(
        "--cross-rank", type=int, default=32, help="SR: rank of the cross-stream"
    )
    ap.add_argument(
        "--init",
        type=Path,
        help="Warm-start from this adapter (e.g. the previous DAgger round's); "
        "its config (kind, rank, targets) is used as saved",
    )
    ap.add_argument("--base", default="ibm-granite/granite-4.1-3b")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=32)
    # alpha = rank (scale 1) and lr 1e-4: at alpha 64 / lr 2e-4 a rank-32
    # all-linear adapter collapsed to the label marginal mid-run.
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--micro", type=int, default=8, help="Micro-batch (memory)")
    ap.add_argument("--every", type=int, default=3, help="Keep every k-th tic")
    ap.add_argument("--max-examples", type=int, default=150_000)
    ap.add_argument("--max-heldout", type=int, default=4000)
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument(
        "--keep-best",
        action="store_true",
        help="Also save the adapter with the lowest held-out soft CE (to <out>/best)",
    )
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.base)
    a = args.adapter
    if a == ROUTER:
        label_ids = route_token_ids(tok)
        pb = PromptBuilder(tok, ROUTER_SYSTEM_PROMPT, align=False)
        rows = load_router_rows(args.data)
        val_rows = load_router_rows(args.eval_data) if args.eval_data else None
    else:
        label_ids = output_token_ids(tok, VOCAB[a])
        pb = PromptBuilder(tok, SYSTEM_PROMPT)
        styles = args.styles or (
            [a]
            if a in OUTPUTS and OUTPUTS[a] == OUTPUTS["fighter"]
            else ["fighter", "cautious", "collector"]
        )
        rows = load_game_rows(args.data, a, styles, tok, args.every)
        val_rows = (
            load_game_rows(args.eval_data, a, styles, tok, args.every)
            if args.eval_data
            else None
        )
    classes = list(label_ids)
    allowed = torch.tensor(list(label_ids.values()), device=device)
    if val_rows is None:
        train_rows, val_rows = split_by_episode(rows, args.val_frac, args.seed)
    else:
        train_rows = rows
    rng = random.Random(args.seed)
    if len(train_rows) > args.max_examples:
        train_rows = rng.sample(train_rows, args.max_examples)
    if len(val_rows) > args.max_heldout:
        val_rows = rng.sample(val_rows, args.max_heldout)

    def prompt(r) -> list[int]:
        stream, n, state = r[0], r[1], r[2]
        if stream is None:
            return pb.ids(state, None)
        return pb.game_ids(stream.ids(n), state, [None])[0]

    def target(r) -> list[float]:
        t = [r[3].get(c, 0.0) for c in classes]
        s = sum(t)
        return [x / s for x in t]

    hard = Counter(max(r[3], key=r[3].get) for r in train_rows)
    lens = [len(prompt(r)) for r in train_rows[:500]]
    print(
        f"{a} ({args.kind}): {len(train_rows)} train / {len(val_rows)} held-out rows, "
        f"prompt {min(lens)}-{max(lens)} tokens (first 500); "
        f"label mix {hard.most_common(6)}",
        flush=True,
    )

    if args.kind == "sr":
        if args.init is not None:
            raise SystemExit("--init is not supported for sr yet")
        from shadow_residual.training.factory import get_shadow_residual_peft_model

        # K/V always come from the base stream: no LoRA can land there.
        targets = [t for t in args.targets.split(",") if t not in ("k_proj", "v_proj")]
        cfg = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            target_modules=[*targets, SR_CROSS],
            rank_pattern={SR_CROSS: args.cross_rank},
            bias="none",
        )
        model = get_shadow_residual_peft_model(
            args.base, cfg, torch_dtype=torch.bfloat16
        ).to(device)
        if args.grad_ckpt:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.enable_input_require_grads()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base, dtype=torch.bfloat16
        ).to(device)
        if args.grad_ckpt:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        if args.init is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(args.init), is_trainable=True)
        else:
            extra = (
                {"alora_invocation_tokens": alora_invocation_ids(tok)}
                if args.kind == "alora"
                else {}
            )
            cfg = LoraConfig(
                task_type="CAUSAL_LM",
                r=args.rank,
                lora_alpha=args.alpha,
                lora_dropout=args.dropout,
                target_modules=args.targets.split(","),
                bias="none",
                **extra,
            )
            model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def evaluate(items: list) -> tuple[dict, list[list[float]]]:
        model.eval()
        probs: list[list[float]] = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for b in batches(items, args.micro, False, 0):
                ids, m, p, _ = collate(
                    [prompt(r) for r in b], [target(r) for r in b], pad_id, device
                )
                probs += torch.softmax(
                    last_logits(model, ids, m, p, allowed), -1
                ).tolist()
        model.train()
        tg = [target(r) for r in items]
        gold = [max(range(len(classes)), key=t.__getitem__) for t in tg]
        pred = [max(range(len(classes)), key=q.__getitem__) for q in probs]
        ce = sum(
            -sum(t_i * math.log(max(q_i, 1e-9)) for t_i, q_i in zip(t, q))
            for t, q in zip(tg, probs)
        ) / max(1, len(items))
        maj = Counter(gold).most_common(1)[0]
        ev = {
            "n": len(items),
            "acc": round(
                sum(g == q for g, q in zip(gold, pred)) / max(1, len(items)), 4
            ),
            "soft_ce": round(ce, 4),
            "majority_class": classes[maj[0]],
            "majority_baseline": round(maj[1] / max(1, len(items)), 4),
            "per_class": {
                c: {
                    "n": sum(g == ci for g in gold),
                    "acc": round(
                        sum(g == q == ci for g, q in zip(gold, pred))
                        / max(1, sum(g == ci for g in gold)),
                        4,
                    ),
                }
                for ci, c in enumerate(classes)
                if any(g == ci for g in gold)
            },
        }
        if a == CRITIC:
            hi, lo = DANGER_LEVELS.index("high"), DANGER_LEVELS.index("low")
            ev["auc_damage_1s"] = auc(
                [1 - q[classes.index(DANGER_LEVELS[lo])] for q in probs],
                [g != classes.index("low") for g in gold],
            )
            ev["auc_high"] = auc(
                [q[classes.index(DANGER_LEVELS[hi])] for q in probs],
                [g == classes.index("high") for g in gold],
            )
        return ev, probs

    steps_per_epoch = math.ceil(len(train_rows) / args.batch)
    total = max(1, int(steps_per_epoch * args.epochs))
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / warm)
        * 0.5
        * (1 + math.cos(math.pi * min(1.0, s / total))),
    )
    step, t0, run_loss, history = 0, time.time(), 0.0, []
    train_curve: list[dict] = []  # training loss, every 50 steps
    best = {"soft_ce": math.inf, "step": None}
    model.train()
    epoch = 0
    while step < total:
        for b in batches(train_rows, args.batch, True, args.seed + epoch):
            for k in range(0, len(b), args.micro):
                mb = b[k : k + args.micro]
                ids, m, p, tgt = collate(
                    [prompt(r) for r in mb], [target(r) for r in mb], pad_id, device
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logp = torch.log_softmax(last_logits(model, ids, m, p, allowed), -1)
                    loss = -(tgt * logp).sum(-1).mean() * len(mb) / len(b)
                loss.backward()
                run_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print(
                    f"step {step}/{total} loss {run_loss / 50:.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s",
                    flush=True,
                )
                train_curve.append({"step": step, "loss": round(run_loss / 50, 5)})
                run_loss = 0.0
            if step % args.eval_every == 0 and val_rows:
                ev, _ = evaluate(val_rows[:1000])
                history.append({"step": step, **{k: ev[k] for k in ("acc", "soft_ce")}})
                if args.keep_best and ev["soft_ce"] < best["soft_ce"]:
                    best = {"soft_ce": ev["soft_ce"], "acc": ev["acc"], "step": step}
                    model.save_pretrained(str(args.out / "best"))
                print(
                    f"  held-out acc {ev['acc']:.4f} soft CE {ev['soft_ce']:.4f} "
                    f"(majority {ev['majority_baseline']:.4f})",
                    flush=True,
                )
            if step >= total:
                break
        epoch += 1

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    if args.kind == "sr":
        # What the Granite Switch composer reads to place an SR control token:
        # it replaces this single token, the last one of the prompt.
        path = args.out / "adapter_config.json"
        acfg = json.loads(path.read_text())
        acfg["last_context_token"] = _EOR
        acfg["last_context_token_id"] = tok.encode(_EOR, add_special_tokens=False)[0]
        acfg["share_moe_routing"] = True  # no-op on a dense base; recorded by SR
        path.write_text(json.dumps(acfg, indent=2))
    ev, probs = evaluate(val_rows)
    with open(args.out / "heldout_preds.jsonl", "w") as f:
        for r, q in zip(val_rows, probs):
            stream, n, state = r[0], r[1], r[2]
            f.write(
                json.dumps(
                    {
                        "history_ids": stream.ids(n) if stream is not None else None,
                        "state": state,
                        "label": max(r[3], key=r[3].get),
                        "peft": classes[max(range(len(q)), key=q.__getitem__)],
                        "peft_probs": {c: round(x, 5) for c, x in zip(classes, q)},
                    }
                )
                + "\n"
            )
    metrics = {
        "adapter": a,
        "kind": args.kind,
        "args": {k: str(v) for k, v in vars(args).items()},
        "train_examples": len(train_rows),
        "steps": step,
        "seconds": round(time.time() - t0, 1),
        "heldout": ev,
        "history": history,
        "train_curve": train_curve,
        "best": best if args.keep_best else None,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(
        f"\n{a} ({args.kind}): held-out acc {ev['acc']:.4f} soft CE {ev['soft_ce']:.4f} "
        f"vs majority {ev['majority_baseline']:.4f} ({ev['majority_class']})"
        + (
            f"; AUC damage-in-1s {ev['auc_damage_1s']:.3f}"
            if a == CRITIC and ev.get("auc_damage_1s")
            else ""
        )
        + f"; saved -> {args.out}"
    )


if __name__ == "__main__":
    main()
