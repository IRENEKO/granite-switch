# SPDX-License-Identifier: Apache-2.0
"""Router data: instructions a player might type, labeled with a behavior.

A larger local Granite model, run in-process with vLLM so no API keys are
needed, writes paraphrases of each play style. The hand-written held-out
instructions below are never shown to it::

    python router_data.py --model ibm-granite/granite-4.1-8b --per-behavior 300 \
        --out data/router

``--offline`` builds a smaller template-based set without a model, which is
enough to exercise the pipeline on a laptop.

Writes ``train.jsonl`` (generated plus seed instructions) and ``heldout.jsonl``
(hand-written only), with rows ``{"text", "label", "ep"}``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from expert import BEHAVIORS

STYLE = {
    "hunter": "aggressive: seek out monsters, attack and kill them; only heal when nearly dead",
    "survivor": "cautious: stay alive, avoid fights, keep distance from monsters, pick up health and armor",
    "scavenger": "collector: gather items, ammo, armor and weapons; only shoot monsters that get very close",
}

SEEDS = {
    "hunter": [
        "go kill everything",
        "hunt them down",
        "attack!",
        "clear the room",
        "be aggressive",
        "shoot every monster you see",
        "go on a rampage",
        "take the fight to them",
        "rip and tear",
        "frag them all",
        "chase down the zombies",
        "no mercy",
        "find something to kill",
        "aggressive mode",
        "fight fight fight",
    ],
    "survivor": [
        "stay alive",
        "play it safe",
        "stop fighting and grab health",
        "avoid the monsters",
        "keep your distance",
        "don't die",
        "heal up",
        "be careful",
        "retreat and recover",
        "defensive mode",
        "run away from them",
        "you're hurt, find a medikit",
        "no risks",
        "survive as long as you can",
        "back off",
    ],
    "scavenger": [
        "collect all the loot",
        "grab the ammo",
        "pick up everything",
        "go get the armor",
        "find weapons",
        "loot the place",
        "gather supplies",
        "get the items",
        "stock up on ammo",
        "scavenge",
        "collect stuff and ignore the zombies",
        "pick up the shotgun shells",
        "go shopping",
        "sweep the room for items",
        "hoard everything",
    ],
}

# Hand-written, never shown to the generator: the router's eval set.
HELDOUT = {
    "hunter": [
        "i want carnage",
        "exterminate the demons",
        "time to go berserk",
        "hit them before they hit you",
        "wipe out the imps",
        "you're the predator now",
        "make them pay",
        "kill count, let's go up",
        "blast whatever moves",
        "seek and destroy",
    ],
    "survivor": [
        "you're almost dead, be smart",
        "lay low for a bit",
        "don't engage, just stay healthy",
        "your health matters more than kills",
        "hang back and patch yourself up",
        "play like a coward",
        "whatever you do, don't get hit",
        "keep away from the fighting",
        "safety first",
        "patch up, stay out of trouble",
    ],
    "scavenger": [
        "empty the map of goodies",
        "hoover up the pickups",
        "go treasure hunting",
        "we need more bullets",
        "grab every box you can find",
        "collector mode on",
        "fill your pockets",
        "ignore them, get the stuff",
        "armor up, pick up whatever is lying around",
        "resupply",
    ],
}

GEN_PROMPT = (
    "A player is typing short instructions to an AI bot that plays Doom. Write {n} different "
    "instructions that all ask the bot to play in this style: {style}.\n"
    "Vary the wording, length (2 to 15 words), tone and slang; include some indirect ones and a few "
    "with typos. Do not number them. One instruction per line, nothing else.\n"
    "Examples:\n{examples}"
)


def clean(line: str) -> str | None:
    t = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"').strip()
    n = len(t.split())
    if not t or n < 2 or n > 20 or len(t) > 120:
        return None
    return t


def offline(per: int, rng: random.Random) -> dict[str, list[str]]:
    """Template recombination of the seeds: a stand-in when no model is available."""
    pre = ["", "please ", "ok ", "now ", "hey, ", "just ", "yo ", "alright "]
    post = ["", " now", " please", "!", " for a while", " ok?", ", thanks", " asap"]
    out = {}
    for b, seeds in SEEDS.items():
        pool = {
            f"{rng.choice(pre)}{s}{rng.choice(post)}".strip()
            for s in seeds
            for _ in range(4)
        }
        out[b] = sorted(pool)[:per]
    return out


def generate(
    model: str, per: int, rng: random.Random, rounds: int
) -> dict[str, list[str]]:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85
    )
    sp = SamplingParams(
        temperature=1.0, top_p=0.95, max_tokens=700, seed=rng.randrange(1 << 30)
    )
    held = {t.lower() for ts in HELDOUT.values() for t in ts}
    out: dict[str, set[str]] = {b: set() for b in BEHAVIORS}
    for r in range(rounds):
        msgs, which = [], []
        for b in BEHAVIORS:
            if len(out[b]) >= per:
                continue
            for _ in range(8):
                ex = "\n".join(rng.sample(SEEDS[b], 5))
                msgs.append(
                    [
                        {
                            "role": "user",
                            "content": GEN_PROMPT.format(
                                n=25, style=STYLE[b], examples=ex
                            ),
                        }
                    ]
                )
                which.append(b)
        if not msgs:
            break
        for b, o in zip(which, llm.chat(msgs, sp, use_tqdm=False)):
            for line in o.outputs[0].text.splitlines():
                t = clean(line)
                if t and t.lower() not in held:
                    out[b].add(t)
        print(
            f"round {r}: " + ", ".join(f"{b} {len(v)}" for b, v in out.items()),
            flush=True,
        )
    return {b: sorted(v)[: per * 2] for b, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="ibm-granite/granite-4.1-8b")
    ap.add_argument("--per-behavior", type=int, default=300)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gen = (
        offline(args.per_behavior, rng)
        if args.offline
        else generate(args.model, args.per_behavior, rng, args.rounds)
    )
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [
        {"text": t, "label": b}
        for b in BEHAVIORS
        for t in [*gen[b][: args.per_behavior], *SEEDS[b]]
    ]
    rng.shuffle(rows)
    with open(args.out / "train.jsonl", "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({**r, "ep": i}) + "\n")
    with open(args.out / "heldout.jsonl", "w") as f:
        for i, (b, ts) in enumerate((b, ts) for b, ts in HELDOUT.items()):
            for j, t in enumerate(ts):
                f.write(
                    json.dumps({"text": t, "label": b, "ep": 10_000 + 100 * i + j})
                    + "\n"
                )
    counts = {b: sum(r["label"] == b for r in rows) for b in BEHAVIORS}
    print(
        f"train {len(rows)} {counts}; held-out {sum(map(len, HELDOUT.values()))} hand-written -> {args.out}"
    )


if __name__ == "__main__":
    main()
