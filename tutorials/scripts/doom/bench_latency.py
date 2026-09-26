# SPDX-License-Identifier: Apache-2.0
"""Per-decision latency of the composed checkpoint on real Doom prompts.

Decision latency is wall clock from "state is available" to "action is
available": prompt-id assembly plus one vLLM engine step, which prefills the
fresh state tokens and emits one action token. Scenarios, all in one engine:

* ``single``: one game, one decision per call, adapters cycling (the demo).
* ``decode``: reference cost of a plain decode step on the same engine.
* ``batch``: N games deciding in the same engine step (N = 1, 2, 4, ...), and
  how many games fit in one 35 Hz Doom tic.
* ``shadow``: all three behaviors on the same state in one step.
* ``router``: one-token instruction routing.
* ``live``: the policy actually plays, unthrottled (turbo), including env time.

Examples::

    python bench_latency.py --model models/standin-switch --json out/bench.json
    python bench_latency.py --model models/standin-switch --engine-loop
    python bench_latency.py --model models/standin-switch --no-prefix-cache
    python bench_latency.py --model /path/granite-4.1-3b --plain-base   # no SWITCH
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doom_env import TIC_MS, DoomEnv
from expert import BEHAVIORS, Expert
from policy import VLLMPolicy

INSTRUCTIONS = [
    "go kill everything",
    "stop fighting and grab health",
    "pick up all the ammo and armor you can find",
    "play it safe for a while",
    "hunt them down",
]


def pct(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 3),
        "p50": round(float(np.percentile(a, 50)), 3),
        "p90": round(float(np.percentile(a, 90)), 3),
        "p99": round(float(np.percentile(a, 99)), 3),
        "max": round(float(a.max()), 3),
    }


def expert_states(episodes: int, seed: int) -> list[str]:
    """Consecutive state texts from expert play: the real prompt distribution."""
    env = DoomEnv(seed=seed, hud=True, resolution="640X480", timeout_tics=60 * 35)
    ex, texts = Expert(), []
    for ep in range(episodes):
        b = BEHAVIORS[ep % len(BEHAVIORS)]
        obs = env.reset(seed=seed + ep)
        ex.reset()
        while not obs.done:
            if not obs.dead:
                texts.append(obs.text)
            obs = env.step(ex.act(obs, b), weapon=ex.weapon(obs))
    env.close()
    return texts


def bench_single(pol: VLLMPolicy, texts: list[str], n: int) -> dict:
    ms, build, engine, fresh, cached = [], [], [], [], []
    for i in range(n):
        d = pol.decide(texts[i % len(texts)], BEHAVIORS[(i // 50) % len(BEHAVIORS)])
        ms.append(d.ms)
        build.append(d.build_ms)
        engine.append(d.engine_ms)
        fresh.append(d.fresh_tokens)
        cached.append(d.cached_tokens)
    tot = sum(fresh) + sum(cached)
    return {
        "decision_ms": pct(ms),
        "build_ms": pct(build),
        "engine_ms": pct(engine),
        "fresh_tokens_mean": round(float(np.mean(fresh)), 1),
        "cached_tokens_mean": round(float(np.mean(cached)), 1),
        "prefix_cache_hit_rate": round(sum(cached) / max(1, tot), 3),
        "decisions_per_s": round(1000.0 / float(np.mean(ms)), 1),
        "within_tic_budget": round(float(np.mean(np.asarray(ms) < TIC_MS)), 4),
    }


def bench_decode(pol: VLLMPolicy, n_tokens: int = 64, reps: int = 10) -> dict:
    """Plain decode steps on this engine, for reference against a decision."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    engine = pol.llm.llm_engine
    sp = SamplingParams(max_tokens=n_tokens, ignore_eos=True, temperature=0.0)
    prompt = pol.pb.ids(
        "hp 100 armor 0 | pistol 50 | arms 2:50 | see nothing | wall l9 f9 r9 b9 | hit 0",
        None,
    )
    steps = []
    for r in range(reps):
        engine.add_request(
            f"decode-{r}-{time.time()}", TokensPrompt(prompt_token_ids=prompt), sp
        )
        first, done = True, False
        while not done:
            t = time.perf_counter()
            outs = engine.step()
            dt = (time.perf_counter() - t) * 1000
            if not first:
                steps.append(dt)
            first = False
            done = any(o.finished for o in outs)
    return {"decode_step_ms": pct(steps)}


def bench_batch(pol: VLLMPolicy, texts: list[str], sizes: list[int], reps: int) -> dict:
    rng = random.Random(0)
    out = {}
    for b in sizes:
        ms = []
        for r in range(reps):
            start = rng.randrange(len(texts) - b)
            batch = texts[start : start + b]
            advs = [rng.choice(BEHAVIORS) for _ in batch]
            ms.append(pol.decide_batch(batch, advs)[0].ms)
        stats = pct(ms)
        stats["games_at_35hz"] = b if stats["p99"] < TIC_MS else 0
        out[str(b)] = stats
        print(
            f"  batch {b:>3}: p50 {stats['p50']:6.2f} ms  p99 {stats['p99']:6.2f} ms",
            flush=True,
        )
    return out


def bench_shadow(pol: VLLMPolicy, texts: list[str], n: int) -> dict:
    ms = [
        pol.decide_batch([t] * len(BEHAVIORS), list(BEHAVIORS))[0].ms for t in texts[:n]
    ]
    return {"three_behaviors_ms": pct(ms)}


def bench_router(pol: VLLMPolicy, n: int) -> dict:
    routes = [pol.route(INSTRUCTIONS[i % len(INSTRUCTIONS)]) for i in range(n)]
    return {"route_ms": pct([r.ms for r in routes])}


def bench_live(pol: VLLMPolicy, seconds: float, seed: int) -> dict:
    """Unthrottled play: the model sets the pace. Env time is reported separately."""
    env = DoomEnv(seed=seed, hud=True, resolution="640X480")
    obs = env.reset(seed=seed)
    dec_ms, env_ms, n, t_end = [], [], 0, time.perf_counter() + seconds
    t_start = time.perf_counter()
    while time.perf_counter() < t_end:
        if obs.done:
            obs = env.reset()
        if obs.dead:
            obs = env.step("wait")
            continue
        d = pol.decide(obs, BEHAVIORS[(n // 350) % len(BEHAVIORS)])
        t = time.perf_counter()
        obs = env.step(d.action)
        env_ms.append((time.perf_counter() - t) * 1000)
        dec_ms.append(d.ms)
        n += 1
    wall = time.perf_counter() - t_start
    env.close()
    return {
        "decision_ms": pct(dec_ms),
        "env_step_ms": pct(env_ms),
        "tics": n,
        "tics_per_s": round(n / wall, 1),
        "x_realtime": round(n / wall / 35.0, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--plain-base",
        action="store_true",
        help="--model is a plain base checkpoint (no control tokens): isolates SWITCH overhead",
    )
    ap.add_argument(
        "--engine-loop",
        action="store_true",
        help="Drive LLMEngine.add_request/step instead of LLM.generate",
    )
    ap.add_argument("--no-prefix-cache", action="store_true")
    ap.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs (for comparison)",
    )
    ap.add_argument(
        "--cudagraph-mode",
        default="FULL",
        choices=("FULL", "FULL_AND_PIECEWISE", "PIECEWISE"),
    )
    ap.add_argument(
        "--n", type=int, default=3000, help="Decisions in the single-game scenario"
    )
    ap.add_argument(
        "--episodes", type=int, default=3, help="Expert episodes to draw prompts from"
    )
    ap.add_argument("--batch-sizes", default="1,2,4,8,16")
    ap.add_argument("--live-seconds", type=float, default=20.0)
    ap.add_argument("--scenarios", default="single,decode,batch,shadow,router,live")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    scen = set(args.scenarios.split(","))
    sizes = [int(x) for x in args.batch_sizes.split(",")]
    t0 = time.time()
    texts = expert_states(args.episodes, args.seed)
    print(f"{len(texts)} expert states in {time.time() - t0:.1f}s", flush=True)

    pol = VLLMPolicy(
        args.model,
        engine_loop=args.engine_loop,
        prefix_caching=not args.no_prefix_cache,
        max_num_seqs=max(8, max(sizes) * 3),
        enforce_eager=args.enforce_eager,
        cudagraph_mode=args.cudagraph_mode,
        base_model=args.plain_base,
        warmup=200,
    )
    import torch
    import vllm

    res: dict = {
        "config": {
            "model": args.model,
            "plain_base": args.plain_base,
            "engine_loop": args.engine_loop,
            "prefix_caching": not args.no_prefix_cache,
            "enforce_eager": args.enforce_eager,
            "cudagraph_mode": args.cudagraph_mode,
            "gpu": torch.cuda.get_device_name(0),
            "vllm": vllm.__version__,
            "torch": torch.__version__,
        }
    }
    runs = [
        ("single", lambda: bench_single(pol, texts, args.n)),
        ("decode", lambda: bench_decode(pol)),
        ("batch", lambda: bench_batch(pol, texts, sizes, reps=200)),
        ("shadow", lambda: bench_shadow(pol, texts, 1000)),
        ("router", lambda: bench_router(pol, 200)),
        ("live", lambda: bench_live(pol, args.live_seconds, args.seed + 99)),
    ]
    for name, fn in runs:
        if name not in scen or (args.plain_base and name in ("shadow", "router")):
            continue
        print(f"\n== {name}", flush=True)
        res[name] = fn()
        print(json.dumps(res[name], indent=1), flush=True)

    if "single" in res:
        s = res["single"]["decision_ms"]
        print(
            f"\nSUMMARY decision p50 {s['p50']:.2f} ms  p99 {s['p99']:.2f} ms  "
            f"({res['single']['decisions_per_s']} decisions/s; tic budget {TIC_MS:.1f} ms; "
            f"gate p50 <= 20 ms: {'PASS' if s['p50'] <= 20 else 'FAIL'})"
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
