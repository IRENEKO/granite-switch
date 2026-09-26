# SPDX-License-Identifier: Apache-2.0
"""Play deathmatches against bots in parallel ViZDoom envs; write jsonl + stats.

Scripted-expert matches, with DART-style noise for wider state coverage::

    python collect.py --policy expert --episodes 50 --timeout-s 120 --dart 0.15 \
        --out data/round0

Scripted-baseline evaluation, 10-minute matches against the default bots::

    python collect.py --policy expert --behaviors fighter --episodes 10 \
        --bots default --stats-only --out out/baseline_default

Student rollouts for DAgger. The student (composed checkpoint) drives, the
expert labels every visited state, and all envs step in lockstep so each
tic's student decisions are one batched vLLM call::

    python collect.py --policy vllm --model ./doom-switch --episodes 30 \
        --beta 0.0 --out data/round1

``--policy random`` drives the same lockstep path with uniform random actions.
It is a stat baseline, and a way to exercise the plumbing without a GPU.

Every policy chooses the weapon through the expert's planner every
``PLAN_EVERY_TICS`` (about 0.5 s). Tics while the player is dead need no
decision and produce no row.

Outputs in ``--out``:

* ``<behavior>.jsonl``: one row per decided tic, ``{"text", "expert", "act",
  "weapon", ...}``; ``expert`` is the training label, ``weapon`` the planner's
  slot on tics where it ran (else null).
* ``stats.jsonl``: one row per match.
* ``videos/<behavior>_ep<k>.mp4``: the first ``--record`` matches per behavior.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doom_env import ACTIONS, BOT_SETS, MATCH_TICS, TIC_HZ, DoomEnv
from expert import BEHAVIORS, PLAN_EVERY_TICS, Expert

RESOLUTION = "640X480"  # one setting everywhere: collection, bench and demo


class _Noise:
    """DART-style noise: bursts of random actions, labels stay the expert's."""

    def __init__(self, rate: float, rng: random.Random, max_len: int = 6):
        self.rate, self.rng, self.max_len = rate, rng, max_len
        self.left, self.action = 0, "wait"

    def __call__(self, action: str) -> str:
        if self.left == 0 and self.rate > 0 and self.rng.random() < self.rate / 3:
            self.left = self.rng.randint(1, self.max_len)
            self.action = self.rng.choice(ACTIONS)
        if self.left > 0:
            self.left -= 1
            return self.action
        return action


def _writer(path: Path):
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        str(path), fps=TIC_HZ, codec="libx264", quality=7, macro_block_size=1
    )


def _make_env(task: dict) -> DoomEnv:
    return DoomEnv(
        seed=task["seed"],
        resolution=RESOLUTION,
        hud=True,
        timeout_tics=task["timeout"],
        bots=task["bots"],
        n_bots=task["n_bots"],
    )


def _plan(expert: Expert, obs, behavior: str) -> int | None:
    """The weapon planner's slot on its cadence, else None (keep pressing the last)."""
    if obs.tick % PLAN_EVERY_TICS:
        return None
    return expert.weapon(obs, behavior)


def _stats(task: dict, env: DoomEnv, **extra) -> dict:
    return {
        "behavior": task["behavior"],
        "ep": task["ep"],
        "seed": task["seed"],
        "bots": task["bots"],
        "policy": task["policy"],
        **extra,
        **env.stats.as_dict(),
    }


# ── Expert path: each worker runs whole matches on its own ──────────────────────
def _expert_episode(task: dict) -> dict:
    env = _make_env(task)
    expert, rng = Expert(), random.Random(task["seed"])
    noise = _Noise(task["dart"], rng)
    b = task["behavior"]
    obs = env.reset(seed=task["seed"])
    rows, video = [], _writer(Path(task["video"])) if task["video"] else None
    while not obs.done:
        if video is not None:
            video.append_data(env.frame())
        if obs.dead:
            obs = env.step("wait")
            continue
        label = expert.act(obs, b)
        weapon = _plan(expert, obs, b)
        act = noise(label)
        if task["keep_rows"]:
            rows.append(
                {
                    "b": b,
                    "ep": task["ep"],
                    "t": obs.tick,
                    "text": obs.text,
                    "expert": label,
                    "act": act,
                    "weapon": weapon,
                }
            )
        obs = env.step(act, weapon=weapon)
    if video is not None:
        video.close()
    env.close()
    return {"rows": rows, "stats": _stats(task, env)}


# ── Lockstep path: the main process decides for every env each tick ─────────────
def _lockstep_worker(conn, tasks: list[dict]) -> None:
    for task in tasks:
        env = _make_env(task)
        expert, rng = Expert(), random.Random(task["seed"])
        b = task["behavior"]
        obs = env.reset(seed=task["seed"])
        rows, agree, decided = [], 0, 0
        video = _writer(Path(task["video"])) if task["video"] else None
        while not obs.done:
            if video is not None:
                video.append_data(env.frame())
            if obs.dead:
                obs = env.step("wait")
                continue
            label = expert.act(obs, b)
            weapon = _plan(expert, obs, b)
            conn.send(("state", obs.text, b))
            student = conn.recv()
            act = label if rng.random() < task["beta"] else student
            agree += student == label
            decided += 1
            if task["keep_rows"]:
                rows.append(
                    {
                        "b": b,
                        "ep": task["ep"],
                        "t": obs.tick,
                        "text": obs.text,
                        "expert": label,
                        "student": student,
                        "act": act,
                        "weapon": weapon,
                    }
                )
            obs = env.step(act, weapon=weapon)
        if video is not None:
            video.close()
        env.close()
        stats = _stats(task, env, agreement=round(agree / max(1, decided), 4))
        conn.send(("episode", {"rows": rows, "stats": stats}))
    conn.send(("done",))


def _run_lockstep(tasks: list[dict], workers: int, decide_batch, on_episode) -> None:
    ctx = mp.get_context("spawn")
    shards = [tasks[i::workers] for i in range(workers)]
    procs, conns = [], []
    for shard in shards:
        if not shard:
            continue
        parent, child = ctx.Pipe()
        p = ctx.Process(target=_lockstep_worker, args=(child, shard), daemon=True)
        p.start()
        procs.append(p)
        conns.append(parent)
    active = list(conns)
    while active:
        pending = []  # (conn, text, behavior)
        for c in list(active):
            while True:
                msg = c.recv()
                if msg[0] == "state":
                    pending.append((c, msg[1], msg[2]))
                    break
                if msg[0] == "episode":
                    on_episode(msg[1])
                    continue
                active.remove(c)
                break
        if pending:
            actions = decide_batch([p[1] for p in pending], [p[2] for p in pending])
            for (c, _, _), a in zip(pending, actions):
                c.send(a)
    for p in procs:
        p.join()


# ── Reporting ──────────────────────────────────────────────────────────────────
def summarize(stats: list[dict]) -> str:
    def per_min(key):
        return lambda s: 60 * s[key] / max(1, s["seconds"])

    cols = [
        ("frags", lambda s: s["frags"]),
        ("deaths", lambda s: s["deaths"]),
        ("margin", lambda s: s["margin"]),
        ("top%", lambda s: 100.0 * s["top"]),
        ("rank", lambda s: s["rank"]),
        ("best bot", lambda s: s["best_bot"][1]),
        ("frags/min", per_min("frags")),
        ("dealt/min", per_min("damage_dealt")),
        ("taken/min", per_min("damage_taken")),
        ("pick/min", per_min("pickups")),
        ("threat m", lambda s: s["threat_dist"]),
    ]
    if any("agreement" in s for s in stats):
        cols.append(("agree%", lambda s: 100.0 * s.get("agreement", 0)))
    head = f"{'behavior':<10} {'bots':<8} {'n':>3} " + " ".join(
        f"{c:>13}" for c, _ in cols
    )
    lines = [head, "-" * len(head)]
    groups = sorted(
        {(s["behavior"], s["bots"]) for s in stats},
        key=lambda g: (g[0] not in BEHAVIORS, g),
    )
    for b, bots in groups:
        ss = [s for s in stats if s["behavior"] == b and s["bots"] == bots]
        cells = []
        for _, f in cols:
            vals = [f(s) for s in ss]
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            cells.append(f"{statistics.mean(vals):7.1f} ±{sd:5.1f}")
        lines.append(
            f"{b:<10} {bots:<8} {len(ss):>3} " + " ".join(f"{c:>13}" for c in cells)
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--policy", choices=("expert", "vllm", "random"), default="expert")
    ap.add_argument("--model", help="Composed checkpoint (for --policy vllm)")
    ap.add_argument(
        "--behaviors", nargs="+", default=list(BEHAVIORS), choices=BEHAVIORS
    )
    ap.add_argument("--episodes", type=int, default=50, help="Per behavior")
    ap.add_argument("--bots", default="default", choices=sorted(BOT_SETS))
    ap.add_argument("--n-bots", type=int, default=7)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument(
        "--timeout-s", type=float, default=MATCH_TICS / TIC_HZ, help="Match length"
    )
    ap.add_argument(
        "--dart", type=float, default=0.0, help="Noise-burst rate (expert only)"
    )
    ap.add_argument(
        "--beta", type=float, default=0.0, help="DAgger: P(execute expert label)"
    )
    ap.add_argument("--record", type=int, default=0, help="MP4s per behavior")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--stats-only", action="store_true", help="Do not write per-tic rows"
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tasks = []
    for bi, b in enumerate(args.behaviors):
        for ep in range(args.episodes):
            tasks.append(
                {
                    "behavior": b,
                    "ep": ep,
                    "seed": args.seed + 1000 * bi + ep,
                    "timeout": int(args.timeout_s * TIC_HZ),
                    "bots": args.bots,
                    "n_bots": args.n_bots,
                    "dart": args.dart,
                    "beta": args.beta,
                    "policy": args.policy,
                    "keep_rows": not args.stats_only,
                    "video": str(args.out / "videos" / f"{b}_ep{ep}.mp4")
                    if ep < args.record
                    else None,
                }
            )

    row_files = (
        {b: open(args.out / f"{b}.jsonl", "w") for b in args.behaviors}
        if not args.stats_only
        else {}
    )
    stats_f = open(args.out / "stats.jsonl", "w")
    all_stats: list[dict] = []
    n_rows = 0

    def on_episode(res: dict) -> None:
        nonlocal n_rows
        s = res["stats"]
        all_stats.append(s)
        stats_f.write(json.dumps(s) + "\n")
        stats_f.flush()
        if res["rows"]:
            f = row_files[s["behavior"]]
            for r in res["rows"]:
                f.write(json.dumps(r) + "\n")
            n_rows += len(res["rows"])
        print(
            f"[{len(all_stats)}/{len(tasks)}] {s['behavior']:<9} ep{s['ep']:<3} "
            f"frags {s['frags']:>3} deaths {s['deaths']:>3} margin {s['margin']:>+4} "
            f"rank {s['rank']} (best bot {s['best_bot'][0]} {s['best_bot'][1]}) "
            f"pickups {s['pickups']:>3}"
            + (f" agree {100 * s['agreement']:.1f}%" if "agreement" in s else ""),
            flush=True,
        )

    t0 = time.time()
    if args.policy == "expert":
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            for res in pool.imap_unordered(_expert_episode, tasks):
                on_episode(res)
    else:
        if args.policy == "vllm":
            from policy import VLLMPolicy

            pol = VLLMPolicy(args.model, max_num_seqs=max(8, args.workers), warmup=5)

            def decide_batch(texts, behaviors):
                return [d.action for d in pol.decide_batch(texts, behaviors)]
        else:
            rng = random.Random(args.seed)

            def decide_batch(texts, behaviors):
                return [rng.choice(ACTIONS) for _ in texts]

        _run_lockstep(tasks, args.workers, decide_batch, on_episode)

    for f in row_files.values():
        f.close()
    stats_f.close()
    table = summarize(all_stats)
    (args.out / "summary.txt").write_text(table + "\n")
    print(
        f"\n{len(all_stats)} matches, {n_rows} rows in {time.time() - t0:.0f}s -> {args.out}\n"
    )
    print(table)


if __name__ == "__main__":
    main()
