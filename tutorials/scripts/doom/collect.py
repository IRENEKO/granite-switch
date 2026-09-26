# SPDX-License-Identifier: Apache-2.0
"""Roll out expert or student policies in parallel ViZDoom envs; write jsonl + stats.

Expert rollouts (round 0), with DART-style noise for wider state coverage::

    python collect.py --policy expert --episodes 50 --dart 0.15 --out data/round0

Student rollouts for DAgger. The student (composed checkpoint) drives, the
expert labels every visited state, and all envs step in lockstep so each
tick's student decisions are one batched vLLM call::

    python collect.py --policy vllm --model ./doom-switch --episodes 30 \
        --beta 0.0 --out data/round1

``--policy random`` drives the same lockstep path with uniform random actions.
It is a stat baseline, and a way to exercise the plumbing without a GPU.

Outputs in ``--out``:

* ``<behavior>.jsonl``: one row per tic, ``{"text", "expert", "act", ...}``;
  ``expert`` is the training label.
* ``stats.jsonl``: one row per episode.
* ``videos/<behavior>_ep<k>.mp4``: the first ``--record`` episodes per behavior.
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

from doom_env import ACTIONS, TIC_HZ, DoomEnv
from expert import BEHAVIORS, Expert

RESOLUTION = "640X480"  # one setting everywhere: collection, bench and demo
SKILL = 3


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


# ── Expert path: each worker runs whole episodes on its own ─────────────────────
def _expert_episode(task: dict) -> dict:
    env = DoomEnv(
        seed=task["seed"],
        resolution=RESOLUTION,
        hud=True,
        timeout_tics=task["timeout"],
        skill=task["skill"],
    )
    expert, rng = Expert(), random.Random(task["seed"])
    noise = _Noise(task["dart"], rng)
    b = task["behavior"]
    obs = env.reset(seed=task["seed"])
    rows, video = [], _writer(Path(task["video"])) if task["video"] else None
    while not obs.done:
        label = expert.act(obs, b)
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
                }
            )
        if video is not None:
            video.append_data(env.frame())
        obs = env.step(act)
    if video is not None:
        video.close()
    env.close()
    stats = {
        "behavior": b,
        "ep": task["ep"],
        "seed": task["seed"],
        "policy": "expert",
        **env.stats.as_dict(),
    }
    return {"rows": rows, "stats": stats}


# ── Lockstep path: the main process decides for every env each tick ─────────────
def _lockstep_worker(conn, tasks: list[dict]) -> None:
    for task in tasks:
        env = DoomEnv(
            seed=task["seed"],
            resolution=RESOLUTION,
            hud=True,
            timeout_tics=task["timeout"],
            skill=task["skill"],
        )
        expert, rng = Expert(), random.Random(task["seed"])
        b = task["behavior"]
        obs = env.reset(seed=task["seed"])
        rows, agree = [], 0
        video = _writer(Path(task["video"])) if task["video"] else None
        while not obs.done:
            label = expert.act(obs, b)
            conn.send(("state", obs.text, b))
            student = conn.recv()
            act = label if rng.random() < task["beta"] else student
            agree += student == label
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
                    }
                )
            if video is not None:
                video.append_data(env.frame())
            obs = env.step(act)
        if video is not None:
            video.close()
        env.close()
        stats = {
            "behavior": b,
            "ep": task["ep"],
            "seed": task["seed"],
            "policy": task["policy"],
            "agreement": round(agree / max(1, env.stats.tics), 4),
            **env.stats.as_dict(),
        }
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
    cols = [
        ("kills", lambda s: s["kills"]),
        ("dmg", lambda s: s["damage_taken"]),
        ("dmg/min", lambda s: 60 * s["damage_taken"] / max(1, s["seconds"])),
        ("pickups", lambda s: s["pickups"]),
        ("pick/min", lambda s: 60 * s["pickups"] / max(1, s["seconds"])),
        ("kills/min", lambda s: 60 * s["kills"] / max(1, s["seconds"])),
        ("shots/min", lambda s: 60 * s["shots"] / max(1, s["seconds"])),
        ("threat m", lambda s: s["threat_dist"]),
        ("alive s", lambda s: s["seconds"]),
        ("died%", lambda s: 100.0 * s["died"]),
    ]
    if any("agreement" in s for s in stats):
        cols.append(("agree%", lambda s: 100.0 * s.get("agreement", 0)))
    head = f"{'behavior':<10} {'n':>3} " + " ".join(f"{c:>13}" for c, _ in cols)
    lines = [head, "-" * len(head)]
    for b in sorted(
        {s["behavior"] for s in stats}, key=lambda b: (b not in BEHAVIORS, b)
    ):
        ss = [s for s in stats if s["behavior"] == b]
        cells = []
        for _, f in cols:
            vals = [f(s) for s in ss]
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            cells.append(f"{statistics.mean(vals):7.1f} ±{sd:5.1f}")
        lines.append(f"{b:<10} {len(ss):>3} " + " ".join(f"{c:>13}" for c in cells))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--policy", choices=("expert", "vllm", "random"), default="expert")
    ap.add_argument("--model", help="Composed checkpoint (for --policy vllm)")
    ap.add_argument(
        "--behaviors", nargs="+", default=list(BEHAVIORS), choices=BEHAVIORS
    )
    ap.add_argument("--episodes", type=int, default=50, help="Per behavior")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--timeout-s", type=float, default=60.0, help="Episode length cap")
    ap.add_argument(
        "--dart", type=float, default=0.0, help="Noise-burst rate (expert only)"
    )
    ap.add_argument(
        "--beta", type=float, default=0.0, help="DAgger: P(execute expert label)"
    )
    ap.add_argument("--record", type=int, default=0, help="MP4s per behavior")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skill", type=int, default=SKILL, help="ViZDoom doom_skill 1-5")
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
                    "skill": args.skill,
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
            f"kills {s['kills']:>2} dmg {s['damage_taken']:>4} pickups {s['pickups']:>3} "
            f"alive {s['seconds']:>5.1f}s{' died' if s['died'] else ''}"
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
        f"\n{len(all_stats)} episodes, {n_rows} rows in {time.time() - t0:.0f}s -> {args.out}\n"
    )
    print(table)


if __name__ == "__main__":
    main()
