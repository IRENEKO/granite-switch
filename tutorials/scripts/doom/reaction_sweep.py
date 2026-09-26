# SPDX-License-Identifier: Apache-2.0
"""Does deciding every tic beat 10 Hz, and does latency (staleness) matter, in game outcomes?

The same scripted teacher plays every condition. Only two things change: when
decisions happen (the cadence), and how old the state each decision was based
on is (the delay, in tics). Real-time rules apply: the game never waits, and the
current action repeats until a newer decision lands. Games are deathmatches
against the default bots::

    python reaction_sweep.py --games 40 --seconds 120 --workers 12
"""

import argparse
import multiprocessing as mp
import os
import statistics as st
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doom_env import ATTACKS, TIC_HZ, TIC_MS, DoomEnv, isolate_workdir
from expert import PLAN_EVERY_TICS, Expert

CONDS = {  # name: (decisions per second, delay in tics)
    "35 Hz, no delay (our demo, 7 ms < 1 tic)": (35, 0),
    "35 Hz, 1-tic delay (~30-57 ms system)": (35, 1),
    "35 Hz, 3-tic delay (100 ms, pipelined)": (35, 3),
    "10 Hz, no delay": (10, 0),
    "10 Hz, 3-tic delay (Jev-like)": (10, 3),
}
BEHAVIORS = ("fighter", "cautious")


def beat(t, hz):
    return hz >= TIC_HZ or t == 0 or (t * hz) // TIC_HZ != ((t - 1) * hz) // TIC_HZ


def run(args):
    name, behavior, seed, seconds = args
    isolate_workdir()
    hz, delay = CONDS[name]
    env, ex = (
        DoomEnv(
            seed=seed, resolution="640X480", hud=True, timeout_tics=seconds * TIC_HZ
        ),
        Expert(),
    )
    obs = env.reset(seed=seed)
    ex.reset()
    pending, current = deque(), "wait"
    last_seen, appeared, reactions = -(10**9), None, []
    while not obs.done:
        t = obs.tick
        if beat(t, hz):
            pending.append((t + delay, ex.act(obs, behavior)))
        while pending and pending[0][0] <= t:
            current = pending.popleft()[1]
        vis = any(o.kind == "enemy" for o in obs.seen)
        if vis:
            if appeared is None and t - last_seen > TIC_HZ:
                appeared = t
            last_seen = t
        elif appeared is not None and t - last_seen > 3:
            appeared = None
        if appeared is not None and current in ATTACKS:
            reactions.append((t - appeared) * TIC_MS)
            appeared = None
        weapon = ex.weapon(obs) if t % PLAN_EVERY_TICS == 0 else None
        obs = env.step(current, weapon=weapon)
    s = env.stats.as_dict()
    env.close()
    return name, behavior, s, reactions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games", type=int, default=40, help="Games per row")
    ap.add_argument("--seconds", type=int, default=120, help="Game length")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    N = args.games
    tasks = [
        (c, b, 5000 + i, args.seconds)
        for c in CONDS
        for b in BEHAVIORS
        for i in range(N)
    ]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        res = pool.map(run, tasks)
    for b in BEHAVIORS:
        print(f"\n{b}  ({N} {args.seconds}-s games per row, same seeds across rows)")
        print(
            f"{'condition':<44}{'frags/min':>10}{'deaths/min':>11}{'dealt/min':>11}"
            f"{'margin':>8}{'reaction ms (p50)':>19}"
        )
        for c in CONDS:
            stats = [s for n, bb, s, _ in res if n == c and bb == b]
            rt = [x for n, bb, _, r in res if n == c and bb == b for x in r]

            def rate(key):
                xs = [60 * s[key] / s["seconds"] for s in stats]
                return st.mean(xs), st.stdev(xs) / len(xs) ** 0.5

            (fm, se_f), (dm, se_d) = rate("frags"), rate("deaths")
            dealt = rate("damage_dealt")[0]
            margin = st.mean(s["margin"] for s in stats)
            react = st.median(rt) if rt else float("nan")
            print(
                f"{c:<44}{fm:6.2f}±{se_f:3.2f}{dm:7.2f}±{se_d:3.2f}{dealt:9.0f}"
                f"{margin:+8.1f}{react:12.0f} (n={len(rt)})"
            )


if __name__ == "__main__":
    main()
