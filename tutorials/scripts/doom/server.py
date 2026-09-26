# SPDX-License-Identifier: Apache-2.0
"""Live Doom reflex demo: Granite decides every tic; the browser watches.

    python server.py --policy expert                          # laptop, no model
    python server.py --policy vllm --model models/doom-switch # GPU host

Then open http://localhost:8000. From a laptop, tunnel to the GPU host:
``ssh -L 8000:<gpu-node>:8000 <login-node>``.

Layout:

* Each game runs in its own worker process: env stepping (a deathmatch against
  seven bots), the three scripted styles (for the expert policy, shadow
  decisions and live teacher agreement), the scripted weapon planner,
  reaction-time tracking and JPEG encoding. The measured decision is therefore
  just prompt build plus the engine step.
* One loop thread owns the policy. Every tic it takes the latest state of every
  game, makes one (batched) decision, and sends the actions back. With N games
  that is still one engine step per tic.
* FastAPI serves ``static/index.html``, and a websocket pushes telemetry (JSON)
  and frames (binary: one byte of game index, then a JPEG).

Modes: ``35hz`` decides every tic at Doom speed, ``10hz`` decides every 100 ms
of game time and repeats the action in between (the cadence the Jev post
describes), ``turbo`` runs unthrottled so the model sets the pace.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doom_env import (
    ACTION_LABELS,
    ATTACKS,
    DISPLAY_ORDER,
    SHORT_LABELS,
    TIC_HZ,
    TIC_MS,
    DoomEnv,
)
from expert import BEHAVIORS, PLAN_EVERY_TICS, Expert
from policy import make_policy

STATIC = Path(__file__).parent / "static"
MODES = ("35hz", "10hz", "turbo")
GRID_SIZES = (1, 4, 9, 16)
REACTION_QUIET_TICS = TIC_HZ  # a bot must be unseen this long to count as "appearing"


# ── Game worker process ────────────────────────────────────────────────────────
class _Reaction:
    """Game-time from a bot appearing on screen to the first shot at it."""

    def __init__(self) -> None:
        self.last_seen = -(10**9)
        self.appeared: int | None = None

    def update(self, tick: int, enemy_visible: bool, action: str) -> float | None:
        out = None
        if enemy_visible:
            if self.appeared is None and tick - self.last_seen > REACTION_QUIET_TICS:
                self.appeared = tick
            self.last_seen = tick
        elif self.appeared is not None and tick - self.last_seen > 3:
            self.appeared = None  # it left the screen before we fired
        if self.appeared is not None and action in ATTACKS:
            out = (tick - self.appeared) * TIC_MS
            self.appeared = None
        return out


def _jpeg(frame: np.ndarray, size: tuple[int, int] | None, quality: int) -> bytes:
    from PIL import Image

    img = Image.fromarray(frame)
    if size is not None and img.size != size:
        img = img.resize(size, Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _worker(conn, seed: int) -> None:
    env = DoomEnv(seed=seed, resolution="640X480", hud=True, timeout_tics=10**7)
    experts = {b: Expert() for b in BEHAVIORS}
    reaction = _Reaction()
    episodes, obs = 0, env.reset(seed=seed)

    def state(extra: dict) -> dict:
        t0 = time.perf_counter()
        labels = {b: experts[b].act(obs, b) for b in BEHAVIORS}
        expert_ms = (time.perf_counter() - t0) * 1000 / len(BEHAVIORS)
        return {
            "text": obs.text,
            "expert": labels,
            "expert_ms": expert_ms,
            "planner": experts[BEHAVIORS[0]].weapon(obs),
            "dead": obs.dead,
            "tick": obs.tick,
            "hud": {
                "hp": obs.hp,
                "armor": obs.armor,
                "ammo": obs.ammo,
                "weapon": obs.weapon,
                "frags": obs.frags,
                "deaths": obs.deaths,
            },
            "stats": {**env.stats.as_dict(), "episodes": episodes},
            **extra,
        }

    conn.send(state({}))
    while True:
        msg = conn.recv()
        if msg[0] == "close":
            break
        extra: dict = {}
        if msg[0] == "reset":
            obs = env.reset()
            for e in experts.values():
                e.reset()
            episodes += 1
        else:
            _, action, weapon, frame_size, quality = msg
            visible = any(o.kind == "enemy" for o in obs.seen)
            r = reaction.update(obs.tick, visible, action)
            if r is not None:
                extra["reaction_ms"] = r
            obs = env.step(action, weapon=weapon)
            if obs.done:
                extra["episode"] = env.stats.as_dict()
                episodes += 1
                obs = env.reset()
                for e in experts.values():
                    e.reset()
                reaction = _Reaction()
            if frame_size is not None:
                extra["jpeg"] = _jpeg(
                    env.frame(), tuple(frame_size) if frame_size else None, quality
                )
        conn.send(state(extra))
    env.close()


class Game:
    def __init__(self, index: int, seed: int, adapter: str):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker, args=(child, seed), daemon=True)
        self.proc.start()
        self.index = index
        self.adapter = adapter
        self.last_action = "wait"
        self.state = self.conn.recv()

    def close(self) -> None:
        try:
            self.conn.send(("close",))
        except (BrokenPipeError, OSError):
            pass
        self.proc.join(timeout=2)
        if self.proc.is_alive():
            self.proc.kill()


# ── Loop thread ────────────────────────────────────────────────────────────────
def _on_10hz_beat(tick: int) -> bool:
    """True on the tic where another 100 ms of game time has begun."""
    return tick == 0 or (tick * 10) // TIC_HZ != ((tick - 1) * 10) // TIC_HZ


class Rolling:
    def __init__(self, n: int = 2000):
        self.ms: deque[float] = deque(maxlen=n)
        self.t: deque[float] = deque(maxlen=n)

    def add(self, ms: float) -> None:
        self.ms.append(ms)
        self.t.append(time.perf_counter())

    def summary(self) -> dict:
        if not self.ms:
            return {"p50": None, "p99": None, "n": 0, "per_s": 0.0}
        a = np.fromiter(self.ms, dtype=np.float64)
        span = self.t[-1] - self.t[0] if len(self.t) > 1 else 0.0
        return {
            "p50": round(float(np.percentile(a, 50)), 2),
            "p99": round(float(np.percentile(a, 99)), 2),
            "n": len(a),
            "per_s": round((len(a) - 1) / span, 1) if span > 0 else 0.0,
        }


class GameLoop(threading.Thread):
    def __init__(
        self, policy_kind: str, model: str | None, publish, seed: int, **policy_kw
    ):
        super().__init__(daemon=True)
        self.policy_kind = policy_kind
        self._policy_args = (policy_kind, model, policy_kw)
        self.policy = None  # built in run(): vLLM is created and used on one thread
        self.publish = publish  # thread-safe: (kind, payload) -> None
        self.seed = seed
        self.cmds: queue.Queue = queue.Queue()
        self.mode = "35hz"
        self.shadow = False
        self.games: list[Game] = []
        self.lat = Rolling()
        self.tics = Rolling(4000)
        self.reaction = {m: deque(maxlen=200) for m in MODES}
        self.agree: deque[int] = deque(maxlen=1000)
        self.route_info: dict | None = None
        # Game 0's per-tic action distribution since the last publish, rows in
        # DISPLAY_ORDER, for the browser's heatmap.
        self.heat: list[dict] = []
        self._resize(1)

    # Commands arrive from the websocket handler on the asyncio thread.
    def command(self, msg: dict) -> None:
        self.cmds.put(msg)

    def _resize(self, n: int) -> None:
        for g in self.games:
            g.close()
        self.games = [
            Game(i, self.seed + 7919 * i, BEHAVIORS[i % len(BEHAVIORS)])
            for i in range(n)
        ]
        self.lat = Rolling()
        self.tics = Rolling(4000)

    def _handle(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "mode" and msg.get("mode") in MODES:
            self.mode = msg["mode"]
            self.lat, self.tics = Rolling(), Rolling(4000)
        elif kind == "adapter" and msg.get("name") in BEHAVIORS:
            for g in self.games:
                g.adapter = msg["name"]
        elif kind == "mix":
            for g in self.games:
                g.adapter = BEHAVIORS[g.index % len(BEHAVIORS)]
        elif kind == "instruction" and msg.get("text", "").strip():
            text = msg["text"].strip()[:300]
            r = self.policy.route(text)
            for g in self.games:
                g.adapter = r.adapter
            self.route_info = {
                "text": text,
                "adapter": r.adapter,
                "prob": round(r.prob, 3),
                "probs": {k: round(v, 3) for k, v in r.probs.items()},
                "ms": round(r.ms, 2),
                "by": "keywords (no model loaded)"
                if self.policy_kind == "expert"
                else "router adapter",
            }
            self.publish("json", {"type": "route", **self.route_info})
        elif kind == "shadow":
            self.shadow = bool(msg.get("on"))
        elif kind == "restart":
            for g in self.games:
                g.conn.send(("reset",))
                g.state = g.conn.recv()
        elif kind == "grid" and msg.get("n") in GRID_SIZES:
            self._resize(int(msg["n"]))

    def _decide(self, due: list[Game]) -> dict[int, dict]:
        """One decision per due game; a single engine step for model policies."""
        out: dict[int, dict] = {}
        if not due:
            return out
        if self.policy_kind == "expert":
            for g in due:
                a = g.state["expert"][g.adapter]
                out[g.index] = {
                    "action": a,
                    "top3": [[a, 1.0]],
                    "probs": {a: 1.0},
                    "ms": g.state["expert_ms"],
                    "build_ms": 0.0,
                    "engine_ms": g.state["expert_ms"],
                    "fresh": 0,
                    "cached": 0,
                }
            return out
        texts = [g.state["text"] for g in due]
        adapters = [g.adapter for g in due]
        if self.shadow and len(due) == 1:
            # The active behavior plus the other two, same state, one step.
            order = [adapters[0], *[b for b in BEHAVIORS if b != adapters[0]]]
            decs = self.policy.decide_batch(texts * len(order), order)
            due[0].state["shadow"] = {b: d.action for b, d in zip(order, decs)}
            decs = decs[:1]
        else:
            decs = self.policy.decide_batch(texts, adapters)
        for g, d in zip(due, decs):
            out[g.index] = {
                "action": d.action,
                "top3": [[a, round(p, 4)] for a, p in d.top3],
                "probs": d.probs,
                "ms": d.ms,
                "build_ms": d.build_ms,
                "engine_ms": d.engine_ms,
                "fresh": d.fresh_tokens,
                "cached": d.cached_tokens,
            }
        return out

    def run(self) -> None:
        kind, model, kw = self._policy_args
        self.policy = make_policy(kind, model, **kw)
        next_t = time.perf_counter()
        last_frame = [0.0] * 64
        last_pub = 0.0
        last_dec: dict[int, dict] = {}
        while True:
            while not self.cmds.empty():
                self._handle(self.cmds.get())
            games = self.games
            due = [
                g
                for g in games
                if not g.state["dead"]
                and (self.mode != "10hz" or _on_10hz_beat(g.state["tick"]))
            ]
            decs = self._decide(due)
            if decs:
                # For a batch every game shares one step, so record it once.
                self.lat.add(next(iter(decs.values()))["ms"])
            now = time.perf_counter()
            single = len(games) == 1
            fps = TIC_HZ if single else 10.0
            for g in games:
                d = decs.get(g.index)
                if d is not None:
                    g.last_action = d["action"]
                    last_dec[g.index] = d
                    teacher = g.state["expert"][g.adapter]
                    if self.policy_kind != "expert":
                        self.agree.append(int(teacher == d["action"]))
                want = now - last_frame[g.index] >= 1.0 / fps
                if want:
                    last_frame[g.index] = now
                size = None if not want else ((640, 480) if single else (224, 168))
                weapon = (
                    g.state["planner"]
                    if g.state["tick"] % PLAN_EVERY_TICS == 0
                    else None
                )
                g.conn.send(("step", g.last_action, weapon, size, 70 if single else 60))
            for g in games:
                g.state = g.conn.recv()
                if "reaction_ms" in g.state and g.adapter == "fighter":
                    self.reaction[self.mode].append(g.state["reaction_ms"])
                jpeg = g.state.pop("jpeg", None)
                if jpeg is not None:
                    self.publish("bytes", bytes([g.index]) + jpeg)
            self.tics.add(0.0)
            self._record_heat(games[0], last_dec.get(0), 0 in decs)

            if now - last_pub >= 1.0 / 30:
                last_pub = now
                self._publish(games, last_dec, bool(decs))

            if self.mode != "turbo":
                next_t += 1.0 / TIC_HZ
                slack = next_t - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    next_t = time.perf_counter()  # fell behind: do not accrue debt
            else:
                next_t = time.perf_counter()

    def _record_heat(self, g: Game, d: dict | None, decided: bool) -> None:
        """One heatmap column per tic. On a tic without a fresh decision (10 Hz
        cadence) the previous distribution repeats, flagged by ``d=False``."""
        probs = (d or {}).get("probs") or {}
        self.heat.append(
            {
                "p": [round(probs.get(a, 0.0), 3) for a in DISPLAY_ORDER],
                "b": BEHAVIORS.index(g.adapter),
                "d": decided,
            }
        )
        del self.heat[:-400]  # a stalled client must not grow this without bound

    def _publish(
        self, games: list[Game], last_dec: dict[int, dict], decided: bool
    ) -> None:
        g0 = games[0]
        d0 = last_dec.get(0, {})
        rt = {}
        for m, xs in self.reaction.items():
            if xs:
                rt[m] = {"median": round(float(np.median(xs)), 1), "n": len(xs)}
        msg = {
            "type": "tick",
            "policy": self.policy_kind,
            "mode": self.mode,
            "grid": len(games),
            "shadow": self.shadow,
            "tick": g0.state["tick"],
            "adapter": g0.adapter,
            "action": g0.last_action,
            "action_label": ACTION_LABELS.get(g0.last_action, g0.last_action),
            "decision": {
                k: (round(v, 3) if isinstance(v, float) else v) for k, v in d0.items()
            },
            "teacher": g0.state["expert"][g0.adapter],
            "shadow_actions": g0.state.get("shadow") if self.shadow else None,
            "text": g0.state["text"],
            "hud": g0.state["hud"],
            "stats": g0.state["stats"],
            "latency": self.lat.summary(),
            "tics_per_s": self.tics.summary()["per_s"],
            "tic_ms": round(TIC_MS, 2),
            "heat": self.heat,
            "order": DISPLAY_ORDER,
            "labels": SHORT_LABELS,
            "behaviors": BEHAVIORS,
            "reaction": rt,
            "agreement": round(sum(self.agree) / len(self.agree), 3)
            if self.agree
            else None,
            "tiles": [
                {
                    "adapter": g.adapter,
                    "action": g.last_action,
                    "hp": g.state["hud"]["hp"],
                    "frags": g.state["stats"]["frags"],
                    "deaths": g.state["stats"]["deaths"],
                }
                for g in games
            ],
        }
        self.heat = []
        self.publish("json", msg)


# ── Web app ────────────────────────────────────────────────────────────────────
def build_app(loop_factory):
    clients: set[asyncio.Queue] = set()
    state: dict = {}

    @asynccontextmanager
    async def lifespan(_app):
        state["aloop"] = asyncio.get_running_loop()
        state["game"] = loop_factory(publish)
        state["game"].start()
        yield

    app = FastAPI(lifespan=lifespan)

    def publish(kind: str, payload) -> None:
        aloop = state.get("aloop")
        if aloop is None:
            return

        def fan_out() -> None:
            for q in list(clients):
                if q.full():
                    try:
                        q.get_nowait()  # drop the oldest message for a slow client
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait((kind, payload))

        aloop.call_soon_threadsafe(fan_out)

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        clients.add(q)
        game: GameLoop = state["game"]
        if game.route_info:
            await sock.send_text(json.dumps({"type": "route", **game.route_info}))

        async def pump() -> None:
            while True:
                kind, payload = await q.get()
                if kind == "bytes":
                    await sock.send_bytes(payload)
                else:
                    await sock.send_text(json.dumps(payload))

        sender = asyncio.create_task(pump())
        try:
            while True:
                game.command(json.loads(await sock.receive_text()))
        except (WebSocketDisconnect, json.JSONDecodeError):
            pass
        finally:
            sender.cancel()
            clients.discard(q)

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--policy", choices=("expert", "vllm"), default="expert")
    ap.add_argument("--model", help="Composed checkpoint (for --policy vllm)")
    ap.add_argument("--engine-loop", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    kw = (
        {"engine_loop": args.engine_loop, "max_num_seqs": 64}
        if args.policy == "vllm"
        else {}
    )

    def factory(publish):
        return GameLoop(args.policy, args.model, publish, args.seed, **kw)

    import uvicorn

    uvicorn.run(build_app(factory), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
