# SPDX-License-Identifier: Apache-2.0
"""Live Doom demo: Granite decides every tic against seven bots; the browser watches.

    python server.py --policy expert                           # laptop, no model
    python server.py --policy rl --teacher runs/rl0/latest.pt  # the RL teacher
    python server.py --policy vllm --model models/doom-switch  # GPU host

Then open http://localhost:8000. From a laptop, tunnel to the GPU host:
``ssh -L 8000:<gpu-node>:8000 <login-node>``.

Layout:

* Each game runs in its own worker process: env stepping (a deathmatch against
  seven bots), the teacher's labels for every style (the scripted player, or
  the RL teacher with ``--teacher``) for the no-model policies, shadow
  decisions and live teacher agreement, the 5 Hz history text, reaction-time
  tracking and JPEG encoding. The measured decision is therefore just prompt
  build plus the engine step.
* One loop thread owns the policy and a tokenized history per game. Every tic
  it takes the latest state of every game and makes one engine step: the style
  adapter and the critic for every game, the weapon planner on its 0.5 s
  cadence, the other styles too with shadow on. All of them read the same
  cached history; only their query suffixes differ.
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
    WEAPON_NAMES,
    DoomEnv,
    isolate_workdir,
)
from expert import BEHAVIORS, PLAN_EVERY_TICS
from history import History
from policy import ARMS, CRITIC, DANGER_LEVELS, make_policy, rule_danger, state_text

STATIC = Path(__file__).parent / "static"
MODES = ("35hz", "10hz", "turbo")
GRID_SIZES = (1, 4, 9, 16)
REACTION_QUIET_TICS = TIC_HZ  # a bot must be unseen this long to count as "appearing"
HISTORY_SHOWN = 12  # history lines sent to the page


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


def _worker(conn, seed: int, teacher: str) -> None:
    from collect import Teacher

    isolate_workdir()
    env = DoomEnv(seed=seed, resolution="640X480", hud=True, timeout_tics=10**7)
    teachers = {b: Teacher(teacher) for b in BEHAVIORS}
    hist = History()
    reaction = _Reaction()
    episodes, obs = 0, env.reset(seed=seed)
    entries: list[str] = []  # appended since the last message

    def state(extra: dict) -> dict:
        nonlocal entries
        t0 = time.perf_counter()
        labels, weapons = {}, {}
        if not obs.dead:
            for b, t in teachers.items():
                move, weapon = t.label(obs, b)
                labels[b] = max(move, key=move.get)
                weapons[b] = int(max(weapon, key=weapon.get))
        teacher_ms = (time.perf_counter() - t0) * 1000 / len(BEHAVIORS)
        new, entries = entries, []
        return {
            "text": state_text(obs),
            "entries": new,
            "history": hist.entries[-HISTORY_SHOWN:],
            "expert": labels,
            "weapons": weapons,
            "danger": rule_danger(obs),
            "expert_ms": teacher_ms,
            "dead": obs.dead,
            "tick": obs.tick,
            "hud": {
                "hp": obs.hp,
                "armor": obs.armor,
                "ammo": obs.ammo,
                "weapon": obs.weapon,
                "slot": obs.slot,
                "frags": obs.frags,
                "deaths": obs.deaths,
            },
            "stats": {**env.stats.as_dict(), "episodes": episodes},
            **extra,
        }

    def new_match() -> None:
        nonlocal obs, reaction
        obs = env.reset()
        for t in teachers.values():
            t.reset()
        hist.reset()
        reaction = _Reaction()
        extra_reset["reset"] = True

    extra_reset: dict = {}
    conn.send(state({"reset": True}))
    while True:
        msg = conn.recv()
        if msg[0] == "close":
            break
        extra: dict = {}
        if msg[0] == "reset":
            new_match()
            episodes += 1
        else:
            _, action, weapon, frame_size, quality = msg
            visible = any(o.kind == "enemy" for o in obs.seen)
            r = reaction.update(obs.tick, visible, action)
            if r is not None:
                extra["reaction_ms"] = r
            entry = hist.observe(obs, None if obs.dead else action)
            if entry is not None:
                entries.append(entry)
            obs = env.step(action, weapon=weapon)
            if obs.done:
                extra["episode"] = env.stats.as_dict()
                episodes += 1
                new_match()
            if frame_size is not None:
                extra["jpeg"] = _jpeg(
                    env.frame(), tuple(frame_size) if frame_size else None, quality
                )
        extra.update(extra_reset)
        extra_reset.clear()
        conn.send(state(extra))
    env.close()


class Game:
    def __init__(self, index: int, seed: int, adapter: str, teacher: str, tok):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(
            target=_worker, args=(child, seed, teacher), daemon=True
        )
        self.proc.start()
        self.index = index
        self.adapter = adapter
        self.last_action = "wait"
        self.tok = tok
        self.hist = History(tok)
        self.state = self.conn.recv()
        self.sync_history()

    def sync_history(self) -> None:
        """Mirror the worker's history (tokenized, for the model's prompt)."""
        if self.state.get("reset"):
            self.hist = History(self.tok)
        for e in self.state["entries"]:
            self.hist.append(e)

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


def _dec(d) -> dict:
    return {
        "action": d.action,
        "top3": [[a, round(p, 4)] for a, p in d.top3],
        "probs": d.probs,
        "ms": d.ms,
        "build_ms": d.build_ms,
        "engine_ms": d.engine_ms,
        "fresh": d.fresh_tokens,
        "cached": d.cached_tokens,
    }


class GameLoop(threading.Thread):
    def __init__(
        self,
        policy_kind: str,
        model: str | None,
        teacher: str,
        publish,
        seed: int,
        **policy_kw,
    ):
        super().__init__(daemon=True)
        self.policy_kind = policy_kind
        self.teacher = teacher
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
        # Game 0's per-tic action distribution and critic since the last
        # publish, for the browser's heatmap and danger strip.
        self.heat: list[dict] = []
        self.plan: dict | None = None  # game 0's last weapon-planner decision

    # Commands arrive from the websocket handler on the asyncio thread.
    def command(self, msg: dict) -> None:
        self.cmds.put(msg)

    def _resize(self, n: int) -> None:
        for g in self.games:
            g.close()
        tok = getattr(self.policy, "tok", None)
        self.games = [
            Game(
                i,
                self.seed + 7919 * i,
                BEHAVIORS[i % len(BEHAVIORS)],
                self.teacher,
                tok,
            )
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
                "by": "router adapter"
                if self.policy_kind == "vllm"
                else "keywords (no model loaded)",
            }
            self.publish("json", {"type": "route", **self.route_info})
        elif kind == "shadow":
            self.shadow = bool(msg.get("on"))
        elif kind == "restart":
            for g in self.games:
                g.conn.send(("reset",))
                g.state = g.conn.recv()
                g.sync_history()
        elif kind == "grid" and msg.get("n") in GRID_SIZES:
            self._resize(int(msg["n"]))

    def _decide(self, due: list[Game]) -> dict[int, dict]:
        """Every due game's outputs for this tic: its style's action, the critic,
        and the weapon plan on planner tics. One engine step for the model."""
        out: dict[int, dict] = {}
        if not due:
            return out
        if self.policy_kind != "vllm":  # the teacher's labels, from the workers
            for g in due:
                s = g.state
                a = s["expert"][g.adapter]
                d = {
                    "action": a,
                    "top3": [[a, 1.0]],
                    "probs": {a: 1.0},
                    "ms": s["expert_ms"],
                    "build_ms": 0.0,
                    "engine_ms": s["expert_ms"],
                    "fresh": 0,
                    "cached": 0,
                    "critic": {s["danger"]: 1.0},
                }
                if s["tick"] % PLAN_EVERY_TICS == 0:
                    slot = s["weapons"][g.adapter]
                    d["plan"] = {"slot": slot, "probs": {str(slot): 1.0}}
                out[g.index] = d
            return out
        games, names = [], []
        for g in due:
            want = [g.adapter, CRITIC]
            if g.state["tick"] % PLAN_EVERY_TICS == 0:
                want.append(ARMS)
            if self.shadow and len(due) == 1:
                want += [b for b in BEHAVIORS if b != g.adapter]
            games.append((g.hist.ids, g.state["text"]))
            names.append(want)
        decs = self.policy.decide_games(games, names)
        for g, want, d in zip(due, names, decs):
            o = _dec(d[g.adapter])
            o["critic"] = d[CRITIC].probs
            if ARMS in d:
                o["plan"] = {"slot": int(d[ARMS].action), "probs": d[ARMS].probs}
            if self.shadow and len(due) == 1:
                g.state["shadow"] = {b: d[b].action for b in BEHAVIORS}
            o["adapters"] = len(want)
            out[g.index] = o
        return out

    def run(self) -> None:
        kind, model, kw = self._policy_args
        self.policy = make_policy("expert" if kind == "rl" else kind, model, **kw)
        self._resize(1)
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
                weapon = None
                if d is not None:
                    g.last_action = d["action"]
                    last_dec[g.index] = d
                    teacher = g.state["expert"].get(g.adapter)
                    if self.policy_kind == "vllm" and teacher is not None:
                        self.agree.append(int(teacher == d["action"]))
                    if "plan" in d:
                        weapon = d["plan"]["slot"]
                        if g.index == 0:
                            self.plan = {**d["plan"], "tick": g.state["tick"]}
                want = now - last_frame[g.index] >= 1.0 / fps
                if want:
                    last_frame[g.index] = now
                size = None if not want else ((640, 480) if single else (224, 168))
                g.conn.send(("step", g.last_action, weapon, size, 70 if single else 60))
            for g in games:
                g.state = g.conn.recv()
                g.sync_history()
                if "reaction_ms" in g.state and g.adapter == "fighter":
                    self.reaction[self.mode].append(g.state["reaction_ms"])
                jpeg = g.state.pop("jpeg", None)
                if jpeg is not None:
                    self.publish("bytes", bytes([g.index]) + jpeg)
            self.tics.add(0.0)
            self._record_heat(games[0], last_dec.get(0), 0 in decs)

            if now - last_pub >= 1.0 / 30:
                last_pub = now
                self._publish(games, last_dec)

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
        """One column per tic: the action distribution, the critic's danger and
        whether this tic had a fresh decision. On a tic without one (10 Hz
        cadence, or dead) the previous distribution repeats, flagged ``d=False``."""
        d = d or {}
        probs = d.get("probs") or {}
        critic = d.get("critic") or {}
        self.heat.append(
            {
                "p": [round(probs.get(a, 0.0), 3) for a in DISPLAY_ORDER],
                "c": [round(critic.get(k, 0.0), 3) for k in DANGER_LEVELS],
                "b": BEHAVIORS.index(g.adapter),
                "d": decided,
            }
        )
        del self.heat[:-400]  # a stalled client must not grow this without bound

    def _publish(self, games: list[Game], last_dec: dict[int, dict]) -> None:
        g0 = games[0]
        d0 = last_dec.get(0, {})
        rt = {}
        for m, xs in self.reaction.items():
            if xs:
                rt[m] = {"median": round(float(np.median(xs)), 1), "n": len(xs)}
        s0 = g0.state["stats"]
        msg = {
            "type": "tick",
            "policy": self.policy_kind,
            "mode": self.mode,
            "grid": len(games),
            "shadow": self.shadow,
            "tick": g0.state["tick"],
            "dead": g0.state["dead"],
            "adapter": g0.adapter,
            "action": g0.last_action,
            "action_label": ACTION_LABELS.get(g0.last_action, g0.last_action),
            "decision": {
                k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in d0.items()
                if k not in ("probs", "critic", "plan")
            },
            "critic": d0.get("critic"),
            "plan": self.plan,
            "weapon_names": WEAPON_NAMES,
            "teacher": g0.state["expert"].get(g0.adapter),
            "shadow_actions": g0.state.get("shadow") if self.shadow else None,
            "text": g0.state["text"],
            "history": g0.state["history"],
            "history_tokens": len(g0.hist.ids) if self.policy_kind == "vllm" else None,
            "hud": g0.state["hud"],
            "stats": s0,
            "scoreboard": s0["scoreboard"],
            "latency": self.lat.summary(),
            "tics_per_s": self.tics.summary()["per_s"],
            "tic_ms": round(TIC_MS, 2),
            "heat": self.heat,
            "order": DISPLAY_ORDER,
            "labels": SHORT_LABELS,
            "behaviors": BEHAVIORS,
            "danger_levels": DANGER_LEVELS,
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
                    "rank": g.state["stats"]["rank"],
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
    ap.add_argument("--policy", choices=("expert", "rl", "vllm"), default="expert")
    ap.add_argument("--model", help="Composed checkpoint (for --policy vllm)")
    ap.add_argument(
        "--teacher",
        default="expert",
        help="Labels for agreement and the no-model policies: 'expert' or an "
        "rl_teacher.py checkpoint (required for --policy rl)",
    )
    ap.add_argument("--engine-loop", action="store_true")
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Style and planner sampling (vllm); the RL teacher plays by sampling",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.policy == "rl" and args.teacher == "expert":
        raise SystemExit("--policy rl needs --teacher <checkpoint>")
    teacher = (
        args.teacher if args.teacher == "expert" else str(Path(args.teacher).resolve())
    )
    kw = (
        {
            "engine_loop": args.engine_loop,
            "max_num_seqs": 128,
            "temperature": args.temperature,
        }
        if args.policy == "vllm"
        else {}
    )

    def factory(publish):
        return GameLoop(args.policy, args.model, teacher, publish, args.seed, **kw)

    import uvicorn

    uvicorn.run(build_app(factory), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
