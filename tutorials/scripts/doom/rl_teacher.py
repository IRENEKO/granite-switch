# SPDX-License-Identifier: Apache-2.0
"""A recurrent PPO teacher for the Doom deathmatch demo, trained against bots.

The policy reads :func:`doom_env.features` (the same player-visible fields the
language model reads as text) plus a style one-hot, through an MLP and a GRU,
and has two heads: a 20-way movement head (``doom_env.ACTIONS``) acting every
tic, and an 8-way weapon head (slots 1-7, or keep the current one) acting on the
weapon planner's cadence (``expert.PLAN_EVERY_TICS``). Frameskip is 1, so the
teacher decides exactly where the model will. The critic is asymmetric: on top
of the GRU state it sees :func:`doom_env.priv_features` (every bot's position,
the frag race, match time), which never reaches the actor.

Reward is Sample Factory's deathmatch shaping (``REWARD_SHAPING_DEATHMATCH_V0``:
frag +1, suicide -1.5, death -0.75, hits, damage, health/armor/weapon/ammo
deltas). Each style reweights it (:data:`STYLE_SHAPING`): cautious pays more
for deaths and damage taken, collector for pickups. One network learns all
three styles; the style can switch mid-episode, as it does in the live demo.

Early learning is kickstarted (Schmitt et al., 2018): a cross-entropy term
toward the scripted player's action for the same style (``expert.py``), with a
weight that decays linearly to zero over ``ks_steps``. RL then has to beat the
scripted player rather than rediscover aiming from scratch.

Envs run in worker processes, several per process, stepped in lockstep with
the learner (the pattern of ``collect.py``). Evaluation matches (10 minutes,
default bots) run in a separate process pool on saved checkpoints, so training
does not wait for them.

Train, then evaluate a checkpoint::

    python rl_teacher.py train --out runs/rl0 --envs 160 --workers 40 --device cuda
    python rl_teacher.py eval --ckpt runs/rl0/latest.pt --matches 12 --workers 12

``--smoke`` shrinks everything for a laptop check.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doom_env import (
    _ACTION_INDEX,
    ACTIONS,
    BOT_SETS,
    FEATURE_DIM,
    MATCH_TICS,
    PRIV_DIM,
    TIC_HZ,
    WEAPON_SLOTS,
    DoomEnv,
    Observation,
    features,
    isolate_workdir,
    priv_features,
)
from expert import BEHAVIORS, PLAN_EVERY_TICS, Expert

STYLES = BEHAVIORS
N_MOVE = len(ACTIONS)
N_WEAPON = len(WEAPON_SLOTS) + 1  # the slots, then "keep the current weapon"
KEEP = len(WEAPON_SLOTS)
OBS_DIM = FEATURE_DIM + len(STYLES) + 1  # + style one-hot + planner-tic flag
WEAPON_PREF = {1: 0.0, 2: 1.0, 3: 5.0, 4: 5.0, 5: 5.0, 6: 10.0, 7: 10.0}


# ── Reward ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Shaping:
    """Reward per unit change (Sample Factory's deathmatch shaping by default)."""

    frag: float = 1.0
    suicide: float = -1.5
    death: float = -0.75
    hit: float = 0.01
    dealt: float = 0.003
    hp_gain: float = 0.005
    hp_loss: float = -0.003
    armor_gain: float = 0.005
    armor_loss: float = -0.001
    weapon: float = 0.02  # x preference, per weapon picked up
    ammo_gain: float = 0.0002  # x preference, per unit
    ammo_loss: float = -0.0001


STYLE_SHAPING = {
    "fighter": Shaping(),
    "cautious": Shaping(death=-2.0, hp_loss=-0.01, armor_loss=-0.005, hp_gain=0.01),
    "collector": Shaping(
        frag=0.5, hp_gain=0.02, armor_gain=0.02, weapon=0.1, ammo_gain=0.001
    ),
}


def shaped_reward(prev: Observation, obs: Observation, w: Shaping) -> float:
    c0, c1 = prev.counters, obs.counters
    df = c1["frags"] - c0["frags"]
    r = w.frag * max(df, 0.0) + w.suicide * max(-df, 0.0)
    r += w.death * (c1["deaths"] - c0["deaths"])
    r += w.hit * (c1["hits"] - c0["hits"]) + w.dealt * (c1["dealt"] - c0["dealt"])
    if prev.dead or obs.dead:
        return r  # a death or a respawn resets the HUD; those deltas are not play
    dh, da = obs.hp - prev.hp, obs.armor - prev.armor
    r += w.hp_gain * max(dh, 0) + w.hp_loss * max(-dh, 0)
    r += w.armor_gain * max(da, 0) + w.armor_loss * max(-da, 0)
    for s, ammo in obs.arms.items():
        if s not in prev.arms:
            r += w.weapon * WEAPON_PREF[s]
        else:
            d = ammo - prev.arms[s]
            r += (w.ammo_gain * d if d > 0 else w.ammo_loss * -d) * WEAPON_PREF[s]
    return r


# ── Environment side ───────────────────────────────────────────────────────────
@dataclass
class Config:
    envs: int = 160
    workers: int = 40
    rollout: int = 128  # steps per env per update
    total_steps: int = 500_000_000
    episode_s: float = 180.0
    # Labels come from the rendered frame; 160x120 sees the same objects as
    # 640x480 (327 vs 332 enemy sightings on one seed) and steps 2.6x faster.
    resolution: str = "160X120"
    bots: str = "default"  # a BOT_SETS name, or "mix" (easy/default/hard per episode)
    style_switch_s: float = 40.0  # mean seconds between mid-episode style switches
    lr: float = 3e-4
    gamma: float = 0.998  # frameskip 1: 0.998^35 = 0.93 per second
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    epochs: int = 4
    minibatches: int = 4
    max_grad_norm: float = 0.5
    ks_coef: float = 1.0  # kickstarting weight at step 0 ...
    ks_steps: int = 40_000_000  # ... decaying linearly to 0 by here
    hidden: int = 256
    eval_every: int = 50  # updates
    eval_matches: int = 8
    eval_workers: int = 8
    eval_seconds: float = MATCH_TICS / TIC_HZ
    seed: int = 0


def obs_vector(obs: Observation, style: str) -> np.ndarray:
    extra = np.zeros(len(STYLES) + 1, dtype=np.float32)
    extra[STYLES.index(style)] = 1.0
    extra[-1] = float(obs.tick % PLAN_EVERY_TICS == 0)
    return np.concatenate([features(obs), extra])


class _Slot:
    """One env inside a worker, with its episode bookkeeping."""

    def __init__(self, cfg: Config, seed: int):
        self.cfg, self.rng = cfg, random.Random(seed)
        self.seed = seed
        self.env = DoomEnv(
            seed=seed,
            resolution=cfg.resolution,
            timeout_tics=int(cfg.episode_s * TIC_HZ),
        )
        self.episode = 0
        self.expert = Expert()
        self.new_episode()

    def _bots(self) -> str:
        if self.cfg.bots == "mix":
            return self.rng.choice(("easy", "default", "default", "hard"))
        return self.cfg.bots

    def new_episode(self) -> None:
        self.episode += 1
        self.bots = self._bots()
        self.obs = self.env.reset(seed=self.seed * 1000 + self.episode, bots=self.bots)
        self.style = self.rng.choice(STYLES)
        self.ret = 0.0
        self.expert.reset()
        self._skip_dead()

    def _skip_dead(self) -> float:
        """Dead tics need no decision: step through them, keeping their reward."""
        r = 0.0
        while self.obs.dead and not self.obs.done:
            prev = self.obs
            self.obs = self.env.step("wait")
            if not self.obs.done:
                r += shaped_reward(prev, self.obs, STYLE_SHAPING[self.style])
        return r

    def step(self, move: int, weapon: int) -> tuple[float, bool, dict | None]:
        prev = self.obs
        slot = None
        if prev.tick % PLAN_EVERY_TICS == 0:
            slot = prev.slot if weapon == KEEP else WEAPON_SLOTS[weapon]
        self.obs = self.env.step(ACTIONS[move], weapon=slot)
        r = 0.0
        if not self.obs.done:
            r = shaped_reward(prev, self.obs, STYLE_SHAPING[self.style])
            r += self._skip_dead()
        self.ret += r
        info = None
        if self.obs.done:
            info = {
                "style": self.style,
                "bots": self.bots,
                "return": round(self.ret, 3),
                **{
                    k: v
                    for k, v in self.env.stats.as_dict().items()
                    if k in ("frags", "deaths", "margin", "rank", "top", "pickups")
                },
            }
            self.new_episode()
            return r, True, info
        # Occasional mid-episode style switches, as the live demo does.
        switch_p = 1.0 / max(1.0, self.cfg.style_switch_s * TIC_HZ)
        if self.rng.random() < switch_p:
            self.style = self.rng.choice(STYLES)
        return r, False, info

    def views(self) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Actor input, critic extra input, and the scripted player's move and
        weapon-head labels for kickstarting."""
        frac = self.obs.tick / max(1, self.env.game.get_episode_timeout())
        move = _ACTION_INDEX[self.expert.act(self.obs, self.style)]
        slot = self.expert.weapon(self.obs, self.style)
        weapon = KEEP if slot == self.obs.slot else WEAPON_SLOTS.index(slot)
        return (
            obs_vector(self.obs, self.style),
            priv_features(self.obs, frac),
            move,
            weapon,
        )


def _env_worker(conn, cfg: Config, seeds: list[int]) -> None:
    isolate_workdir()
    slots = [_Slot(cfg, s) for s in seeds]

    def gather():
        v = [s.views() for s in slots]
        return (
            np.stack([x[0] for x in v]),
            np.stack([x[1] for x in v]),
            np.asarray([x[2] for x in v]),
            np.asarray([x[3] for x in v]),
        )

    conn.send(gather())
    while True:
        msg = conn.recv()
        if msg is None:
            break
        moves, weapons = msg
        t0 = time.perf_counter()
        rews, dones, infos = [], [], []
        for s, m, w in zip(slots, moves, weapons):
            r, d, info = s.step(int(m), int(w))
            rews.append(r)
            dones.append(d)
            if info:
                infos.append(info)
        views = gather()
        busy = time.perf_counter() - t0
        conn.send(
            (*views, np.asarray(rews, np.float32), np.asarray(dones), infos, busy)
        )
    for s in slots:
        s.env.close()


class VecEnv:
    """``cfg.envs`` envs across ``cfg.workers`` processes, stepped in lockstep."""

    def __init__(self, cfg: Config):
        ctx = mp.get_context("spawn")
        per = [
            cfg.envs // cfg.workers + (i < cfg.envs % cfg.workers)
            for i in range(cfg.workers)
        ]
        self.conns, self.procs, self.sizes = [], [], []
        seed = cfg.seed * 100_000
        for n in per:
            if n == 0:
                continue
            parent, child = ctx.Pipe()
            seeds = list(range(seed, seed + n))
            seed += n
            p = ctx.Process(target=_env_worker, args=(child, cfg, seeds), daemon=True)
            p.start()
            self.conns.append(parent)
            self.procs.append(p)
            self.sizes.append(n)

    def reset(self) -> tuple[np.ndarray, ...]:
        """(obs, priv, expert move, expert weapon) for every env."""
        parts = [c.recv() for c in self.conns]
        return tuple(np.concatenate([p[k] for p in parts]) for k in range(4))

    def step(self, moves: np.ndarray, weapons: np.ndarray):
        """-> (obs, priv, expert move, expert weapon, reward, done, infos)."""
        i = 0
        for c, n in zip(self.conns, self.sizes):
            c.send((moves[i : i + n], weapons[i : i + n]))
            i += n
        parts = [c.recv() for c in self.conns]
        arrays = tuple(np.concatenate([p[k] for p in parts]) for k in range(6))
        self.busy = [p[7] for p in parts]  # seconds each worker spent stepping
        return (*arrays, [x for p in parts for x in p[6]])

    def close(self) -> None:
        for c in self.conns:
            c.send(None)
        for p in self.procs:
            p.join(timeout=10)


# ── Model ──────────────────────────────────────────────────────────────────────
def _torch():
    import torch
    from torch import nn

    return torch, nn


def build_agent(hidden: int = 256):
    torch, nn = _torch()

    class Agent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(OBS_DIM, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()
            )
            self.gru = nn.GRU(256, hidden)
            self.pi_move = nn.Linear(hidden, N_MOVE)
            self.pi_weapon = nn.Linear(hidden, N_WEAPON)
            self.v = nn.Sequential(
                nn.Linear(hidden + PRIV_DIM, 256), nn.ReLU(), nn.Linear(256, 1)
            )
            for m in (self.pi_move, self.pi_weapon):
                nn.init.orthogonal_(m.weight, 0.01)
                nn.init.zeros_(m.bias)

        def initial_state(self, n: int, device=None):
            return torch.zeros(1, n, self.gru.hidden_size, device=device)

        def sequence(self, obs, starts, h):
            """obs [T, B, OBS_DIM], starts [T, B] (1 = first tic of an episode) ->
            GRU outputs [T, B, H] and the final state. The state is zeroed at
            every episode start.

            Episode starts are rare (one per ~6000 tics per env), so the time
            axis is cut only where some env starts one, and each piece is one
            cuDNN call instead of a Python loop over tics.
            """
            x = self.enc(obs)
            T = obs.shape[0]
            cuts = [0]
            if T > 1:
                inner = torch.nonzero(starts[1:].any(dim=1)).flatten() + 1
                cuts += inner.tolist()
            cuts.append(T)
            outs = []
            for a, b in itertools.pairwise(cuts):
                h = h * (1.0 - starts[a]).view(1, -1, 1)
                o, h = self.gru(x[a:b], h)
                outs.append(o)
            return torch.cat(outs), h

        def heads(self, z, priv):
            v = self.v(torch.cat([z, priv], -1)).squeeze(-1)
            return self.pi_move(z), self.pi_weapon(z), v

    return Agent()


def save(agent, cfg: Config, path: Path, extra: dict) -> None:
    torch, _ = _torch()
    tmp = path.with_suffix(".tmp")
    torch.save(
        {"state_dict": agent.state_dict(), "config": asdict(cfg), **extra}, str(tmp)
    )
    os.replace(tmp, path)


def load(path: str | Path, device: str = "cpu"):
    torch, _ = _torch()
    ck = torch.load(str(path), map_location=device, weights_only=False)
    cfg = Config(**ck["config"])
    agent = build_agent(cfg.hidden).to(device)
    agent.load_state_dict(ck["state_dict"])
    agent.eval()
    return agent, cfg, ck


class RLPolicy:
    """A trained teacher for one game: call :meth:`step` on every live tic.

    Keeps the GRU state across tics (dead tics are skipped, as in training).
    Returns the movement and weapon distributions; the weapon one only matters
    on planner tics.
    """

    def __init__(self, ckpt: str | Path, device: str = "cpu"):
        torch, _ = _torch()
        torch.set_num_threads(1)
        self.torch = torch
        self.agent, self.cfg, _ = load(ckpt, device)
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.h = self.agent.initial_state(1, self.device)
        self.first = True

    def step(self, obs: Observation, style: str) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        with torch.no_grad():
            x = torch.as_tensor(obs_vector(obs, style), device=self.device).view(
                1, 1, -1
            )
            s = torch.tensor([[1.0 if self.first else 0.0]], device=self.device)
            z, self.h = self.agent.sequence(x, s, self.h)
            zm = z[0]
            pm = torch.softmax(self.agent.pi_move(zm), -1)[0].cpu().numpy()
            pw = torch.softmax(self.agent.pi_weapon(zm), -1)[0].cpu().numpy()
        self.first = False
        return pm, pw

    @staticmethod
    def weapon_slot(pw: np.ndarray, obs: Observation, sample: bool, rng) -> int:
        i = rng.choices(range(N_WEAPON), weights=pw)[0] if sample else int(pw.argmax())
        return obs.slot if i == KEEP else WEAPON_SLOTS[i]


# ── Evaluation ─────────────────────────────────────────────────────────────────
def play_match(args: tuple) -> dict:
    """One full match with a checkpoint on CPU; the collect.py stats row."""
    ckpt, seed, style, bots, seconds, sample = args
    isolate_workdir()
    pol = RLPolicy(ckpt)
    rng = random.Random(seed)
    env = DoomEnv(seed=seed, timeout_tics=int(seconds * TIC_HZ), bots=bots)
    obs = env.reset(seed=seed)
    while not obs.done:
        if obs.dead:
            obs = env.step("wait")
            continue
        pm, pw = pol.step(obs, style)
        m = rng.choices(range(N_MOVE), weights=pm)[0] if sample else int(pm.argmax())
        slot = None
        if obs.tick % PLAN_EVERY_TICS == 0:
            slot = RLPolicy.weapon_slot(pw, obs, sample, rng)
        obs = env.step(ACTIONS[m], weapon=slot)
    env.close()
    return {
        "behavior": style,
        "ep": seed,
        "seed": seed,
        "bots": bots,
        "policy": f"rl:{Path(ckpt).name}",
        **env.stats.as_dict(),
    }


def eval_tasks(ckpt, matches: int, styles, bots: str, seconds: float, seed: int):
    return [
        (str(ckpt), seed + i, st, bots, seconds, True)
        for st in styles
        for i in range(matches)
    ]


# ── Training ───────────────────────────────────────────────────────────────────
def train(cfg: Config, out: Path, device: str, resume: Path | None) -> None:
    torch, nn = _torch()
    torch.manual_seed(cfg.seed)
    if device == "cpu":
        torch.set_num_threads(4)  # leave the cores to the env workers
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    agent = build_agent(cfg.hidden).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)
    update, steps = 0, 0
    if resume is not None:
        ck = torch.load(str(resume), map_location=device, weights_only=False)
        agent.load_state_dict(ck["state_dict"])
        update, steps = ck.get("update", 0), ck.get("steps", 0)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
    log = open(out / "train.jsonl", "a")
    evlog = open(out / "eval.jsonl", "a")
    eval_pool = mp.get_context("spawn").Pool(cfg.eval_workers)
    pending_evals: list[tuple[int, object]] = []

    venv = VecEnv(cfg)
    B, T = cfg.envs, cfg.rollout
    obs_np, priv_np, exm_np, exw_np = venv.reset()
    next_obs = torch.as_tensor(obs_np, device=device)
    next_priv = torch.as_tensor(priv_np, device=device)
    next_exm = torch.as_tensor(exm_np, device=device)
    next_exw = torch.as_tensor(exw_np, device=device)
    next_start = torch.ones(B, device=device)
    h = agent.initial_state(B, device)

    obs_b = torch.zeros(T, B, OBS_DIM, device=device)
    priv_b = torch.zeros(T, B, PRIV_DIM, device=device)
    move_b = torch.zeros(T, B, dtype=torch.long, device=device)
    weap_b = torch.zeros(T, B, dtype=torch.long, device=device)
    exm_b = torch.zeros(T, B, dtype=torch.long, device=device)
    exw_b = torch.zeros(T, B, dtype=torch.long, device=device)
    logp_b = torch.zeros(T, B, device=device)
    plan_b = torch.zeros(T, B, device=device)
    rew_b = torch.zeros(T, B, device=device)
    start_b = torch.zeros(T, B, device=device)
    val_b = torch.zeros(T, B, device=device)
    episodes: list[dict] = []
    t_start, steps_start = time.time(), steps
    n_updates = cfg.total_steps // (B * T)

    while update < n_updates:
        frac = 1.0 - update / n_updates
        for g in opt.param_groups:
            g["lr"] = cfg.lr * frac
        h0 = h.clone()
        t_env, t_roll = 0.0, time.time()
        busy_mean, busy_max, t_policy = 0.0, 0.0, 0.0
        for t in range(T):
            obs_b[t], priv_b[t], start_b[t] = next_obs, next_priv, next_start
            exm_b[t], exw_b[t] = next_exm, next_exw
            plan = next_obs[:, -1]
            plan_b[t] = plan
            tp = time.time()
            with torch.no_grad():
                z, h = agent.sequence(next_obs[None], next_start[None], h)
                lm, lw, v = agent.heads(z[0], next_priv)
                dm = torch.distributions.Categorical(logits=lm)
                dw = torch.distributions.Categorical(logits=lw)
                m, w = dm.sample(), dw.sample()
                logp_b[t] = dm.log_prob(m) + plan * dw.log_prob(w)
            move_b[t], weap_b[t], val_b[t] = m, w, v
            mc, wc = m.cpu().numpy(), w.cpu().numpy()
            te = time.time()
            t_policy += te - tp
            o, p, em, ew, r, d, infos = venv.step(mc, wc)
            t_env += time.time() - te
            busy_mean += float(np.mean(venv.busy))
            busy_max += float(np.max(venv.busy))
            episodes += infos
            next_obs = torch.as_tensor(o, device=device)
            next_priv = torch.as_tensor(p, device=device)
            next_exm = torch.as_tensor(em, device=device)
            next_exw = torch.as_tensor(ew, device=device)
            rew_b[t] = torch.as_tensor(r, device=device)
            next_start = torch.as_tensor(d.astype(np.float32), device=device)
        steps += B * T
        t_roll = time.time() - t_roll

        # GAE. start_b[t+1] marks that step t ended its episode.
        with torch.no_grad():
            z, _ = agent.sequence(next_obs[None], next_start[None], h.clone())
            _, _, next_v = agent.heads(z[0], next_priv)
            adv = torch.zeros_like(rew_b)
            last = torch.zeros(B, device=device)
            for t in reversed(range(T)):
                nonterm = 1.0 - (next_start if t == T - 1 else start_b[t + 1])
                nv = next_v if t == T - 1 else val_b[t + 1]
                delta = rew_b[t] + cfg.gamma * nv * nonterm - val_b[t]
                last = delta + cfg.gamma * cfg.gae_lambda * nonterm * last
                adv[t] = last
            ret = adv + val_b

        # PPO over whole-env sequences (the GRU is re-run from h0).
        env_idx = np.arange(B)
        mb = B // cfg.minibatches
        ks = cfg.ks_coef * max(0.0, 1.0 - steps / max(1, cfg.ks_steps))
        stats = {
            "pg": 0.0,
            "vf": 0.0,
            "ent_m": 0.0,
            "ent_w": 0.0,
            "kl": 0.0,
            "clipfrac": 0.0,
            "ks_ce": 0.0,
            "expert_agree": 0.0,
        }
        n_mb = 0
        for _ in range(cfg.epochs):
            np.random.shuffle(env_idx)
            for k in range(0, B, mb):
                ix = torch.as_tensor(env_idx[k : k + mb], device=device)
                z, _ = agent.sequence(obs_b[:, ix], start_b[:, ix], h0[:, ix])
                lm, lw, v = agent.heads(z, priv_b[:, ix])
                dm = torch.distributions.Categorical(logits=lm)
                dw = torch.distributions.Categorical(logits=lw)
                pl = plan_b[:, ix]
                newlogp = dm.log_prob(move_b[:, ix]) + pl * dw.log_prob(weap_b[:, ix])
                ratio = (newlogp - logp_b[:, ix]).exp()
                a = adv[:, ix]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg = torch.max(
                    -a * ratio, -a * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)
                ).mean()
                vf = 0.5 * ((v - ret[:, ix]) ** 2).mean()
                ent_m = dm.entropy().mean()
                ent_w = (pl * dw.entropy()).sum() / pl.sum().clamp(min=1)
                ce_m = torch.nn.functional.cross_entropy(
                    lm.reshape(-1, N_MOVE), exm_b[:, ix].reshape(-1)
                )
                ce_w_all = torch.nn.functional.cross_entropy(
                    lw.reshape(-1, N_WEAPON), exw_b[:, ix].reshape(-1), reduction="none"
                )
                ce_w = (pl.reshape(-1) * ce_w_all).sum() / pl.sum().clamp(min=1)
                loss = pg + cfg.vf_coef * vf - cfg.ent_coef * (ent_m + 0.5 * ent_w)
                if ks > 0:
                    loss = loss + ks * (ce_m + ce_w)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    lr_ = newlogp - logp_b[:, ix]
                    stats["kl"] += float(((lr_.exp() - 1) - lr_).mean())
                    stats["clipfrac"] += float(
                        ((ratio - 1).abs() > cfg.clip).float().mean()
                    )
                stats["pg"] += pg.item()
                stats["vf"] += vf.item()
                stats["ent_m"] += ent_m.item()
                stats["ent_w"] += ent_w.item()
                stats["ks_ce"] += ce_m.item()
                stats["expert_agree"] += float(
                    (lm.argmax(-1) == exm_b[:, ix]).float().mean()
                )
                n_mb += 1
        update += 1

        rec = {
            "update": update,
            "steps": steps,
            "sps": round((steps - steps_start) / (time.time() - t_start)),
            "env_frac": round(t_env / max(1e-9, t_roll), 3),
            # per rollout step, ms: waiting on workers, their mean and slowest
            # busy time, and the policy forward
            "ms_env": round(1000 * t_env / T, 2),
            "ms_busy_mean": round(1000 * busy_mean / T, 2),
            "ms_busy_max": round(1000 * busy_max / T, 2),
            "ms_policy": round(1000 * t_policy / T, 2),
            "s_rollout": round(t_roll, 2),
            "lr": round(cfg.lr * frac, 7),
            "ks": round(ks, 4),
            **{k: round(v / n_mb, 5) for k, v in stats.items()},
        }
        if episodes:
            for st in STYLES:
                es = [e for e in episodes if e["style"] == st]
                if es:
                    rec[f"{st}_return"] = round(
                        float(np.mean([e["return"] for e in es])), 3
                    )
                    rec[f"{st}_frags"] = round(
                        float(np.mean([e["frags"] for e in es])), 2
                    )
                    rec[f"{st}_deaths"] = round(
                        float(np.mean([e["deaths"] for e in es])), 2
                    )
            rec["episodes"] = len(episodes)
            rec["margin"] = round(float(np.mean([e["margin"] for e in episodes])), 2)
            rec["top"] = round(float(np.mean([e["top"] for e in episodes])), 3)
            episodes = []
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

        if update % cfg.eval_every == 0 or update == n_updates:
            ck = out / f"ckpt_{update:06d}.pt"
            save(agent, cfg, ck, {"update": update, "steps": steps})
            save(
                agent,
                cfg,
                out / "latest.pt",
                {"update": update, "steps": steps, "opt": opt.state_dict()},
            )
            tasks = eval_tasks(
                ck, cfg.eval_matches, ("fighter",), "default", cfg.eval_seconds, 777
            )
            pending_evals.append((update, eval_pool.map_async(play_match, tasks)))
        for u, res in list(pending_evals):
            if res.ready():
                pending_evals.remove((u, res))
                rows = res.get()
                summary = summarize_eval(rows)
                evlog.write(
                    json.dumps({"update": u, **summary, "matches": rows}) + "\n"
                )
                evlog.flush()
                print(f"EVAL update {u}: {json.dumps(summary)}", flush=True)
    venv.close()
    for u, res in pending_evals:
        rows = res.get()
        evlog.write(
            json.dumps({"update": u, **summarize_eval(rows), "matches": rows}) + "\n"
        )
    eval_pool.close()
    log.close()
    evlog.close()


def summarize_eval(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "frags": round(float(np.mean([r["frags"] for r in rows])), 2),
        "deaths": round(float(np.mean([r["deaths"] for r in rows])), 2),
        "best_bot": round(float(np.mean([r["best_bot"][1] for r in rows])), 2),
        "margin": round(float(np.mean([r["margin"] for r in rows])), 2),
        "top": round(float(np.mean([r["top"] for r in rows])), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--device", default="cuda")
    t.add_argument("--resume", type=Path)
    t.add_argument("--smoke", action="store_true", help="Tiny run for a laptop")
    for f in Config.__dataclass_fields__.values():
        t.add_argument(
            f"--{f.name.replace('_', '-')}", type=type(f.default), default=None
        )
    e = sub.add_parser("eval")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--matches", type=int, default=12)
    e.add_argument("--styles", nargs="+", default=["fighter"], choices=STYLES)
    e.add_argument("--bots", default="default", choices=sorted(BOT_SETS))
    e.add_argument("--seconds", type=float, default=MATCH_TICS / TIC_HZ)
    e.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    e.add_argument("--seed", type=int, default=5000)
    e.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.cmd == "train":
        args.out = args.out.resolve()
        cfg = Config()
        if args.smoke:
            cfg = replace(
                cfg,
                envs=4,
                workers=2,
                rollout=32,
                total_steps=4 * 32 * 3,
                episode_s=20.0,
                eval_every=2,
                eval_matches=1,
                eval_workers=1,
                eval_seconds=20.0,
            )
        overrides = {
            k: getattr(args, k)
            for k in Config.__dataclass_fields__
            if getattr(args, k, None) is not None
        }
        cfg = replace(cfg, **overrides)
        if cfg.bots != "mix" and cfg.bots not in BOT_SETS:
            raise SystemExit(f"--bots must be mix or one of {sorted(BOT_SETS)}")
        train(cfg, args.out, args.device, args.resume)
        return

    from collect import summarize

    args.ckpt = str(Path(args.ckpt).resolve())
    tasks = eval_tasks(
        args.ckpt, args.matches, args.styles, args.bots, args.seconds, args.seed
    )
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = []
        for r in pool.imap_unordered(play_match, tasks):
            rows.append(r)
            print(
                f"[{len(rows)}/{len(tasks)}] {r['behavior']:<9} frags {r['frags']:>3} "
                f"deaths {r['deaths']:>3} margin {r['margin']:>+4} rank {r['rank']}",
                flush=True,
            )
    print(summarize(rows))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    main()
