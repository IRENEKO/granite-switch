# SPDX-License-Identifier: Apache-2.0
"""Scripted deathmatch player: the baseline, and the fallback teacher.

The expert decides almost entirely from the player-visible fields of an
:class:`~doom_env.Observation` (the same values that appear in ``obs.text``), so
a student reading only the text can in principle match it. Beyond that it keeps
a little state of its own: which way it is dodging and for how long, and a
blacklist of items it failed to reach, so it never circles an unreachable
pickup forever.

Three styles, sharing the motor primitives below:

* ``fighter``: takes the best weapon for the range, keeps the nearest bot
  centred, fires while strafing (flipping direction every half second or at a
  wall), charges with the shotgun and backs off with the launcher, and hunts
  the last place a bot was seen. Picks up weapons first when nothing is in
  sight; heals below 40 hp only when no bot is close.
* ``cautious``: fights only bots within 10 m, or when hit; backs away while
  firing; below 60 hp leaves fights for health and armor.
* ``collector``: gathers items nearest-first; fights only bots within 6 m, or
  when hit.

:meth:`Expert.weapon` is the weapon planner: a slot for the current range and
ammo. Callers ask it about every half second (``PLAN_EVERY_TICS``) and pass the
slot to :meth:`doom_env.DoomEnv.step`.
"""

from __future__ import annotations

from doom_env import TIC_HZ, Observation, Seen

BEHAVIORS: tuple[str, ...] = ("fighter", "cautious", "collector")
PLAN_EVERY_TICS = TIC_HZ // 2  # the weapon planner's cadence, ~0.5 s

_STALL_TICS = 2 * TIC_HZ  # no progress toward an item for this long ...
_STALL_PROGRESS_M = 0.5
_BLACKLIST_TICS = 10 * TIC_HZ  # ... blacklists it for this long
_DODGE_TICS = (12, 24)  # hold a strafe direction for this many tics
_AIM_ON = 3  # |bearing| at which a shot is on target
_AIM_FINE = 12  # |bearing| within which to fine-aim while firing
_PICKUP_KINDS = {
    "fighter": ("weapon", "ammo", "armor", "health"),
    "cautious": ("health", "armor", "weapon", "ammo"),
    "collector": None,  # everything, nearest first
}


# ── Motor primitives (pure functions of the visible state) ──────────────────────
def turn(b: int) -> str:
    return "left" if b < 0 else "right"


def avoid(obs: Observation) -> str:
    """Turn in place toward the more open side, continuing a turn already begun."""
    if obs.last[-1] in ("left", "right"):
        return obs.last[-1]
    l, _, r, _ = obs.walls
    return "left" if l >= r else "right"


def roam(obs: Observation) -> str:
    """Run through open space, veering away from walls."""
    l, f, r, _ = obs.walls
    if f <= 2:
        return avoid(obs)
    if f <= 5:
        return "fl" if l > r else "fr"
    if l == 0:
        return "fr"
    if r == 0:
        return "fl"
    return "forward"


def steer(obs: Observation, target: Seen) -> str:
    """Run toward a target, unless a wall is in the way before it."""
    f = obs.walls[1]
    if f <= 2 and target.dist > f + 1.5:
        return avoid(obs)
    b = target.b
    if abs(b) > 30:
        return turn(b)
    if abs(b) > 5:
        return "fl" if b < 0 else "fr"
    return "forward"


def weapon_for(obs: Observation, dist: float | None) -> int:
    """Best owned weapon with ammo for a bot at ``dist`` metres (None: none seen)."""
    arms = obs.arms
    d = 10.0 if dist is None else dist

    def has(slot: int, n: int = 1) -> bool:
        return arms.get(slot, 0) >= n

    if has(6):
        return 6  # plasma: fast, accurate, no splash
    if has(3, 2) and d <= 8:
        return 3  # super shotgun up close
    if has(5) and d >= 6:
        return 5  # launcher only where its splash cannot reach us
    if has(7, 40) and d >= 6:
        return 7
    if has(4):
        return 4
    if has(3, 2):
        return 3
    if has(2):
        return 2
    return 1


class Expert:
    """Scripted deathmatch player. One instance per environment; ``reset`` per match."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._blacklist: dict[int, int] = {}  # item id -> tick it expires
        self._track: tuple[int, float, int] | None = None  # id, best dist, tick
        self._dodge = "l"
        self._dodge_until = 0

    # ── Public interface ───────────────────────────────────────────────────────
    def act(self, obs: Observation, behavior: str) -> str:
        if obs.dead:
            return "wait"
        if behavior not in _PICKUP_KINDS:
            raise ValueError(f"unknown behavior {behavior!r}")
        foes = [o for o in obs.seen if o.kind == "enemy"]
        target = self._target(foes)
        engage = target is not None and self._engages(obs, behavior, target)
        if engage:
            return self._fight(obs, behavior, target)
        heal = self._items(obs, ("health", "armor"))
        if heal and self._wants_heal(obs, behavior, target):
            return self._go_to(obs, heal[0])
        if target is not None and behavior == "cautious":
            return self._evade(obs, target)
        if obs.hit > 0 and behavior != "collector":
            return self._find_shooter(obs)
        items = self._items(obs, _PICKUP_KINDS[behavior])
        items = [o for o in items if self._useful(obs, o)]
        if items:
            return self._go_to(obs, items[0])
        if obs.enemy_mem is not None and behavior == "fighter":
            b = obs.enemy_mem[0]
            return turn(b) if abs(b) > 10 else roam(obs)
        if obs.hit > 0:
            return self._find_shooter(obs)
        return roam(obs)

    def weapon(self, obs: Observation, behavior: str | None = None) -> int:
        """The weapon planner: a slot for the nearest visible bot's range."""
        foes = [o for o in obs.seen if o.kind == "enemy"]
        return weapon_for(obs, foes[0].dist if foes else None)

    # ── Decisions ─────────────────────────────────────────────────────────────
    @staticmethod
    def _target(foes: list[Seen]) -> Seen | None:
        """The bot to shoot: the most centred one among those near the nearest."""
        if not foes:
            return None
        near = [o for o in foes if o.dist <= foes[0].dist * 1.5 + 1]
        return min(near, key=lambda o: abs(o.bearing))

    @staticmethod
    def _engages(obs: Observation, behavior: str, m: Seen) -> bool:
        if behavior == "fighter":
            return not (obs.hp < 40 and m.d > 12)
        if behavior == "cautious":
            return m.d <= 10 or obs.hit > 0
        return m.d <= 6 or obs.hit > 0  # collector

    @staticmethod
    def _wants_heal(obs: Observation, behavior: str, m: Seen | None) -> bool:
        threshold = {"fighter": 40, "cautious": 60, "collector": 80}[behavior]
        close = m is not None and m.d <= 6
        return (obs.hp < threshold or obs.armor < 20) and not close

    def _strafe(self, obs: Observation) -> str:
        """Current dodge side, flipped on a timer or when a wall is in the way."""
        l, _, r, _ = obs.walls
        if obs.tick >= self._dodge_until:
            self._dodge = "r" if self._dodge == "l" else "l"
            span = _DODGE_TICS[obs.tick % 2]
            self._dodge_until = obs.tick + span
        if self._dodge == "l" and l == 0:
            self._dodge = "r"
        elif self._dodge == "r" and r == 0:
            self._dodge = "l"
        return self._dodge

    def _fight(self, obs: Observation, behavior: str, m: Seen) -> str:
        b, d = m.b, m.dist
        if abs(b) > _AIM_FINE:
            return turn(b)
        if abs(b) > _AIM_ON:
            return "al" if b < 0 else "ar"
        if obs.weapon == "fist":
            return "charge"
        if obs.weapon == "launcher" and d < 4 and obs.walls[3] >= 2:
            return "bf"  # too close for splash: back off while firing
        if behavior == "cautious" and d < 8 and obs.walls[3] >= 2:
            return "bf"
        if behavior == "fighter" and obs.weapon == "shotgun" and d > 6:
            return "charge" if obs.walls[1] > 2 else "fire"
        return "cl" if self._strafe(obs) == "l" else "cr"

    def _evade(self, obs: Observation, m: Seen) -> str:
        """A far bot the cautious player is not fighting: get out of its line."""
        if abs(m.b) < 60:
            return "dl" if self._strafe(obs) == "l" else "dr"
        return roam(obs)

    @staticmethod
    def _find_shooter(obs: Observation) -> str:
        if obs.enemy_mem is not None and abs(obs.enemy_mem[0]) > 10:
            return turn(obs.enemy_mem[0])
        return "right"  # shot by something unseen: turn to find it

    # ── Item bookkeeping ──────────────────────────────────────────────────────
    def _items(
        self, obs: Observation, kinds: tuple[str, ...] | None = None
    ) -> list[Seen]:
        t = obs.tick
        items = [
            o
            for o in obs.seen
            if o.kind not in ("enemy", "missile")
            and (kinds is None or o.kind in kinds)
            and self._blacklist.get(o.id, -1) < t
        ]
        if kinds is not None:
            items.sort(key=lambda o: (kinds.index(o.kind), o.dist))
        return items

    @staticmethod
    def _useful(obs: Observation, o: Seen) -> bool:
        """Skip ammo for weapons we do not own, and top-ups we do not need."""
        if o.kind == "health":
            return obs.hp < 90
        if o.kind == "armor":
            return obs.armor < 90
        if o.kind == "ammo":
            owned = {"bullets": 4, "shells": 3, "rockets": 5, "cells": 6}
            slot = owned.get(o.label)
            return slot is None or slot in obs.arms or (slot == 4 and 2 in obs.arms)
        return True

    def _go_to(self, obs: Observation, item: Seen) -> str:
        """Steer to an item, blacklisting it if we stop making progress."""
        t = obs.tick
        if self._track is None or self._track[0] != item.id:
            self._track = (item.id, item.dist, t)
        elif item.dist < self._track[1] - _STALL_PROGRESS_M:
            self._track = (item.id, item.dist, t)
        elif t - self._track[2] > _STALL_TICS:
            self._blacklist[item.id] = t + _BLACKLIST_TICS
            self._track = None
            return roam(obs)
        return steer(obs, item)
