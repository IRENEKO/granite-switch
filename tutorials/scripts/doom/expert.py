# SPDX-License-Identifier: Apache-2.0
"""Scripted teacher for the Doom reflex demo: three behaviors, one action per tic.

The expert decides almost entirely from the player-visible fields of an
:class:`~doom_env.Observation` (the same values that appear in ``obs.text``), so
a student reading only the text can in principle match it. The wall clearances
are computed from privileged sector geometry but are serialized for the student
too. The only privileged state the expert keeps for itself is an item
blacklist, which drops items it has failed to reach, so it never circles an
unreachable pickup forever.

Behaviors:

* ``hunter``: seeks the nearest monster, aims, fires; picks up health only
  when hp < 25.
* ``survivor``: keeps distance by backing away while facing the threat,
  sidesteps far ones, fights only when cornered (a monster within 3 m, or
  within 6 m with a wall at its back), and prioritizes health and armor.
* ``scavenger``: collects items nearest-first; fires only at monsters within 5 m.
"""

from __future__ import annotations

from doom_env import TIC_HZ, Observation, Seen

BEHAVIORS: tuple[str, ...] = ("hunter", "survivor", "scavenger")

_STALL_TICS = 2 * TIC_HZ  # no progress toward an item for this long ...
_STALL_PROGRESS_M = 0.5
_BLACKLIST_TICS = 10 * TIC_HZ  # ... blacklists it for this long


# ── Motor primitives (pure functions of the visible state) ──────────────────────
def aim_fire(b: int) -> str:
    """Fire when on target, fine-aim while firing when close, turn otherwise."""
    if abs(b) <= 3:
        return "fire"
    if abs(b) <= 12:
        return "al" if b < 0 else "ar"
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
        return "left" if b < 0 else "right"
    if abs(b) > 5:
        return "fl" if b < 0 else "fr"
    return "forward"


def turn_to(b: int) -> str:
    if abs(b) <= 10:
        return "forward"
    return "left" if b < 0 else "right"


def face(b: int, tolerance: int) -> str | None:
    """Turn toward bearing ``b`` if it is more than ``tolerance`` off-centre."""
    if abs(b) <= tolerance:
        return None
    return "left" if b < 0 else "right"


def sidestep(obs: Observation, b: int) -> str:
    """Strafe away from the side a threat at bearing ``b`` is on, if there is room."""
    l, _, r, _ = obs.walls
    side = "sr" if b <= 0 else "sl"
    if side == "sr" and r == 0:
        return "sl"
    if side == "sl" and l == 0:
        return "sr"
    return side


def back_off(obs: Observation) -> str:
    l, _, r, bk = obs.walls
    if bk >= 2:
        return "back"
    return "sl" if l >= r else "sr"


class Expert:
    """Scripted teacher. Use one instance per environment and ``reset`` per episode."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._blacklist: dict[int, int] = {}  # item id -> tick it expires
        self._track: tuple[int, float, int] | None = None  # id, best dist, tick

    def act(self, obs: Observation, behavior: str) -> str:
        if behavior == "hunter":
            return self._hunter(obs)
        if behavior == "survivor":
            return self._survivor(obs)
        if behavior == "scavenger":
            return self._scavenger(obs)
        raise ValueError(f"unknown behavior {behavior!r}")

    # ── Target bookkeeping ────────────────────────────────────────────────────
    def _items(
        self, obs: Observation, kinds: tuple[str, ...] | None = None
    ) -> list[Seen]:
        t = obs.tick
        return [
            o
            for o in obs.seen
            if o.kind != "monster"
            and (kinds is None or o.kind in kinds)
            and self._blacklist.get(o.id, -1) < t
        ]

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

    # ── Behaviors ─────────────────────────────────────────────────────────────
    def _hunter(self, obs: Observation) -> str:
        monsters = [o for o in obs.seen if o.kind == "monster"]
        health = self._items(obs, ("health",))
        if obs.hp < 25 and health:
            return self._go_to(obs, health[0])
        if monsters:
            m = monsters[0]
            if obs.weapon == "fist" and m.d > 2:
                return steer(obs, m)
            if m.d > 16 and abs(m.b) <= 5 and obs.walls[1] > 2:
                return "forward"  # close in: hitscan spread wastes long shots
            return aim_fire(m.b)
        weapons = self._items(obs, ("weapon",))
        if weapons:
            return self._go_to(obs, weapons[0])
        ammo = self._items(obs, ("ammo",))
        if ammo and (obs.weapon == "fist" or obs.ammo < 20):
            return self._go_to(obs, ammo[0])
        if obs.enemy_mem is not None:
            return turn_to(obs.enemy_mem[0])
        if obs.hit > 0:
            return "right"  # shot by something unseen: turn to find it
        return roam(obs)

    def _survivor(self, obs: Observation) -> str:
        monsters = [o for o in obs.seen if o.kind == "monster"]
        protect = self._items(obs, ("health", "armor", "bonus"))
        if monsters:
            m = monsters[0]
            cornered = obs.walls[3] <= 1 and m.d <= 6
            if m.d <= 3 or cornered:
                return aim_fire(m.b)
            if m.d < 14:  # too close: back away, keeping it in view
                return face(m.b, 25) or back_off(obs)
            if protect and protect[0].d < m.d - 4:
                return self._go_to(obs, protect[0])
            return sidestep(obs, m.b)  # far: stay out of its line of fire
        if obs.hit > 0:
            return "right"  # shot by something unseen: find it
        if obs.enemy_mem is not None:
            return face(obs.enemy_mem[0], 10) or back_off(obs)
        if protect:
            return self._go_to(obs, protect[0])
        return roam(obs)

    def _scavenger(self, obs: Observation) -> str:
        monsters = [o for o in obs.seen if o.kind == "monster"]
        if monsters and monsters[0].d <= 5:
            return aim_fire(monsters[0].b)
        items = self._items(obs)
        if items:
            return self._go_to(obs, items[0])
        return roam(obs)
