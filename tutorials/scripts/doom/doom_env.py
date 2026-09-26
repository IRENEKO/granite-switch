# SPDX-License-Identifier: Apache-2.0
"""ViZDoom wrapper for the Granite Switch Doom reflex demo.

One :meth:`DoomEnv.step` is one Doom tic (1/35 s) and one decision. Every
:class:`Observation` carries two views of the same moment:

* **Player-visible** (``obs.text`` and the fields it is built from): HUD values,
  the objects currently on screen (read from the labels buffer) as bearing and
  distance, three wall clearances, and a small memory the wrapper keeps (last two
  actions, damage taken in the last second, last-seen enemy bearing). This is
  the only thing the model reads.
* **Privileged** (``obs.priv``): every object in the level with world positions,
  the player's exact pose, and raw ray distances. Only the scripted expert may
  use it.

Conventions: bearings are degrees relative to the player's heading, negative to
the left and positive to the right, so ``left`` shrinks a negative bearing.
Distances are metres at 32 map units per metre.
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import vizdoom as vzd
from vizdoom import Button, GameVariable

TIC_HZ = 35
TIC_MS = 1000.0 / TIC_HZ
UNITS_PER_M = 32.0  # Doomguy is 56 units tall, about 1.75 m

# ── Action space ───────────────────────────────────────────────────────────────
# Each name is one token in the Granite 4.1 tokenizer, with no leading space
# (the action follows <|end_of_role|> directly). policy.check_action_tokens
# asserts this against the real tokenizer.
ACTIONS: tuple[str, ...] = (
    "forward",
    "back",
    "left",
    "right",
    "sl",
    "sr",
    "fl",
    "fr",
    "fire",
    "al",
    "ar",
    "wait",
)
ACTION_LABELS = {
    "forward": "forward",
    "back": "back",
    "left": "turn left",
    "right": "turn right",
    "sl": "strafe left",
    "sr": "strafe right",
    "fl": "forward + left",
    "fr": "forward + right",
    "fire": "fire",
    "al": "fire + aim left",
    "ar": "fire + aim right",
    "wait": "wait",
}

TURN_DEG = 7.0  # left / right: fast turn in place
VEER_DEG = 4.0  # fl / fr: turn while running
AIM_DEG = 2.5  # al / ar: fine aim while firing

# name -> (forward, back, strafe_left, strafe_right, attack, turn_delta_deg)
# A positive TURN_LEFT_RIGHT_DELTA turns right, matching the bearing sign.
_MACROS: dict[str, tuple[int, int, int, int, int, float]] = {
    "forward": (1, 0, 0, 0, 0, 0.0),
    "back": (0, 1, 0, 0, 0, 0.0),
    "left": (0, 0, 0, 0, 0, -TURN_DEG),
    "right": (0, 0, 0, 0, 0, TURN_DEG),
    "sl": (0, 0, 1, 0, 0, 0.0),
    "sr": (0, 0, 0, 1, 0, 0.0),
    "fl": (1, 0, 0, 0, 0, -VEER_DEG),
    "fr": (1, 0, 0, 0, 0, VEER_DEG),
    "fire": (0, 0, 0, 0, 1, 0.0),
    "al": (0, 0, 0, 0, 1, -AIM_DEG),
    "ar": (0, 0, 0, 0, 1, AIM_DEG),
    "wait": (0, 0, 0, 0, 0, 0.0),
}
assert set(_MACROS) == set(ACTIONS)

_BUTTONS = [
    Button.SPEED,  # always held: Doom players run (16.7 vs 8.3 units/tic)
    Button.ATTACK,
    Button.MOVE_FORWARD,
    Button.MOVE_BACKWARD,
    Button.MOVE_LEFT,
    Button.MOVE_RIGHT,
    Button.TURN_LEFT_RIGHT_DELTA,
    *[getattr(Button, f"SELECT_WEAPON{i}") for i in range(1, 8)],
]
_N_MOVE_BUTTONS = 7  # everything before the weapon-select buttons

_VARS = [
    GameVariable.HEALTH,
    GameVariable.ARMOR,
    GameVariable.SELECTED_WEAPON,
    GameVariable.SELECTED_WEAPON_AMMO,
    GameVariable.KILLCOUNT,
    GameVariable.DAMAGE_TAKEN,
    GameVariable.POSITION_X,
    GameVariable.POSITION_Y,
    GameVariable.ANGLE,
    *[getattr(GameVariable, f"WEAPON{i}") for i in range(1, 8)],
    *[getattr(GameVariable, f"AMMO{i}") for i in range(1, 8)],
]
_V = {v: i for i, v in enumerate(_VARS)}

WEAPON_NAMES = {
    0: "fist",
    1: "fist",
    2: "pistol",
    3: "shotgun",
    4: "chaingun",
    5: "rocket",
    6: "plasma",
    7: "bfg",
}
# Auto-selected in this order. The rocket launcher and BFG are skipped: splash
# damage at close range would make the scripted teacher kill itself.
_WEAPON_PRIORITY = (6, 4, 3, 2, 1)
_WEAPON_SWITCH_COOLDOWN = 20  # tics; re-pressing slot 3 would toggle SSG/shotgun

# ── Object vocabulary ──────────────────────────────────────────────────────────
MONSTERS = {
    "Zombieman": "zombie",
    "ShotgunGuy": "shotgunner",
    "ChaingunGuy": "chaingunner",
    "WolfensteinSS": "nazi",
    "DoomImp": "imp",
    "Demon": "demon",
    "Spectre": "spectre",
    "LostSoul": "lostsoul",
    "Cacodemon": "cacodemon",
    "HellKnight": "knight",
    "BaronOfHell": "baron",
    "Arachnotron": "arachnotron",
    "PainElemental": "pain",
    "Revenant": "revenant",
    "Fatso": "mancubus",
    "Archvile": "archvile",
    "SpiderMastermind": "mastermind",
    "Cyberdemon": "cyberdemon",
}
# class name -> (text label, kind)
ITEMS: dict[str, tuple[str, str]] = {
    "Medikit": ("medikit", "health"),
    "Stimpack": ("stim", "health"),
    "Soulsphere": ("soulsphere", "health"),
    "Megasphere": ("megasphere", "health"),
    "Berserk": ("berserk", "health"),
    "HealthBonus": ("bonus", "bonus"),
    "ArmorBonus": ("bonus", "bonus"),
    "GreenArmor": ("armor", "armor"),
    "BlueArmor": ("armor", "armor"),
    **{
        n: ("ammo", "ammo")
        for n in (
            "Clip",
            "ClipBox",
            "Shell",
            "ShellBox",
            "RocketAmmo",
            "RocketBox",
            "Cell",
            "CellPack",
            "Backpack",
        )
    },
    **{
        n: ("weapon", "weapon")
        for n in (
            "Chainsaw",
            "Shotgun",
            "SuperShotgun",
            "Chaingun",
            "RocketLauncher",
            "PlasmaRifle",
            "BFG9000",
        )
    },
}
_ALWAYS_PICKABLE_HEALTH = {"Soulsphere", "Megasphere", "Berserk"}

MAX_SEEN_MONSTERS = 3
MAX_SEEN_TOTAL = 5
MIN_SEEN_ITEMS = 2
ENEMY_MEMORY_TICS = 5 * TIC_HZ

# Wall probes: (label, bearing) with a small fan per probe so a pillar edge
# just off-centre still registers. Clearance subtracts the player radius. The
# rear probe stands in for the player knowing what is behind them.
WALL_PROBES = (("l", -35.0), ("f", 0.0), ("r", 35.0), ("b", 180.0))
_PROBE_FAN = (-8.0, 0.0, 8.0)
_PLAYER_RADIUS = 16.0
_MAX_STEP_UP = 24.0
_PLAYER_HEIGHT = 56.0
WALL_CAP_M = 9


def _wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


@dataclass
class Seen:
    """One object on screen, as the player sees it."""

    id: int
    cls: str  # ViZDoom class name, e.g. "Zombieman"
    label: str  # text label, e.g. "zombie"
    kind: str  # "monster" | "health" | "armor" | "ammo" | "weapon" | "bonus"
    bearing: float  # degrees, negative left
    dist: float  # metres

    @property
    def b(self) -> int:
        return round(self.bearing)

    @property
    def d(self) -> int:
        return round(self.dist)


@dataclass
class Privileged:
    """Ground truth the expert may use and the model never sees."""

    x: float
    y: float
    angle: float  # Doom world angle, degrees, counter-clockwise from +x
    objects: list[tuple[int, str, float, float]]  # (id, class, x, y)
    seen_all: list[Seen]  # every on-screen object, before capping
    walls_raw: tuple[float, ...]  # l, f, r, b clearance in metres


@dataclass
class Observation:
    tick: int
    text: str
    hp: int
    armor: int
    ammo: int
    weapon: str
    seen: list[Seen]  # the capped subset that appears in ``text``
    walls: tuple[int, int, int, int]  # l, f, r, b clearance, whole metres, capped
    hit: int  # damage taken in the last second
    enemy_mem: tuple[int, int] | None  # (bearing, seconds ago), only if none seen
    last: tuple[str, str]  # last two executed actions, oldest first
    priv: Privileged
    done: bool = False


@dataclass
class EpisodeStats:
    tics: int = 0
    kills: int = 0
    damage_taken: int = 0
    died: bool = False
    pickups: dict[str, int] = field(default_factory=dict)
    shots: int = 0
    threat_tics: int = 0  # tics with a monster on screen
    threat_dist_sum: float = 0.0  # sum of nearest on-screen monster distance

    @property
    def pickups_total(self) -> int:
        return sum(self.pickups.values())

    def as_dict(self) -> dict:
        return {
            "tics": self.tics,
            "seconds": round(self.tics / TIC_HZ, 1),
            "kills": self.kills,
            "damage_taken": self.damage_taken,
            "died": self.died,
            "pickups": self.pickups_total,
            "pickups_by_kind": dict(self.pickups),
            "shots": self.shots,
            "threat_s": round(self.threat_tics / TIC_HZ, 1),
            "threat_dist": round(self.threat_dist_sum / max(1, self.threat_tics), 2),
        }


def serialize(
    hp: int,
    armor: int,
    ammo: int,
    weapon: str,
    seen: list[Seen],
    walls: tuple[int, int, int, int],
    hit: int,
    enemy_mem: tuple[int, int] | None,
    last: tuple[str, str],
) -> str:
    """Build the player-visible state line the model reads (about 40-75 tokens)."""
    objs = ", ".join(f"{o.label} {o.b:+d} {o.d}m" for o in seen) or "nothing"
    l, f, r, b = walls
    text = (
        f"hp {hp} armor {armor} ammo {ammo} {weapon} | see {objs} | "
        f"wall l{l} f{f} r{r} b{b} | hit {hit}"
    )
    if enemy_mem is not None:
        text += f" | enemy {enemy_mem[0]:+d} {enemy_mem[1]}s"
    return text + f" | last {last[0]} {last[1]}"


def pickable(cls: str, kind: str, hp: int, armor: int) -> bool:
    """Whether a player with this HUD would pick the item up by touching it.

    Uses only HUD values so the rule is learnable from the text. Ammo and
    weapons are assumed pickable; the expert blacklists the rare full-ammo case.
    """
    if kind == "health":
        return hp < 100 or cls in _ALWAYS_PICKABLE_HEALTH
    if kind == "armor":
        return armor < 100
    return kind in ("ammo", "weapon", "bonus")


class DoomEnv:
    """Synchronous single-player ViZDoom on the bundled ``deathmatch.wad``."""

    def __init__(
        self,
        *,
        seed: int | None = None,
        resolution: str = "320X240",
        hud: bool = False,
        timeout_tics: int = 60 * TIC_HZ,
        skill: int = 3,
        scenario: str = "deathmatch.cfg",
    ):
        g = vzd.DoomGame()
        g.load_config(os.path.join(vzd.scenarios_path, scenario))
        g.set_available_buttons(_BUTTONS)
        g.set_available_game_variables(_VARS)
        g.set_objects_info_enabled(True)
        g.set_sectors_info_enabled(True)
        g.set_labels_buffer_enabled(True)
        g.set_screen_resolution(getattr(vzd.ScreenResolution, f"RES_{resolution}"))
        g.set_screen_format(vzd.ScreenFormat.RGB24)
        g.set_render_hud(hud)
        g.set_render_crosshair(hud)
        g.set_render_weapon(True)
        g.set_window_visible(False)
        g.set_sound_enabled(False)
        g.set_mode(vzd.Mode.PLAYER)
        g.set_episode_timeout(timeout_tics)
        g.set_doom_skill(skill)
        if seed is not None:
            g.set_seed(seed)
        g.init()
        self.game = g
        self._frame: np.ndarray | None = None
        self._segments: np.ndarray | None = None
        self._segments_tick = -(10**9)
        self.stats = EpisodeStats()
        self.obs: Observation | None = None

    # ── Episode control ──────────────────────────────────────────────────────
    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self.game.set_seed(seed)
        self.game.new_episode()
        self.stats = EpisodeStats()
        self._tick = 0
        self._last: deque[str] = deque(["wait", "wait"], maxlen=2)
        self._dmg_hist: deque[float] = deque(maxlen=TIC_HZ + 1)
        self._enemy_seen: tuple[float, float, int] | None = None  # x, y, tick
        self._items_prev: dict[int, tuple[str, float, float]] = {}
        self._next_switch = 0
        self._segments_tick = -(10**9)
        self.obs = self._observe()
        return self.obs

    def step(self, action: str) -> Observation:
        """Execute one macro action for one tic and return the next observation."""
        assert self.obs is not None and not self.obs.done, "call reset() first"
        fwd, back, sl, sr, atk, turn = _MACROS[action]
        buttons = [1, atk, fwd, back, sl, sr, turn] + [0] * 7
        slot = self._weapon_to_select()
        if slot is not None:
            buttons[_N_MOVE_BUTTONS + slot - 1] = 1
            self._next_switch = self._tick + _WEAPON_SWITCH_COOLDOWN
        self.game.make_action(buttons, 1)
        self._tick += 1
        self._last.append(action)
        self.stats.shots += atk
        if self.game.is_episode_finished():
            self.stats.died = self.game.is_player_dead()
            self.stats.tics = self._tick
            if self.stats.died:  # the killing blow lands after the last state
                self.stats.damage_taken += self.obs.hp
            self.obs.done = True
            return self.obs
        self.obs = self._observe()
        return self.obs

    def frame(self) -> np.ndarray | None:
        """Latest RGB frame (H, W, 3), or None before the first reset."""
        return self._frame

    def close(self) -> None:
        self.game.close()

    # ── Internals ────────────────────────────────────────────────────────────
    def _weapon_to_select(self) -> int | None:
        if self._tick < self._next_switch or self.obs is None:
            return None
        gv = self._gv
        for slot in _WEAPON_PRIORITY:
            owned = gv[_V[getattr(GameVariable, f"WEAPON{slot}")]] > 0
            ammo = gv[_V[getattr(GameVariable, f"AMMO{slot}")]]
            if slot == 1 or (owned and ammo > 0):
                current = int(gv[_V[GameVariable.SELECTED_WEAPON]])
                return None if current == slot else slot
        return None

    def _refresh_segments(self, sectors) -> None:
        """Collect impassable line segments as an (N, 4) array of x1, y1, x2, y2.

        A two-sided line blocks when the floor step exceeds what the player can
        climb or the opening is lower than the player. Drop-offs count as walls
        too, which keeps the teacher off ledges it could not climb back up.
        """
        owners: dict[tuple, tuple[list[bool], list]] = {}
        for sec in sectors:
            for ln in sec.lines:
                key = tuple(sorted(((ln.x1, ln.y1), (ln.x2, ln.y2))))
                flags, secs = owners.setdefault(key, ([], []))
                flags.append(ln.is_blocking)
                secs.append(sec)
        segs = []
        for (p1, p2), (flags, secs) in owners.items():
            if len(secs) < 2 or any(flags):
                block = True
            else:
                a, b = secs[0], secs[1]
                step = abs(a.floor_height - b.floor_height)
                opening = min(a.ceiling_height, b.ceiling_height) - max(
                    a.floor_height, b.floor_height
                )
                block = step > _MAX_STEP_UP or opening < _PLAYER_HEIGHT
            if block:
                segs.append((*p1, *p2))
        self._segments = np.asarray(segs, dtype=np.float64).reshape(-1, 4)

    def _raycast(self, x: float, y: float, angles_deg: np.ndarray) -> np.ndarray:
        """Distance in map units to the nearest blocking segment along each angle."""
        segs = self._segments
        th = np.radians(angles_deg)[:, None]
        dx, dy = np.cos(th), np.sin(th)
        ax, ay = segs[:, 0] - x, segs[:, 1] - y
        ex, ey = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
        den = dx * ey - dy * ex
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (ax * ey - ay * ex) / den
            u = (ax * dy - ay * dx) / den
        hit = (np.abs(den) > 1e-9) & (t > 0) & (u >= 0) & (u <= 1)
        t = np.where(hit, t, np.inf)
        return t.min(axis=1)

    def _observe(self) -> Observation:
        st = self.game.get_state()
        gv = st.game_variables
        self._gv = gv
        self._frame = st.screen_buffer
        hp = max(0, int(gv[_V[GameVariable.HEALTH]]))
        armor = int(gv[_V[GameVariable.ARMOR]])
        wslot = int(gv[_V[GameVariable.SELECTED_WEAPON]])
        weapon = WEAPON_NAMES.get(wslot, "fist")
        ammo = (
            0 if wslot <= 1 else max(0, int(gv[_V[GameVariable.SELECTED_WEAPON_AMMO]]))
        )
        px, py = gv[_V[GameVariable.POSITION_X]], gv[_V[GameVariable.POSITION_Y]]
        heading = gv[_V[GameVariable.ANGLE]]
        dmg_total = gv[_V[GameVariable.DAMAGE_TAKEN]]
        self.stats.kills = int(gv[_V[GameVariable.KILLCOUNT]])
        self.stats.damage_taken = int(dmg_total)
        self.stats.tics = self._tick

        # Damage in the last second.
        self._dmg_hist.append(dmg_total)
        hit = int(self._dmg_hist[-1] - self._dmg_hist[0])

        # Privileged object list; the player's own object is the DoomPlayer
        # nearest the player's position (the map also holds passive dolls).
        objects = [(o.id, o.name, o.position_x, o.position_y) for o in st.objects]
        players = [o for o in objects if o[1] == "DoomPlayer"]
        self_id = (
            min(players, key=lambda o: (o[2] - px) ** 2 + (o[3] - py) ** 2)[0]
            if players
            else -1
        )

        # Pickups: an item that vanished while within touching range last tic.
        items_now = {
            oid: (ITEMS[name][1], ox, oy)
            for oid, name, ox, oy in objects
            if name in ITEMS
        }
        for oid, (kind, ox, oy) in self._items_prev.items():
            if oid not in items_now and math.hypot(ox - px, oy - py) < 64.0:
                self.stats.pickups[kind] = self.stats.pickups.get(kind, 0) + 1
        self._items_prev = items_now

        # On-screen objects from the labels buffer.
        seen_all: list[Seen] = []
        seen_ids = set()
        for lab in st.labels:
            name, oid = lab.object_name, lab.object_id
            if oid == self_id or oid in seen_ids:
                continue
            if name in MONSTERS:
                label, kind = MONSTERS[name], "monster"
            elif name in ITEMS:
                label, kind = ITEMS[name]
            else:
                continue
            seen_ids.add(oid)
            dx, dy = lab.object_position_x - px, lab.object_position_y - py
            bearing = -_wrap180(math.degrees(math.atan2(dy, dx)) - heading)
            dist = math.hypot(dx, dy) / UNITS_PER_M
            seen_all.append(Seen(oid, name, label, kind, bearing, dist))
        seen_all.sort(key=lambda o: o.dist)

        monsters = [o for o in seen_all if o.kind == "monster"][:MAX_SEEN_MONSTERS]
        if monsters:
            self.stats.threat_tics += 1
            self.stats.threat_dist_sum += monsters[0].dist
        items = [
            o
            for o in seen_all
            if o.kind != "monster" and pickable(o.cls, o.kind, hp, armor)
        ]
        items = items[: max(MIN_SEEN_ITEMS, MAX_SEEN_TOTAL - len(monsters))]
        seen = monsters + items

        # Enemy memory: remember where the nearest monster was last seen and
        # report its bearing from where the player stands now.
        if monsters:
            m = monsters[0]
            th = math.radians(heading - m.bearing)
            d = m.dist * UNITS_PER_M
            self._enemy_seen = (
                px + d * math.cos(th),
                py + d * math.sin(th),
                self._tick,
            )
            enemy_mem = None
        elif (
            self._enemy_seen is not None
            and self._tick - self._enemy_seen[2] <= ENEMY_MEMORY_TICS
        ):
            ex, ey, t0 = self._enemy_seen
            b = -_wrap180(math.degrees(math.atan2(ey - py, ex - px)) - heading)
            enemy_mem = (round(b), round((self._tick - t0) / TIC_HZ))
        else:
            enemy_mem = None

        # Wall clearances.
        if self._tick - self._segments_tick >= 10:
            self._refresh_segments(st.sectors)
            self._segments_tick = self._tick
        angles = np.array(
            [heading - (b + f) for _, b in WALL_PROBES for f in _PROBE_FAN]
        )
        dist = self._raycast(px, py, angles).reshape(len(WALL_PROBES), -1).min(axis=1)
        walls_raw = tuple(
            float(max(0.0, d - _PLAYER_RADIUS) / UNITS_PER_M) for d in dist
        )
        walls = tuple(min(WALL_CAP_M, int(w)) for w in walls_raw)

        last = (self._last[0], self._last[1])
        text = serialize(hp, armor, ammo, weapon, seen, walls, hit, enemy_mem, last)
        priv = Privileged(px, py, heading, objects, seen_all, walls_raw)
        return Observation(
            tick=self._tick,
            text=text,
            hp=hp,
            armor=armor,
            ammo=ammo,
            weapon=weapon,
            seen=seen,
            walls=walls,
            hit=hit,
            enemy_mem=enemy_mem,
            last=last,
            priv=priv,
        )
