# SPDX-License-Identifier: Apache-2.0
"""Record the model playing, with a telemetry panel, as an MP4.

A script of segments drives the video. Each segment is either an adapter name
or a free-text instruction (routed by the router adapter), plus seconds of game
time::

    python record_video.py --model models/doom-round0 --out out/granite_doom.mp4 \
        --segment "hunter" 20 \
        --segment "stop fighting and grab health" 20 \
        --segment "collect all the loot" 20

Every action in the video comes from the composed checkpoint, one token per tic.
The panel shows the measured wall-clock time of each decision against the 28.6 ms
tic. The game runs synchronously and the video plays at 35 fps, so it is real
game speed.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doom_env import ACTION_LABELS, TIC_HZ, TIC_MS, DoomEnv
from expert import BEHAVIORS
from policy import VLLMPolicy

W_GAME, H = 640, 480
W_PANEL = 360
BG = (22, 22, 22)
LAYER = (38, 38, 38)
LAYER2 = (57, 57, 57)
TEXT = (244, 244, 244)
TEXT2 = (198, 198, 198)
HELP = (168, 168, 168)
BLUE = (15, 98, 254)
TEAL = (8, 189, 186)
AMBER = (210, 161, 6)
COLORS = {"hunter": (250, 77, 86), "survivor": TEAL, "scavenger": AMBER}


def font(size: int, mono: bool = False):
    for path in (
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf"
        if mono
        else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
        if mono
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


class Panel:
    def __init__(self) -> None:
        self.f_title = font(15)
        self.f_big = font(34)
        self.f = font(15)
        self.f_small = font(12)
        self.f_mono = font(12, mono=True)

    def draw(self, img: Image.Image, t: dict) -> None:
        d = ImageDraw.Draw(img)
        x0 = W_GAME
        d.rectangle([x0, 0, x0 + W_PANEL, H], fill=BG)
        x = x0 + 18
        y = 14
        d.text((x, y), "GRANITE SWITCH  ·  DOOM", font=self.f_title, fill=TEXT2)
        y += 26
        d.text((x, y), "one output token per tic", font=self.f_small, fill=HELP)
        y += 26

        # Behavior chip
        c = COLORS.get(t["adapter"], TEXT)
        d.rounded_rectangle([x, y, x + 150, y + 28], radius=14, fill=LAYER2)
        d.ellipse([x + 12, y + 10, x + 20, y + 18], fill=c)
        d.text((x + 28, y + 5), t["adapter"], font=self.f, fill=TEXT)
        d.text((x + 162, y + 6), "aLoRA adapter", font=self.f_small, fill=HELP)
        y += 38
        if t.get("instruction"):
            d.text((x, y), f"“{t['instruction'][:40]}”", font=self.f_small, fill=TEXT2)
            y += 16
            d.text(
                (x, y),
                f"router -> {t['routed']} ({100 * t['route_p']:.0f}%, {t['route_ms']:.1f} ms)",
                font=self.f_small,
                fill=HELP,
            )
            y += 20
        else:
            y += 36

        # Decision latency
        d.text((x, y), "DECISION", font=self.f_small, fill=HELP)
        y += 16
        num = f"{t['ms']:.1f}"
        d.text((x, y), num, font=self.f_big, fill=TEXT)
        d.text(
            (x + d.textlength(num, font=self.f_big) + 6, y + 16),
            "ms",
            font=self.f,
            fill=TEXT2,
        )
        d.text(
            (x + 140, y + 4), f"p50 {t['p50']:.1f} ms", font=self.f_small, fill=TEXT2
        )
        d.text(
            (x + 140, y + 22), f"p99 {t['p99']:.1f} ms", font=self.f_small, fill=TEXT2
        )
        y += 46
        # Tic budget bar: 0..100 ms
        bw = W_PANEL - 36
        d.rectangle([x, y, x + bw, y + 10], fill=LAYER2)
        d.rectangle([x, y, x + int(bw * min(1.0, t["ms"] / 100.0)), y + 10], fill=TEAL)
        tx = x + int(bw * TIC_MS / 100.0)
        d.line([tx, y - 4, tx, y + 14], fill=TEXT, width=1)
        d.text((tx + 4, y + 13), "1 tic 28.6 ms", font=self.f_small, fill=TEXT2)
        d.text((x + bw - 50, y + 13), "100 ms", font=self.f_small, fill=HELP)
        y += 36

        # Action + top-3
        d.text((x, y), "ACTION", font=self.f_small, fill=HELP)
        y += 16
        d.text(
            (x, y), ACTION_LABELS.get(t["action"], t["action"]), font=self.f, fill=TEXT
        )
        d.text((x + 200, y + 2), t["action"], font=self.f_mono, fill=HELP)
        y += 24
        for a, p in t["top3"]:
            d.text((x, y), ACTION_LABELS.get(a, a)[:16], font=self.f_small, fill=TEXT2)
            d.rectangle([x + 120, y + 3, x + 120 + 170, y + 11], fill=LAYER2)
            d.rectangle([x + 120, y + 3, x + 120 + int(170 * p), y + 11], fill=BLUE)
            d.text((x + 296, y), f"{100 * p:.0f}%", font=self.f_small, fill=TEXT2)
            y += 18
        y += 8

        # Stats
        d.text((x, y), "EPISODE", font=self.f_small, fill=HELP)
        y += 16
        s = t["stats"]
        d.text(
            (x, y),
            f"hp {t['hp']}  armor {t['armor']}  {t['weapon']}",
            font=self.f_small,
            fill=TEXT2,
        )
        y += 16
        d.text(
            (x, y),
            f"kills {s['kills']}  pickups {s['pickups']}  deaths {t['deaths']}",
            font=self.f_small,
            fill=TEXT2,
        )
        y += 24

        # What the model reads
        d.text((x, y), "MODEL INPUT", font=self.f_small, fill=HELP)
        y += 16
        words, line, lines = t["text"].split(" "), "", []
        for w in words:
            if len(line) + len(w) + 1 > 46:
                lines.append(line)
                line = w
            else:
                line = f"{line} {w}".strip()
        lines.append(line)
        for ln in lines:
            if y > H - 40:
                break
            d.text((x, y), ln, font=self.f_mono, fill=TEXT2)
            y += 14
        d.text(
            (x, H - 22), f"tic {t['tick']}  ·  {t['gpu']}", font=self.f_small, fill=HELP
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--segment",
        nargs=2,
        action="append",
        metavar=("ADAPTER_OR_INSTRUCTION", "SECONDS"),
        required=True,
    )
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    import imageio.v2 as imageio
    import torch

    gpu = torch.cuda.get_device_name(0).replace("NVIDIA ", "")
    pol = VLLMPolicy(args.model, warmup=100)
    env = DoomEnv(seed=args.seed, resolution="640X480", hud=True, timeout_tics=10**7)
    obs = env.reset(seed=args.seed)
    panel = Panel()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(args.out), fps=TIC_HZ, codec="libx264", quality=8, macro_block_size=1
    )
    lat: deque[float] = deque(maxlen=1000)
    deaths, all_ms = 0, []
    for what, secs in args.segment:
        info: dict = {"instruction": None}
        if what in BEHAVIORS:
            adapter = what
        else:
            r = pol.route(what)
            adapter = r.adapter
            info = {
                "instruction": what,
                "routed": r.adapter,
                "route_p": r.prob,
                "route_ms": r.ms,
            }
        print(f"segment: {what!r} -> {adapter} for {secs}s", flush=True)
        for _ in range(int(float(secs) * TIC_HZ)):
            d = pol.decide(obs, adapter)
            lat.append(d.ms)
            all_ms.append(d.ms)
            frame = env.frame()
            text, hud = obs.text, (obs.hp, obs.armor, obs.weapon)
            obs = env.step(d.action)
            if obs.done:
                deaths += env.stats.died
                obs = env.reset()
            img = Image.new("RGB", (W_GAME + W_PANEL, H), BG)
            img.paste(Image.fromarray(frame), (0, 0))
            a = np.fromiter(lat, dtype=np.float64)
            panel.draw(
                img,
                {
                    **info,
                    "adapter": adapter,
                    "ms": d.ms,
                    "p50": float(np.percentile(a, 50)),
                    "p99": float(np.percentile(a, 99)),
                    "action": d.action,
                    "top3": d.top3,
                    "stats": env.stats.as_dict(),
                    "hp": hud[0],
                    "armor": hud[1],
                    "weapon": hud[2],
                    "deaths": deaths,
                    "text": text,
                    "tick": obs.tick,
                    "gpu": gpu,
                },
            )
            writer.append_data(np.asarray(img))
    writer.close()
    env.close()
    a = np.asarray(all_ms)
    print(
        f"wrote {args.out}: {a.size} decisions, p50 {np.percentile(a, 50):.2f} ms, "
        f"p99 {np.percentile(a, 99):.2f} ms, max {a.max():.2f} ms"
    )


if __name__ == "__main__":
    main()
