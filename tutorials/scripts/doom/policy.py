# SPDX-License-Identifier: Apache-2.0
"""Policies for the Doom reflex demo, plus direct prompt-id assembly.

Every policy answers ``decide(obs, adapter) -> Decision`` and
``route(instruction) -> Route``, so rollouts, the latency bench and the live
server can use any of them:

* :class:`ExpertPolicy`: the scripted teacher plus a keyword router. Needs no
  model, so the whole demo runs on a laptop.
* :class:`VLLMPolicy`: the composed Granite Switch checkpoint served in-process
  by vLLM. One engine step per decision: prefill the fresh state tokens and
  emit one action token.

``decide`` takes the full :class:`~doom_env.Observation` because the expert
reads privileged fields. Model policies read ``obs.text`` only.

Prompt ids are assembled directly each tick: a pre-tokenized prefix, the
tokenized state, and a pre-tokenized per-adapter suffix. This skips Jinja
rendering. ``python policy.py --check-template <model_dir>`` confirms the ids
are identical to ``apply_chat_template(adapter_name=...)``.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field

from doom_env import ACTIONS, WEAPON_SLOTS, Observation
from expert import BEHAVIORS, Expert

ARMS = "arms"  # weapon planner: one slot digit
CRITIC = "critic"  # danger of taking damage or dying within 1 s
ROUTER = "router"
ADAPTERS: tuple[str, ...] = (*BEHAVIORS, ARMS, CRITIC, ROUTER)
WEAPON_TOKENS: tuple[str, ...] = tuple(str(s) for s in WEAPON_SLOTS)
DANGER_LEVELS: tuple[str, ...] = ("low", "mid", "high")

SYSTEM_PROMPT = (
    "You play Doom deathmatch against bots. The game state gives health, armor, the "
    "selected weapon and its ammo, owned weapon slots with ammo; objects on screen as "
    "name, bearing in degrees (negative is left) and distance; wall clearance left, "
    "front, right and behind in metres; damage taken in the last second; where an "
    "enemy was last seen; your last two actions. Reply with what is asked: one action ("
    + " ".join(ACTIONS)
    + "), a weapon slot (1-7), or the danger of being hit soon (low mid high)."
)
ROUTER_SYSTEM_PROMPT = (
    "Pick the Doom deathmatch play style that best follows the player's instruction: "
    "fighter (hunt and frag the bots), cautious (avoid damage, fight only up close, "
    "heal), collector (collect items, armor and weapons). Reply with the style name."
)

_SOR, _EOR, _EOT = "<|start_of_role|>", "<|end_of_role|>", "<|end_of_text|>"

# CUDA-graph capture sizes, in scheduled tokens per engine step. A decision
# prefills ~45 fresh tokens per game, so the sizes must reach past that for one
# game and past N x 45 for an N-game batch. vLLM otherwise caps capture at
# 2 x max_num_seqs, and a larger step runs eagerly and launch-bound: 6 ms -> 17 ms
# on an H100 for one game.
CAPTURE_SIZES = [
    *(1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256),
    *(320, 384, 448, 512, 640, 768, 896, 1024),
]


@dataclass
class Decision:
    action: str
    top3: list[tuple[str, float]]
    ms: float  # state available -> action available, wall clock
    fresh_tokens: int = 0
    cached_tokens: int = 0
    build_ms: float = 0.0  # prompt-id assembly (tokenizing the state)
    engine_ms: float = 0.0  # vLLM: submit -> output (prefill + one token)
    probs: dict[str, float] = field(default_factory=dict)  # every action


@dataclass
class Route:
    adapter: str
    prob: float
    ms: float
    probs: dict[str, float] = field(default_factory=dict)


# ── Prompt-id assembly ─────────────────────────────────────────────────────────
def control_token(adapter: str) -> str:
    return f"<|{adapter}|>"


def single_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{text!r} is {len(ids)} tokens {ids}, expected exactly 1")
    return ids[0]


def output_token_ids(tokenizer, words: tuple[str, ...]) -> dict[str, int]:
    """Map each output word to its single token id; fail loudly if any is not one
    token or two share one."""
    ids = {w: single_token_id(tokenizer, w) for w in words}
    if len(set(ids.values())) != len(ids):
        raise ValueError(f"output tokens collide: {ids}")
    return ids


def action_token_ids(tokenizer) -> dict[str, int]:
    return output_token_ids(tokenizer, ACTIONS)


def weapon_token_ids(tokenizer) -> dict[str, int]:
    return output_token_ids(tokenizer, WEAPON_TOKENS)


def danger_token_ids(tokenizer) -> dict[str, int]:
    return output_token_ids(tokenizer, DANGER_LEVELS)


def route_token_ids(tokenizer) -> dict[str, int]:
    """Map each behavior to the first token of its name (the router's one output token)."""
    ids = {b: tokenizer.encode(b, add_special_tokens=False)[0] for b in BEHAVIORS}
    if len(set(ids.values())) != len(ids):
        raise ValueError(f"behavior first tokens collide: {ids}")
    return ids


class PromptBuilder:
    """Pre-tokenized prompt pieces; ``ids`` concatenates them with the fresh text.

    ``adapter=None`` gives the base-model prompt (what PEFT trains on).
    Otherwise the adapter's control token takes the place of the
    ``<|start_of_role|>`` that opens the assistant header, exactly as the
    composed chat template renders an aLoRA whose invocation sequence is
    ``<|start_of_role|>assistant<|end_of_role|>``. At runtime the switch gives
    the control token ``<|start_of_role|>``'s embedding, so the model sees the
    sequence it was trained on.
    """

    def __init__(self, tokenizer, system: str, adapters: tuple[str, ...] = ()):
        self.tok = tokenizer
        self.prefix = self._enc(f"{_SOR}system{_EOR}{system}{_EOT}\n{_SOR}user{_EOR}")
        self.suffix = {None: self._enc(f"{_EOT}\n{_SOR}assistant{_EOR}")}
        for a in adapters:
            self.suffix[a] = self._enc(f"{_EOT}\n{control_token(a)}assistant{_EOR}")
        self.system = system

    def _enc(self, s: str) -> list[int]:
        return self.tok.encode(s, add_special_tokens=False)

    def ids(self, text: str, adapter: str | None) -> list[int]:
        return self.prefix + self._enc(text) + self.suffix[adapter]

    def n_fixed(self) -> int:
        return len(self.prefix)


def alora_invocation_ids(tokenizer) -> list[int]:
    return tokenizer.encode(f"{_SOR}assistant{_EOR}", add_special_tokens=False)


def check_template(model_dir: str, texts: list[str]) -> None:
    """Assert direct id assembly == ``apply_chat_template`` for every adapter."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    composed = control_token(BEHAVIORS[0]) in tok.get_vocab()
    adapters = ADAPTERS if composed else ()
    for system, names in (
        (SYSTEM_PROMPT, (*BEHAVIORS, ARMS, CRITIC, None)),
        (ROUTER_SYSTEM_PROMPT, (ROUTER, None)),
    ):
        pb = PromptBuilder(tok, system, adapters)
        for text in texts:
            msgs = [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ]
            for a in names if composed else (None,):
                kw = {"adapter_name": a} if a else {}
                rendered = tok.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False, **kw
                )
                ref = tok(rendered, add_special_tokens=False).input_ids
                got = pb.ids(text, a)
                assert got == ref, f"adapter={a}\n got={got}\n ref={ref}\n{rendered!r}"
    check_output_tokens(tok)
    print(
        f"OK: prompt ids match apply_chat_template for "
        f"{'adapters ' + ', '.join(ADAPTERS) + ' and base' if composed else 'the base template'} "
        f"on {len(texts)} states."
    )


def check_output_tokens(tok) -> None:
    """Every output vocabulary is single tokens, distinct within its adapter, and
    unchanged when it directly follows ``<|end_of_role|>`` (where it is emitted)."""
    groups = {
        "actions": action_token_ids(tok),
        "weapon slots": weapon_token_ids(tok),
        "danger levels": danger_token_ids(tok),
        "route first tokens": route_token_ids(tok),
    }
    eor = tok.encode(_EOR, add_special_tokens=False)
    for name, ids in groups.items():
        if name == "route first tokens":
            continue
        for w, i in ids.items():
            got = tok.encode(_EOR + w, add_special_tokens=False)
            if got != [*eor, i]:
                raise ValueError(f"{w!r} after {_EOR} tokenizes as {got}")
    for name, ids in groups.items():
        print(f"  {name}: {len(ids)} single, distinct tokens")


# ── Keyword router (stand-in when no model is loaded) ───────────────────────────
_KEYWORDS = {
    "fighter": r"kill|hunt|fight|attack|shoot|aggress|destroy|frag|slay|murder|rampage|clear",
    "cautious": r"surviv|safe|careful|avoid|run away|flee|hide|retreat|heal|health|defen|cautious|stay alive|don.t die",
    "collector": r"collect|loot|pick|gather|item|ammo|armor|scaveng|grab|supplies|weapon",
}


_NEGATION = r"(?:stop|don.?t|do not|no|never|quit|avoid|without)\s+(?:\w+\s+){0,2}?"


def keyword_route(instruction: str) -> Route:
    t0 = time.perf_counter()
    text = instruction.lower()
    # "stop fighting" is a vote for cautious, not fighter.
    negated_fights = len(
        re.findall(_NEGATION + "(?:" + _KEYWORDS["fighter"] + ")", text)
    )
    text = re.sub(_NEGATION + "(?:" + _KEYWORDS["fighter"] + ")\\w*", " ", text)
    scores = {b: len(re.findall(p, text)) for b, p in _KEYWORDS.items()}
    scores["cautious"] += negated_fights
    total = sum(scores.values())
    if total == 0:
        probs = {b: 1 / len(BEHAVIORS) for b in BEHAVIORS}
    else:
        probs = {b: s / total for b, s in scores.items()}
    best = max(probs, key=lambda b: (probs[b], b == "fighter"))
    return Route(best, probs[best], (time.perf_counter() - t0) * 1000, probs)


# ── Policies ───────────────────────────────────────────────────────────────────
class ExpertPolicy:
    """The scripted teacher behind the same interface as the model."""

    name = "expert"

    def __init__(self) -> None:
        self.expert = Expert()

    def reset(self) -> None:
        self.expert.reset()

    def decide(self, obs: Observation, adapter: str) -> Decision:
        t0 = time.perf_counter()
        action = self.expert.act(obs, adapter)
        ms = (time.perf_counter() - t0) * 1000
        return Decision(action, [(action, 1.0)], ms, probs={action: 1.0})

    def decide_many(
        self, obs: Observation, adapters: tuple[str, ...]
    ) -> dict[str, Decision]:
        return {a: self.decide(obs, a) for a in adapters}

    def route(self, instruction: str) -> Route:
        return keyword_route(instruction)


class VLLMPolicy:
    """Composed Granite Switch checkpoint served in-process by vLLM.

    Args:
        model: Composed checkpoint directory.
        engine_loop: Drive ``LLMEngine.add_request/step`` directly instead of
            ``LLM.generate``. The latency test picks whichever is faster.
        prefix_caching: Keep on; the system prompt is then prefilled once.
        logprobs_mode: ``processed_logprobs`` so the probabilities are over allowed
            actions, renormalized.
        cudagraph_mode: ``FULL`` captures the whole forward, SWITCH kernels
            included, for prefill steps too; ``FULL_AND_PIECEWISE`` is vLLM's
            default.
        gc_freeze: After warmup, move every live object to the permanent
            generation so a full GC pass does not walk vLLM's object heap in the
            middle of a decision.
        async_scheduling: Passed to vLLM when set; ``None`` keeps its default.
    """

    name = "vllm"

    def __init__(
        self,
        model: str,
        *,
        engine_loop: bool = False,
        prefix_caching: bool = True,
        max_num_seqs: int = 8,
        gpu_memory_utilization: float = 0.5,
        max_model_len: int = 512,
        logprobs_mode: str = "processed_logprobs",
        enforce_eager: bool = False,
        warmup: int = 20,
        base_model: bool = False,
        cudagraph_mode: str = "FULL",
        gc_freeze: bool = True,
        async_scheduling: bool | None = None,
    ):
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=model,
            dtype="bfloat16",
            max_model_len=max_model_len,
            enable_prefix_caching=prefix_caching,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            max_logprobs=len(ACTIONS),
            logprobs_mode=logprobs_mode,
            enforce_eager=enforce_eager,
            compilation_config={
                "cudagraph_capture_sizes": CAPTURE_SIZES,
                "cudagraph_mode": cudagraph_mode,
            },
            **(
                {}
                if async_scheduling is None
                else {"async_scheduling": async_scheduling}
            ),
        )
        self.tok = self.llm.get_tokenizer()
        # A plain base model (the SWITCH-overhead baseline) has no control
        # tokens: every adapter then maps to the base prompt.
        self.base_model = base_model
        adapters = () if base_model else ADAPTERS
        self.pb = PromptBuilder(self.tok, SYSTEM_PROMPT, adapters)
        self.rb = PromptBuilder(self.tok, ROUTER_SYSTEM_PROMPT, adapters)
        self.action_ids = action_token_ids(self.tok)
        self.id_to_action = {v: k for k, v in self.action_ids.items()}
        self.route_ids = route_token_ids(self.tok)
        self.id_to_route = {v: k for k, v in self.route_ids.items()}
        self.sp_action = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            allowed_token_ids=list(self.action_ids.values()),
            logprobs=len(ACTIONS),  # the full distribution, for the heatmap
        )
        self.sp_route = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            allowed_token_ids=list(self.route_ids.values()),
            logprobs=len(BEHAVIORS),
        )
        self.engine_loop = engine_loop
        self._warmup(warmup)
        if gc_freeze:
            gc.collect()
            gc.freeze()

    def reset(self) -> None:
        pass

    # ── Engine calls ───────────────────────────────────────────────────────────
    def _run(self, prompts: list[list[int]], sp) -> list:
        from vllm.inputs import TokensPrompt

        if not self.engine_loop:
            return self.llm.generate(
                [TokensPrompt(prompt_token_ids=p) for p in prompts], sp, use_tqdm=False
            )
        engine = self.llm.llm_engine
        ids = [uuid.uuid4().hex for _ in prompts]
        for rid, p in zip(ids, prompts):
            engine.add_request(rid, TokensPrompt(prompt_token_ids=p), sp)
        done: dict[str, object] = {}
        while len(done) < len(ids):
            for out in engine.step():
                if out.finished:
                    done[out.request_id] = out
        return [done[rid] for rid in ids]

    def _adapter(self, adapter: str | None) -> str | None:
        return None if self.base_model else adapter

    @staticmethod
    def _dist(out, id_map: dict[int, str]) -> list[tuple[str, float]]:
        lp = out.outputs[0].logprobs[0]
        pairs = [(id_map[i], math.exp(v.logprob)) for i, v in lp.items() if i in id_map]
        pairs.sort(key=lambda p: -p[1])
        return pairs

    def _decision(self, out, t0: float, t1: float, t2: float) -> Decision:
        dist = self._dist(out, self.id_to_action)
        top = dist[:3]
        action = self.id_to_action[out.outputs[0].token_ids[0]]
        n_prompt = len(out.prompt_token_ids)
        cached = out.num_cached_tokens or 0
        return Decision(
            action,
            top,
            (time.perf_counter() - t0) * 1000,
            n_prompt - cached,
            cached,
            build_ms=(t1 - t0) * 1000,
            engine_ms=(t2 - t1) * 1000,
            probs=dict(dist),
        )

    # ── Policy interface ───────────────────────────────────────────────────────
    def decide(self, obs: Observation | str, adapter: str) -> Decision:
        t0 = time.perf_counter()
        text = obs if isinstance(obs, str) else obs.text
        ids = self.pb.ids(text, self._adapter(adapter))
        t1 = time.perf_counter()
        out = self._run([ids], self.sp_action)[0]
        return self._decision(out, t0, t1, time.perf_counter())

    def decide_batch(self, texts: list[str], adapters: list[str]) -> list[Decision]:
        """One engine call for many states (DAgger rollouts). ``ms`` is per batch."""
        t0 = time.perf_counter()
        prompts = [self.pb.ids(t, self._adapter(a)) for t, a in zip(texts, adapters)]
        t1 = time.perf_counter()
        outs = self._run(prompts, self.sp_action)
        t2 = time.perf_counter()
        return [self._decision(o, t0, t1, t2) for o in outs]

    def decide_many(
        self, obs: Observation, adapters: tuple[str, ...]
    ) -> dict[str, Decision]:
        """All ``adapters`` on the same state in one engine step (shadow decisions)."""
        decs = self.decide_batch([obs.text] * len(adapters), list(adapters))
        return dict(zip(adapters, decs))

    def route(self, instruction: str) -> Route:
        t0 = time.perf_counter()
        out = self._run(
            [self.rb.ids(instruction, self._adapter(ROUTER))], self.sp_route
        )[0]
        probs = dict(self._dist(out, self.id_to_route))
        best = self.id_to_route[out.outputs[0].token_ids[0]]
        return Route(
            best, probs.get(best, 0.0), (time.perf_counter() - t0) * 1000, probs
        )

    def _warmup(self, n: int) -> None:
        text = (
            "hp 100 armor 0 | pistol 50 | arms 2:50 | see bot -12 8m, medikit +40 3m | "
            "wall l3 f9 r9 b2 | hit 0 | last forward forward"
        )
        for i in range(n):
            self.decide(
                text.replace("-12", f"{-12 - i:+d}"), BEHAVIORS[i % len(BEHAVIORS)]
            )
        self.route("go kill everything")


def make_policy(kind: str, model: str | None = None, **kw):
    if kind == "expert":
        return ExpertPolicy()
    if kind == "vllm":
        if not model:
            raise SystemExit("--model is required for --policy vllm")
        return VLLMPolicy(model, **kw)
    raise SystemExit(f"unknown policy {kind!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--check-template",
        metavar="MODEL_DIR",
        required=True,
        help="Composed model dir (or a base model id for the base-template check)",
    )
    ap.add_argument(
        "--episodes", type=int, default=2, help="Episodes of states to check"
    )
    args = ap.parse_args()

    from doom_env import DoomEnv

    env = DoomEnv(seed=0)
    pol = ExpertPolicy()
    texts = []
    for ep in range(args.episodes):
        obs = env.reset(seed=ep)
        pol.reset()
        while not obs.done and len(texts) < 400 * (ep + 1):
            if not obs.dead:
                texts.append(obs.text)
            obs = env.step(pol.decide(obs, BEHAVIORS[ep % len(BEHAVIORS)]).action)
    env.close()
    texts.extend(
        ["go kill everything", "stay alive, grab health", "collect all the loot"]
    )
    check_template(args.check_template, texts[::7])


if __name__ == "__main__":
    main()
