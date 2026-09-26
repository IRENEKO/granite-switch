# Doom Reflex Demo: Granite Switch Decides Every Tic

Granite plays Doom from a structured text game state, one output token per
decision, on every Doom tic (35 Hz). Three behaviors (`hunter`, `survivor`,
`scavenger`) and a one-token instruction router are aLoRA adapters embedded in
one Granite Switch checkpoint. Switching behavior live changes only a control
token.

The code lives in [`tutorials/scripts/doom/`](../scripts/doom/doom_env.py).

## What the demo measures, and what it claims

- **Decision latency** is wall clock in the game loop, from "state available" to
  "action available". It covers building the prompt ids plus one vLLM engine
  step, which prefills the ~45 fresh state tokens and emits one action token.
  That is time-to-first-token, not a decode step. Frame streaming is outside it.
- **Many behaviors, one checkpoint, one server.** Switching behavior changes the
  control token and nothing else: no adapter loading or swapping. The prefill of
  the shared system prompt comes from the prefix cache for every adapter.
- **A language interface.** A router adapter maps free text ("stop fighting and
  grab health") to a behavior with one output token.
- **Not claimed:** judgment beyond the scripted teacher (the students imitate it,
  so it sets their ceiling), or a like-for-like speed comparison with the Jev
  demo. The Jev post describes about 10 decisions per second but does not state
  its hardware or model. The wording we use: *"decides every tic at X ms p50 on
  one H100; the Jev post describes ~10 decisions/sec."*

## Results

### Decision latency (one H100 80GB HBM3, vLLM 0.19.1, granite-4.1-3b)

Measured by [`bench_latency.py`](../scripts/doom/bench_latency.py) on real
states from expert play (mean 44.5 fresh tokens per decision, 70% of prompt
tokens served from the prefix cache). The first run used random-weight stand-in
aLoRAs, which cost the same compute as trained ones, so the go/no-go could come
before any training. The trained round-0 checkpoint then measured p50 7.1 ms and
p99 12.7 ms for one decision, and the narrated video below made 2,800 decisions
at p50 6.6 ms and p99 14.1 ms.

| Scenario | p50 | p99 | Notes |
|---|---:|---:|---|
| One decision, composed model (4 aLoRAs) | 7.6 ms | 13.3 ms | 123 decisions/s; 99.97% within one 28.6 ms tic |
| Same prompts, plain base model (no SWITCH) | 6.1 ms | 12.3 ms | SWITCH adds ~1.5 ms |
| Plain decode step, for reference | 3.9 ms | | |
| 16 games in one batched step | 12.9 ms | 18.6 ms | every game decides every tic |
| All 3 behaviors on the same state, one step | 8.2 ms | | "shadow decisions" |
| Router (instruction to behavior) | 6.4 ms | | |
| Turbo play (unthrottled, env in the loop) | 6.6 ms | 12.7 ms | 87.5 tics/s, 2.5x real time; env step 4.1 ms |
| One decision, no prefix cache | 9.4 ms | 13.0 ms | 151 fresh tokens |

Run-to-run and node-to-node variation is about ±1 ms at p50.

Two settings decide these numbers, and neither is optional:

1. **CUDA-graph capture sizes.** vLLM caps its largest CUDA-graph capture size
   at 2 x `max_num_seqs`. With `max_num_seqs=8` that is 16 tokens, so a 45-token
   prefill runs eagerly. It is then bound by kernel-launch overhead: 17 ms
   instead of 6 ms. [`policy.py`](../scripts/doom/policy.py) sets explicit
   capture sizes up to 1024 tokens (so 16-game batches are also graphed) and
   `cudagraph_mode=FULL`, which captures the whole forward, SWITCH kernels
   included, for prefill steps too.
2. **CPU allocation.** An in-process vLLM engine with a starved CPU gives 60–90 ms
   decisions with huge tails. On LSF, request cores on one host
   (`-n 16 -R "span[hosts=1]"`) and cap the thread pools (`OMP_NUM_THREADS=8`).

`LLM.generate` was about 1 ms faster than driving `LLMEngine.add_request/step`
directly, so `VLLMPolicy` uses `LLM.generate` by default (`--engine-loop` switches).

### Behaviors (scripted teacher, 50 episodes each, 60 s cap, skill 3)

| | kills/min | pickups | health pickups/min | shots/min | alive (s) |
|---|---:|---:|---:|---:|---:|
| hunter | **15.2** | 16.8 | 2.1 | 848 | 32 |
| survivor | 8.2 | 17.1 | **6.3** | 268 | 40 |
| scavenger | 11.5 | **31.2** | 4.2 | 306 | 44 |

The hunter kills fastest, the scavenger collects the most, and the survivor
fights least and heals most. On `deathmatch.wad` the survivor does **not** take
less damage per minute than the others. The map keeps spawning hitscan monsters
into an open arena, and hitscan can't be dodged. Avoiding fights lets monsters
accumulate (6.1 alive on average around a passive agent, 2.7 around the hunter),
so passivity costs more than it saves. More than ten evasive designs and a
DECORATE mod swapping hitscanners for imps and demons did not change this; the
survivor's signature is therefore its behavior, not its damage rate.

### Trained students (round 0)

Each behavior aLoRA (rank 16, q/k/v/o, one epoch, about 10 minutes on one H100)
was trained on 60 noisy teacher episodes per behavior. The table gives accuracy
against the teacher on held-out episodes, then the check that the composed
checkpoint in vLLM picks the same action as the PEFT adapter it was built from:

| Adapter | held-out accuracy | majority-class baseline | composed == PEFT |
|---|---:|---:|---:|
| hunter | 96.4% | 29.6% | 99.47% |
| survivor | 91.4% | 24.7% | 98.97% |
| scavenger | 98.7% | 32.1% | 99.82% |
| router (30 hand-written instructions) | 96.7% | 33.3% | 100% |

The survivor misses the 99% parity gate by 0.03 points. Its likeliest cause is
near-ties (such as strafe left vs right) flipping on bf16 differences between
PEFT and the fused SWITCH kernels, which are not bit-exact (see CLAUDE.md
gotcha 9); the disagreeing states have not been inspected yet.

With the student driving (30 one-minute games per behavior, teacher labeling
each state it reaches):

| | kills/min | pickups | health pickups/min | shots/min | agrees with teacher |
|---|---:|---:|---:|---:|---:|
| hunter | **15.6** | 19.9 | 0.9 | 830 | 85.1% |
| survivor | 6.7 | 15.9 | **5.0** | **267** | 81.4% |
| scavenger | 12.3 | **29.5** | 2.3 | 306 | 97.2% |

The students' profiles separate the same way the teacher's do. Agreement is
lower than the held-out accuracy because the student's own mistakes take it to
states the teacher never visited; DAgger round 1 (training on those states,
already collected) targets exactly that.

### Does reacting faster help? (scripted hunter, 40 games per row)

The same teacher under real-time rules: the game never waits, and the last
action repeats until a newer decision lands. Only the cadence and the age of
the state each decision uses change.

| Condition | kills/min | damage/min | median reaction |
|---|---:|---:|---:|
| Every tic, no delay (this system: 7 ms fits inside a tic) | **16.0** | 215 | **143 ms** |
| Every tic, 1-tic delay | 13.8 | 206 | 171 ms |
| Every tic, 3-tic delay (100 ms, pipelined) | 12.0 | 224 | 229 ms |
| 10 Hz, no delay | 15.2 | 218 | 214 ms |
| 10 Hz, 3-tic delay | 7.6 | 305 | 357 ms |

Reaction is game time from a monster appearing on screen to the first shot at
it; most of the 143 ms is turning at 7° per tic. Below one tic, faster buys
headroom (batching, shadow decisions, a bigger model), not faster reflexes,
because Doom reads input 35 times a second. Above a tic, staleness costs far
more than a lower cadence. The teacher was designed for every-tic control, so
the 10 Hz rows are an upper bound on the penalty for a policy designed for
10 Hz, not a measurement of any other system.

### Why aLoRA, honestly

Every decision here is a fresh prompt: the fixed system prompt, the current
state line, the control token. The only prefix shared across adapters is the
system prompt, which plain LoRA could also cache per adapter, and a 45-token
prefill costs about the same whether it is shared or not. A merged model or a
single multi-task fine-tune (behavior named in the prompt) would likely match
these results at the plain-base latency. aLoRA's advantage, reusing a long
base-model context across adapters without re-prefilling it, only appears once
the prompt carries game history. That design, and a direct comparison, are open.

## How it works

### State and actions

[`doom_env.py`](../scripts/doom/doom_env.py) wraps ViZDoom 1.3.1 on the
bundled `deathmatch.wad` in synchronous `PLAYER` mode, one step per tic, with
the best weapon auto-selected. The model reads one line of player-visible text
(p99 72 tokens):

```
hp 64 armor 0 ammo 23 shotgun | see zombie -12 8m, medikit +40 3m | wall l3 f9 r9 b2 | hit 10 | enemy -140 2s | last left forward
```

That is HUD values, the objects in the labels buffer as bearing (negative is
left) and distance, wall clearance left/front/right/back, damage in the last
second, the last-seen enemy's bearing, and the last two actions. The twelve
macro actions are each one token of the Granite 4.1 tokenizer:
`forward back left right sl sr fl fr fire al ar wait`. Decoding uses
`max_tokens=1`, `allowed_token_ids=<actions>`, `logprobs=3` and
`logprobs_mode=processed_logprobs`, so the top-3 are renormalized over actions.

### Prompt layout

```
<|start_of_role|>system<|end_of_role|>{system prompt, cached}<|end_of_text|>
<|start_of_role|>user<|end_of_role|>{state, fresh}<|end_of_text|>
<|hunter|>assistant<|end_of_role|>  ->  one action token
```

Each adapter's aLoRA invocation sequence is
`<|start_of_role|>assistant<|end_of_role|>`. The composed chat template puts
the control token in place of that `<|start_of_role|>`, and at runtime the switch
gives the control token `<|start_of_role|>`'s embedding. The adapter therefore
sees exactly its training sequence, and everything before the control token is
base-model prefill that every adapter shares. `PromptBuilder` assembles these ids
directly each tick, with no Jinja rendering;
`python policy.py --check-template <composed-model>` asserts that they are
identical to `apply_chat_template(adapter_name=...)` for every adapter.

### Teacher and students

[`expert.py`](../scripts/doom/expert.py) decides from the same visible fields
the student reads; the only privileged state it keeps is a blacklist for items it
cannot reach. [`collect.py`](../scripts/doom/collect.py) runs it in parallel
envs with DART-style noise bursts, labeling every state with the teacher's
action. [`train_alora.py`](../scripts/doom/train_alora.py) trains each aLoRA
with PEFT (`LoraConfig(alora_invocation_tokens=...)`), with the loss on the one
output token. DAgger rounds then let the student drive while the teacher labels
the states it visits. [`router_data.py`](../scripts/doom/router_data.py) has a
larger local Granite model (`granite-4.1-8b`, in-process vLLM) write ~300
paraphrases per behavior; 30 hand-written instructions are held out for eval.

## Running it

### On a laptop (no model)

```bash
uv sync --extra doom --no-default-groups
cd tutorials/scripts/doom
python collect.py --policy expert --episodes 50 --stats-only --out /tmp/doom-stats
python server.py --policy expert          # http://localhost:8000
```

The UI then runs on the scripted teacher (and a keyword router), so everything
except the model can be developed and checked without a GPU.

### On a GPU host

```bash
uv sync --extra doom            # includes the default vllm19 group
cd tutorials/scripts/doom
BASE=ibm-granite/granite-4.1-3b

# 1. Latency go/no-go with stand-in adapters
python build_model.py standin --base $BASE --out models/standin-switch
python bench_latency.py --model models/standin-switch --json out/bench.json
python bench_latency.py --model $BASE --plain-base --scenarios single,decode,batch

# 2. Data: teacher rollouts with noise, router paraphrases
python collect.py --policy expert --episodes 60 --dart 0.15 --record 2 --out data/round0
python router_data.py --model ibm-granite/granite-4.1-8b --out data/router

# 3. Train, compose, check parity (composed vLLM vs PEFT, gate >= 99%)
for b in hunter survivor scavenger; do
  python train_alora.py --adapter $b --data data/round0/$b.jsonl --base $BASE --out runs/round0/$b
done
python train_alora.py --adapter router --data data/router/train.jsonl \
  --eval-data data/router/heldout.jsonl --epochs 4 --batch 32 --base $BASE --out runs/router/router
python build_model.py compose --runs runs/round0 --router-runs runs/router --base $BASE --out models/doom-round0
python build_model.py verify --runs runs/round0 --router-runs runs/router --model models/doom-round0

# 4. DAgger: the student drives, the teacher labels; retrain on the union
python collect.py --policy vllm --model models/doom-round0 --episodes 30 --out data/round1

# 5. Live demo
python server.py --policy vllm --model models/doom-round0 --host 0.0.0.0

# 6. A narrated video: telemetry panel, behavior switches through the router
python record_video.py --model models/doom-round0 --out out/granite_doom.mp4 \
  --segment hunter 25 --segment "stop fighting and grab health" 20 \
  --segment "collect all the loot" 20 --segment "go kill everything" 15
```

From a laptop, tunnel to the GPU host: `ssh -L 8000:<gpu-node>:8000 <login-node>`.

On an LSF cluster, give every GPU step cores on one host and a thread cap
(`bsub ... -n 16 -R "span[hosts=1]"`, `OMP_NUM_THREADS=8`), and start the
`bsub -I` client from `tmux` so a dropped ssh connection does not kill the job.

### The UI

- The game, or a grid of 4, 9 or 16 games that all decide in one batched engine
  step every tic.
- An instruction box (the router's pick, probability and latency), plus behavior
  chips that force an adapter.
- The **cadence** toggle: *every tic* (35 Hz), *10 Hz* (decide every 100 ms of
  game time and repeat the action in between, the cadence the Jev post
  describes), and *turbo* (unthrottled, so the model sets the pace).
- Decision latency p50/p99, decisions per second, the tic-budget bar (0–100 ms,
  with markers at one tic and at ~100 ms), a sparkline, and the per-decision
  breakdown: prompt build, engine step, and fresh vs cached tokens.
- Reaction time for each cadence: game time from a monster appearing on screen
  to the hunter's first shot at it.
- Action probabilities, live agreement with the teacher, shadow decisions (what
  the other behaviors would do on the same state), and the exact text the model
  reads.
