# Doom Stage 2 Report: A Granite Switch Player Against ZDoom Bots

Branch `feature/doom-deathmatch-bots`, work through 2026-09-28. Methods,
commands and the full tables are in the demo guide,
[`tutorials/guides/doom_demo.md`](../tutorials/guides/doom_demo.md); the code is
in [`tutorials/scripts/doom/`](../tutorials/scripts/doom/doom_env.py). This
report collects what was done, what was measured, and what we conclude.

## Conclusions

1. **Granite plays deathmatch well, deciding every tic.** A LoRA student
   distilled from an RL teacher scores 120.3 frags per ten-minute match against
   seven ZDoom bots. That is +96.1 ahead of the best bot, and 91% of the teacher's
   margin. It decides on each of Doom's 35 tics per second, in about 9-13 ms on
   one H100.
2. **aLoRA and Shadow Residual (SR) pay off only when several adapters read the
   same context.** With one adapter, LoRA is as fast or faster. With five
   adapters on one state, aLoRA:
   - keeps every tic under the 28.6 ms budget, including window resets, which
     LoRA misses (p99 30.8 ms);
   - switches behavior with no stall (9.9 ms against 19.5 ms);
   - holds 4.7x more games in the same KV memory.

   Playing many independent games at once, like a simultaneous-exhibition chess
   player, shows off vLLM batching, not aLoRA.
3. **SR keeps learning when the data is diverse.** On 2,000 randomized matches,
   SR's held-out loss falls to the end of a 320k-example pass, with no gap
   between training and held-out loss. The earlier "SR peaks at step 500, then
   overfits" came from too little, too correlated data, and was measured on a
   biased held-out slice (a bug, now fixed).
4. **SR ties aLoRA and trails LoRA by a constant margin.**
   - KL to the teacher: LoRA 0.176, aLoRA 0.229, SR 0.221 (at its own recipe).
   - Twice the parameters barely helps SR (0.218), and a lower learning rate ends
     worse.
   - In the field, SR and aLoRA both sample to about 104 frags, against LoRA's
     120. The aLoRA fighter was not given SR's tuned recipe, so the tie is best
     read as aLoRA at least matching SR.
5. **The cost of late invocation is structural.** On a history-only question
   (where was the last bot seen?):
   - LoRA reaches 0.902;
   - aLoRA 0.730 at best;
   - SR 0.657 at best.

   No rank, learning rate or capacity closes it. The history is pure base-model
   KV, which is what makes sharing it cheap, and no adapter can reshape it.
6. **Served SR was silently wrong until this work.** vLLM 0.19.1's
   FlashAttention 3 schedule does not fit SR's doubled query heads, so CUDA-graph
   steps returned wrong logits (0/24 agreement with PEFT). FlashAttention 2
   restores parity. Every SR number before the fix came from PEFT or HF, not from
   a served model.
7. **The RL teacher converged at about 130 frags.** The scripted player scores
   23.4; the teacher reaches 92.8 at 51M steps, 114.9 at 163M, and 130.5 at 706M.
   The last 100M steps changed nothing measurable. A wider teacher trained on
   mixed conditions did not do better.
8. **Under our rules, a single policy is enough.** Items respawn and weapons stay
   on the floor, so ammo is effectively unlimited. The teacher fires 27-39% of
   its shots at nothing, and 53% of them with the BFG. Phase-specialized behavior
   (forage when weak, hunt when armed) would only matter under scarcer rules.
   Scarce rules make the current teacher collapse (5.8 frags). The middle ground,
   Doom's classic rules, crashes ViZDoom 1.3.1.

## What was built

**The arena.** cig.wad MAP02, a free-for-all against the seven default ZDoom
bots, in synchronous `Mode.PLAYER`.

- **Disclosed differences from the ViZDoom competition:**
  - `+viz_nocheat` is off, because the structured state needs labels and
    objects;
  - vertical autoaim is on;
  - the respawn delay is 1 s rather than 10 s.
- **Actions:** 20 single-token actions, including fire-while-moving,
  strafe-running and circle-strafing. A separate weapon planner picks a slot
  every 0.5 s.

**The prompt.** Every adapter reads one prompt:

```
[system] [history: 5 Hz entries covering 10 s] [current state] <control token> -> one output token
```

**Six adapters in one Granite Switch checkpoint** (Granite 4.1 3B):

| Adapter | When it runs | Output |
|---|---|---|
| fighter, cautious, collector | every tic | one of 20 actions |
| arms (weapon planner) | every 0.5 s | a weapon slot |
| critic | every tic | danger in the next second (UI only) |
| router | on a typed instruction | a style |

**The RL teacher** is PPO with a GRU:
- asymmetric critic;
- 20-way movement head and weapon head;
- style inputs;
- Sample Factory's reward shaping;
- 160 envs on CPU workers with one GPU learner.

**Distillation.** The students imitate the teacher's action distribution (soft
labels), refined by DAgger rounds. Training is PEFT (LoRA, aLoRA or SR), with DDP
and length bucketing.

**Tooling** (in [`tutorials/scripts/doom/`](../tutorials/scripts/doom/doom_env.py)):

| Script | What it does |
|---|---|
| `doom_env.py` | arena, actions, state text, rule sets |
| `history.py` | the append-only history |
| `policy.py` | prompt assembly, vLLM serving, the FA2 selection for SR |
| `rl_teacher.py` | the teacher: training and evaluation |
| `collect.py` | teacher and DAgger data, field tests |
| `train_alora.py` | LoRA, aLoRA and SR training |
| `build_model.py` | compose (including SR) and parity verification |
| `bench_latency.py` | latency scenarios |
| `server.py`, `record_video.py` | the live demo and videos |

## Results

### Environment and latency (stages 0 and 1)

**Environment throughput:** 424 tics/s per process at 320x240 and 212 at
640x480, against a gate of 105.

**The scripted baseline** is fighter 23.4 frags against the best bot's 35.3.
Bot skill barely matters: the bots mostly frag each other.

**Latency per tic on one H100** (vLLM 0.19.1, FULL CUDA graphs), with stand-in
adapters:

| Per tic, ms | aLoRA | LoRA |
|---|---:|---:|
| fighter only, p50 / p99 | 9.0 / 11.5 | 8.6 / 10.3 |
| fighter + critic + planner, p99 | 12.6 | 16.6 |
| ... on window-reset tics, p99 | 19.5 | 30.8 |
| switch to an adapter idle for 10 s, p50 | 9.9 | 19.5 |
| 5 adapters on an uncached history, p50 | 25.8 | 86.4 |
| KV blocks per game, 5 adapters | 90 | 425 |

### The teacher (stage 2)

All results are 12 ten-minute matches against the default bots with the same
seeds.

| Teacher | Frags | Deaths | Margin |
|---|---:|---:|---:|
| scripted fighter | 23.4 | 7.4 | -11.9 |
| rl0, 51M steps | 92.8 | 19.2 | +65 |
| rl0, 163M | 114.9 | | |
| **rl0, 706M** (teacher for round f) | **130.5** | **14.8** | **+105.3** |
| rl0, 778M-800M (four checkpoints) | 129.3-133.2 | 12.0-15.4 | +104.7 to +110.9 |
| rl2, 584M (512 wide, mixed conditions) | 112.9 | 14.4 | +88.7 |

- **Reaction time and reactions per second both matter, and reaction time more.**
  The 51M-step teacher, 20 two-minute games per row, same seeds:

  | Reactions per second | Reaction time | Frags/min | Margin per game |
  |---:|---|---:|---:|
  | 35 | under 1 tic (this demo) | 9.09 | +10.2 |
  | 35 | 1 tic (29 ms) | 6.46 | +5.2 |
  | 35 | 3 tics (86 ms) | 2.89 | -3.0 |
  | 10 | under 1 tic | 3.60 | -1.7 |
  | 10 | 3 tics (86 ms) | 0.33 | -8.2 |

  - **In a sequential loop the two are one number:** about 100 ms per decision
    means about 10 reactions per second, each about 3 tics old.
  - **Pipelining raises the rate, not the reaction time.** Only a decision faster
    than a tic reaches the first row.
  - **The table overstates the loss for a slower player.** This teacher was
    trained for the first row only; a player trained for slower timing is
    untested.
- **Two rule changes were tried and dropped:**
  - The 10 s respawn cut deaths by about a third but did not stop firing at
    nothing.
  - Scarce items collapsed the teacher.

### Distillation (stage 3)

Fighter students, sampled (T = 1) and greedy:

| Round | Kind | Data | Matches | Sampled: frags / margin | Greedy: frags / margin |
|---|---|---|---:|---:|---:|
| b0 | aLoRA | 120 two-minute teacher matches (51M teacher) | 8 | 71.8 / +42.2 | 36.8 / +5.1 |
| c | aLoRA | + DAgger from b0, more teacher matches | 8 | 79.6 / +53.2 | 50.2 / +18.6 |
| d | aLoRA | + DAgger from c | 8 | 88.4 / +63.4 | 65.4 / +31.5 |
| d | LoRA | the same data | 8 | 95.9 / +68.2 | 74.9 / +48.0 |
| e | LoRA | 900 ten-minute teacher matches (163M teacher) + 300 DAgger | 8 | 100.5 / +76.6 | 114.6 / +90.1 |
| **f** | **LoRA** | **2,000 randomized fighter matches (706M teacher)** | 12 | **120.3 / +96.1** | 115.9 / +92.6 |
| f | SR | the same | 12 | 104.5 / +81.7 | 108.2 / +82.2 |
| f | aLoRA | the same | 12 | 104.3 / +80.2 | 82.2 / +55.7 |

What the rounds show:
- **More, better data helped every round.**
- **LoRA led aLoRA on identical data** (round d: +68.2 against +63.4).
- **Greedy decoding catches up as students improve**, but late-invocation
  students still stall in some greedy matches: SR in 2 of 12, aLoRA in 4.
- **The student's positions are recoverable.** When the teacher takes over
  student-started matches, it scores as it does from the start (11.4 against
  12.1 frags in 90 s). What students lack is decisions.

### The SR deep dive

**Serving.** SR attention in the granite-switch vLLM backend is one layer with
`2 * num_heads` query heads. vLLM 0.19.1 sizes FlashAttention 3's
ahead-of-time schedule from the model's head count:
- eager steps raise `scheduler_metadata must have shape (metadata_size)`;
- CUDA-graph steps return wrong logits without an error.

The same checkpoint under HF matched PEFT (TV 0.017). With FlashAttention 2 (or
the Triton backend), composed SR matches PEFT on every clear state (fighter TV
0.010). `VLLMPolicy` now picks FA2 for dual-stream checkpoints.

**The learning-curve study.** Seven fighter adapters were trained on the same
320k rows (from 2,000 matches, randomized over bot tier, 3-7 bots and 2-10
minutes), in the same order: one pass of 10k steps. They were scored every 500
steps on 3,000 rows from 120 separate matches. The teacher's entropy on those
rows (2.075 nats) is the floor, and KL is the distance above it.

| Arm | Held-out soft CE at 10k | KL to teacher | Top-choice agreement | Best step |
|---|---:|---:|---:|---:|
| LoRA r32, lr 1e-4 | 2.251 | **0.176** | **0.575** | 10,000 |
| aLoRA r32, lr 1e-4 | 2.304 | 0.229 | 0.531 | 10,000 |
| aLoRA r64 | 2.292 | 0.217 | 0.534 | 10,000 |
| SR r32, own recipe (lr 2e-4, α 2r, wd 0.01) | 2.296 | 0.221 | 0.534 | 9,000 |
| SR r64, cross-stream 77 (2x parameters), own recipe | 2.293 | 0.218 | 0.534 | 9,000 |
| SR r32, lr 1e-4 | 2.333 | 0.258 | 0.495 | 9,500 |
| SR r32, lr 5e-5 | 2.369 | 0.294 | 0.468 | 10,000 |

What the curves show:
- **The SR-LoRA gap is steady.** It stays at 0.03-0.045 nats throughout.
- **SR flattens first.** Over the last 40% of training it gains 0.011 nats, while
  LoRA and aLoRA gain 0.024.
- **Parameter counts are matched.** SR at r32 with cross-stream 32 is within 2%
  of LoRA's count (60.9M against 62.3M): under grouped-query attention, the k/v
  LoRA SR cannot have is small.

**The history probe.** Final full-set accuracy, where the majority class scores
0.474:

| | Accuracy |
|---|---:|
| LoRA r32 | 0.902 |
| aLoRA r32, lr 1e-4 / lr 2e-4 / r64 / r128 | 0.686 / 0.730 / 0.672 / 0.640 |
| SR, cross-stream 32, every recipe tried (lr 5e-5 to 2e-4, α r or 2r) | 0.613-0.633 |
| SR, cross-stream 96, lr 1e-4 / own recipe | 0.657 / 0.624 |

aLoRA and SR can tell whether a bot appears in the history, but not where.

### Rules and phases

With the 706M teacher, 12 matches per rule set:

| Rules | Frags | Time weak (nothing above the pistol) | Frags/min armed | Frags/min weak |
|---|---:|---:|---:|---:|
| standard | 130.5 | 8.1% | 14.1 | 2.0 |
| scarce | 5.8 | 83.9% | 3.8 | 0.03 |
| classic | crashes | | | |

**The classic crash.** It happens in ViZDoom's objects-info buffer:
- with that buffer off, the same matches run;
- every other rule set runs with it on;
- it happens even with the player standing still.

The trigger is a picked-up weapon hidden until it respawns, which only classic
rules do. Freedoom's DEHACKED patch is not the cause: it patches no Things, only
seven frames and text. The objects buffer only feeds the RL teacher's critic
(every bot's position), so disabling it under classic rules is a workable fix.
The root cause inside ViZDoom is still open.

## Caveats

- **The aLoRA-SR field comparison is not recipe-matched.** SR used its own
  recipe, about 4x aLoRA's effective step size. The planners are matched (same
  recipe and rows); the critics are not, but they only feed the UI.
- **Early mid-run curves used a biased held-out slice.** When the held-out set
  was smaller than `--max-heldout`, it was not shuffled, so the mid-run slice
  covered only a few matches. That affected the probe sweep's mid-run curves;
  final numbers were on the full set. It is fixed, and the round-f study used a
  shuffled 3,000-row set throughout.
- **Six-match evaluations are noisy.** One read 141.8 frags for a checkpoint that
  scores 129.3 over 12 matches, so only 12-match numbers are used for decisions.
- **SR in vLLM runs its adapter stream over every token,** shared context
  included. It shares the context across adapters but pays about double compute
  on prefill, which will show in throughput comparisons.
- **The phase measurement's re-arm time was not recorded** (the respawn event
  falls on a dead tic the script skipped). The weak-time and frag-rate columns
  are unaffected.

## Open work (not started)

- **A recipe-matched aLoRA-SR comparison:**
  - an aLoRA fighter trained with SR's recipe;
  - a field test of the SR fighter trained with aLoRA's recipe.
- **A DAgger round for SR.** `train_alora.py` cannot warm-start SR yet (`--init`).
- **The capacity grid:** G games x N adapters per game, for LoRA, aLoRA and SR,
  all on FA2, reporting the most games inside the tic budget.
- **A showcase where adapters share a context and each one matters:** phase
  specialists (hunter, forager) with a selector that runs with them in the same
  engine step. It needs rules where ammo binds, so the classic crash has to be
  worked around first.
- **Decoding for late-invocation students:** a low temperature or a stuck
  detector, to keep greedy's sharpness without the stalls.
- **Stage 4:** the live demo, a narrated video and the final guide.

## Commits

The branch commits after `9197adc`, as of this report:

| Commit | Change |
|---|---|
| `92d7cf8` | heatmap UI |
| `e325ae9` | deathmatch arena |
| `45d07d6` | history prompts, multi-adapter steps, RL teacher |
| `91ffb81` | serving plus stage-1 latency |
| `c61fd91` | video, whole-distribution parity |
| `7a607d0` | teacher beats the bots; students sample |
| `3693072` | sampling in the demo and video |
| `624a1c1` | distillation; aLoRA, LoRA and SR comparison |
| `96811bd` | takeover test |
| `74a1b4c` | teacher and data scaling |
| `ed47c43` | respawn and item-rule options |
| `c57e4f3` | SR serving fix, SR compose, trainer logging, 1 s respawn |
| `f91b165` | guide through round f, held-out shuffle fix |

Data, runs and models are on the cluster under
`/proj/dmfexp/lastrasl/doom-demo`:

- `data/f0`, `data/f0h`, `data/f1`: round-f training, held-out and DAgger data;
- `runs/study/*`: the seven study arms;
- `runs/f-*`: the planners and critics;
- `models/doom-f-{lora,alora,sr}`: the composed students.
