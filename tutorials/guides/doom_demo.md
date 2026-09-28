# Doom Reflex Demo: Granite Switch Decides Every Tic

Granite plays Doom from a structured text game state, one output token per
decision, on every Doom tic (35 Hz). Three behaviors (`hunter`, `survivor`,
`scavenger`) and a one-token instruction router are aLoRA adapters embedded in
one Granite Switch checkpoint. Switching behavior live changes only a control
token.

The code lives in [`tutorials/scripts/doom/`](../scripts/doom/doom_env.py).

> **Status.** Everything below, from "What the demo measures" on, is round 0:
> monsters on `deathmatch.wad`, 12 actions, and the behaviors `hunter`,
> `survivor` and `scavenger`, measured at commit `9197adc`. The scripts have
> since moved to stage 2, described in the next section, and the round-0
> commands no longer reproduce those numbers. Stage 2 is current through
> distillation round f (2026-09-28): a teacher that converged at about 130 frags
> per ten-minute match, and LoRA, aLoRA and Shadow Residual students trained on
> the same 2,000 matches.

## Stage 2 (in progress): deathmatch against built-in bots

The arena is cig.wad MAP02, a full deathmatch against seven ZDoom bots, hosted
in synchronous `Mode.PLAYER`. It differs from the ViZDoom competition rules in
two disclosed ways: `+viz_nocheat` is off, because the labels, objects and
sector data the state is built from need it off; and vertical autoaim is on,
because the action space is 2D. The model reads structured state, not pixels.
The action vocabulary grows to 20 single tokens, adding charge, fire while
backing or strafing, strafe-running and circle-strafing. Weapons are chosen by a
separate planner every 0.5 s (slot digits `1`-`7`). The styles are `fighter`,
`cautious` and `collector`.

Stage 0 measurements, on an M-series MacBook with 14 cores:

- **Environment throughput** with 7 bots and labels, objects and sectors on,
  running the full wrapper (state text, features, scripted policy), 12
  processes at once: 424 tics/s per process at 320x240, 212 tics/s at
  640x480. The gate is 3 x 35 = 105.
- **Output tokens.** All 20 actions, the weapon slots `1`-`7` and the danger
  levels `low`/`mid`/`high` are single, distinct Granite 4.1 tokens, also
  directly after `<|end_of_role|>` (`policy.py --check-template`).
- **Scripted baseline** ([`expert.py`](../scripts/doom/expert.py)) against the
  default bots: 12 ten-minute matches for `fighter`, 6 each for the other
  styles.

  | Style | Frags | Deaths | Best bot's frags | Margin | Top frag count |
  |---|---:|---:|---:|---:|---:|
  | fighter | 23.4 ± 11.5 | 7.4 ± 4.4 | 35.3 ± 8.5 | -11.9 ± 11.9 | 2 of 12 |
  | cautious | 14.5 ± 7.2 | 3.7 ± 3.9 | 29.3 ± 6.1 | -14.8 ± 12.1 | 1 of 6 |
  | collector | 9.2 ± 8.4 | 3.0 ± 2.5 | 33.3 ± 4.3 | -24.2 ± 11.8 | 0 of 6 |

  The fighter's frags track how long a bot is on screen. In its worst matches
  (0-5 frags) a bot was in view for under 35 of 600 s: the scripted player gets
  stuck in quiet parts of the map.
- **Bot skill matters little.** In matches mixing three skill-20 and three
  skill-100 bots (8 matches, 6 minutes each), the two groups scored 272 and 303
  frags. In ZDoom free-for-all the bots mostly frag each other. The benchmark is
  therefore the **default bots**, the first seven of the `bots.cfg` shipped with
  ViZDoom, unchanged ([`bots.cfg`](../scripts/doom/bots.cfg)).

### Stage 1: history, several adapters, and what aLoRA buys

Every adapter now reads one prompt: the system prompt, a history of the last
10 s, and the current state, then its own query suffix:

```
[system] [history: 5 Hz entries, append-only, up to ~1.3k tokens] [pad]
[now t14.2 | current state, ~80 tokens] [pad] <|adapter|>assistant<|end_of_role|>
```

History entries ([`history.py`](../scripts/doom/history.py)) carry absolute
match time (`t12.4 hp 64 face 135 | bot +10 8m | did cl`), so the history only
grows and stays in the prefix cache from tic to tic. The current-state line
starts with `now t14.2`. At 10 s the window drops its oldest half; that tic
re-prefills about 700 tokens. The padding (newlines to the next 16-token KV
block) leaves only the 5-token suffix as per-adapter work.

The same six stand-in adapters (rank 32, every linear layer, random weights)
are composed twice: as aLoRA, and as plain LoRA, whose control token sits at
position 0 so each adapter has its own KV from the first token. Both run the same
prompts: a 90 s trace of scripted play replayed with its history, including
window resets. Results are from [`bench_latency.py`](../scripts/doom/bench_latency.py)
on one H100 (vLLM 0.19.1, `cudagraph_mode=FULL`, capture sizes and
`max_num_batched_tokens` up to 2048), both runs on the same node:

| Per tic, ms | aLoRA | LoRA |
|---|---:|---:|
| Fighter only, p50 / p99 | 9.0 / 11.5 | 8.6 / 10.3 |
| Fighter + critic every tic, planner every 0.5 s, p50 / p99 | 10.2 / 12.6 | 10.6 / 16.6 |
| ... on window-reset tics, p99 | 19.5 | **30.8** |
| 5 adapters every tic, p50 / p99 | 13.4 / 16.5 | 16.9 / 24.7 |
| Behavior switch to an adapter idle for 10 s, p50 (steady: 9.7 / 8.9) | 9.9 | **19.5** |
| 5 adapters on a history not yet cached, p50 | 25.8 | 86.4 |

| KV per game (16-token blocks) | aLoRA | LoRA |
|---|---:|---:|
| 1 adapter | 86 | 85 |
| 5 adapters | 90 | 425 |
| Games that fit in the KV cache (0.5 GPU memory), 5 adapters | 226 | 48 |

What this shows:

- **With one adapter, aLoRA buys nothing.** LoRA is 0.4 ms faster; both prefill
  the same ~85 fresh tokens a tic.
- **Every extra adapter on the same state is nearly free with aLoRA.** vLLM
  shares the history even between requests of one step (it caches blocks when it
  allocates them): on a history no request has seen, requests 2 to 5 find about
  1,200 of their ~1,230 prompt tokens cached. LoRA adapters each prefill their
  own copy, so five of them on a cold history cost 3.3x as much.
- **Switching behavior is free with aLoRA and doubles the tic with LoRA,**
  because the LoRA adapter's own copy of the history is stale after 10 s.
- **Memory is 5x with LoRA at 5 adapters,** so it fits a fifth of the games.
- Both pass the latency gate with one game (p99 under the 28.6 ms tic), aLoRA
  with margin at every tic including window resets. LoRA with three adapters
  goes over the tic on resets.

### Stage 2: an RL teacher that beats the bots

[`rl_teacher.py`](../scripts/doom/rl_teacher.py) is PPO with a GRU (CleanRL
style, written for this demo rather than taken from Sample Factory, whose PyPI
release pins `gymnasium<1.0`). Its design:

- **Inputs.** The actor reads the same player-visible fields as the model's
  text, as numbers. The critic also sees every bot's position and the frag race.
- **Outputs.** It picks one of the 20 actions every tic, and a weapon slot every
  0.5 s.
- **Reward.** Sample Factory's deathmatch shaping, reweighted per style: cautious
  pays more for deaths and damage taken, collector for pickups.
- **Kickstarting.** Early on, a cross-entropy term pulls it toward the scripted
  player. Its weight decays to zero at 40M steps.
- **Throughput.** 160 envs in 40 worker processes run at 160x120, which sees the
  same objects as 640x480. That is about 10k steps/s on one H100 with 48 cores.

Evaluation uses ten-minute matches against the default bots, with the teacher
sampling from its distribution:

| Steps | Matches | Frags | Deaths | Best bot | Top frag count |
|---:|---:|---:|---:|---:|---:|
| 10M | 6 | 19.8 | 4.7 | 33.7 | 1 of 6 |
| 20M | 6 | 21.8 | 7.7 | 33.0 | 0 of 6 |
| 30M | 6 | 27.8 | 6.2 | 34.3 | 2 of 6 |
| 41M | 12 | 43.0 ± 8.3 | 10.2 | 29.8 | 9 of 12 |
| **51M** | **12** | **92.8 ± 7.7** | **19.2** | **27.8** | **12 of 12** |
| 61M-174M (12 evaluations) | 6 each | 97-117 | 11-20 | 20-28 | 6 of 6 each |

Kickstarting holds it near the scripted player until the imitation term fades.
Within 10M steps after that it triples its frags: the scripted fighter gets 23.4
per match, this teacher 92.8. From 61M steps it plateaus at about 100-117 frags,
with the bots' best falling to 20-28 as it takes their frags. The students are
distilled from the 51M-step checkpoint. Playing its most likely action instead
of sampling gives 84.5 ± 35.8 frags, top in 11 of 12, so the students sample too.

**Does reacting every tic matter for this player?** A rerun of
[`reaction_sweep.py`](../scripts/doom/reaction_sweep.py) with the 51M-step
teacher (20 two-minute games per row, same seeds across rows; "margin" is frags
ahead of the best bot):

| Condition | Fighter frags/min | Deaths/min | Margin |
|---|---:|---:|---:|
| Every tic, no delay (the demo: decisions well under a tic) | 9.09 ± 0.57 | 1.93 | +10.2 |
| Every tic, 1-tic delay | 6.46 ± 0.49 | 2.26 | +5.2 |
| Every tic, 3-tic delay (100 ms) | 2.89 ± 0.41 | 2.51 | -3.0 |
| 10 Hz, no delay | 3.60 ± 0.49 | 2.14 | -1.7 |
| 10 Hz, 3-tic delay (Jev-like) | 0.33 ± 0.15 | 2.04 | -8.2 |

One tic of staleness costs this player 29% of its frags. At 10 Hz with 100 ms of
latency it stops scoring and falls behind the bots. The scripted teacher was
much less sensitive (in round 0, staleness above a tic halved its kills). One
reason is that the RL teacher was trained to act every tic with no delay and
never saw one, so the table overstates what a player trained for 10 Hz would
lose. It is the right table for this demo: the model imitates exactly this
player.

The styles separate in the intended directions (51M steps, 6 matches each for
cautious and collector):

| Style | Frags | Deaths | Pickups/min |
|---|---:|---:|---:|
| fighter | 92.8 | 19.2 | 16.1 |
| cautious | 84.0 | **14.8** | 18.0 |
| collector | 84.2 | 17.0 | **21.2** |

**Training longer converges at about 130 frags.** All at 1 s respawn, 12
ten-minute matches against the default bots, the same seeds for every row
(steps = updates x 20,480):

| Teacher | Frags | Deaths | Margin |
|---|---:|---:|---:|
| rl0, 51M steps (rounds b-d) | 92.8 | 19.2 | +65 |
| rl0, 163M (round e) | 114.9 | | |
| **rl0, 706M (round f)** | **130.5 ± 11.4** | **14.8** | **+105.3** |
| rl0, 778M / 788M / 799M | 129.3 / 133.2 / 133.0 | 15.4 / 12.4 / 12.0 | +104.7 / +109.8 / +110.9 |
| rl0, 800M (end of run, lr annealed to 0) | 132.2 ± 9.1 | 12.9 | +105.8 |
| rl2, 584M (512 units wide, trained on mixed conditions) | 112.9 ± 13.6 | 14.4 | +88.7 |

The last five rl0 rows are within noise of each other (standard error about 4
frags); a 6-match reading of 141.8 at 778M did not hold up over 12 matches.
rl2 was trained on every bot tier, 3-7 bots and 2-10-minute matches, to
generalize. On a grid of 5-minute matches (4 tiers x 3 or 7 bots, 6 matches a
cell), rl0 at 400M steps still had the most frags in every cell; rl2 died least
in most cells.

**What the teacher does that looks odd.** 27-39% of its shots are fired with no
bot on screen, and 53% of its shots are the BFG. Under the standard rules firing
costs nothing: items respawn every 30 s and weapons stay on the floor, so ammo is
effectively unlimited. Two rule changes were tried and dropped:

- **The competition's 10 s respawn.** The 480M-step teacher scores 91.2 frags
  under it. Fine-tuned for 60M steps at 10 s (rl3), it ends at about 90 frags
  with 5.3-6 deaths a match, but fires at nothing more often, not less.
- **Scarce items** (nothing respawns, weapons do not stay): rl3 collapses to 5.9
  frags. A fine-tune (rl4) was recovering (16.8 frags, top in 4 of 6 at 15M
  steps) when it was stopped.

Everything reported uses the standard rules and 1 s respawn.

### Stage 3: distilling the teacher into Granite (in progress)

The 51M-step teacher plays 40 two-minute matches per style. Every live tic
becomes a training row: its history (rebuilt from the match's stream), the
state, and the teacher's whole move distribution. Planner tics also get its
weapon distribution. The critic's label is the outcome of the next second.

Adapters are rank 32 on every linear layer, α = rank, lr 1e-4, and use about
50k examples each. On held-out matches:

| Adapter | Agreement with the teacher's top choice | Majority baseline |
|---|---:|---:|
| fighter | 0.597 | 0.163 |
| cautious | 0.614 | 0.146 |
| collector | 0.583 | 0.177 |
| arms (weapon planner) | 0.736 | 0.352 |
| critic (danger in the next second) | 0.795, AUC 0.790 for "any damage" | 0.795 |
| router | 0.933 | 0.333 |

Top-choice agreement is capped by the teacher itself: its most likely action
carries 47% on average (entropy 1.60 nats). The fighter's soft cross-entropy of
1.85 nats is about 0.25 nats above that floor.

**The first student beats the bots, but at 65% of the teacher's margin.** It is
composed from these six aLoRAs and samples like the teacher. In 8 ten-minute
matches per style against the default bots:

| Student (sampling) | Frags | Deaths | Best bot | Margin | Top frag count |
|---|---:|---:|---:|---:|---:|
| fighter | 71.8 ± 15.1 | 16.9 | 29.5 | +42.2 | 8 of 8 |
| cautious | 61.1 ± 9.7 | 15.1 | 31.5 | +29.6 | 8 of 8 |
| collector | 66.6 ± 12.4 | 16.1 | 25.2 | +41.4 | 8 of 8 |
| fighter, greedy | 36.8 ± 22.5 | 8.1 | 31.6 | +5.1 | 5 of 8 |

The gate is 80% of the teacher's +65 margin, and the fighter reaches +42.2
(65%). It also looks less smart than its score suggests, especially early in a
match and on bad spawns. On six new seeds it scored 1.8 frags in the first
minute; on the evaluation seeds it scored 4.8, against the teacher's 6.8. It
deals half the teacher's damage (665 vs 1,292 per minute). Lowering the sampling
temperature does not help (0.7 to 1.0 all score the same within noise). So the
gap is in the decisions: imperfect imitation of a stochastic teacher compounds
over the several tics that aiming takes. The fix under way is DAgger. The
student plays, the teacher labels the states the student reaches (on those
states the student's action matches the teacher's top choice 33% of the time,
against 47% for the teacher's own samples), and the adapters are fine-tuned on
all the data.

Composed and PEFT agree on 99.6-99.97% of held-out states where PEFT's top
choice leads by at least 0.05 (mean total-variation distance 0.005-0.018).
Overall argmax agreement for the style adapters is 95.9-96.3%. The gap is
near-ties in flat distributions.

**DAgger rounds c-e.** Each round adds matches the previous student played,
labeled by the teacher, and fine-tunes from the previous round's adapters. The
field numbers are fighter students, 8 ten-minute matches against the default
bots:

| Round | Kind | Training data (matches) | Sampled: frags / margin | Greedy: frags / margin |
|---|---|---|---:|---:|
| b0 | aLoRA | 40 two-minute teacher matches per style (51M teacher) | 71.8 / +42.2 | 36.8 / +5.1 |
| c | aLoRA | + 20 two-minute DAgger matches per style from b0, 40 more two-minute teacher matches per style | 79.6 / +53.2 | 50.2 / +18.6 |
| d | aLoRA | + 20 two-minute DAgger matches per style from c | 88.4 / +63.4 | 65.4 / +31.5 |
| d | LoRA | the same, trained from scratch | 95.9 / +68.2 | 74.9 / +48.0 |
| e | LoRA | 300 ten-minute teacher matches per style (163M teacher), 100 DAgger matches per style from d-LoRA | 100.5 / +76.6 | **114.6 / +90.1** |

- **Greedy decoding catches up with more data.** An imperfect student's argmax
  loops (b0 greedy scores half its sampled frags); by round e greedy is ahead.
- **The student's states are not traps.** When the teacher takes over matches
  the student started, it scores 11.4 frags in the next 90 s, against 12.1 in
  matches it played from the start. What the student lacks is decisions, not
  position.

**aLoRA against LoRA and Shadow Residual on the same data.** The probe is a
history-only question: where, relative to the current heading, was the last
bot that appears in the history? It is asked only when none is on screen and
the state line's 5-second enemy memory has expired. All numbers are final and
use the full held-out set:

| | aLoRA r32 | aLoRA r64 | aLoRA r128 | Shadow Residual (cross-stream 32 / 96) | LoRA r32 |
|---|---:|---:|---:|---:|---:|
| probe accuracy (majority 0.474) | 0.686 | 0.672 | 0.640 | 0.628 / 0.657 | **0.902** |
| fighter agreement | 0.597 | | | 0.568 (step ~2000) | **0.673** |
| critic AUC, damage in 1 s | 0.790 | | | | 0.801 |

- **aLoRA and SR can tell whether a bot appears in the history, but not where.**
  On the probe's `wait` class (no bot in the last 10 s) both are at 100%. On
  the four directions, aLoRA is at 27-50% (chance is 25%) and LoRA at 75-85%.
  Finding the direction takes arithmetic across fields: the old entry's heading
  and bearing against the current heading. LoRA can encode that into the
  history tokens as it reads them. aLoRA and SR must do it at the last few
  positions against fixed base-model keys and values.
- **Rank doesn't fix it:** 64 and 128 are no better than 32. The limit is
  structural. It is the same property that makes aLoRA cheap (Stage 1): the
  history is pure base-model KV, so every adapter can share it and none can
  reshape it.
- **A learning-rate sweep helps aLoRA a little and SR not at all.** Final probe
  accuracy on the full held-out set (LoRA r32 at lr 1e-4: 0.902):

  | lr, α | 5e-5, 32 | 1e-4, 32 | 1e-4, 64 | 2e-4, 32 | 2e-4, 64 |
  |---|---:|---:|---:|---:|---:|
  | aLoRA r32 | 0.626 | 0.686 | 0.690 | **0.730** | 0.722 |
  | SR r32, cross-stream 32 | 0.613 | 0.628 | 0.632 | 0.619 | 0.633 (wd 0.01, SR's recipe) |

  SR's recipe with cross-stream 96 ends at 0.624. The mid-run evaluations in
  this sweep used the first 1,000 held-out rows, and the held-out set was not
  shuffled when nothing had to be dropped, so those rows came from only a few
  matches. The early peak SR showed on them (step 500 of 1,593) is therefore a
  weak signal; the study below re-tests it on a shuffled held-out set from 120
  matches. `train_alora.py` now always shuffles.

Two measurement notes on stage 1. vLLM 0.19.1 fails with "scheduler_metadata must have
shape (metadata_size)" on steps larger than the largest CUDA-graph capture
size, which only the LoRA baseline produces. Capping `max_num_batched_tokens`
at 2048 fixes it for both variants by chunking such prefills across steps.
And tail latency depends on the node: two nodes shared with other jobs showed
p90s near 20 ms for the same single-adapter run that measures 11.6 ms p90 on
quiet nodes. Both columns above ran on the same quiet node.

#### Round f: one teacher, one dataset, three kinds of adapter

**Data.** The 706M-step teacher plays 2,000 fighter matches and 600 for each
other style, each with a random bot tier, 3-7 bots and a length of 2-10 minutes.
Rows are every tenth tic plus every planner tic: 3.78M fighter rows. Another 120
matches per style, from the same conditions with other seeds, are the held-out
set every run is scored on.

**Serving Shadow Residual.** The first composed SR student agreed with its PEFT
adapter on 0-4% of held-out states under vLLM, with NaNs in some slots. The same
checkpoint under the HF backend matched it (mean total-variation distance 0.017).
The cause:

- SR's vLLM attention is one layer with twice the model's query heads,
  `Attention(2 * num_heads, num_kv_heads)`.
- vLLM 0.19.1 sizes FlashAttention 3's ahead-of-time schedule from the model's
  head count.
- So eager steps raise "scheduler_metadata must have shape (metadata_size)", and
  CUDA-graph steps return wrong logits without an error.

`VLLMPolicy` now selects FlashAttention 2 for dual-stream checkpoints
(`attention_config={"flash_attn_version": 2}`); the Triton backend also works.
With it, composed and PEFT agree on every state where the top choice is clear.
The fighter's mean total-variation distance is 0.010-0.018, and the planner,
critic and router agree on 99.4-100% of all states. Before this fix, no SR
number came from a served model; they were all PEFT evaluations.

**Does SR keep learning? A controlled study.** Seven fighter adapters are
trained under the same conditions:

- one pass over the same 320k rows in the same order, about 160 per match;
- batch 32, 10,000 steps, cosine schedule, rank 32 unless noted;
- scored every 500 steps on the same 3,000 held-out rows, as soft cross-entropy
  against the teacher's distribution.

The teacher's own entropy on those rows, 2.075 nats, is the floor. KL is the
distance above it.

| Arm (held-out soft CE at step) | 1k | 2k | 4k | 6k | 8k | 10k | KL to teacher | Top-choice agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LoRA, lr 1e-4 | 2.452 | 2.369 | 2.309 | 2.275 | 2.257 | **2.251** | **0.176** | **0.575** |
| aLoRA, lr 1e-4 | 2.509 | 2.429 | 2.361 | 2.328 | 2.308 | 2.304 | 0.229 | 0.531 |
| aLoRA x2 (rank 64) | 2.487 | 2.409 | 2.348 | 2.316 | 2.296 | 2.292 | 0.217 | 0.534 |
| SR, its own recipe (lr 2e-4, α = 2r, wd 0.01) | 2.493 | 2.403 | 2.338 | 2.307 | 2.296 | 2.296 | 0.221 | 0.534 |
| SR x2 (rank 64, cross-stream 77), own recipe | 2.491 | 2.409 | 2.345 | 2.308 | 2.294 | 2.293 | 0.218 | 0.534 |
| SR, lr 1e-4 | 2.493 | 2.421 | 2.364 | 2.340 | 2.334 | 2.333 | 0.258 | 0.495 |
| SR, lr 5e-5 | 2.552 | 2.456 | 2.396 | 2.375 | 2.369 | 2.369 | 0.294 | 0.468 |

- **With diverse data, SR learns to the end and does not overfit.** Its
  held-out loss falls until step 9,000 (288k examples). For every run, the loss
  on each fresh training batch stays within about 0.01 of the held-out loss.
- **SR's own recipe is its best; a lower learning rate ends worse.**
- **SR ties aLoRA and trails LoRA by a steady margin.** Both late-invocation
  kinds land at a KL of 0.22, LoRA at 0.18. The SR-LoRA gap is 0.03-0.045 nats
  throughout, and SR flattens first: over the last 40% of training it gains
  0.011 nats, while LoRA and aLoRA gain 0.024.
- **Doubling capacity does not close the gap.** At twice LoRA's parameters, SR
  gains 0.003 nats and aLoRA 0.012. (For SR, "x2" means rank 64 with the
  cross-stream sized to LoRA's parameter count.) At rank 32, a cross-stream of
  32 is already within 2% of LoRA's parameter count (60.9M vs 62.3M). The k/v
  LoRA SR cannot have is small under grouped-query attention.
- **Weapon planners trained on the same rows** agree with the teacher on 92-93%
  of held-out planner tics for all three kinds: SR 0.922, LoRA 0.931, aLoRA
  0.928.

**In the field.** Each round-f student combines that kind's fighter and its
planner (the same recipe and rows for all three) with a router. The SR student
also has a new SR critic; the LoRA and aLoRA students reuse critics from rounds
e and d, which only feed the UI. The other styles are copies of the fighter.
Each student played 12 ten-minute matches against the default bots, the same
seeds for all:

| Student | Greedy: frags / deaths / margin | Top | Sampled (T = 1): frags / deaths / margin | Top | Sampled margin / teacher's |
|---|---:|---:|---:|---:|---:|
| **LoRA** | 115.9 ± 28.7 / 7.6 / +92.6 | 12 of 12 | **120.3 ± 13.1 / 14.8 / +96.1** | 12 of 12 | **91%** |
| SR | 108.2 ± 40.9 / 8.6 / +82.2 | 11 of 12 | 104.5 ± 9.2 / 12.7 / +81.7 | 12 of 12 | 78% |
| aLoRA | 82.2 ± 50.6 / 4.9 / +55.7 | 9 of 12 | 104.3 ± 8.7 / 15.7 / +80.2 | 12 of 12 | 76% |
| teacher (706M) | | | 130.5 ± 11.4 / 14.8 / +105.3 | 12 of 12 | |

- **The LoRA student passes the stage gate** (80% of the teacher's margin): it
  reaches 91%, with every sampled match at 105 frags or more.
- **SR and aLoRA tie when sampling**, just under the gate, before any DAgger
  round on the round-f data.
- **Greedy decoding gets the late-invocation students stuck.** The SR student
  stalled in two matches (26 and 36 frags); in the other ten it averaged 123.6.
  The aLoRA student stalled in four (0, 21, 25 and 41). The LoRA student had one
  weak match (45). Sampling never stalled.
- **The aLoRA-SR comparison is not recipe-matched.** The SR fighter used SR's
  own recipe (lr 2e-4, α = 2r, wd 0.01), about four times the effective step of
  aLoRA's (lr 1e-4, α = r). On the probe, aLoRA gains from lr 2e-4, and at
  aLoRA's settings SR does worse than aLoRA on held-out (KL 0.258 against
  0.229). The tie is therefore best read as aLoRA at least matching SR.

**The item rules decide whether play has phases.** The 706M teacher, 12 matches
per rule set ("weak" means no usable weapon above the pistol):

| Rules | Frags | Time weak | Frags/min armed | Frags/min weak |
|---|---:|---:|---:|---:|
| standard | 130.5 | 8.1% | 14.1 | 2.0 |
| scarce | 5.8 | 83.9% | 3.8 | 0.03 |

Armed, the teacher scores seven times faster than weak, but under the standard
rules it is weak only briefly, because weapons stay on the floor. Under scarce
rules, trained where ammo is plentiful, it never learns to forage. Doom's
"classic" (altdeath) rules, where items respawn but weapons do not stay, crash
ViZDoom 1.3.1 within about 1.4 s in every match. The crash is in its objects-info
buffer: with that buffer off the same matches run, and every other rule set runs
with it on. The trigger is a picked-up weapon being hidden until it respawns,
which only classic does. Freedoom's DEHACKED patch is not involved; it touches no
Things, only seven frames and text.

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

Stage 2, round f (the deathmatch results above):

```bash
# Teacher: PPO-GRU against the bots (1 GPU learner, ~46 cores of env workers)
python rl_teacher.py train --out runs/rl0 --device cuda --envs 160 --workers 40 \
  --eval-workers 6 --eval-matches 6 --eval-every 500 --total-steps 800000000 --lr 2e-4
python rl_teacher.py eval --ckpt runs/rl0/ckpt_034500.pt --matches 12 --styles fighter \
  --respawn-s 1 --workers 40 --seed 5000
T=runs/rl0/ckpt_034500.pt

# Data: randomized matches, sparse rows, and a held-out set from other seeds
V="--bots all --n-bots-min 3 --n-bots 7 --timeout-min-s 120 --timeout-s 600 --row-every 10"
python collect.py --policy teacher --teacher $T --behaviors fighter --episodes 2000 --seed 100000 $V --out data/f0/fighter
python collect.py --policy teacher --teacher $T --behaviors cautious collector --episodes 600 --seed 200000 $V --out data/f0/cc
python collect.py --policy teacher --teacher $T --episodes 120 --seed 900000 $V --out data/f0h

# One fighter per kind on the same rows; SR with its own recipe
C="--eval-data data/f0h --every 1 --max-examples 320000 --max-heldout 3000 --eval-n 3000 --batch 32 --keep-best"
python train_alora.py --adapter fighter --kind lora $C --data data/f0/fighter --base $BASE --out runs/f-lora/fighter
python train_alora.py --adapter fighter --kind sr --lr 2e-4 --alpha 64 --weight-decay 0.01 --grad-ckpt --micro 8 \
  $C --data data/f0/fighter --base $BASE --out runs/f-sr/fighter
# (arms and critic the same way, on data/f0/fighter data/f0/cc; the router on data/router)

# Compose (the composer reads SR from the weights), check parity, play
python build_model.py compose --kind sr --runs runs/f-sr --router-runs runs/router-sr --base $BASE --out models/doom-f-sr
python build_model.py verify --runs runs/f-sr --router-runs runs/router-sr --model models/doom-f-sr
python collect.py --policy vllm --model models/doom-f-sr --teacher expert --behaviors fighter \
  --episodes 12 --timeout-s 600 --stats-only --out out/field_f-sr
```

Under DDP, prefix `train_alora.py` with `python -m torch.distributed.run --standalone
--nproc_per_node N`; each rank takes a slice of every batch.

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
