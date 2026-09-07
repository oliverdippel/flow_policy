# flow-policy

## Project framing

This is **not** a VLA. It is a small, honest artifact: a language-conditioned
imitation-learning policy with a flow-matching action head, trained on a toy
manipulation task, wrapped in a ROS2 interface. The point is to demonstrate real
engineering decisions (conditioning, multimodal action generation, action
chunking, deployment plumbing) at a scale that's actually verifiable in a
weekend — not to overclaim scope.

## At a glance

A CLIP-conditioned flow-matching policy learns to push a T-shaped block to
one of two named targets on the PushT board, from 800 scripted-expert
demonstrations, entirely on CPU.

| | |
|---|---|
| **Proof the flow-matching head works at all** | On a synthetic bimodal target (`x1 = +-2`), sampled points split ~45/54% across the true modes with 0.4% collapsed to the mean — the exact failure mode plain MSE regression has ([Milestone 4](#milestone-4--flow-matching-action-head)). |
| **Real closed-loop rollouts** | Seeded spot-checks push the block with visible, deliberate motion, verified with multi-frame contact sheets, not just final frames ([Milestone 6](#milestone-6--inference-ode-sampling--action-chunking)). |
| **Real evaluation numbers (N=25/variant)** | Success 0% (left) / 12% (right); instruction-confusion 0% on both; mean coverage ~0.32-0.35, in the same range as the scripted expert's own. These are corrected numbers — an earlier pass reported 76%/64% against a position metric with a real bug in it. See [Rotation fix + dataset-size ablation](#rotation-fix--dataset-size-ablation) for the full story, including why more data stopped helping around 400-800 episodes. |

![left target rollout](assets/rollout_push_to_the_left_target.gif)
![right target rollout](assets/rollout_push_to_the_right_target.gif)

### Architecture

```
instruction (str) --> frozen CLIP text encoder --> cached embed (512) --\
                                                                          >--> concat --> proj --> cond (128)
state obs (5,)    --> small MLP encoder --> obs embed (64) -------------/                              |
                                                                                                          v
noise x0 ~ N(0,I) (16,) --\                                                              v_theta(xt, t, cond)
                            >--> xt = (1-t)x0 + t*x1 -------------------------------------> (MLP, predicts velocity)
ground truth x1 (16,) ----/                                                                              |
                                                                                                          v
                                                          Euler-integrate dx/dt = v_theta, t: 0 -> 1, 10 steps
                                                                                                          |
                                                                                                          v
                                                        action chunk (H=8, 2) --> receding horizon (execute 4, replan)
                                                                                                          |
                                                                                                          v
                                                                                            gym-pusht env / ROS2 topics
```

`x1` is a flattened `H=8`-step action chunk (`16` = `8*2`); training samples
`t` and `x0` fresh per example and regresses `v_theta` toward `x1 - x0` via
MSE — see [Milestone 4](#milestone-4--flow-matching-action-head)
for why this reproduces multimodal action distributions instead of averaging
them away.

### Reproduce

```bash
uv sync
uv run python scripts/smoke_test_env.py     # M0: sanity-check the env
uv run python scripts/collect_data.py       # M1: 800 scripted-expert episodes
uv run python scripts/inspect_batch.py      # M2: dataset shape/index checks
uv run python scripts/test_conditioning.py  # M3: conditioning encoder unit test
uv run python scripts/test_flow_matching_bimodal.py  # M4: bimodal proof-of-correctness
uv run python scripts/train.py              # M5: train (8000 epochs, ~27 min on CPU)
uv run python scripts/rollout.py            # M6: closed-loop rollout -> GIFs
uv run python scripts/evaluate.py           # M7: N=25/variant evaluation harness
```

### What I'd do with more time or compute

- **A more precise expert, not just a rotation-aware one.** The rotate/
  translate expert in [Rotation fix + dataset-size ablation](#rotation-fix--dataset-size-ablation)
  gets the environment's real 95%-coverage criterion to ~4.7% -- a real
  crack in a number that was a hard 0% all project, but its
  phase-hysteresis makes its own actions non-Markovian, which measurably
  hurt the policy trained on its demonstrations. A version without hidden
  phase state (e.g. a single continuous control law that blends rotation
  and translation correction as a function of the *current* state alone,
  not a mode switch with memory) could plausibly transfer the coverage
  improvement to the learned policy instead of just the expert.
- **Given the dataset-size ablation came back flat (200-1600 episodes,
  success 0.06-0.12 throughout, no trend), data volume is not the current
  lever** -- the policy has converged to the expert's own precision
  ceiling. A materially better expert or a different architecture/
  observation space matters more now than collecting more of the same.
- **Pixel observations + a real vision backbone.** The plan's low-dim-state
  path was the right first cut for a weekend, but a small CNN (or a frozen
  pretrained backbone) over the 96x96 RGB observation is the natural next
  step, and would make FiLM conditioning (instead of concatenation) worth
  its complexity.
- **A harder, genuinely multimodal real task.** The two PushT variants prove
  conditioning routes correctly, but neither one actually needs multimodal
  action generation to solve — the synthetic bimodal test is still the only
  place this project demonstrates *why* flow matching over MSE regression.
  A task with real left/right/either-works ambiguity at a single state would
  make that case on real data, not just a toy target.
- **More eval seeds.** N=25/variant is small enough that the confusion rate
  hitting exactly 0% on both variants could plausibly move with more seeds;
  worth confirming it holds at N=100+ before leaning on it too hard.
- **Actually run the Milestone 8 ROS2 nodes** against a real ROS2 install
  rather than a syntax-checked, never-executed stub.

A scope note in the same spirit: the plan's own budget was ~800 lines of
code total; this repo is at ~1610 (`src/flow_policy` + `scripts` +
`ros2_nodes`), including a 9-way milestone split where each gets its own
argparse-CLI script, and ~160 lines of never-executed ROS2 stub. The
scripted expert itself went through several rewritten strategies while
debugging Milestone 1 and again in the rotation-fix follow-up (see those
sections for why), but only the final, working version — 162 lines, now
covering both the position-only and rotation-aware modes — is in the repo;
the throwaway tuning scripts were deleted once they'd done their job. Noting
the overage rather than quietly not mentioning it.

## Status

- [x] Milestone 0 — environment setup
- [x] Milestone 1 — task env + scripted expert data collection
- [x] Milestone 2 — dataset + dataloader
- [x] Milestone 3 — vision + language conditioning encoder
- [x] Milestone 4 — flow-matching action head
- [x] Milestone 5 — training loop + sanity checks
- [x] Milestone 6 — inference: ODE sampling + action chunking
- [x] Milestone 7 — evaluation harness + metrics
- [~] Milestone 8 — ROS2 wrapper (documented stub, not run — see below)
- [x] Milestone 9 — README, video, polish

## Performance iteration (post-Milestone 9)

The Milestone 7 numbers (28% / 8% success) were honestly reported but weak,
and weren't good enough as a final result. Three changes, cheapest first:

1. **Switched the default `replan_every` from 4 to the full chunk size (8)**
   — i.e. commit to the whole predicted chunk before replanning, instead of
   replanning every 4 of 8 steps as the plan originally specified. Free (no
   retraining): on the Milestone 7 checkpoint this alone moved average
   success from 18% to 26%, at the cost of a worse confusion rate on one
   variant (8%→24% on left, 20%→12% on right) — a real but mixed win, not a
   clean one.
2. **Recollected a bigger dataset**: 200 → 800 episodes (400/variant instead
   of 100). While doing this, `collect_data.py` turned out to have a real
   bug: episode filenames are `ep_{index}_{variant}.npz` with `index`
   assigned sequentially across both variants, so the same index can land on
   a different variant across two runs with different `--episodes-per-variant`
   values — old files then don't get overwritten, they pile up alongside the
   new ones. Training briefly ran on a 900-file dataset that silently included
   100 duplicate episodes from the original 200-episode run (harmless in
   content, but a real hygiene bug and a reproducibility risk). Fixed by
   clearing the output directory at the start of every collection run, then
   recollected and retrained clean.
3. **Bigger model**: the flow-matching MLP's hidden width/depth went from
   256/3 layers to 512/4. Trained 8000 epochs on the clean 800-episode
   dataset (~27 minutes on CPU — still cheap, just no longer near-instant).

| metric | Milestone 7 (200 ep, small model, replan=4) | after (800 ep, bigger model, replan=8) |
|---|---|---|
| success rate (left / right) | 28% / 8% | **76% / 64%** |
| instruction-confusion rate | 8% / 20% | **0% / 0%** |
| mean final distance | 85px / 95px | 29px / 29px |
| final training loss | ~0.16 | ~0.10 |

Re-verified with contact sheets (frames sampled across each episode, not
just the last one) that this is genuine pushing from far-away spawns, not a
statistical artifact of lucky initial block placement. The rollout GIFs and
`assets/evaluation_results.json` at the top of this README are from this
final checkpoint, not the original Milestone 7 run.

One number that *didn't* move: the environment's own native 95%-coverage
success rate is still 0% throughout all of this. Every change here targeted
getting the block to the right *position* more reliably; none of it touches
the scripted expert's inability to control the block's final *rotation*
(Milestone 1), which is the actual gap between this project's relaxed
success metric and PushT's real one.

**Superseded, in part, by the next section.** Tackling that rotation gap
turned up a real bug in the position-success metric itself (not just the
expert) that was inflating every number above. The 76%/64% here was real
relative to the metric it was measured against, but that metric turns out
not to have been quite the right one. Read on.

## Rotation fix + dataset-size ablation

Two follow-up questions after the performance iteration above: could the
scripted expert's rotation blind spot (Milestone 1) actually be fixed, and
was 800 episodes enough, or would more data help further? Both
investigations, done together, turned up something more important than
either answer on its own.

### The metric bug this surfaced

Building a rotation-aware expert meant computing the block's true target
pose precisely, which meant looking closely at exactly how `gym_pusht`
places its goal polygon. It places the goal polygon's *origin* at
`goal_pose[:2]` directly — the same raw origin the state observation
reports (`obs[2:4]`), **not** the block's centroid. Every position-success
number in this README up to this point (Milestones 1, 6, 7, and the
performance iteration above) compared the block's **centroid** to
`goal_pos` instead — a self-consistent yardstick (the expert aimed at the
same point the metric measured against), but not the one that determines
actual task success. Concretely, for one identical final state:

```
origin-based dist (correct, matches env coverage): 34.7px
centroid-based dist (the old, inflated metric):     11.5px
```

That's a ~23px gap on a single state, and it isn't a fixed offset -- it
varies with the block's rotation (the centroid sits ~45px from the origin,
in a direction that rotates with the block), so it can't be corrected with
a simple constant adjustment; it has to be fixed at the source. Fixed by
comparing `obs[2:4]` to `goal_pos` directly everywhere a position-success
number is computed (`collect_data.py`, `rollout.py`, `evaluation.py`), and
by giving the scripted expert's translate phase the right aim point too
(`goal_pos` shifted by the centroid offset, evaluated at *goal* angle, not
left at `goal_pos` itself — see `flow_policy/expert.py`).

This means every number reported earlier in this README understated how
far the block actually was from the target. It doesn't mean the work done
in Milestones 1-9 or the performance iteration was wrong -- the relative
comparisons within each of those sections (before/after a specific change)
remain valid, since the same yardstick was used on both sides of each one.
It means the *absolute* numbers were more flattering than the real task.
The numbers from here on use the corrected metric.

### The rotation-aware expert: a genuine improvement that didn't transfer

`flow_policy/expert.py` now alternates between two phases: **rotate** --
push tangentially at the block's stem tip (~75px lever arm from the
centroid) in whichever direction produces torque toward the goal angle
(the push direction is always the goal-angle-error-signed +90-degree
rotation of the centroid-to-tip direction, which analytically guarantees
the correct torque sign regardless of current orientation) -- and
**translate**, the original position-only strategy. A hysteresis band
(switch to rotate above a 0.35 rad angle error, back to translate below
0.05 rad) avoids flip-flopping at the boundary.

Tuned and measured over N=150/variant: this expert nearly doubles mean
coverage (0.44-0.49, vs. 0.29-0.32 for position-only) and achieves the
environment's actual 95%-coverage success **~4.7% of the time** -- a real,
reproducible crack in a number that had been a hard 0% for the entire
project up to this point.

Then the actual test: recollect the primary dataset with this expert,
retrain the same policy architecture, re-evaluate. Result: **worse**, not
better -- policy success dropped, confusion rate got worse, and the
policy's own native-success rate stayed at 0% despite the expert's own
occasional success. The likely cause, on inspection: the rotate/translate
hysteresis makes the expert's action a function of *phase history*, not
just the current `(obs, instruction)` -- right in the hysteresis band, the
same visible state can produce either phase's action depending on which
side the expert entered from, a hidden variable the policy never observes.
That's textbook non-Markovian demonstration data, and behavior cloning
degrades when the same visible state maps to conflicting target actions.
This is a real, if unglamorous, finding: a scripted expert that performs
better on the task is not automatically a better *data generator* for
imitation learning if its own policy has hidden state the learner can't see.

Given that, the rotation phase is implemented and available
(`ScriptedExpertConfig(enable_rotation_phase=True)`) but **off by default**
-- the primary dataset and checkpoint use position-only demonstrations,
which is what the numbers throughout this README (post-metric-fix) reflect.

### Dataset-size ablation: 200 / 400 / 800 / 1600 episodes

Trained the same architecture on four dataset sizes, each for a *matched*
~200,000 gradient steps (more episodes -> more batches/epoch -> fewer
epochs needed, rather than confounding "more data" with "more compute"):

| episodes | epochs | avg success rate | avg coverage |
|---|---|---|---|
| 200 | 33,333 | 0.08 | 0.32 |
| 400 | 16,666 | 0.10 | 0.29 |
| 800 | 8,000 | 0.06 | 0.34 |
| 1600 | 4,000 | 0.12 | 0.29 |

Flat. No dataset size in this range does meaningfully better than any
other -- success rate bounces between 0.06 and 0.12 with no trend, and
coverage sits in a tight 0.29-0.34 band throughout. **More data of this
expert's quality is not the current bottleneck.** The tell: the scripted
expert's own coverage (0.29-0.32 without rotation correction) is in the
same range the trained policies land in -- the policy has essentially
converged to imitating the expert's own precision ceiling, and no amount
of additional demonstrations at that precision raises the ceiling. The
lever that would actually move this number is a *better expert*
(precision, not just task-completion rate) or a materially different
architecture/observation space (pixels, a larger model) -- not more data
collection at the current setup. Full per-size results:
`assets/eval_ablation_{200,400,1600}.json` (800 is
`assets/evaluation_results.json`, the primary checkpoint).

### Where the primary checkpoint landed

```
[push to the left target]  (25 rollouts)
  success rate:              0.00
  mean final dist:           44.5px
  mean env coverage:         0.35
  instruction-confusion rate: 0.00

[push to the right target]  (25 rollouts)
  success rate:              0.12
  mean final dist:           48.1px
  mean env coverage:         0.32
  instruction-confusion rate: 0.00
```

Weaker than the pre-metric-fix 76%/64%, and that's the honest number now.
Confusion rate is still 0% on both -- conditioning routes the push
correctly essentially always, even though precise landing on target is the
harder, still-unsolved part. The rollout GIFs and `evaluation_results.json`
linked at the top of this README are from this checkpoint.

## Milestone 0 — Environment setup

Task/env: [`gym-pusht`](https://github.com/huggingface/gym-pusht) (PushT), installed via `uv`.

```bash
uv sync
uv run python scripts/smoke_test_env.py
```

Confirms `reset`/`step`/`render` all work end-to-end and saves a GIF of a
random-action rollout to `assets/smoke_test_random_rollout.gif`.

Note: `gym-pusht` declares `pymunk>=6.6.0`, but pymunk 7.x removed
`Space.add_collision_handler`, which `gym-pusht==0.1.6` still relies on. Pinned
`pymunk<7` in `pyproject.toml` to work around this.

![random rollout](assets/smoke_test_random_rollout.gif)

## Milestone 1 — Task variants + scripted expert + data collection

Two task variants, differing only in goal *position* (`src/flow_policy/envs.py`):

- **"push to the left target"** — goal pose `(156, 256, 45°)`
- **"push to the right target"** — goal pose `(356, 256, 45°)`

`TargetPushTEnv` wraps the base env and overrides `goal_pose` right after each
reset (PushT reads `self.goal_pose` fresh every step, so this is enough to
redirect reward/success/rendering). Crucially, the 5-dim state observation
(agent + block pose) never reveals which target is active — only the
instruction does.

### On the success metric (read before trusting the numbers)

PushT's native success criterion is >=95% polygon-intersection coverage
between the current and goal block poses — which requires matching the
block's *rotation*, not just its position. Precisely controlling a T-shape's
rotation through non-prehensile pushing is a genuinely hard control problem
(it's why the original PushT dataset was collected via human teleoperation,
not a scripted controller) and is out of scope for a heuristic proportional
controller. Since these two task variants only differ by goal *position*,
this project measures the scripted expert's success by **position only**:
the block's raw origin (`obs[2:4]`, matching how `goal_pos` is defined --
see [Rotation fix + dataset-size ablation](#rotation-fix--dataset-size-ablation)
for a metric bug this project shipped for a while, comparing the block's
*centroid* to `goal_pos` instead) within `POSITION_SUCCESS_RADIUS = 30px` of
the goal position (`src/flow_policy/envs.py`), ignoring final rotation. The
environment's native `coverage` and `is_success` are still recorded in every
saved episode for reference.

```bash
uv run python scripts/collect_data.py
```

```
[push to the left target] 400 episodes -- position-success rate: 0.28 (radius=30.0px), mean env coverage: 0.34
[push to the right target] 400 episodes -- position-success rate: 0.26 (radius=30.0px), mean env coverage: 0.34
```

800 episodes (400/variant — bumped from 100/variant during the
[performance iteration](#performance-iteration-post-milestone-9), same rates
either way), randomized initial agent/block poses, saved to
`data/episodes/ep_XXXXX_vN.npz` (gitignored — regenerate with the command
above; it clears the directory first, since re-running with a different
`--episodes-per-variant` used to leave stale orphaned files behind — see the
performance-iteration section). Each file: `observations (T, 5)`,
`actions (T, 2)`, `variant_id`, `instruction`, `position_success`,
`final_position_dist`, `final_coverage`, `env_is_success`.

The scripted expert (`src/flow_policy/expert.py`) circles around the block's
true centroid (recovered from the body-origin state via a fixed local-frame
offset — the origin sits at the top of the T, not its center) at a safe
standoff radius until angularly aligned opposite the goal direction, then
drives straight through the centroid toward the goal. Circling first, rather
than aiming straight at a "point behind the block," avoids cutting across the
block from the wrong side and knocking it in a random direction — an earlier,
naive version of this controller did exactly that and had a 0% success rate.
A later rotation-aware version of this same expert is documented in
[Rotation fix + dataset-size ablation](#rotation-fix--dataset-size-ablation) —
it's a genuine capability improvement for the expert, but disabled by
default because it measurably hurt the policy trained on its demonstrations.

## Milestone 2 — Dataset + dataloader

`PushTChunkDataset` (`src/flow_policy/dataset.py`) loads all episodes into
memory and, per `__getitem__`, samples one episode and a random timestep `t`
within it, returning that step's observation, the episode's instruction, and
an `H=8` action chunk `actions[t : t+H]` (padded by repeating the final action
if the episode ends first).

```bash
uv run python scripts/inspect_batch.py
```

```
dataset has 800 episodes
observation: shape=(16, 5) dtype=torch.float32
instruction: list of 16 strings, e.g. 'push to the left target'
action_chunk: shape=(16, 8, 2) dtype=torch.float32
shape assertions passed.

spot-check: batch obs matched ep_00318_v0.npz at timestep t=280
  ...
spot-check passed: action_chunk exactly matches actions[t : t+H] from the source episode.
```

The spot-check re-locates the sampled observation in its source `.npz` file
and confirms the returned action chunk is byte-for-byte `actions[t:t+H]` —
catches off-by-one indexing bugs directly rather than trusting shapes alone.
Since every collected episode runs the full 300 steps (the scripted expert
never triggers the env's own 95%-coverage termination — see Milestone 1), a
single random batch essentially never exercises the padding branch (only the
last 7 of 300 timesteps trigger it), so it's verified separately by forcing
`t = T-1` and checking the chunk equals the final action repeated 8 times.

## Milestone 3 — Conditioning encoder

`src/flow_policy/conditioning.py`:

- `ObservationEncoder` — small MLP, `obs (5,) -> 64 -> 64`.
- `InstructionEncoder` — frozen CLIP ViT-B-32 (`open_clip`, `openai` weights,
  `quickgelu` variant to match how those weights were trained). Only the two
  fixed instructions from `flow_policy.envs` ever appear, so both are embedded
  once at construction and cached as a buffer — there's no CLIP forward pass
  at training or inference time. An instruction outside that fixed set raises
  `KeyError` rather than silently falling back to a live encode.
- `ConditioningEncoder` — concatenates the two embeddings and projects to a
  fixed-size `cond` vector (concatenation chosen over FiLM for simplicity —
  FiLM would earn its keep more with a higher-capacity vision backbone than a
  5-dim state vector).

```bash
uv run python scripts/test_conditioning.py
```

```
CLIP text embedding L2 distance (left vs right): 2.802
cond shape: (8, 128)
cond L2 distance (left vs right instruction, same obs): [0.507, 0.504, ...]
all conditioning encoder assertions passed.
```

Confirms: `cond` has the expected shape, an identical `(obs, instruction)`
pair always gives the exact same cached embedding, and swapping the
instruction while holding `obs` fixed measurably changes `cond` (L2 distance
~0.5, well above the epsilon) — conditioning is actually doing something.

## Milestone 4 — Flow-matching action head

`src/flow_policy/flow_matching.py` implements the linear (optimal-transport)
conditional flow-matching path: `x0 ~ N(0,I)`, `t ~ U(0,1)`, `xt = (1-t)x0 +
t*x1`, regress `v_theta(xt, t, cond)` toward `v_target = x1 - x0` via MSE;
sample by Euler-integrating `dx/dt = v_theta` from `t=0` to `t=1`. `cond` is
optional throughout, so the head is fully testable before it's wired to the
real conditioning encoder.

The reason this milestone matters: plain MSE-regression BC collapses
multimodal target distributions to their mean, which for an actual bimodal
target is a sample that never occurs. Flow matching's whole point is
reproducing the distribution instead — so before touching real (messy,
partially-successful) robot data, that claim is checked on a synthetic target
where the right answer is known exactly.

```bash
uv run python scripts/test_flow_matching_bimodal.py
```

```
fraction of samples near +2.0: 0.454
fraction of samples near -2.0: 0.538
fraction collapsed near the mean (0): 0.004

PASS: samples are bimodal, not collapsed to the mean.
```

![bimodal test histogram](assets/flow_matching_bimodal_test.png)

`x1` is +2 or -2 with equal probability (no conditioning). After 3000 training
steps, 1000 Euler-sampled points from the learned model split ~45/54% between
the two true modes with only 0.4% landing near the mean — the collapsed-mean
failure mode a plain MSE regressor would produce. This is the milestone's
proof of correctness; the head is only wired into the real task starting
Milestone 5.

## Milestone 5 — Training loop + sanity checks

`src/flow_policy/policy.py` ties the conditioning encoder and flow-matching
head into `FlowMatchingPolicy`, and adds an `EMA` helper (decay 0.999) for a
smoothed copy of the weights to use at inference. One catch found here:
PushT's action space is raw pixel coordinates in `[0, 512]`, and flow matching
interpolates actions against `N(0,1)` noise — feeding in unnormalized actions
made the regression target's scale ~500x too large (loss ~80000 instead of
~1). Fixed by rescaling actions to `[-1, 1]` using the environment's known,
fixed bounds (`normalize_action` / `denormalize_action` in `policy.py`) before
they ever reach the flow-matching head.

`scripts/train.py` wires in the dataset (M2): each batch computes `cond` from
`(observation, instruction)`, samples `t` and `x0`, computes the flow-matching
loss, backprops with grad-norm clipping (max 1.0), steps Adam (lr 2e-4), and
updates the EMA copy. Checkpoints (model + EMA state, plus the model's own
capacity kwargs so a checkpoint always reloads with the exact architecture it
was trained with) saved periodically.

```bash
uv run python scripts/train.py
```

```
dataset: 800 episodes, 25 batches/epoch, batch_size=32
epoch    0  loss ~1.1
epoch 2000  loss ~0.14
epoch 6000  loss ~0.10
epoch 7999  loss 0.1035
```

![training loss curve](assets/training_loss.png)

**Correction, found while building Milestone 6:** this originally ran for 200
epochs (~1200 gradient steps) on a 200-episode dataset, and the loss curve
alone looked fine — smooth and decreasing. But feeding a real held-out state
through the trained model and comparing its predicted action chunk to the
expert's showed pure noise (predicted coordinates like `597, -46` — outside
the board entirely), and closed-loop rollouts just drove the agent to wander
while the block sat at its spawn point untouched. Loss trending down is not
the same as samples being any good; only checking actual samples caught it.
Retrained for 10,000 epochs, which fixed it (loss ~1.19 → ~0.16, visibly
plateauing) — full details on that first fix are in the git history. The
numbers and config above are from a second round of training, described in
[Performance iteration](#performance-iteration-post-milestone-9): 800
episodes (not 200) and a bigger model (512-hidden/4-layer, not
256/3-layer), 8000 epochs, final loss ~0.10. Checkpoints land in
`checkpoints/` (gitignored — retrain with the command above; ~27 minutes
on CPU at this size, not the ~2 minutes the original smaller run took).

## Milestone 6 — Inference: ODE sampling + action chunking

`src/flow_policy/rollout.py`'s `RecedingHorizonController`: each planning call
samples a full 8-step action chunk from the EMA-weight policy (Euler ODE,
10 steps), then executes some number of those actions before re-observing
and re-predicting. An optional temporal-ensembling mode blends overlapping
chunks (a chunk generated `replan_every` steps ago still covers the newest
chunk's steps, just at a larger in-chunk offset), weighted down by how stale
that prediction is.

```bash
uv run python scripts/rollout.py
```

```
[push to the left target] position_success=False (dist=38.7px) env_coverage=0.39
[push to the right target] position_success=False (dist=47.3px) env_coverage=0.37
```

![left target rollout](assets/rollout_push_to_the_left_target.gif)
![right target rollout](assets/rollout_push_to_the_right_target.gif)

Both rollouts show real, deliberate pushing (confirmed by inspecting frames
across the episode, not just the final one) — the agent circles to the far
side of the block and drives it toward the correct target, not a coincidence
of where the block happened to spawn — just short of this project's 30px bar
on this particular seed, with coverage in the high-0.3s (numbers here are
post-metric-fix; see [Rotation fix + dataset-size ablation](#rotation-fix--dataset-size-ablation)
for what changed and why these are lower than an earlier pass reported). One
seed each is a spot-check, not a statistic — Milestone 7 runs N=25 per
variant for the real numbers.

**A responsiveness/smoothness tradeoff, originally left as the plan's
specified default, later revisited:** the plan specifies replanning every 4
of the 8 predicted steps, which the first version of this milestone used as
the default. In side-by-side testing on that checkpoint, committing to the
*full* 8-step chunk before replanning (`replan_every = chunk_size`) did
better (both variants under the 30px bar) than either 4-step replanning or
temporal ensembling. Original writeup here reasoned that a still-imperfect
model made more frequent replanning more vulnerable to chunk-to-chunk
sampling noise interrupting a committed push, and kept the plan's specified
default rather than switching to whatever scored best on one checkpoint.
That reasoning turned out right in a way that mattered: revisited during the
[performance iteration](#performance-iteration-post-milestone-9) below, full-chunk
commit is now the default, worth exactly the tradeoff described above rather
than a workaround for an undertrained model.

## Milestone 7 — Evaluation harness + metrics

`scripts/evaluate.py` runs 25 rollouts per variant (distinct seeds from data
collection and the M6 spot-checks) and reports: success rate (this project's
position-only metric), mean time-to-success (first step the block came within
30px of goal, among rollouts that ever got there), and the
**instruction-confusion rate** — how often the block ended up closer to the
*other* target than the one instructed. That last number is the point of
having two variants at all: it checks conditioning actually steered the push,
not just whether the policy can push competently.

```bash
uv run python scripts/evaluate.py
```

```
[push to the left target]  (25 rollouts)
  success rate:              0.00
  mean final dist:           44.5px
  mean env coverage:         0.35
  mean time-to-success:      131.1 steps
  never reached target:      0.72
  instruction-confusion rate: 0.00

[push to the right target]  (25 rollouts)
  success rate:              0.12
  mean final dist:           48.1px
  mean env coverage:         0.32
  mean time-to-success:      171.4 steps
  never reached target:      0.60
  instruction-confusion rate: 0.00
```

Full per-episode results in `assets/evaluation_results.json`.

**Read this next to Milestone 6, not instead of it** — one seed each there
is a demo, not a statistic; this N=25 harness is the real number. This
milestone's numbers moved twice after first being reported: up to 76%/64%
during the [performance iteration](#performance-iteration-post-milestone-9)
(more data, a bigger model, a better replan default), then back down to the
0%/12% above once [a real bug in the position-success metric itself](#rotation-fix--dataset-size-ablation)
was found and fixed -- the metric had been comparing the block's centroid to
the goal position instead of its actual origin, which is what the
environment itself uses. What hasn't moved across any of these three passes:
instruction-confusion has stayed at or near 0% throughout, including the
very first, weakest pass (which was 8%/20%, not 0%, but still mostly
"missed the target" rather than "pushed to the wrong side"). Conditioning
routing the push correctly is the one claim in this project that's held up
under every revision.

## Milestone 8 — ROS2 wrapper (documented stub, not run)

No ROS2 in this environment: no `ros2` CLI, no `/opt/ros`, `ROS_DISTRO`
unset, and `import rclpy` fails in both the system Python and the project's
venv. Per the plan's own guidance for this case, the intended design is
documented here with real code (`ros2_nodes/policy_node.py`,
`ros2_nodes/pusht_bridge.py`) rather than skipped silently — but neither
file has been executed or even successfully imported, since `rclpy` isn't
installed. Treat them as a design document with code in it, not a tested
artifact.

- **`policy_node.py`** subscribes to `/pusht/observation`
  (`Float32MultiArray`, the 5-dim state) and `/pusht/instruction`
  (`String`), runs Milestone 6's `RecedingHorizonController` internally
  (same replanning logic as the Python-only rollout — this node is a
  transport layer around it, not a second implementation), and publishes to
  `/pusht/action` (`Float32MultiArray`) at 10Hz.
- **`pusht_bridge.py`** owns a live `gym-pusht` env, publishes the current
  observation and instruction, and steps the env each time an action
  arrives on `/pusht/action` — so a real run would prove the full loop
  (env → topic → policy → topic → env) running through ROS2 message
  passing, not just direct Python calls.

If ROS2 becomes available: `ros2 run` both nodes (or a launch file starting
both), and the action topic should visibly publish in response to the
observation topic, with `pusht_bridge.py`'s logger showing episode resets.
