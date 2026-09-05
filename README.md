# flow-policy

## Project framing

This is **not** a VLA. It is a small, honest artifact: a language-conditioned
imitation-learning policy with a flow-matching action head, trained on a toy
manipulation task, wrapped in a ROS2 interface. The point is to demonstrate real
engineering decisions (conditioning, multimodal action generation, action
chunking, deployment plumbing) at a scale that's actually verifiable in a
weekend — not to overclaim scope.

## Status

- [x] Milestone 0 — environment setup
- [x] Milestone 1 — task env + scripted expert data collection
- [ ] Milestone 2 — dataset + dataloader
- [ ] Milestone 3 — vision + language conditioning encoder
- [ ] Milestone 4 — flow-matching action head
- [ ] Milestone 5 — training loop + sanity checks
- [ ] Milestone 6 — inference: ODE sampling + action chunking
- [ ] Milestone 7 — evaluation harness + metrics
- [ ] Milestone 8 — ROS2 wrapper
- [ ] Milestone 9 — README, video, polish

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
block centroid within `POSITION_SUCCESS_RADIUS = 30px` of the goal position
(`src/flow_policy/envs.py`), ignoring final rotation. The environment's
native `coverage` and `is_success` are still recorded in every saved episode
for reference.

```bash
uv run python scripts/collect_data.py
```

```
[push to the left target] 100 episodes -- position-success rate: 0.67 (radius=30.0px), mean env coverage: 0.34
[push to the right target] 100 episodes -- position-success rate: 0.74 (radius=30.0px), mean env coverage: 0.32
```

200 episodes (100/variant), randomized initial agent/block poses, saved to
`data/episodes/ep_XXXXX_vN.npz` (gitignored — regenerate with the command
above). Each file: `observations (T, 5)`, `actions (T, 2)`, `variant_id`,
`instruction`, `position_success`, `final_position_dist`, `final_coverage`,
`env_is_success`.

The scripted expert (`src/flow_policy/expert.py`) circles around the block's
true centroid (recovered from the body-origin state via a fixed local-frame
offset — the origin sits at the top of the T, not its center) at a safe
standoff radius until angularly aligned opposite the goal direction, then
drives straight through the centroid toward the goal. Circling first, rather
than aiming straight at a "point behind the block," avoids cutting across the
block from the wrong side and knocking it in a random direction — an earlier,
naive version of this controller did exactly that and had a 0% success rate.
