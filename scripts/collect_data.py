"""Milestone 1: run the scripted expert to collect demonstration episodes.

For each task variant ("push to the left target" / "push to the right
target"), runs the scripted expert for N episodes with randomized initial
agent/block poses (the environment's default reset behavior), and saves each
episode to disk as one .npz file.

Success is measured by this project's own position-only metric (block
centroid within `POSITION_SUCCESS_RADIUS` px of the goal position) rather than
the environment's native 95%-coverage criterion, which also requires matching
the block's final rotation -- see `flow_policy.expert` and
`flow_policy.envs.POSITION_SUCCESS_RADIUS` for why that's out of scope here.
The environment's native `coverage` / `is_success` are still recorded per
episode for reference.
"""

import argparse
import pathlib

import numpy as np

from flow_policy.envs import POSITION_SUCCESS_RADIUS, TASK_VARIANTS, block_centroid_world, make_variant_env
from flow_policy.expert import ScriptedPushTExpert

DEFAULT_OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "episodes"
EPISODES_PER_VARIANT = 100
MAX_STEPS = 300


def collect_episode(variant, seed, max_steps=MAX_STEPS):
    env = make_variant_env(variant)
    obs, info = env.reset(seed=seed)
    expert = ScriptedPushTExpert(variant.goal_pose)

    observations = []
    actions = []
    for _ in range(max_steps):
        action = expert.act(obs)
        observations.append(obs.astype(np.float32))
        actions.append(action.astype(np.float32))
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()

    centroid = block_centroid_world(obs[2:4], obs[4])
    final_dist = float(np.linalg.norm(centroid - variant.goal_pose[:2]))
    position_success = final_dist <= POSITION_SUCCESS_RADIUS

    return {
        "observations": np.stack(observations),  # (T, 5)
        "actions": np.stack(actions),  # (T, 2)
        "variant_id": variant.variant_id,
        "instruction": variant.instruction,
        "position_success": position_success,
        "final_position_dist": final_dist,
        "final_coverage": float(info["coverage"]),
        "env_is_success": bool(info["is_success"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-variant", type=int, default=EPISODES_PER_VARIANT)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    episode_idx = 0
    for variant in TASK_VARIANTS:
        successes = []
        coverages = []
        for i in range(args.episodes_per_variant):
            seed = args.seed_offset + variant.variant_id * 100_000 + i
            episode = collect_episode(variant, seed=seed)
            successes.append(episode["position_success"])
            coverages.append(episode["final_coverage"])

            path = args.out_dir / f"ep_{episode_idx:05d}_v{variant.variant_id}.npz"
            np.savez(
                path,
                observations=episode["observations"],
                actions=episode["actions"],
                variant_id=episode["variant_id"],
                instruction=episode["instruction"],
                position_success=episode["position_success"],
                final_position_dist=episode["final_position_dist"],
                final_coverage=episode["final_coverage"],
                env_is_success=episode["env_is_success"],
            )
            episode_idx += 1

        success_rate = float(np.mean(successes))
        mean_coverage = float(np.mean(coverages))
        summary[variant.instruction] = {"success_rate": success_rate, "mean_coverage": mean_coverage}
        print(
            f"[{variant.instruction}] {args.episodes_per_variant} episodes -- "
            f"position-success rate: {success_rate:.2f} (radius={POSITION_SUCCESS_RADIUS}px), "
            f"mean env coverage: {mean_coverage:.2f}"
        )

    print(f"\nSaved {episode_idx} episodes to {args.out_dir}")
    return summary


if __name__ == "__main__":
    main()
