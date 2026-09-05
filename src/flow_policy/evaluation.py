"""Milestone 7: evaluation harness -- run many rollouts per variant and
compute success rate, time-to-success, and the language-conditioning
confusion check (did "push left" actually push the block right?)."""

import numpy as np

from flow_policy.envs import LEFT_TARGET, RIGHT_TARGET, TaskVariant
from flow_policy.policy import FlowMatchingPolicy
from flow_policy.rollout import run_rollout

_SIDE_GOAL_POS = {
    "left": LEFT_TARGET.goal_pose[:2],
    "right": RIGHT_TARGET.goal_pose[:2],
}


def classify_achieved_side(block_centroid: np.ndarray) -> str:
    """Which of the two fixed target positions the block ended up closer to,
    independent of which one was instructed."""
    dist_left = np.linalg.norm(block_centroid - _SIDE_GOAL_POS["left"])
    dist_right = np.linalg.norm(block_centroid - _SIDE_GOAL_POS["right"])
    return "left" if dist_left < dist_right else "right"


def evaluate_variant(
    policy: FlowMatchingPolicy,
    variant: TaskVariant,
    num_episodes: int,
    seed_start: int,
    **rollout_kwargs,
) -> list[dict]:
    instructed_side = "left" if variant.variant_id == LEFT_TARGET.variant_id else "right"

    episodes = []
    for i in range(num_episodes):
        result = run_rollout(policy, variant, seed=seed_start + i, render=False, **rollout_kwargs)
        achieved_side = classify_achieved_side(result.final_block_centroid)
        episodes.append(
            {
                "seed": seed_start + i,
                "num_steps": result.num_steps,
                "position_success": result.position_success,
                "first_success_step": result.first_success_step,
                "final_position_dist": result.final_position_dist,
                "final_coverage": result.final_coverage,
                "env_is_success": result.env_is_success,
                "instructed_side": instructed_side,
                "achieved_side": achieved_side,
                "confused": achieved_side != instructed_side,
            }
        )
    return episodes


def summarize_episodes(episodes: list[dict]) -> dict:
    successes = [e["position_success"] for e in episodes]
    success_steps = [e["first_success_step"] for e in episodes if e["first_success_step"] is not None]
    return {
        "num_episodes": len(episodes),
        "success_rate": float(np.mean(successes)),
        "mean_final_dist": float(np.mean([e["final_position_dist"] for e in episodes])),
        "mean_coverage": float(np.mean([e["final_coverage"] for e in episodes])),
        "env_native_success_rate": float(np.mean([e["env_is_success"] for e in episodes])),
        "mean_time_to_success": float(np.mean(success_steps)) if success_steps else None,
        "frac_never_reached_target": 1.0 - len(success_steps) / len(episodes),
        "confusion_rate": float(np.mean([e["confused"] for e in episodes])),
    }
