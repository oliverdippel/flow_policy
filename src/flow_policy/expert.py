"""Scripted proportional-controller expert for PushT.

The environment's native success criterion (>=95% polygon-intersection
coverage between the current and goal block poses) requires matching the
block's rotation as well as its position -- and rotation is genuinely hard to
control with a simple heuristic pusher (this is why PushT's original dataset
was collected via human teleoperation rather than a scripted oracle). This
project's task variants only differ by goal *position* ("left" vs "right"
target), so this expert -- and this project's own success metric, see
`flow_policy.envs.POSITION_SUCCESS_RADIUS` -- targets position only and
ignores the block's final orientation.

The observation's block pose (block_x, block_y, block_angle) is the T-shape's
body *origin* (where the crossbar meets the stem), not its centroid; see
`flow_policy.envs.block_centroid_world` for the fixed local-frame offset that
recovers it.

Strategy each step: circle around the block's centroid at a safe standoff
radius until angularly aligned with the direction from the centroid to the
goal, then drive straight through the centroid toward the goal. Circling
first (rather than aiming straight at a point behind the block) avoids
cutting across the block from the wrong side and knocking it in a random
direction.
"""

import dataclasses

import numpy as np

from flow_policy.envs import block_centroid_world


def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclasses.dataclass
class ScriptedExpertConfig:
    transit_radius: float = 110.0  # standoff radius while circling around the block's centroid
    push_lead: float = 30.0  # how far past the centroid to aim while pushing
    align_angle_thresh: float = 0.3  # radians; switch to push phase once within this angle
    max_step_angle: float = 0.12  # radians; max angular slew of the transit target per step


class ScriptedPushTExpert:
    """Proportional controller toward a fixed goal position (position only, ignores angle)."""

    def __init__(self, goal_pose: np.ndarray, config: ScriptedExpertConfig | None = None):
        self.goal_pos = np.asarray(goal_pose[:2], dtype=np.float64)
        self.config = config or ScriptedExpertConfig()

    def act(self, obs: np.ndarray) -> np.ndarray:
        cfg = self.config
        agent_pos = np.asarray(obs[0:2], dtype=np.float64)
        block_pos = np.asarray(obs[2:4], dtype=np.float64)
        block_angle = float(obs[4])
        centroid = block_centroid_world(block_pos, block_angle)

        to_goal = self.goal_pos - centroid
        dist_to_goal = np.linalg.norm(to_goal)
        push_dir = to_goal / dist_to_goal if dist_to_goal > 1e-6 else np.array([1.0, 0.0])

        centroid_to_agent = agent_pos - centroid
        dist_centroid_to_agent = np.linalg.norm(centroid_to_agent)
        current_angle = (
            float(np.arctan2(centroid_to_agent[1], centroid_to_agent[0]))
            if dist_centroid_to_agent > 1e-6
            else float(np.arctan2(-push_dir[1], -push_dir[0]))
        )
        desired_angle = float(np.arctan2(-push_dir[1], -push_dir[0]))
        angle_diff = _wrap_angle(desired_angle - current_angle)

        is_aligned = (
            abs(angle_diff) < cfg.align_angle_thresh and dist_centroid_to_agent < cfg.transit_radius * 1.3
        )
        if is_aligned:
            target = centroid + push_dir * cfg.push_lead
        else:
            step = np.clip(angle_diff, -cfg.max_step_angle, cfg.max_step_angle)
            new_angle = current_angle + step
            target = centroid + cfg.transit_radius * np.array([np.cos(new_angle), np.sin(new_angle)])

        return np.clip(target, 0.0, 512.0).astype(np.float32)
