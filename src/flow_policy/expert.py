"""Scripted proportional-controller expert for PushT.

The environment's native success criterion (>=95% polygon-intersection
coverage between the current and goal block poses) requires matching the
block's rotation as well as its position. An earlier version of this expert
gave up on rotation entirely (see git history / README Milestone 1) after a
naive "always push through the centroid, with a small rotation-correcting
offset" approach failed to generalize -- rotation and position corrections
fought each other when applied simultaneously through a single contact
point.

This version instead alternates between two phases, each reusing the same
circle-around-an-anchor-then-push maneuver, just aimed at a different point:

- **rotate**: push tangentially at the block's stem tip (a fixed point ~75
  units from the centroid -- far, for good leverage) in whichever direction
  produces a rotation toward the goal angle, via r x F = |r| * sign(F ⟂ r) --
  push direction is always the +90-degree rotation of the centroid-to-tip
  direction, scaled by the sign of the needed rotation, which analytically
  guarantees torque of the correct sign regardless of current orientation.
- **translate**: push through the centroid toward the goal *centroid* (the
  goal pose's origin, `goal_pos`, shifted by the same centroid offset --
  rotated by `goal_angle` -- since the environment's coverage metric compares
  the block's raw origin to `goal_pos` directly, not the centroid).

A phase switches from translate to rotate once the angle error exceeds
`rotation_enter_thresh`, and back once it drops below the smaller
`rotation_exit_thresh` -- the hysteresis gap avoids flip-flopping every step
right at the boundary.

The observation's block pose (block_x, block_y, block_angle) is the T-shape's
body *origin* (where the crossbar meets the stem), not its centroid; see
`flow_policy.envs.block_centroid_world` for the fixed local-frame offset that
recovers it.
"""

import dataclasses

import numpy as np

from flow_policy.envs import block_centroid_world

# Stem-tip offset from the centroid, in the block's local (pre-rotation) frame.
# Local origin->stem-tip is (0, 120); local origin->centroid is (0, 45) (see
# flow_policy.envs.BLOCK_LOCAL_CENTROID); their difference is (0, 75).
_LOCAL_ROTATION_LEVER = np.array([0.0, 75.0])


def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _rot_matrix(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def _perp90(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]])


@dataclasses.dataclass
class ScriptedExpertConfig:
    transit_radius: float = 110.0  # standoff radius while circling around the current anchor point
    push_lead: float = 30.0  # how far past the anchor to aim while pushing
    align_angle_thresh: float = 0.2  # radians; switch to push phase once within this angle of the anchor
    max_step_angle: float = 0.12  # radians; max angular slew of the transit target per step
    rotation_enter_thresh: float = 0.35  # radians; switch translate -> rotate once angle error exceeds this
    rotation_exit_thresh: float = 0.05  # radians; switch rotate -> translate once angle error drops below this
    enable_rotation_phase: bool = False
    # ^ Off by default. The rotate/translate hysteresis makes the expert's action a
    # function of *phase history*, not just the current (obs, instruction) -- the
    # same visible state can map to either phase's action depending on which side
    # of the hysteresis band the expert entered from, a hidden variable the policy
    # never sees. That's genuine improvement in what the *expert* can do (coverage
    # nearly doubles, occasional native-success episodes -- see README), but it
    # measurably hurt the downstream imitation-learned policy (classic BC failure
    # mode: non-Markovian demonstrations corrupt the training signal). Set True to
    # reproduce that experiment; the default reflects what's actually used for the
    # primary dataset/checkpoint.


class ScriptedPushTExpert:
    """Alternates between rotating the block toward the goal angle and translating it toward the goal position."""

    def __init__(self, goal_pose: np.ndarray, config: ScriptedExpertConfig | None = None):
        self.goal_pos = np.asarray(goal_pose[:2], dtype=np.float64)
        self.goal_angle = float(goal_pose[2])
        self.config = config or ScriptedExpertConfig()
        self._phase = "translate"
        # The environment's coverage metric places the goal polygon's *origin* at
        # goal_pos directly (see flow_policy.envs module docstring / README) -- so
        # the block's origin, not centroid, should end up at goal_pos. The correct
        # centroid target is therefore goal_pos shifted by the centroid offset
        # rotated by the *goal* angle, not left at goal_pos itself (an earlier
        # version of this expert aimed the centroid at goal_pos directly, a
        # systematic ~45px error that silently capped achievable coverage).
        self._goal_centroid = block_centroid_world(self.goal_pos, self.goal_angle)

    def _steer(self, agent_pos: np.ndarray, anchor: np.ndarray, push_dir: np.ndarray, lead: float) -> np.ndarray:
        cfg = self.config
        anchor_to_agent = agent_pos - anchor
        dist = np.linalg.norm(anchor_to_agent)
        current_angle = (
            float(np.arctan2(anchor_to_agent[1], anchor_to_agent[0]))
            if dist > 1e-6
            else float(np.arctan2(-push_dir[1], -push_dir[0]))
        )
        desired_angle = float(np.arctan2(-push_dir[1], -push_dir[0]))
        angle_diff = _wrap_angle(desired_angle - current_angle)

        is_aligned = abs(angle_diff) < cfg.align_angle_thresh and dist < cfg.transit_radius * 1.3
        if is_aligned:
            target = anchor + push_dir * lead
        else:
            step = np.clip(angle_diff, -cfg.max_step_angle, cfg.max_step_angle)
            new_angle = current_angle + step
            target = anchor + cfg.transit_radius * np.array([np.cos(new_angle), np.sin(new_angle)])
        return target

    def act(self, obs: np.ndarray) -> np.ndarray:
        cfg = self.config
        agent_pos = np.asarray(obs[0:2], dtype=np.float64)
        block_pos = np.asarray(obs[2:4], dtype=np.float64)
        block_angle = float(obs[4])
        centroid = block_centroid_world(block_pos, block_angle)
        angle_error = _wrap_angle(self.goal_angle - block_angle)

        if cfg.enable_rotation_phase:
            if self._phase == "translate" and abs(angle_error) > cfg.rotation_enter_thresh:
                self._phase = "rotate"
            elif self._phase == "rotate" and abs(angle_error) < cfg.rotation_exit_thresh:
                self._phase = "translate"

        if self._phase == "rotate":
            lever = _rot_matrix(block_angle) @ _LOCAL_ROTATION_LEVER
            anchor = centroid + lever
            r_hat = lever / np.linalg.norm(lever)
            sign = 1.0 if angle_error >= 0.0 else -1.0
            push_dir = sign * _perp90(r_hat)
            lead = cfg.push_lead
        else:
            anchor = centroid
            # Aiming at the goal-angle-adjusted centroid only reduces origin error
            # when the angle is *also* being corrected -- with rotation disabled the
            # block's angle is uncontrolled, and (R(goal_angle) - R(actual_angle))
            # applied to the centroid offset is, on average, a *larger* random-
            # direction error than just aiming at goal_pos directly (whose error is
            # a constant-magnitude offset regardless of the random angle). So only
            # use the angle-adjusted target when rotation correction is active.
            target_centroid = self._goal_centroid if cfg.enable_rotation_phase else self.goal_pos
            to_goal = target_centroid - centroid
            dist_to_goal = np.linalg.norm(to_goal)
            push_dir = to_goal / dist_to_goal if dist_to_goal > 1e-6 else np.array([1.0, 0.0])
            # A full push_lead overshoots straight past the goal once close, and the
            # controller then has to circle back around -- causing the distance to
            # oscillate instead of settling. Cap the lead by the remaining distance
            # so the push target never lands past the goal itself.
            lead = min(cfg.push_lead, dist_to_goal * 0.5)

        target = self._steer(agent_pos, anchor, push_dir, lead)
        return np.clip(target, 0.0, 512.0).astype(np.float32)
