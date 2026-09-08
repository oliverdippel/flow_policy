"""Scripted proportional-controller expert for PushT.

The environment's native success criterion (>=95% polygon-intersection
coverage between the current and goal block poses) requires matching the
block's rotation as well as its position. This file has been through three
designs for that, each a response to the previous one's failure mode:

1. **none** -- position only, never corrects rotation. Simple and stable,
   but caps achievable coverage well below what the criterion needs.
2. **hysteresis** -- alternates between a rotate phase (tangential push at
   the stem tip, torque-signed toward the goal angle) and the original
   translate phase, switching between them with a hysteresis band. Nearly
   doubles the expert's own coverage and gets occasional native successes --
   but the hysteresis makes the expert's action a function of *phase
   history*, not just the current `(obs, instruction)`: right at the
   boundary, the same visible state can produce either phase's action
   depending on which side the expert entered from. That's non-Markovian
   demonstration data, and it measurably hurt the policy trained on it (see
   README).
3. **continuous_blend** -- no phase, no memory: every step blends a
   translation force (toward the goal centroid) with a rotation-correcting
   tangential force, weighted continuously by the current angle error
   (saturating, not thresholded). Since it's a pure function of the current
   state with no stored phase, it's structurally immune to the non-Markovian
   failure mode above -- the open question this mode exists to answer is
   whether it can recover option 2's coverage improvement without also
   recovering its problem.

The observation's block pose (block_x, block_y, block_angle) is the T-shape's
body *origin* (where the crossbar meets the stem), not its centroid; see
`flow_policy.envs.block_centroid_world` for the fixed local-frame offset that
recovers it. All three modes share a circle-around-an-anchor-then-push
maneuver (`_steer`) that avoids cutting across the block from the wrong side.
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


def _normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else fallback


@dataclasses.dataclass
class ScriptedExpertConfig:
    transit_radius: float = 110.0  # standoff radius while circling around the current anchor point
    push_lead: float = 30.0  # how far past the anchor to aim while pushing
    align_angle_thresh: float = 0.2  # radians; switch to push phase once within this angle of the anchor
    max_step_angle: float = 0.12  # radians; max angular slew of the transit target per step

    rotation_mode: str = "continuous_blend"  # "none", "hysteresis", or "continuous_blend" -- see module docstring

    # hysteresis mode only:
    rotation_enter_thresh: float = 0.35  # radians; switch translate -> rotate once angle error exceeds this
    rotation_exit_thresh: float = 0.05  # radians; switch rotate -> translate once angle error drops below this

    # continuous_blend mode only:
    rotation_blend_gain: float = 0.5  # scales the rotation-correcting component relative to translation
    rotation_saturation_angle: float = 0.9  # radians; angle error beyond this gets full blend weight (no more)


class ScriptedPushTExpert:
    """Pushes the block toward a goal pose, per `config.rotation_mode` (see module docstring)."""

    def __init__(self, goal_pose: np.ndarray, config: ScriptedExpertConfig | None = None):
        self.goal_pos = np.asarray(goal_pose[:2], dtype=np.float64)
        self.goal_angle = float(goal_pose[2])
        self.config = config or ScriptedExpertConfig()
        self._phase = "translate"  # hysteresis mode only
        # The environment's coverage metric places the goal polygon's *origin* at
        # goal_pos directly (see flow_policy.envs module docstring / README) -- so
        # the block's origin, not centroid, should end up at goal_pos. The correct
        # centroid target is therefore goal_pos shifted by the centroid offset
        # rotated by the *goal* angle, not left at goal_pos itself. This only pays
        # off when rotation is also being corrected (mode != "none") -- see
        # _translate_target below.
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

    def _translate_target(self) -> np.ndarray:
        # See __init__: the angle-adjusted centroid target only helps once rotation
        # is actually being corrected; "none" mode aims at raw goal_pos instead.
        return self.goal_pos if self.config.rotation_mode == "none" else self._goal_centroid

    def act(self, obs: np.ndarray) -> np.ndarray:
        cfg = self.config
        agent_pos = np.asarray(obs[0:2], dtype=np.float64)
        block_pos = np.asarray(obs[2:4], dtype=np.float64)
        block_angle = float(obs[4])
        centroid = block_centroid_world(block_pos, block_angle)
        angle_error = _wrap_angle(self.goal_angle - block_angle)

        to_goal = self._translate_target() - centroid
        dist_to_goal = np.linalg.norm(to_goal)
        translate_dir = to_goal / dist_to_goal if dist_to_goal > 1e-6 else np.array([1.0, 0.0])
        # A full push_lead overshoots straight past the goal once close, and the
        # controller then has to circle back around -- causing the distance to
        # oscillate instead of settling. Cap the lead by the remaining distance
        # so the push target never lands past the goal itself.
        translate_lead = min(cfg.push_lead, dist_to_goal * 0.5)

        if cfg.rotation_mode == "hysteresis":
            if self._phase == "translate" and abs(angle_error) > cfg.rotation_enter_thresh:
                self._phase = "rotate"
            elif self._phase == "rotate" and abs(angle_error) < cfg.rotation_exit_thresh:
                self._phase = "translate"

            if self._phase == "rotate":
                lever = _rot_matrix(block_angle) @ _LOCAL_ROTATION_LEVER
                anchor = centroid + lever
                r_hat = _normalize(lever, np.array([0.0, 1.0]))
                sign = 1.0 if angle_error >= 0.0 else -1.0
                push_dir = sign * _perp90(r_hat)
                lead = cfg.push_lead
            else:
                anchor, push_dir, lead = centroid, translate_dir, translate_lead

        elif cfg.rotation_mode == "continuous_blend":
            # No stored phase: this is a pure function of the current state, so
            # (unlike "hysteresis") the same (obs, instruction) always produces the
            # same action -- Markovian demonstrations by construction.
            anchor = centroid
            perp = _perp90(translate_dir)
            rot_scale = np.clip(angle_error / cfg.rotation_saturation_angle, -1.0, 1.0)
            blended = translate_dir + cfg.rotation_blend_gain * rot_scale * perp
            push_dir = _normalize(blended, translate_dir)
            lead = translate_lead

        else:  # "none"
            anchor, push_dir, lead = centroid, translate_dir, translate_lead

        target = self._steer(agent_pos, anchor, push_dir, lead)
        return np.clip(target, 0.0, 512.0).astype(np.float32)
