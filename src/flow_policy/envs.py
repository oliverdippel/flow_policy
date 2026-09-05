"""Task variants over the PushT environment.

PushT's default goal pose is hardcoded to the center of the board. To get two
geometrically distinct task variants ("push to the left target" / "push to the
right target"), we wrap the env and override `goal_pose` right after each
reset — `PushTEnv._get_coverage` / `_get_info` read `self.goal_pose` fresh on
every call, so this is sufficient to redirect reward, success, and rendering
without touching env internals.

Note this also means the raw observation (agent + block pose, 5-dim) never
reveals which target is active — the instruction is the only signal that
disambiguates the two variants, which is the point of this project.
"""

import dataclasses

import gymnasium as gym
import numpy as np


@dataclasses.dataclass(frozen=True)
class TaskVariant:
    variant_id: int
    instruction: str
    goal_pose: np.ndarray


# The T-shape's body *origin* (as reported in the state observation) sits at
# the top of the T, where the crossbar meets the stem -- not the shape's
# centroid. This is the fixed local-frame offset to the true centroid,
# matching gym_pusht.envs.pusht.PushTEnv.add_tee's
# `body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2`.
BLOCK_LOCAL_CENTROID = np.array([0.0, 45.0])

# This project's own success metric: the block's centroid within this many
# pixels of the goal position, ignoring the block's final rotation. The
# environment's native criterion (>=95% polygon-intersection coverage)
# requires matching rotation too, which needs much more sophisticated
# non-prehensile push control than this project's scope -- see
# `flow_policy.expert` for why. Since the two task variants here only differ
# by goal *position*, this metric directly measures what actually matters for
# this project: whether the instruction steered the push to the right target.
POSITION_SUCCESS_RADIUS = 30.0


def _rot_matrix(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def block_centroid_world(block_pos: np.ndarray, block_angle: float) -> np.ndarray:
    return np.asarray(block_pos, dtype=np.float64) + _rot_matrix(block_angle) @ BLOCK_LOCAL_CENTROID


LEFT_TARGET = TaskVariant(
    variant_id=0,
    instruction="push to the left target",
    goal_pose=np.array([156.0, 256.0, np.pi / 4]),
)
RIGHT_TARGET = TaskVariant(
    variant_id=1,
    instruction="push to the right target",
    goal_pose=np.array([356.0, 256.0, np.pi / 4]),
)
TASK_VARIANTS = [LEFT_TARGET, RIGHT_TARGET]
INSTRUCTION_BY_ID = {v.variant_id: v.instruction for v in TASK_VARIANTS}


class TargetPushTEnv(gym.Wrapper):
    """Wraps gym_pusht/PushT-v0 to fix the goal pose to a chosen task variant."""

    def __init__(self, env: gym.Env, variant: TaskVariant):
        super().__init__(env)
        self.variant = variant

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.env.unwrapped.goal_pose = self.variant.goal_pose.copy()
        info["goal_pose"] = self.env.unwrapped.goal_pose
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["goal_pose"] = self.env.unwrapped.goal_pose
        return obs, reward, terminated, truncated, info


def make_variant_env(variant: TaskVariant, render_mode: str = "rgb_array") -> TargetPushTEnv:
    import gym_pusht  # noqa: F401  (registers gym_pusht/PushT-v0)

    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=render_mode)
    return TargetPushTEnv(env, variant)
