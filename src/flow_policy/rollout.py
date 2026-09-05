"""Closed-loop inference: ODE sampling + receding-horizon action chunking.

Each planning call predicts a full `chunk_size`-step action chunk via the
flow-matching head's Euler sampler, but only the first `replan_every` actions
are executed before re-observing and re-predicting -- committing to the full
chunk blindly would ignore how much the world can drift from the plan over a
whole chunk's horizon.

Optional temporal ensembling: when replanning, several recently-generated
chunks still overlap (chunk generated `replan_every` steps ago also covers
the steps the newest chunk covers, just at a larger in-chunk offset). Blending
the overlapping predictions -- weighting a prediction down the more steps old
its plan is -- trades a bit of responsiveness for smoother actions.
"""

import dataclasses

import numpy as np
import torch

from flow_policy.envs import POSITION_SUCCESS_RADIUS, TaskVariant, block_centroid_world, make_variant_env
from flow_policy.policy import FlowMatchingPolicy


@dataclasses.dataclass
class RolloutResult:
    frames: list[np.ndarray] | None
    num_steps: int
    final_position_dist: float
    final_coverage: float
    env_is_success: bool
    position_success: bool
    final_block_centroid: np.ndarray
    first_success_step: int | None  # first t at which the block was within the success radius, if ever


class RecedingHorizonController:
    def __init__(
        self,
        policy: FlowMatchingPolicy,
        replan_every: int = 4,
        n_euler_steps: int = 10,
        temporal_ensemble: bool = False,
        ensemble_decay: float = 0.5,
    ):
        self.policy = policy
        self.replan_every = replan_every
        self.n_euler_steps = n_euler_steps
        self.temporal_ensemble = temporal_ensemble
        self.ensemble_decay = ensemble_decay
        self._chunk_history: list[tuple[int, np.ndarray]] = []  # (start_t, chunk) pairs

    def reset(self) -> None:
        self._chunk_history.clear()

    def _plan(self, obs: np.ndarray, instruction: str, t: int) -> None:
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        chunk = self.policy.sample_action_chunk(obs_tensor, [instruction], n_steps=self.n_euler_steps)
        chunk = chunk[0].numpy()  # (H, action_dim)
        self._chunk_history.append((t, chunk))
        max_age_chunks = -(-self.policy.chunk_size // self.replan_every)  # ceil(H / replan_every)
        self._chunk_history = self._chunk_history[-max_age_chunks:]

    def act(self, obs: np.ndarray, instruction: str, t: int) -> np.ndarray:
        if t % self.replan_every == 0:
            self._plan(obs, instruction, t)

        if not self.temporal_ensemble:
            start_t, chunk = self._chunk_history[-1]
            return chunk[t - start_t]

        weighted_sum = None
        weight_total = 0.0
        for start_t, chunk in self._chunk_history:
            offset = t - start_t
            if 0 <= offset < chunk.shape[0]:
                weight = float(np.exp(-self.ensemble_decay * offset))
                weighted_sum = chunk[offset] * weight if weighted_sum is None else weighted_sum + chunk[offset] * weight
                weight_total += weight
        return weighted_sum / weight_total


def run_rollout(
    policy: FlowMatchingPolicy,
    variant: TaskVariant,
    seed: int,
    max_steps: int = 300,
    replan_every: int = 4,
    n_euler_steps: int = 10,
    temporal_ensemble: bool = False,
    position_success_radius: float = POSITION_SUCCESS_RADIUS,
    render: bool = True,
) -> RolloutResult:
    env = make_variant_env(variant)
    obs, info = env.reset(seed=seed)
    controller = RecedingHorizonController(policy, replan_every=replan_every, n_euler_steps=n_euler_steps,
                                            temporal_ensemble=temporal_ensemble)
    controller.reset()

    frames = [env.render()] if render else None
    first_success_step = None
    for t in range(max_steps):
        action = controller.act(obs, variant.instruction, t)
        obs, reward, terminated, truncated, info = env.step(action)
        if render:
            frames.append(env.render())

        centroid = block_centroid_world(obs[2:4], obs[4])
        dist = float(np.linalg.norm(centroid - variant.goal_pose[:2]))
        if first_success_step is None and dist <= position_success_radius:
            first_success_step = t

        if terminated or truncated:
            break
    env.close()

    return RolloutResult(
        frames=frames,
        num_steps=t + 1,
        final_position_dist=dist,
        final_coverage=float(info["coverage"]),
        env_is_success=bool(info["is_success"]),
        position_success=dist <= position_success_radius,
        final_block_centroid=centroid,
        first_success_step=first_success_step,
    )


def load_policy_from_checkpoint(checkpoint_path, instructions: list[str]) -> FlowMatchingPolicy:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy = FlowMatchingPolicy(instructions, chunk_size=checkpoint["chunk_size"])
    policy.load_state_dict(checkpoint["ema_state_dict"])
    policy.eval()
    return policy
