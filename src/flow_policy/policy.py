"""Ties the conditioning encoder (Milestone 3) and flow-matching head
(Milestone 4) together into the trainable policy, and an EMA helper for
tracking a smoothed copy of its weights during training."""

import copy

import torch
import torch.nn as nn

from flow_policy.conditioning import ConditioningEncoder
from flow_policy.flow_matching import FlowMatchingHead

# PushT's action space is Box(0, 512) (a target x/y position on the board).
# Flow matching interpolates between actions and N(0, I) noise, so unnormalized
# actions blow up the regression target's scale by ~2 orders of magnitude
# (e.g. loss ~80000 instead of ~1) -- rescale to [-1, 1] using the env's known,
# fixed bounds so the network trains on a sane scale.
ACTION_LOW = 0.0
ACTION_HIGH = 512.0


def normalize_action(action: torch.Tensor) -> torch.Tensor:
    mid = (ACTION_HIGH + ACTION_LOW) / 2.0
    half_range = (ACTION_HIGH - ACTION_LOW) / 2.0
    return (action - mid) / half_range


def denormalize_action(action_norm: torch.Tensor) -> torch.Tensor:
    mid = (ACTION_HIGH + ACTION_LOW) / 2.0
    half_range = (ACTION_HIGH - ACTION_LOW) / 2.0
    return action_norm * half_range + mid


class FlowMatchingPolicy(nn.Module):
    def __init__(
        self,
        instructions: list[str],
        obs_dim: int = 5,
        action_dim: int = 2,
        chunk_size: int = 8,
        cond_dim: int = 128,
        **flow_head_kwargs,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.conditioning_encoder = ConditioningEncoder(instructions, obs_dim=obs_dim, cond_dim=cond_dim)
        self.flow_head = FlowMatchingHead(x_dim=chunk_size * action_dim, cond_dim=cond_dim, **flow_head_kwargs)

    def compute_loss(self, obs: torch.Tensor, instructions: list[str], action_chunk: torch.Tensor) -> torch.Tensor:
        cond = self.conditioning_encoder(obs, instructions)
        x1 = normalize_action(action_chunk).reshape(action_chunk.shape[0], -1)
        return self.flow_head.training_loss(x1, cond=cond)

    @torch.no_grad()
    def sample_action_chunk(self, obs: torch.Tensor, instructions: list[str], n_steps: int = 10) -> torch.Tensor:
        cond = self.conditioning_encoder(obs, instructions)
        chunk_norm = self.flow_head.sample_action_chunk(
            cond, chunk_shape=(self.chunk_size, self.action_dim), n_steps=n_steps
        )
        return denormalize_action(chunk_norm)


class EMA:
    """Exponential moving average of a model's weights, for use at inference
    (standard practice for diffusion/flow models: the raw weights are noisier
    than their running average)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            shadow_value = self.shadow[name]
            if value.dtype.is_floating_point:
                shadow_value.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                shadow_value.copy_(value)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)
