"""Conditional flow-matching action head (linear / optimal-transport path).

Following the standard flow-matching formulation (Lipman et al.; used in pi0's
action expert): let `x1` be the ground-truth action chunk (flattened to a
vector of dim `H * action_dim`) and `x0 ~ N(0, I)` a noise sample of the same
dimension. Sample `t ~ Uniform(0, 1)` per example, define the interpolated
point `xt = (1 - t) * x0 + t * x1`, and regress a network `v_theta(xt, t,
cond)` toward the constant target velocity `v_target = x1 - x0` via MSE. At
inference, integrate the learned ODE `dx/dt = v_theta(x, t, cond)` forward
from `t=0` (pure noise) to `t=1` with simple Euler steps.

`cond` is optional (pass `cond_dim=0` and `cond=None` throughout) so this
module can be verified standalone on a synthetic target before it's wired to
the real conditioning encoder -- see `scripts/test_flow_matching_bimodal.py`.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: (B,) in [0, 1] -> (B, dim). Standard transformer-style embedding;
    t is scaled up first since sinusoids of a [0,1]-ranged input would
    otherwise barely vary across most frequency bands."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
    args = (t.unsqueeze(-1) * 1000.0) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class VectorFieldMLP(nn.Module):
    """v_theta(x, t, cond) -> predicted velocity, same shape as x."""

    def __init__(
        self,
        x_dim: int,
        cond_dim: int = 0,
        time_embed_dim: int = 32,
        hidden_dim: int = 256,
        num_hidden_layers: int = 3,
    ):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        input_dim = x_dim + time_embed_dim + cond_dim

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(prev_dim, hidden_dim), nn.SiLU()]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, x_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        t_embed = sinusoidal_time_embedding(t, self.time_embed_dim)
        parts = [x, t_embed] if cond is None else [x, t_embed, cond]
        return self.net(torch.cat(parts, dim=-1))


class FlowMatchingHead(nn.Module):
    def __init__(self, x_dim: int, cond_dim: int = 0, **velocity_net_kwargs):
        super().__init__()
        self.x_dim = x_dim
        self.velocity_net = VectorFieldMLP(x_dim, cond_dim=cond_dim, **velocity_net_kwargs)

    def training_loss(self, x1: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x1.shape[0]
        x0 = torch.randn_like(x1)
        t = torch.rand(batch_size, device=x1.device, dtype=x1.dtype)
        xt = (1.0 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
        v_target = x1 - x0
        v_pred = self.velocity_net(xt, t, cond)
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor | None = None,
        n_steps: int = 10,
        batch_size: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if batch_size is None:
            batch_size = cond.shape[0] if cond is not None else 1
        if device is None:
            device = cond.device if cond is not None else next(self.parameters()).device

        x = torch.randn(batch_size, self.x_dim, device=device)
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = torch.full((batch_size,), step * dt, device=device)
            x = x + dt * self.velocity_net(x, t, cond)
        return x

    def sample_action_chunk(
        self,
        cond: torch.Tensor | None,
        chunk_shape: tuple[int, int],
        n_steps: int = 10,
    ) -> torch.Tensor:
        """Convenience wrapper: sample() and reshape the flat vector to (B, H, action_dim)."""
        batch_size = cond.shape[0] if cond is not None else 1
        flat = self.sample(cond=cond, n_steps=n_steps, batch_size=batch_size)
        horizon, action_dim = chunk_shape
        return flat.view(batch_size, horizon, action_dim)
