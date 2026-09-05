"""Vision (well, low-dim state) + language conditioning encoder.

Two pieces feed a single `cond` vector: a small MLP over the low-dim PushT
state observation, and a frozen CLIP text encoder over the instruction.
Combined by concatenation + a linear projection (simpler than FiLM, and
sufficient here -- FiLM would matter more with a higher-capacity vision
backbone than a 5-dim state vector).

Only the instructions attached to `flow_policy.envs.TASK_VARIANTS` ever appear
in this project, so the text encoder embeds them once at construction time and
caches the result as a buffer -- there is no CLIP forward pass at training or
inference time, matching the milestone's "don't recompute every batch"
requirement. Passing an instruction outside that fixed set is a bug, not a
supported use case, so it raises `KeyError` rather than silently falling back
to a live CLIP call.
"""

import open_clip
import torch
import torch.nn as nn

TEXT_MODEL_NAME = "ViT-B-32-quickgelu"
TEXT_PRETRAINED = "openai"


class InstructionEncoder(nn.Module):
    """Frozen, pre-cached CLIP text embeddings for a fixed set of instructions."""

    def __init__(self, instructions: list[str], model_name: str = TEXT_MODEL_NAME, pretrained: str = TEXT_PRETRAINED):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
        model.eval()
        with torch.no_grad():
            tokens = tokenizer(list(instructions))
            embeddings = model.encode_text(tokens).float()
        del model  # only needed to build the cache -- the instruction set is fixed

        self.embed_dim = embeddings.shape[1]
        self._instruction_to_idx = {s: i for i, s in enumerate(instructions)}
        self.register_buffer("cache", embeddings)  # (num_instructions, embed_dim)

    def forward(self, instructions: list[str]) -> torch.Tensor:
        idxs = [self._instruction_to_idx[s] for s in instructions]
        return self.cache[idxs]


class ObservationEncoder(nn.Module):
    """Small MLP over the low-dim PushT state observation."""

    def __init__(self, obs_dim: int = 5, hidden_dim: int = 64, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ConditioningEncoder(nn.Module):
    """Combines the observation embedding and instruction embedding into `cond`."""

    def __init__(
        self,
        instructions: list[str],
        obs_dim: int = 5,
        obs_hidden_dim: int = 64,
        obs_embed_dim: int = 64,
        cond_dim: int = 128,
    ):
        super().__init__()
        self.obs_encoder = ObservationEncoder(obs_dim, obs_hidden_dim, obs_embed_dim)
        self.instruction_encoder = InstructionEncoder(instructions)
        combined_dim = obs_embed_dim + self.instruction_encoder.embed_dim
        self.proj = nn.Sequential(nn.Linear(combined_dim, cond_dim), nn.ReLU())
        self.cond_dim = cond_dim

    def forward(self, obs: torch.Tensor, instructions: list[str]) -> torch.Tensor:
        obs_embed = self.obs_encoder(obs)
        instr_embed = self.instruction_encoder(instructions)
        return self.proj(torch.cat([obs_embed, instr_embed], dim=-1))
