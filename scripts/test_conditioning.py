"""Milestone 3: unit test for the conditioning encoder.

Checks the `cond` vector's shape, that swapping the instruction measurably
changes it (conditioning actually has an effect), and that an identical
instruction always yields the same cached CLIP embedding (deterministic,
no recompute).
"""

import torch

from flow_policy.conditioning import ConditioningEncoder, InstructionEncoder
from flow_policy.envs import LEFT_TARGET, RIGHT_TARGET

INSTRUCTIONS = [LEFT_TARGET.instruction, RIGHT_TARGET.instruction]
BATCH_SIZE = 8
OBS_DIM = 5
COND_DIM = 128
EPSILON = 1e-3


def main():
    torch.manual_seed(0)

    # --- raw instruction encoder: cached, deterministic, and instruction-sensitive ---
    instruction_encoder = InstructionEncoder(INSTRUCTIONS)
    left_embed = instruction_encoder([LEFT_TARGET.instruction])[0]
    left_embed_again = instruction_encoder([LEFT_TARGET.instruction])[0]
    right_embed = instruction_encoder([RIGHT_TARGET.instruction])[0]

    assert torch.equal(left_embed, left_embed_again), "same instruction should give the exact same cached embedding"
    text_dist = torch.linalg.norm(left_embed - right_embed).item()
    print(f"CLIP text embedding L2 distance (left vs right): {text_dist:.3f}")
    assert text_dist > EPSILON, "different instructions should give measurably different CLIP embeddings"

    # --- full conditioning encoder: shape + instruction sensitivity end-to-end ---
    encoder = ConditioningEncoder(INSTRUCTIONS, obs_dim=OBS_DIM, cond_dim=COND_DIM)
    encoder.eval()

    obs = torch.randn(BATCH_SIZE, OBS_DIM)
    with torch.no_grad():
        cond_left = encoder(obs, [LEFT_TARGET.instruction] * BATCH_SIZE)
        cond_right = encoder(obs, [RIGHT_TARGET.instruction] * BATCH_SIZE)
        cond_left_again = encoder(obs, [LEFT_TARGET.instruction] * BATCH_SIZE)

    print(f"cond shape: {tuple(cond_left.shape)}")
    assert cond_left.shape == (BATCH_SIZE, COND_DIM), cond_left.shape

    assert torch.equal(cond_left, cond_left_again), "same (obs, instruction) should give the exact same cond vector"

    cond_dist = torch.linalg.norm(cond_left - cond_right, dim=-1)
    print(f"cond L2 distance (left vs right instruction, same obs): {cond_dist.tolist()}")
    assert (cond_dist > EPSILON).all(), "swapping the instruction should measurably change cond"

    print("all conditioning encoder assertions passed.")


if __name__ == "__main__":
    main()
