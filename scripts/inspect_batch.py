"""Milestone 2: load one batch from PushTChunkDataset and sanity-check it.

Prints/asserts expected shapes, and spot-checks one raw action chunk against
its source episode file to confirm the timestep indexing has no off-by-one.
"""

import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader

from flow_policy.dataset import PushTChunkDataset, collate_fn

EPISODES_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "episodes"
BATCH_SIZE = 16
CHUNK_SIZE = 8
OBS_DIM = 5
ACTION_DIM = 2


def main():
    dataset = PushTChunkDataset(EPISODES_DIR, chunk_size=CHUNK_SIZE)
    print(f"dataset has {len(dataset)} episodes")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))

    obs = batch["observation"]
    instructions = batch["instruction"]
    action_chunk = batch["action_chunk"]

    print(f"observation: shape={tuple(obs.shape)} dtype={obs.dtype}")
    print(f"instruction: {type(instructions).__name__} of {len(instructions)} strings, e.g. {instructions[0]!r}")
    print(f"action_chunk: shape={tuple(action_chunk.shape)} dtype={action_chunk.dtype}")

    assert obs.shape == (BATCH_SIZE, OBS_DIM), obs.shape
    assert isinstance(instructions, list) and len(instructions) == BATCH_SIZE
    assert all(isinstance(s, str) for s in instructions)
    assert action_chunk.shape == (BATCH_SIZE, CHUNK_SIZE, ACTION_DIM), action_chunk.shape
    print("shape assertions passed.")

    # Spot-check indexing: re-fetch one raw episode from disk and manually find a
    # timestep whose observation matches the batch's, then confirm the following
    # actions line up exactly with the sampled action_chunk (no off-by-one).
    ep_paths = sorted(EPISODES_DIR.glob("*.npz"))
    sample_obs = obs[0].numpy()
    sample_chunk = action_chunk[0].numpy()
    match = None
    for path in ep_paths:
        data = np.load(path, allow_pickle=True)
        raw_obs = data["observations"]
        hits = np.where(np.all(np.isclose(raw_obs, sample_obs), axis=1))[0]
        if len(hits) > 0:
            match = (path, data, int(hits[0]))
            break

    assert match is not None, "couldn't find the sampled observation in any source episode"
    path, data, t = match
    raw_actions = data["actions"]
    expected_chunk = raw_actions[t : t + CHUNK_SIZE]
    if expected_chunk.shape[0] < CHUNK_SIZE:
        pad_len = CHUNK_SIZE - expected_chunk.shape[0]
        expected_chunk = np.concatenate([expected_chunk, np.repeat(raw_actions[-1:], pad_len, axis=0)], axis=0)

    print(f"\nspot-check: batch obs matched {path.name} at timestep t={t}")
    print(f"  sampled action_chunk[0]:  {sample_chunk[0]}")
    print(f"  raw episode action[t]:    {raw_actions[t]}")
    print(f"  sampled action_chunk[-1]: {sample_chunk[-1]}")
    print(f"  expected padded chunk[-1]:{expected_chunk[-1]}")
    assert np.allclose(sample_chunk, expected_chunk), "action chunk does not match source episode -- off-by-one?"
    print("spot-check passed: action_chunk exactly matches actions[t : t+H] from the source episode.")


if __name__ == "__main__":
    main()
