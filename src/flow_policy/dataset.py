"""PyTorch Dataset over collected PushT episodes (Milestone 1's .npz files).

Each `__getitem__` samples one episode and one random timestep within it,
returning that timestep's observation, the episode's instruction, and the
next `chunk_size` actions starting there. If the episode ends before a full
chunk is available, the last action is repeated to pad it out -- padding
with a held final action is the natural choice for chunked receding-horizon
control (the policy learns "keep doing the last thing" past episode end,
rather than being fed an arbitrary/undefined action).
"""

import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_CHUNK_SIZE = 8


class PushTChunkDataset(Dataset):
    def __init__(self, episodes_dir: pathlib.Path | str, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self.chunk_size = chunk_size
        paths = sorted(pathlib.Path(episodes_dir).glob("*.npz"))
        if not paths:
            raise ValueError(f"no episode files found in {episodes_dir}")

        self.episodes = []
        for path in paths:
            data = np.load(path, allow_pickle=True)
            self.episodes.append(
                {
                    "observations": data["observations"].astype(np.float32),  # (T, obs_dim)
                    "actions": data["actions"].astype(np.float32),  # (T, action_dim)
                    "instruction": str(data["instruction"]),
                }
            )

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict:
        episode = self.episodes[idx]
        observations = episode["observations"]
        actions = episode["actions"]
        num_steps = observations.shape[0]

        t = np.random.randint(0, num_steps)
        chunk_end = t + self.chunk_size
        chunk = actions[t:chunk_end]
        if chunk.shape[0] < self.chunk_size:
            pad_len = self.chunk_size - chunk.shape[0]
            pad = np.repeat(actions[-1:], pad_len, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)

        return {
            "observation": torch.from_numpy(observations[t]),
            "instruction": episode["instruction"],
            "action_chunk": torch.from_numpy(chunk),
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "observation": torch.stack([item["observation"] for item in batch]),
        "instruction": [item["instruction"] for item in batch],
        "action_chunk": torch.stack([item["action_chunk"] for item in batch]),
    }
