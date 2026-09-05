"""Milestone 0 smoke test: confirm gym-pusht reset/step/render work end-to-end.

Runs a random-action rollout and saves a GIF so the environment is verified
before any modeling starts.
"""

import pathlib

import gymnasium as gym
import gym_pusht  # noqa: F401  (registers gym_pusht/PushT-v0)
import imageio.v2 as imageio
import numpy as np

OUT_PATH = pathlib.Path(__file__).resolve().parent.parent / "assets" / "smoke_test_random_rollout.gif"
NUM_STEPS = 100
SEED = 0


def main() -> None:
    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
    obs, info = env.reset(seed=SEED)
    print(f"reset ok. obs shape={obs.shape} dtype={obs.dtype}")
    print(f"action_space={env.action_space} observation_space={env.observation_space}")

    frames = [env.render()]
    rng = np.random.default_rng(SEED)
    total_reward = 0.0
    for t in range(NUM_STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        frames.append(env.render())
        if terminated or truncated:
            print(f"episode ended at step {t} (terminated={terminated}, truncated={truncated})")
            break

    env.close()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(OUT_PATH, frames, fps=20)

    print(f"collected {len(frames)} frames, total_reward={total_reward:.3f}")
    print(f"saved rollout GIF to {OUT_PATH}")


if __name__ == "__main__":
    main()
