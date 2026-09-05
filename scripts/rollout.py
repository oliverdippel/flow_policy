"""Milestone 6: closed-loop rollout of the trained policy, both instructions.

Runs the EMA-weights policy end-to-end in the environment via receding-
horizon control (replan every 4 steps from an 8-step predicted chunk), for
both task variants, saving a GIF per variant and logging whether each run
reached its instructed target.
"""

import argparse
import pathlib

import imageio.v2 as imageio
import torch

from flow_policy.envs import LEFT_TARGET, POSITION_SUCCESS_RADIUS, RIGHT_TARGET
from flow_policy.rollout import load_policy_from_checkpoint, run_rollout

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "policy_epoch10000.pt"
DEFAULT_OUT_DIR = REPO_ROOT / "assets"
INSTRUCTIONS = [LEFT_TARGET.instruction, RIGHT_TARGET.instruction]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replan-every", type=int, default=4)
    parser.add_argument("--n-euler-steps", type=int, default=10)
    parser.add_argument("--temporal-ensemble", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)  # Euler sampling draws x0 ~ N(0,I) fresh each replan; fix for reproducible results
    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy = load_policy_from_checkpoint(args.checkpoint, INSTRUCTIONS)
    print(f"loaded EMA policy from {args.checkpoint}")

    for variant in (LEFT_TARGET, RIGHT_TARGET):
        result = run_rollout(
            policy,
            variant,
            seed=args.seed,
            replan_every=args.replan_every,
            n_euler_steps=args.n_euler_steps,
            temporal_ensemble=args.temporal_ensemble,
        )

        safe_name = variant.instruction.replace(" ", "_")
        gif_path = args.out_dir / f"rollout_{safe_name}.gif"
        imageio.mimsave(gif_path, result.frames, fps=20)

        print(
            f"[{variant.instruction}] steps={result.num_steps} "
            f"position_success={result.position_success} (dist={result.final_position_dist:.1f}px, "
            f"radius={POSITION_SUCCESS_RADIUS}px) env_coverage={result.final_coverage:.2f} "
            f"env_is_success={result.env_is_success}"
        )
        print(f"  saved {gif_path}")


if __name__ == "__main__":
    main()
