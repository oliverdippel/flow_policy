"""Milestone 7: evaluation harness.

Runs N rollouts per instruction variant with randomized initial conditions
and reports: success rate (this project's position-only metric, see
Milestone 1), time-to-success, and the instruction-confusion rate -- how
often the block ended up closer to the *other* target than the one
instructed, which is the language-conditioning failure mode this project's
two variants exist to catch (checks conditioning worked, not just task
competence).
"""

import argparse
import json
import pathlib

import torch

from flow_policy.envs import LEFT_TARGET, RIGHT_TARGET
from flow_policy.evaluation import evaluate_variant, summarize_episodes
from flow_policy.rollout import load_policy_from_checkpoint

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "policy_epoch10000.pt"
DEFAULT_OUT_PATH = REPO_ROOT / "assets" / "evaluation_results.json"
INSTRUCTIONS = [LEFT_TARGET.instruction, RIGHT_TARGET.instruction]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-path", type=pathlib.Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--num-episodes", type=int, default=25)
    parser.add_argument("--seed-start", type=int, default=50_000)
    parser.add_argument("--replan-every", type=int, default=4)
    parser.add_argument("--n-euler-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    policy = load_policy_from_checkpoint(args.checkpoint, INSTRUCTIONS)
    print(f"loaded EMA policy from {args.checkpoint}\n")

    results = {}
    for variant in (LEFT_TARGET, RIGHT_TARGET):
        episodes = evaluate_variant(
            policy,
            variant,
            num_episodes=args.num_episodes,
            seed_start=args.seed_start + variant.variant_id * 10_000,
            replan_every=args.replan_every,
            n_euler_steps=args.n_euler_steps,
        )
        summary = summarize_episodes(episodes)
        results[variant.instruction] = {"summary": summary, "episodes": episodes}

        print(f"[{variant.instruction}]  ({summary['num_episodes']} rollouts)")
        print(f"  success rate:              {summary['success_rate']:.2f}")
        print(f"  mean final dist:           {summary['mean_final_dist']:.1f}px")
        print(f"  mean env coverage:         {summary['mean_coverage']:.2f}")
        print(f"  env native success rate:   {summary['env_native_success_rate']:.2f}")
        mts = summary["mean_time_to_success"]
        print(f"  mean time-to-success:      {mts:.1f} steps" if mts is not None else "  mean time-to-success:      n/a (never reached)")
        print(f"  never reached target:      {summary['frac_never_reached_target']:.2f}")
        print(f"  instruction-confusion rate: {summary['confusion_rate']:.2f}")
        print()

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved full results to {args.out_path}")


if __name__ == "__main__":
    main()
