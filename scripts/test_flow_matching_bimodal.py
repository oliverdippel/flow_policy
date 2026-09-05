"""Milestone 4: verify the flow-matching head on a synthetic bimodal target.

x1 is either +2 or -2 with equal probability, no conditioning. This is the
one case a plain MSE-regression action head provably fails on -- it would
regress to the mean (0), which isn't even a valid sample. Flow matching's
whole point is to reproduce the bimodal distribution instead, so this must be
verified in isolation before touching real (and much messier) robot data.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from flow_policy.flow_matching import FlowMatchingHead

MODES = (-2.0, 2.0)
NUM_TRAIN_STEPS = 3000
BATCH_SIZE = 256
NUM_SAMPLES = 2000
N_EULER_STEPS = 10
OUT_PATH = "assets/flow_matching_bimodal_test.png"

# A sample "near" a mode vs. collapsed to the mean (0): half the mode spacing.
NEAR_MODE_THRESH = (MODES[1] - MODES[0]) / 4  # = 1.0
COLLAPSED_THRESH = 0.5


def sample_target(batch_size: int) -> torch.Tensor:
    signs = torch.randint(0, 2, (batch_size, 1), dtype=torch.float32) * 2 - 1  # +/-1
    return signs * 2.0


def main():
    torch.manual_seed(0)
    head = FlowMatchingHead(x_dim=1, cond_dim=0, hidden_dim=64, time_embed_dim=16, num_hidden_layers=3)
    optimizer = torch.optim.Adam(head.parameters(), lr=2e-3)

    for step in range(NUM_TRAIN_STEPS):
        x1 = sample_target(BATCH_SIZE)
        loss = head.training_loss(x1, cond=None)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0 or step == NUM_TRAIN_STEPS - 1:
            print(f"step {step:5d}  loss {loss.item():.4f}")

    head.eval()
    samples = head.sample(cond=None, n_steps=N_EULER_STEPS, batch_size=NUM_SAMPLES).squeeze(-1)

    frac_near_pos = (samples > NEAR_MODE_THRESH).float().mean().item()
    frac_near_neg = (samples < -NEAR_MODE_THRESH).float().mean().item()
    frac_collapsed = (samples.abs() < COLLAPSED_THRESH).float().mean().item()

    print(f"\nfraction of samples near +{MODES[1]}: {frac_near_pos:.3f}")
    print(f"fraction of samples near {MODES[0]}: {frac_near_neg:.3f}")
    print(f"fraction collapsed near the mean (0): {frac_collapsed:.3f}")

    plt.figure(figsize=(6, 4))
    plt.hist(samples.numpy(), bins=60, range=(-4, 4))
    plt.axvline(MODES[0], color="green", linestyle="--", label="true modes")
    plt.axvline(MODES[1], color="green", linestyle="--")
    plt.axvline(0.0, color="red", linestyle=":", label="collapsed-to-mean failure mode")
    plt.title("Flow-matching samples on a synthetic bimodal target")
    plt.xlabel("x1 sample")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=120)
    print(f"saved histogram to {OUT_PATH}")

    assert frac_near_pos > 0.35, f"too few samples near +{MODES[1]}: {frac_near_pos:.3f}"
    assert frac_near_neg > 0.35, f"too few samples near {MODES[0]}: {frac_near_neg:.3f}"
    assert frac_collapsed < 0.15, f"samples collapsed toward the mean: {frac_collapsed:.3f}"
    print("\nPASS: samples are bimodal, not collapsed to the mean.")


if __name__ == "__main__":
    main()
