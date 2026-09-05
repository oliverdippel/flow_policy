"""Milestone 5: train the flow-matching policy on the real collected episodes.

Wires together the dataset (M2), conditioning encoder (M3), and flow-matching
head (M4): each batch computes `cond` from (observation, instruction),
samples t and x0, computes the flow-matching loss, and backprops. Tracks an
EMA copy of the weights for use at inference (M6).
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from flow_policy.dataset import PushTChunkDataset, collate_fn
from flow_policy.envs import LEFT_TARGET, RIGHT_TARGET
from flow_policy.policy import EMA, FlowMatchingPolicy

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_EPISODES_DIR = REPO_ROOT / "data" / "episodes"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
DEFAULT_LOSS_PLOT_PATH = REPO_ROOT / "assets" / "training_loss.png"

INSTRUCTIONS = [LEFT_TARGET.instruction, RIGHT_TARGET.instruction]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-dir", type=pathlib.Path, default=DEFAULT_EPISODES_DIR)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--loss-plot-path", type=pathlib.Path, default=DEFAULT_LOSS_PLOT_PATH)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.loss_plot_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = PushTChunkDataset(args.episodes_dir, chunk_size=args.chunk_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    print(f"dataset: {len(dataset)} episodes, {len(loader)} batches/epoch, batch_size={args.batch_size}")

    policy = FlowMatchingPolicy(INSTRUCTIONS, chunk_size=args.chunk_size)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    ema = EMA(policy, decay=0.999)

    epoch_losses = []
    for epoch in range(args.epochs):
        batch_losses = []
        for batch in loader:
            loss = policy.compute_loss(batch["observation"], batch["instruction"], batch["action_chunk"])

            assert torch.isfinite(loss), f"non-finite loss at epoch {epoch}: {loss.item()}"

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(policy)

            batch_losses.append(loss.item())

        epoch_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(epoch_loss)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d}  loss {epoch_loss:.4f}")

        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            ckpt_path = args.checkpoint_dir / f"policy_epoch{epoch + 1:04d}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": policy.state_dict(),
                    "ema_state_dict": ema.shadow,
                    "chunk_size": args.chunk_size,
                },
                ckpt_path,
            )
            print(f"  saved checkpoint to {ckpt_path}")

    plt.figure(figsize=(6, 4))
    plt.plot(epoch_losses)
    plt.xlabel("epoch")
    plt.ylabel("flow-matching loss (mean over epoch)")
    plt.title("Training loss")
    plt.tight_layout()
    plt.savefig(args.loss_plot_path, dpi=120)
    print(f"\nsaved loss curve to {args.loss_plot_path}")


if __name__ == "__main__":
    main()
