"""Train the MLP while a cubic schedule removes connections.

Pruning starts after epoch 1, runs every 100 steps until 75% of training, then
the mask is frozen and the remaining steps only fine-tune. Sparsity is global
over weights. Biases stay dense.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nn.mlp import MLP
from optim.adam import Adam
from prune.criteria import criterion_from_name, layer_sparsity, mask_sparsity, prune_to_sparsity
from prune.schedule import sparsity_at
from train.data import load_mnist
from train.loop import StepCounter, evaluate, train_epoch
from train.seed import set_seed

ROOT = Path(__file__).resolve().parents[1]
SIZES = [784, 128, 64, 10]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an MLP that prunes itself.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-size", type=int, default=60_000)
    parser.add_argument("--sparsity", type=float, default=0.9)
    parser.add_argument("--criterion", choices=("taylor", "magnitude"), default="taylor")
    parser.add_argument("--prune-every", type=int, default=100)
    parser.add_argument("--ema-beta", type=float, default=0.9)
    args = parser.parse_args()
    if args.prune_every < 1:
        raise SystemExit("--prune-every must be positive")
    if not 0.0 <= args.sparsity < 1.0:
        raise SystemExit("--sparsity must lie in [0, 1)")

    rng = set_seed(args.seed)
    x_train, y_train, x_test, y_test = load_mnist()
    if not 1 <= args.train_size <= x_train.shape[0]:
        raise SystemExit(f"--train-size must be in 1..{x_train.shape[0]}")
    x_train = x_train[: args.train_size]
    y_train = y_train[: args.train_size]

    steps_per_epoch = math.ceil(x_train.shape[0] / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    start = steps_per_epoch
    end = int(0.75 * total_steps)
    if end <= start:
        raise SystemExit("the prune window is empty; train for more epochs")

    model = MLP(SIZES, rng)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = criterion_from_name(args.criterion, beta=args.ema_beta)
    clock = StepCounter()
    events = []

    def on_backward(step: int) -> None:
        del step
        criterion.observe(model.layers)

    def on_step(step: int) -> None:
        if step < start or step > end:
            return
        if step % args.prune_every != 0 and step != end:
            return
        target = sparsity_at(step, initial=0.0, final=args.sparsity, start=start, end=end)
        achieved = prune_to_sparsity(model.layers, criterion.score, target, optimizer)
        events.append({"step": step, "target_sparsity": target, "achieved_sparsity": achieved})

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model,
            optimizer,
            x_train,
            y_train,
            args.batch_size,
            rng,
            clock=clock,
            on_backward=on_backward,
            on_step=on_step,
        )
        test_loss, test_accuracy = evaluate(model, x_test, y_test, args.batch_size)
        achieved, nonzero, total = mask_sparsity(model.layers)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "sparsity": achieved,
            "nnz": nonzero,
            "numel": total,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}  train_loss {train_loss:.4f}  train_acc {train_accuracy:.4f}  "
            f"test_loss {test_loss:.4f}  test_acc {test_accuracy:.4f}  sparsity {achieved:.4f}",
            flush=True,
        )

    achieved, nonzero, total = mask_sparsity(model.layers)
    exact_zeros = all(np.all(layer.weight.data[layer.mask.data == 0.0] == 0.0) for layer in model.layers)
    report = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "architecture": SIZES,
        "init": "he_normal",
        "criterion": args.criterion,
        "ema_beta": args.ema_beta if args.criterion == "taylor" else None,
        "target_sparsity": args.sparsity,
        "achieved_sparsity": achieved,
        "nnz": nonzero,
        "numel": total,
        "weights_exactly_zero_where_masked": exact_zeros,
        "prune_every": args.prune_every,
        "prune_start_step": start,
        "prune_end_step": end,
        "total_steps": total_steps,
        "train_size": int(x_train.shape[0]),
        "test_size": int(x_test.shape[0]),
        "layers": layer_sparsity(model.layers),
        "prune_events": events,
        "history": history,
        "test_accuracy": history[-1]["test_accuracy"],
        "test_loss": history[-1]["test_loss"],
    }
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    tag = f"prune_{args.criterion}_s{args.sparsity:g}_seed{args.seed}"
    out = results / f"{tag}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    _plot(history, results / f"{tag}.png")
    print(f"wrote {out}")
    if abs(achieved - args.sparsity) > 0.005:
        raise SystemExit(f"achieved sparsity {achieved:.6f} is more than 0.5 points from {args.sparsity}")


def _plot(history: list[dict], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["test_loss"] for row in history], label="test")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="train")
    axes[1].plot(epochs, [row["test_accuracy"] for row in history], label="test")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[2].plot(epochs, [row["sparsity"] for row in history])
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("sparsity")
    axes[2].set_ylim(0.0, 1.0)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


if __name__ == "__main__":
    main()
