"""Train the dense MLP. The mask stays all ones for this run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nn.mlp import MLP
from optim.adam import Adam
from train.data import load_mnist
from train.loop import evaluate, train_epoch
from train.seed import set_seed

ROOT = Path(__file__).resolve().parents[1]
SIZES = [784, 128, 64, 10]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a dense MLP on MNIST.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-size", type=int, default=60_000)
    args = parser.parse_args()

    rng = set_seed(args.seed)
    x_train, y_train, x_test, y_test = load_mnist()
    if not 1 <= args.train_size <= x_train.shape[0]:
        raise SystemExit(f"--train-size must be in 1..{x_train.shape[0]}")
    x_train = x_train[: args.train_size]
    y_train = y_train[: args.train_size]

    model = MLP(SIZES, rng)
    optimizer = Adam(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_epoch(model, optimizer, x_train, y_train, args.batch_size, rng)
        test_loss, test_accuracy = evaluate(model, x_test, y_test, args.batch_size)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}  train_loss {train_loss:.4f}  train_acc {train_accuracy:.4f}  "
            f"test_loss {test_loss:.4f}  test_acc {test_accuracy:.4f}",
            flush=True,
        )

    report = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "architecture": SIZES,
        "init": "he_normal",
        "train_size": int(x_train.shape[0]),
        "test_size": int(x_test.shape[0]),
        "history": history,
        "test_accuracy": history[-1]["test_accuracy"],
        "test_loss": history[-1]["test_loss"],
    }
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    out = results / f"dense_seed{args.seed}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    _plot(history, results / f"dense_seed{args.seed}.png")
    print(f"wrote {out}")


def _plot(history: list[dict], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.5))
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
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


if __name__ == "__main__":
    main()
