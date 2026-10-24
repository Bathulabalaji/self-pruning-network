"""Sweep sparsity, criterion, and seed, then write the Pareto claim from the files.

A run that already has a matching JSON is not repeated. Sparsity 0 still goes
through the pruning loop with a target of zero, so it is the same code path
as a positive target.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SPARSITIES = (0.0, 0.5, 0.75, 0.9, 0.95)
CRITERIA = ("taylor", "magnitude")
SEEDS = (0, 1, 2)
BATCH_FOR_FLOPS = 128


def result_path(criterion: str, sparsity: float, seed: int) -> Path:
    return RESULTS / f"prune_{criterion}_s{sparsity:g}_seed{seed}.json"


def summarize(rows: list[dict]) -> dict:
    """Aggregate finished runs. ``rows`` are the JSON reports from ``train_prune``."""
    grouped: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        key = (row["criterion"], float(row["target_sparsity"]))
        grouped.setdefault(key, []).append(row)

    points = []
    for (criterion, target), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        accuracy = np.array([float(item["test_accuracy"]) for item in group], dtype=np.float64)
        achieved = np.array([float(item["achieved_sparsity"]) for item in group], dtype=np.float64)
        nnz = np.array([float(item["nnz"]) for item in group], dtype=np.float64)
        numel = int(group[0]["numel"])
        mean_nnz = float(nnz.mean())
        points.append(
            {
                "criterion": criterion,
                "target_sparsity": target,
                "seeds": [int(item["seed"]) for item in group],
                "test_accuracy_mean": float(accuracy.mean()),
                "test_accuracy_std": float(accuracy.std(ddof=1)) if accuracy.size > 1 else 0.0,
                "test_accuracy_per_seed": [float(value) for value in accuracy],
                "achieved_sparsity_mean": float(achieved.mean()),
                "achieved_sparsity_per_seed": [float(value) for value in achieved],
                "nnz_mean": mean_nnz,
                "numel": numel,
                "sparse_flops_batch_128": int(2 * round(mean_nnz) * BATCH_FOR_FLOPS),
                "dense_flops_batch_128": int(2 * numel * BATCH_FOR_FLOPS),
            }
        )

    claim, distinguishable = _claim_at(points, target=0.9)
    return {
        "points": points,
        "claim": claim,
        "gap_exceeds_sum_of_seed_stds": distinguishable,
        "cost_note": (
            "sparse_flops_batch_128 counts 2 * nnz * 128 and is the theoretical matmul cost. "
            "results/cost_taylor_s0.9.json shows the gather kernel was slower than a dense GEMM "
            "at this width, so these FLOPs are not a measured latency win."
        ),
    }


def _claim_at(points: list[dict], target: float) -> tuple[str, bool]:
    by_name = {point["criterion"]: point for point in points if point["target_sparsity"] == target}
    taylor = by_name["taylor"]
    magnitude = by_name["magnitude"]
    n_seeds = len(taylor["seeds"])
    gap = abs(taylor["test_accuracy_mean"] - magnitude["test_accuracy_mean"])
    band = taylor["test_accuracy_std"] + magnitude["test_accuracy_std"]
    distinguishable = bool(gap > band)
    claim = (
        f"At {target:.0%} target sparsity, Taylor pruning retains "
        f"{taylor['test_accuracy_mean']:.4f} ± {taylor['test_accuracy_std']:.4f} test accuracy "
        f"versus {magnitude['test_accuracy_mean']:.4f} ± {magnitude['test_accuracy_std']:.4f} "
        f"for magnitude pruning, across {n_seeds} seeds."
    )
    if distinguishable:
        claim += " The gap is larger than the sum of the two seed standard deviations."
    else:
        claim += " The seed intervals overlap, so this gap is not distinguishable from seed noise at this sample size."
    return claim, distinguishable


def _run_one(criterion: str, sparsity: float, seed: int) -> Path:
    path = result_path(criterion, sparsity, seed)
    if path.exists():
        report = json.loads(path.read_text())
        if (
            report.get("criterion") == criterion
            and float(report.get("target_sparsity")) == sparsity
            and int(report.get("seed")) == seed
        ):
            print(f"reuse {path.name}", flush=True)
            return path
    logs = RESULTS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{path.stem}.log"
    command = [
        sys.executable,
        "-m",
        "experiments.train_prune",
        "--criterion",
        criterion,
        "--sparsity",
        str(sparsity),
        "--seed",
        str(seed),
    ]
    print(f"start {criterion} sparsity {sparsity:g} seed {seed}", flush=True)
    with log_path.open("w") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
    print(f"done {path.name}", flush=True)
    return path


def _plot(summary: dict, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    for criterion in CRITERIA:
        rows = [point for point in summary["points"] if point["criterion"] == criterion]
        (line,) = axis.plot(
            [point["achieved_sparsity_mean"] for point in rows],
            [point["test_accuracy_mean"] for point in rows],
            marker="o",
            label=f"{criterion} mean",
        )
        for point in rows:
            axis.scatter(
                point["achieved_sparsity_per_seed"],
                point["test_accuracy_per_seed"],
                s=18,
                alpha=0.75,
                color=line.get_color(),
            )
    axis.set_xlabel("achieved sparsity")
    axis.set_ylabel("test accuracy")
    axis.set_xlim(-0.02, 1.0)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparsity-accuracy sweep for both criteria.")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")

    specs = [(criterion, sparsity, seed) for seed in SEEDS for criterion in CRITERIA for sparsity in SPARSITIES]
    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(_run_one, criterion, sparsity, seed) for criterion, sparsity, seed in specs]
        for future in as_completed(futures):
            paths.append(future.result())

    rows = [json.loads(path.read_text()) for path in sorted(result_path(c, s, seed) for c, s, seed in specs)]
    summary = summarize(rows)
    summary["runs"] = [path.name for path in sorted(result_path(c, s, seed) for c, s, seed in specs)]
    out = RESULTS / "pareto.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    figure = RESULTS / "pareto.png"
    _plot(summary, figure)
    print(summary["claim"], flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
