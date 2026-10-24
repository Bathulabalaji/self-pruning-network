# Self-pruning network

A from-scratch NumPy reverse-mode autodiff engine and a classifier that removes its own connections while it trains. The tape, the backward pass, Adam, the pruning rule, and the training loop are in this repository. No autodiff or pruning library is used.

Python 3.10 or newer. Dependencies are installed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Commands

Gradient checks, including broadcast and fused cross-entropy:

```bash
uv run pytest tests/test_gradcheck.py
```

The masked-weight contract and the Adam revival check:

```bash
uv run pytest tests/test_masked.py
uv run pytest tests/test_adam_revival.py
```

Dense MNIST training. Writes `results/dense_seed0.json` and `results/dense_seed0.png`.

```bash
uv run python -m experiments.train_dense --seed 0
```

Self-pruning run at 90% sparsity with the Taylor score. Writes `results/prune_taylor_s0.9_seed0.json`.

```bash
uv run python -m experiments.train_prune --criterion taylor --sparsity 0.9 --seed 0
```

Sparse-forward cost on that sparsity pattern. Writes `results/cost_taylor_s0.9.json`.

```bash
uv run python -m experiments.measure_cost
```

Pareto sweep: sparsities 0, 0.50, 0.75, 0.90, 0.95, criteria Taylor and magnitude, seeds 0, 1, 2. A JSON that is already present is not trained again. Writes `results/pareto.json` and `results/pareto.png`, and prints the claim.

```bash
uv run python -m experiments.pareto
```

## Claim

Copied from `results/pareto.json`, which the sweep prints itself:

> At 90% target sparsity, Taylor pruning retains 0.9758 ± 0.0004 test accuracy versus 0.9784 ± 0.0009 for magnitude pruning, across 3 seeds. The gap is larger than the sum of the two seed standard deviations.

Magnitude is ahead. The absolute gap is about 0.26 percentage points. The unpruned mean on the same three seeds is 0.9772. Details and the FLOP table are in [DESIGN.md](DESIGN.md).

## Data

MNIST is the official idx split, 60,000 train and 10,000 test, pixels scaled to `[0, 1]`, cached at `data/mnist.npz` on first use. `train/data.py` downloads the idx files because `sklearn.datasets.fetch_openml` does not complete here: OpenML returns HTTP 412, and this machine's Python trust store rejects the TLS handshake. scikit-learn is listed as a dependency because the brief allows it for dataset loading. It is not imported. It is not used for the model, the loss, the gradients, or the optimizer.

## Layout

- `engine/` — tensors, ops, fused cross-entropy
- `nn/` — He initialization, `Linear`, `MLP`
- `optim/` — Adam with a per-connection step counter
- `prune/` — Taylor and magnitude scores, cubic schedule, sparse forward
- `train/` — data, seeding, mini-batch loop
- `experiments/` — the commands above
- `tests/` — gradcheck, mask, revival, prune, sparse match, claim formatting
- `results/` — curves, the Pareto file, and the cost measurement
