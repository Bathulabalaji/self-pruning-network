# Self-pruning network

A from-scratch NumPy reverse-mode autodiff engine and a classifier that removes its own connections during training. Training, gradients, and pruning are not delegated to a library.

MNIST is the official idx split cached by `train/data.py`. scikit-learn is not used for training, models, or gradients.

```bash
uv sync
uv run pytest tests/test_gradcheck.py
uv run pytest tests/test_masked.py
uv run pytest tests/test_adam_revival.py
uv run python -m experiments.train_dense --seed 0
uv run python -m experiments.train_prune --criterion taylor --sparsity 0.9 --seed 0
uv run python -m experiments.measure_cost
uv run python -m experiments.pareto
```
