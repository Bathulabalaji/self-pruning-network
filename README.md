# Self-pruning network

A from-scratch NumPy reverse-mode autodiff engine and a classifier that removes its own connections during training. scikit-learn is used only to download MNIST. Training, gradients, and pruning are not delegated to a library.

Run instructions for dense training, pruning, and the Pareto sweep are added with those milestones.

```bash
uv sync
uv run pytest tests/test_gradcheck.py
uv run pytest tests/test_masked.py
```
