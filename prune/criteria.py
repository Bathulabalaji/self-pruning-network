"""Importance scores and a global prune that writes an exact 0/1 mask.

Removing connection ``i`` changes the loss by about ``-g_i w_i``, with ``g_i``
the pre-mask gradient. A minibatch flips the sign of that term, so the Taylor
keep-score is an exponential moving average of ``|w_i g_i|``. Magnitude uses
``|w_i|`` and ignores the gradient. Biases are not scored and not pruned.

Masked entries are not candidates. A prune only turns currently active
connections off, and it zeros the weight and the Adam state at once.
"""

from __future__ import annotations

import numpy as np

from nn.layers import Linear
from optim.adam import Adam


class Magnitude:
    name = "magnitude"

    def observe(self, layers: list[Linear]) -> None:
        del layers

    def score(self, layer: Linear) -> np.ndarray:
        return np.abs(layer.weight.data)


class TaylorSaliency:
    """EMA of ``|weight * pre-mask gradient|``. ``beta`` is the keep rate."""

    name = "taylor"

    def __init__(self, beta: float = 0.9) -> None:
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must lie in [0, 1)")
        self.beta = float(beta)
        self._scores: dict[int, np.ndarray] = {}

    def observe(self, layers: list[Linear]) -> None:
        for layer in layers:
            gradient = layer.weight.saliency_grad
            if gradient is None:
                continue
            current = self._scores.get(id(layer.weight))
            if current is None:
                current = np.zeros_like(layer.weight.data)
            signal = np.abs(layer.weight.data * gradient)
            active = layer.mask.data == 1.0
            updated = self.beta * current + (1.0 - self.beta) * signal
            self._scores[id(layer.weight)] = np.where(active, updated, 0.0)

    def score(self, layer: Linear) -> np.ndarray:
        stored = self._scores.get(id(layer.weight))
        if stored is None:
            return np.zeros_like(layer.weight.data)
        return stored


def criterion_from_name(name: str, beta: float = 0.9):
    if name == "magnitude":
        return Magnitude()
    if name == "taylor":
        return TaylorSaliency(beta=beta)
    raise ValueError(f"unknown criterion {name!r}")


def mask_sparsity(layers: list[Linear]) -> tuple[float, int, int]:
    """Return ``(1 - nnz / numel, nnz, numel)`` over weight masks. Biases are excluded."""
    nonzero = 0
    total = 0
    for layer in layers:
        total += int(layer.weight.data.size)
        nonzero += int(np.count_nonzero(layer.mask.data))
    if total == 0:
        raise ValueError("the model has no weights to score")
    return 1.0 - nonzero / total, nonzero, total


def layer_sparsity(layers: list[Linear]) -> list[dict]:
    rows = []
    for index, layer in enumerate(layers):
        total = int(layer.weight.data.size)
        nonzero = int(np.count_nonzero(layer.mask.data))
        rows.append(
            {
                "layer": index,
                "shape": list(layer.weight.shape),
                "numel": total,
                "nnz": nonzero,
                "sparsity": 1.0 - nonzero / total,
            }
        )
    return rows


def prune_to_sparsity(layers: list[Linear], score_of, target: float, optimizer: Adam) -> float:
    """Drop the lowest-scoring active weights until the global target is met.

    ``score_of(layer)`` returns a keep-score with the same shape as the weight.
    Higher means more important. Already-masked weights stay masked.
    """
    if not 0.0 <= target <= 1.0:
        raise ValueError("target sparsity must lie in [0, 1]")
    grouped = []
    for layer in layers:
        score = np.asarray(score_of(layer), dtype=np.float64)
        if score.shape != layer.weight.shape:
            raise ValueError(f"score shape {score.shape} does not match weight shape {layer.weight.shape}")
        active = np.flatnonzero(layer.mask.data.ravel() == 1.0)
        grouped.append((layer, score.ravel(), active))
    total = sum(int(layer.weight.data.size) for layer, _, _ in grouped)
    keep = int(round((1.0 - target) * total))
    active_count = sum(int(active.size) for _, _, active in grouped)
    drop = active_count - keep
    remove = np.zeros(active_count, dtype=bool)
    if drop > 0:
        scores = np.concatenate([score[active] for _, score, active in grouped])
        victims = np.argpartition(scores, drop - 1)[:drop]
        remove[victims] = True
    offset = 0
    for layer, _, active in grouped:
        count = int(active.size)
        chosen = active[remove[offset : offset + count]]
        offset += count
        if chosen.size:
            mask = layer.mask.data.copy()
            mask.ravel()[chosen] = 0.0
            layer.set_mask(mask)
        layer.weight.data *= layer.mask.data
        optimizer.constrain(layer.weight)
    achieved, _, _ = mask_sparsity(layers)
    return achieved
