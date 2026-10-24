"""Mini-batch training and evaluation on the autodiff tape."""

from __future__ import annotations

import numpy as np

from engine.loss import cross_entropy
from engine.tensor import Tensor
from nn.mlp import MLP
from optim.adam import Adam


def iter_batches(images: np.ndarray, labels: np.ndarray, batch_size: int, rng: np.random.Generator | None):
    count = images.shape[0]
    order = np.arange(count) if rng is None else rng.permutation(count)
    for start in range(0, count, batch_size):
        index = order[start : start + batch_size]
        yield images[index], labels[index]


def train_epoch(
    model: MLP,
    optimizer: Adam,
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    total_loss = 0.0
    correct = 0
    seen = 0
    for batch_images, batch_labels in iter_batches(images, labels, batch_size, rng):
        model.zero_grad()
        logits = model(Tensor(batch_images))
        loss = cross_entropy(logits, batch_labels)
        if not np.isfinite(loss.data):
            raise RuntimeError("training loss is not finite")
        loss.backward()
        optimizer.step()
        batch_count = batch_labels.shape[0]
        total_loss += float(loss.data) * batch_count
        correct += int(np.sum(np.argmax(logits.data, axis=1) == batch_labels))
        seen += batch_count
    return total_loss / seen, correct / seen


def evaluate(model: MLP, images: np.ndarray, labels: np.ndarray, batch_size: int) -> tuple[float, float]:
    total_loss = 0.0
    correct = 0
    seen = 0
    for batch_images, batch_labels in iter_batches(images, labels, batch_size, None):
        logits = model(Tensor(batch_images))
        loss = cross_entropy(logits, batch_labels)
        if not np.isfinite(loss.data):
            raise RuntimeError("evaluation loss is not finite")
        batch_count = batch_labels.shape[0]
        total_loss += float(loss.data) * batch_count
        correct += int(np.sum(np.argmax(logits.data, axis=1) == batch_labels))
        seen += batch_count
    return total_loss / seen, correct / seen
