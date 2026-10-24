"""One dense step on a tiny set must be finite and must reduce the loss."""

import numpy as np

from nn.mlp import MLP
from optim.adam import Adam
from train.loop import evaluate, train_epoch
from train.seed import set_seed


def test_one_epoch_reduces_loss_without_nans():
    rng = set_seed(0)
    images = rng.normal(size=(64, 8))
    labels = rng.integers(0, 3, size=(64,))
    model = MLP([8, 16, 3], rng)
    before, _ = evaluate(model, images, labels, batch_size=32)
    optimizer = Adam(model.parameters(), lr=1e-2)
    train_loss, train_accuracy = train_epoch(model, optimizer, images, labels, batch_size=32, rng=rng)
    after, accuracy = evaluate(model, images, labels, batch_size=32)
    assert np.isfinite(before) and np.isfinite(after) and np.isfinite(train_loss)
    assert after < before
    assert 0.0 <= train_accuracy <= 1.0
    assert 0.0 <= accuracy <= 1.0
