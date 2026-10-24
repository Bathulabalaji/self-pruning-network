"""The cubic ramp, both criteria, and the exact-zero mask."""

import numpy as np

from engine import Tensor
from nn import MLP
from optim import Adam
from prune import criterion_from_name, mask_sparsity, prune_to_sparsity, sparsity_at
from train.loop import StepCounter, evaluate, train_epoch
from train.seed import set_seed


def test_cubic_ramp_endpoints_and_midpoint():
    assert sparsity_at(10, initial=0.0, final=0.9, start=10, end=20) == 0.0
    assert sparsity_at(20, initial=0.0, final=0.9, start=10, end=20) == 0.9
    assert sparsity_at(5, initial=0.0, final=0.9, start=10, end=20) == 0.0
    mid = sparsity_at(15, initial=0.0, final=0.9, start=10, end=20)
    assert np.allclose(mid, 0.9 + (0.0 - 0.9) * (0.5**3))
    earlier = sparsity_at(12, initial=0.0, final=0.9, start=10, end=20)
    later = sparsity_at(18, initial=0.0, final=0.9, start=10, end=20)
    assert earlier < mid < later


def test_global_magnitude_prune_zeros_the_smallest_weights():
    rng = set_seed(0)
    model = MLP([2, 2, 1], rng)
    model.layers[0].weight.data[:] = np.array([[0.1, 4.0], [0.2, 3.0]])
    model.layers[1].weight.data[:] = np.array([[5.0], [0.05]])
    optimizer = Adam(model.parameters(), lr=1e-3)
    model.layers[0].weight.grad = np.ones_like(model.layers[0].weight.data)
    model.layers[1].weight.grad = np.ones_like(model.layers[1].weight.data)
    model.layers[0].bias.grad = np.ones_like(model.layers[0].bias.data)
    model.layers[1].bias.grad = np.ones_like(model.layers[1].bias.data)
    optimizer.step()
    before = optimizer.state[id(model.layers[0].weight)]["m"].copy()
    assert np.all(before != 0.0)

    achieved = prune_to_sparsity(model.layers, criterion_from_name("magnitude").score, 0.5, optimizer)
    # Six weights. Keeping half leaves the three largest: 5, 4, 3.
    sparsity, nonzero, total = mask_sparsity(model.layers)
    assert total == 6
    assert nonzero == 3
    assert achieved == sparsity
    first = model.layers[0].weight.data
    second = model.layers[1].weight.data
    assert first[0, 0] == 0.0 and first[1, 0] == 0.0
    assert second[1, 0] == 0.0
    assert first[0, 1] != 0.0 and first[1, 1] != 0.0 and second[0, 0] != 0.0
    assert np.all(model.layers[0].mask.data[first == 0.0] == 0.0)
    state = optimizer.state[id(model.layers[0].weight)]
    assert state["m"][0, 0] == 0.0 and state["v"][0, 0] == 0.0 and state["t"][0, 0] == 0
    assert state["m"][0, 1] != 0.0
    assert np.all(model.layers[0].bias.data != 0.0)
    assert not hasattr(model.layers[0].bias, "mask")


def test_taylor_score_uses_the_pre_mask_gradient():
    layer_inputs = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    model = MLP([2, 1], set_seed(1))
    weight = np.array([[0.5], [-0.25]])
    model.layers[0].weight.data[:] = weight
    model.layers[0].bias.data[:] = 0.0
    model.zero_grad()
    model(layer_inputs).sum().backward()
    criterion = criterion_from_name("taylor", beta=0.9)
    criterion.observe(model.layers)
    expected = 0.1 * np.abs(weight * model.layers[0].weight.saliency_grad)
    assert np.allclose(criterion.score(model.layers[0]), expected)
    assert np.allclose(model.layers[0].weight.saliency_grad, layer_inputs.data.T @ np.ones((2, 1)))


def test_short_run_hits_the_target_and_stays_zero():
    rng = set_seed(2)
    images = rng.normal(size=(20, 4))
    labels = rng.integers(0, 3, size=(20,))
    model = MLP([4, 8, 3], rng)
    optimizer = Adam(model.parameters(), lr=1e-2)
    criterion = criterion_from_name("taylor")
    target = 0.5
    steps_per_epoch = 2
    epochs = 4
    total_steps = steps_per_epoch * epochs
    start, end = steps_per_epoch, int(0.75 * total_steps)
    clock = StepCounter()

    def on_backward(step):
        del step
        criterion.observe(model.layers)

    def on_step(step):
        if step < start or step > end:
            return
        if step % 2 == 0 or step == end:
            prune_to_sparsity(
                model.layers,
                criterion.score,
                sparsity_at(step, initial=0.0, final=target, start=start, end=end),
                optimizer,
            )

    for _ in range(epochs):
        train_epoch(
            model,
            optimizer,
            images,
            labels,
            10,
            rng,
            clock=clock,
            on_backward=on_backward,
            on_step=on_step,
        )
    sparsity, nonzero, total = mask_sparsity(model.layers)
    assert nonzero == int(round((1.0 - target) * total))
    assert np.isfinite(sparsity)
    for layer in model.layers:
        assert np.all(layer.weight.data[layer.mask.data == 0.0] == 0.0)
        state = optimizer.state[id(layer.weight)]
        assert np.all(state["m"][layer.mask.data == 0.0] == 0.0)
        assert np.all(state["v"][layer.mask.data == 0.0] == 0.0)
        assert np.all(state["t"][layer.mask.data == 0.0] == 0)
    loss, accuracy = evaluate(model, images, labels, 10)
    assert np.isfinite(loss)
    assert 0.0 <= accuracy <= 1.0
