"""Adam must not keep state for a masked connection, and revival starts at step 1."""

import numpy as np

from engine import Tensor
from nn import Linear
from optim import Adam

LR = 1e-3
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8


def first_adam_step(value: float, grad: float) -> float:
    """Closed form of one Adam update from a zero moment and step counter."""
    moment = (1.0 - BETA1) * grad
    second = (1.0 - BETA2) * grad * grad
    moment_hat = moment / (1.0 - BETA1)
    second_hat = second / (1.0 - BETA2)
    return value - LR * moment_hat / (np.sqrt(second_hat) + EPS)


def test_unmasked_step_matches_the_closed_form():
    weight = Tensor(np.array([0.4, -0.2]), requires_grad=True)
    weight.grad = np.array([0.5, -1.5])
    optimizer = Adam([weight], lr=LR, betas=(BETA1, BETA2), eps=EPS)
    optimizer.step()
    assert np.allclose(weight.data[0], first_adam_step(0.4, 0.5))
    assert np.allclose(weight.data[1], first_adam_step(-0.2, -1.5))
    assert np.array_equal(optimizer.state[id(weight)]["t"], np.array([1, 1]))


def test_masked_connection_is_cleared_in_one_step_and_stays_clear():
    weight = Tensor(np.array([1.0, 1.0]), requires_grad=True)
    weight.mask = np.array([1.0, 1.0])
    optimizer = Adam([weight], lr=LR, betas=(BETA1, BETA2), eps=EPS)

    for _ in range(30):
        weight.grad = np.array([1.0, 1.0])
        optimizer.step()

    stored = optimizer.state[id(weight)]
    assert stored["t"][1] == 30
    assert stored["m"][1] != 0.0
    assert stored["v"][1] != 0.0

    weight.mask = np.array([1.0, 0.0])
    weight.grad = np.array([0.3, 0.7])
    optimizer.step()
    stored = optimizer.state[id(weight)]
    assert weight.data[1] == 0.0
    assert stored["m"][1] == 0.0
    assert stored["v"][1] == 0.0
    assert stored["t"][1] == 0
    assert stored["t"][0] == 31

    for step in range(10):
        weight.grad = np.array([0.2, 10.0 + step])
        optimizer.step()
        stored = optimizer.state[id(weight)]
        assert weight.data[1] == 0.0
        assert stored["m"][1] == 0.0
        assert stored["v"][1] == 0.0
        assert stored["t"][1] == 0


def test_revived_connection_matches_a_new_parameter():
    weight = Tensor(np.array([1.0, 1.0]), requires_grad=True)
    weight.mask = np.array([1.0, 1.0])
    optimizer = Adam([weight], lr=LR, betas=(BETA1, BETA2), eps=EPS)
    for _ in range(30):
        weight.grad = np.array([1.0, 1.0])
        optimizer.step()
    weight.mask = np.array([1.0, 0.0])
    for _ in range(10):
        weight.grad = np.array([0.2, 4.0])
        optimizer.step()

    weight.mask = np.array([1.0, 1.0])
    weight.grad = np.array([0.25, -0.4])
    optimizer.step()
    stored = optimizer.state[id(weight)]

    fresh = Tensor(np.array([0.0]), requires_grad=True)
    fresh.grad = np.array([-0.4])
    Adam([fresh], lr=LR, betas=(BETA1, BETA2), eps=EPS).step()

    assert stored["t"][1] == 1
    assert stored["t"][0] == 41
    assert np.allclose(weight.data[1], fresh.data[0])
    assert np.allclose(weight.data[1], first_adam_step(0.0, -0.4))
    assert not np.allclose(weight.data[0], first_adam_step(1.0, 0.25))


def test_linear_mask_keeps_the_pruned_weight_at_zero_through_adam():
    layer = Linear(2, 1, weight=np.array([[0.5], [-0.25]]), bias=False)
    layer.set_mask(np.array([[1.0], [0.0]]))
    inputs = Tensor(np.array([[1.0, -2.0], [0.5, 0.25]]))
    optimizer = Adam(layer.parameters(), lr=LR, betas=(BETA1, BETA2), eps=EPS)

    for _ in range(5):
        layer.zero_grad()
        layer(inputs).sum().backward()
        optimizer.step()

    stored = optimizer.state[id(layer.weight)]
    assert layer.weight.data[1, 0] == 0.0
    assert stored["m"][1, 0] == 0.0
    assert stored["v"][1, 0] == 0.0
    assert stored["t"][1, 0] == 0
    assert layer.weight.grad[1, 0] == 0.0
    assert stored["t"][0, 0] == 5

    layer.set_mask(np.array([[1.0], [1.0]]))
    layer.zero_grad()
    layer(inputs).sum().backward()
    revived_grad = layer.weight.grad[1, 0]
    assert revived_grad != 0.0
    optimizer.step()
    assert np.allclose(layer.weight.data[1, 0], first_adam_step(0.0, revived_grad))
    assert optimizer.state[id(layer.weight)]["t"][1, 0] == 1
    assert optimizer.state[id(layer.weight)]["t"][0, 0] == 6
