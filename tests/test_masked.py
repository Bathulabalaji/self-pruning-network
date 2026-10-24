"""The masked-weight contract: a 0 in the mask removes the connection from the forward and from the parameter gradient."""

import numpy as np
import pytest

from engine import Tensor, cross_entropy
from nn import Linear
from train.seed import set_seed


def test_masked_forward_matches_weights_that_were_zeroed():
    rng = set_seed(0)
    weight = rng.normal(size=(3, 2))
    mask = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    inputs = Tensor(rng.normal(size=(4, 3)))

    masked = Linear(3, 2, weight=weight, bias=False)
    masked.set_mask(mask)
    zeroed = weight.copy()
    zeroed[mask == 0.0] = 0.0
    reference = Linear(3, 2, weight=zeroed, bias=False)

    assert np.allclose(masked(inputs).data, reference(inputs).data)


def test_mask_gates_the_parameter_gradient_and_keeps_saliency():
    rng = set_seed(1)
    weight = rng.normal(size=(3, 2))
    inputs = Tensor(rng.normal(size=(5, 3)))
    mask = np.ones((3, 2))
    mask[0, 1] = 0.0
    mask[2, 0] = 0.0

    dense = Linear(3, 2, weight=weight, bias=False)
    sparse = Linear(3, 2, weight=weight, bias=False)
    sparse.set_mask(mask)

    dense(inputs).sum().backward()
    sparse(inputs).sum().backward()

    expected = inputs.data.T @ np.ones((5, 2))
    assert np.allclose(dense.weight.grad, expected)
    assert np.allclose(sparse.weight.saliency_grad, expected)
    assert np.allclose(sparse.weight.grad, expected * mask)
    assert np.all(sparse.weight.grad[mask == 0.0] == 0.0)
    assert np.all(sparse.weight.saliency_grad[mask == 0.0] != 0.0)


def test_a_stored_zero_without_a_mask_still_receives_a_gradient():
    """Zeroing W is not pruning. dL/dW for a matmul does not depend on W."""
    inputs = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    weight = np.zeros((2, 1))
    layer = Linear(2, 1, weight=weight, bias=False)
    layer(inputs).sum().backward()
    expected = inputs.data.T @ np.ones((2, 1))
    assert np.allclose(layer.weight.grad, expected)
    assert np.any(layer.weight.grad != 0.0)


def test_masked_weight_gradient_matches_finite_differences():
    rng = set_seed(2)
    weight = rng.normal(size=(4, 3))
    mask = np.ones((4, 3))
    mask[1, 0] = 0.0
    mask[3, 2] = 0.0
    layer = Linear(4, 3, weight=weight)
    layer.set_mask(mask)
    inputs = Tensor(rng.normal(size=(6, 4)))
    targets = rng.integers(0, 3, size=(6,))

    def loss():
        return cross_entropy(layer(inputs), targets)

    layer.zero_grad()
    loss().backward()
    analytical_weight = layer.weight.grad.copy()
    analytical_bias = layer.bias.grad.copy()
    saliency = layer.weight.saliency_grad.copy()

    eps = 1e-5
    numerical = np.empty_like(weight)
    for index in np.ndindex(weight.shape):
        original = layer.weight.data[index]
        layer.weight.data[index] = original + eps
        plus = loss().data
        layer.weight.data[index] = original - eps
        minus = loss().data
        layer.weight.data[index] = original
        numerical[index] = (plus - minus) / (2.0 * eps)

    assert np.allclose(analytical_weight, numerical, rtol=1e-4, atol=1e-5)
    assert np.allclose(numerical[mask == 0.0], 0.0, atol=1e-6)
    assert np.any(np.abs(saliency[mask == 0.0]) > 1e-8)
    assert np.all(analytical_weight[mask == 0.0] == 0.0)

    numerical_bias = np.empty_like(layer.bias.data)
    for index in np.ndindex(layer.bias.shape):
        original = layer.bias.data[index]
        layer.bias.data[index] = original + eps
        plus = loss().data
        layer.bias.data[index] = original - eps
        minus = loss().data
        layer.bias.data[index] = original
        numerical_bias[index] = (plus - minus) / (2.0 * eps)
    assert np.allclose(analytical_bias, numerical_bias, rtol=1e-4, atol=1e-5)


def test_saliency_accumulates_and_zero_grad_clears_it():
    inputs = Tensor(np.ones((2, 2)))
    layer = Linear(2, 1, weight=np.ones((2, 1)), bias=False)
    layer.set_mask(np.array([[0.0], [1.0]]))
    loss = layer(inputs).sum()
    loss.backward()
    once = layer.weight.saliency_grad.copy()
    loss.backward()
    assert np.allclose(layer.weight.saliency_grad, 2.0 * once)
    assert np.allclose(layer.weight.grad, 2.0 * once * np.array([[0.0], [1.0]]))
    layer.zero_grad()
    assert layer.weight.grad is None
    assert layer.weight.saliency_grad is None


def test_set_mask_rejects_soft_values():
    layer = Linear(2, 2, weight=np.ones((2, 2)), bias=False)
    with pytest.raises(ValueError):
        layer.set_mask(np.full((2, 2), 1e-8))
