"""Finite-difference checks for every operation in the autodiff engine."""

import numpy as np
import pytest

from engine import Tensor, cross_entropy, mean, relu, sum, tanh
from train.seed import set_seed

EPS = 1e-5
RTOL = 1e-4
ATOL = 1e-5


def gradcheck(fn, inputs, eps=EPS, rtol=RTOL, atol=ATOL):
    tensors = [value for value in inputs if isinstance(value, Tensor) and value.requires_grad]
    for tensor in tensors:
        tensor.grad = None
    value = fn(*inputs)
    assert value.shape == (), "gradcheck target must be a scalar"
    value.backward()

    for tensor in tensors:
        analytical = np.array(tensor.grad, copy=True)
        numerical = np.empty_like(tensor.data)
        for index in np.ndindex(tensor.data.shape):
            original = tensor.data[index]
            tensor.data[index] = original + eps
            plus = fn(*inputs).data
            tensor.data[index] = original - eps
            minus = fn(*inputs).data
            tensor.data[index] = original
            numerical[index] = (plus - minus) / (2.0 * eps)
        if not np.allclose(analytical, numerical, rtol=rtol, atol=atol):
            diff = np.max(np.abs(analytical - numerical))
            raise AssertionError(f"gradcheck failed, max abs diff {diff}")


def test_add_broadcast_matches_finite_differences():
    rng = set_seed(0)
    left = Tensor(rng.normal(size=(4, 3)), requires_grad=True)
    right = Tensor(rng.normal(size=(3,)), requires_grad=True)
    gradcheck(lambda a, b: (a + b).sum(), (left, right))


def test_sub_mul_div_broadcast():
    rng = set_seed(1)
    left = Tensor(rng.normal(size=(3, 2, 4)), requires_grad=True)
    right = Tensor(rng.normal(size=(2, 1)) + 2.0, requires_grad=True)
    gradcheck(lambda a, b: (a - b).sum(), (left, right))
    gradcheck(lambda a, b: (a * b).sum(), (left, right))
    gradcheck(lambda a, b: (a / b).sum(), (left, right))


def test_square_uses_both_mul_paths_for_one_tensor():
    rng = set_seed(2)
    value = Tensor(rng.normal(size=(5,)), requires_grad=True)
    gradcheck(lambda x: (x * x).sum(), (value,))


@pytest.mark.parametrize(
    "left_shape,right_shape",
    [
        ((4, 3), (3, 5)),
        ((3,), (3, 5)),
        ((4, 3), (3,)),
        ((2, 4, 3), (3,)),
        ((3,), (3,)),
        ((2, 4, 3), (3, 5)),
        ((4, 3), (2, 3, 5)),
        ((2, 4, 3), (2, 3, 5)),
        ((1, 4, 3), (2, 3, 5)),
    ],
)
def test_matmul_shapes(left_shape, right_shape):
    rng = set_seed(3)
    left = Tensor(rng.normal(size=left_shape), requires_grad=True)
    right = Tensor(rng.normal(size=right_shape), requires_grad=True)
    gradcheck(lambda a, b: (a @ b).sum(), (left, right))


def test_sum_and_mean_axes():
    rng = set_seed(4)
    value = Tensor(rng.normal(size=(2, 3, 4)), requires_grad=True)
    gradcheck(lambda x: x.sum(), (value,))
    gradcheck(lambda x: x.mean(), (value,))
    gradcheck(lambda x: x.sum(axis=0).sum(), (value,))
    gradcheck(lambda x: x.sum(axis=-1).mean(), (value,))
    gradcheck(lambda x: x.mean(axis=(0, 2)).sum(), (value,))
    gradcheck(lambda x: sum(x, axis=1, keepdims=True).sum(), (value,))
    gradcheck(lambda x: mean(x, keepdims=True).sum(), (value,))


def test_relu_off_the_kink():
    rng = set_seed(5)
    data = rng.normal(size=(6, 4))
    data[np.abs(data) < 0.25] += 0.5
    value = Tensor(data, requires_grad=True)
    gradcheck(lambda x: relu(x).sum(), (value,))


def test_tanh():
    rng = set_seed(6)
    value = Tensor(rng.normal(size=(5, 3)), requires_grad=True)
    gradcheck(lambda x: tanh(x).sum(), (value,))


def test_cross_entropy_matches_finite_differences():
    rng = set_seed(7)
    logits = Tensor(rng.normal(size=(5, 4)), requires_grad=True)
    targets = rng.integers(0, 4, size=(5,))
    gradcheck(lambda z: cross_entropy(z, targets), (logits,))


def test_cross_entropy_is_shift_invariant_and_finite_for_large_logits():
    logits = Tensor(np.array([[1000.0, 1000.0, 1000.0], [5.0, -4.0, 20.0]]), requires_grad=True)
    targets = np.array([1, 2])
    loss = cross_entropy(logits, targets)
    assert np.isfinite(loss.data)
    loss.backward()
    assert np.all(np.isfinite(logits.grad))
    assert np.allclose(logits.grad.sum(axis=1), 0.0, atol=1e-8)

    shifted = Tensor(logits.data + 17.0, requires_grad=True)
    shifted_loss = cross_entropy(shifted, targets)
    assert np.allclose(loss.data, shifted_loss.data, atol=1e-8)


def test_backward_accumulates_until_zero_grad():
    value = Tensor(np.array([1.5, -2.0]), requires_grad=True)
    first = (value * value).sum()
    first.backward()
    once = value.grad.copy()
    first.backward()
    assert np.allclose(value.grad, 2.0 * once)
    first.zero_grad()
    assert value.grad is None
    (value * value).sum().backward()
    assert np.allclose(value.grad, once)


def test_non_scalar_backward_requires_an_explicit_gradient():
    value = Tensor(np.ones((2, 2)), requires_grad=True)
    with pytest.raises(RuntimeError):
        value.backward()
