"""Differentiable NumPy operations.

Each output tensor stores a closure. The closure receives only the gradient
that flowed into the output on this backward pass, and it adds the resulting
parent contributions into both `.grad` and the pass-local flow.
"""

from __future__ import annotations

import numpy as np

from engine.tensor import Tensor, accumulate, unbroadcast


def _as_tensor(value) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return Tensor(value, requires_grad=False)


def _needs_grad(*nodes: Tensor) -> bool:
    return any(node.requires_grad for node in nodes)


def _restore_reduced(grad: np.ndarray, axis, keepdims: bool, shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast a reduction gradient back to the unreduced operand shape."""
    if axis is None or keepdims:
        return np.broadcast_to(np.asarray(grad, dtype=np.float64), shape).copy()
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    axes = tuple(size if size >= 0 else size + len(shape) for size in axes)
    restored = np.asarray(grad, dtype=np.float64)
    for reduced_axis in sorted(axes):
        restored = np.expand_dims(restored, axis=reduced_axis)
    return np.broadcast_to(restored, shape).copy()


def add(a, b) -> Tensor:
    a, b = _as_tensor(a), _as_tensor(b)
    out = Tensor(a.data + b.data, requires_grad=_needs_grad(a, b), parents=(a, b))

    def _backward(upstream: np.ndarray, flow: dict[int, np.ndarray]) -> None:
        accumulate(a, unbroadcast(upstream, a.shape), flow)
        accumulate(b, unbroadcast(upstream, b.shape), flow)

    out._backward = _backward
    return out


def sub(a, b) -> Tensor:
    a, b = _as_tensor(a), _as_tensor(b)
    out = Tensor(a.data - b.data, requires_grad=_needs_grad(a, b), parents=(a, b))

    def _backward(upstream: np.ndarray, flow: dict[int, np.ndarray]) -> None:
        accumulate(a, unbroadcast(upstream, a.shape), flow)
        accumulate(b, unbroadcast(-upstream, b.shape), flow)

    out._backward = _backward
    return out


def mul(a, b) -> Tensor:
    a, b = _as_tensor(a), _as_tensor(b)
    out = Tensor(a.data * b.data, requires_grad=_needs_grad(a, b), parents=(a, b))

    def _backward(upstream: np.ndarray, flow: dict[int, np.ndarray]) -> None:
        accumulate(a, unbroadcast(upstream * b.data, a.shape), flow)
        accumulate(b, unbroadcast(upstream * a.data, b.shape), flow)

    out._backward = _backward
    return out


def div(a, b) -> Tensor:
    a, b = _as_tensor(a), _as_tensor(b)
    out = Tensor(a.data / b.data, requires_grad=_needs_grad(a, b), parents=(a, b))

    def _backward(upstream: np.ndarray, flow: dict[int, np.ndarray]) -> None:
        accumulate(a, unbroadcast(upstream / b.data, a.shape), flow)
        accumulate(b, unbroadcast(-upstream * a.data / np.square(b.data), b.shape), flow)

    out._backward = _backward
    return out


def matmul(a, b) -> Tensor:
    a, b = _as_tensor(a), _as_tensor(b)
    out = Tensor(a.data @ b.data, requires_grad=_needs_grad(a, b), parents=(a, b))

    def _backward(upstream: np.ndarray, flow: dict[int, np.ndarray]) -> None:
        # Two vectors are an inner product. Every other vector is a matrix
        # with a dummy axis, which NumPy removes from the result. Put that
        # axis back, differentiate the 2D product, then drop it again.
        # The dummy axis on the right is trailing, so unbroadcast (which only
        # sums leading axes) cannot remove it.
        if a.data.ndim == 1 and b.data.ndim == 1:
            accumulate(a, unbroadcast(upstream * b.data, a.shape), flow)
            accumulate(b, unbroadcast(upstream * a.data, b.shape), flow)
            return
        left = a.data
        right = b.data
        grad = upstream
        left_was_vector = left.ndim == 1
        right_was_vector = right.ndim == 1
        if left_was_vector:
            left = left[None, :]
            grad = np.expand_dims(grad, axis=-2 if grad.ndim else 0)
        if right_was_vector:
            right = right[:, None]
            grad = np.expand_dims(grad, axis=-1)
        grad_left = grad @ np.swapaxes(right, -1, -2)
        grad_right = np.swapaxes(left, -1, -2) @ grad
        if left_was_vector:
            grad_left = np.squeeze(grad_left, axis=-2)
        if right_was_vector:
            grad_right = np.squeeze(grad_right, axis=-1)
        accumulate(a, unbroadcast(grad_left, a.shape), flow)
        accumulate(b, unbroadcast(grad_right, b.shape), flow)

    out._backward = _backward
    return out


def sum(x, axis=None, keepdims: bool = False) -> Tensor:
    x = _as_tensor(x)
    out = Tensor(
        np.sum(x.data, axis=axis, keepdims=keepdims),
        requires_grad=x.requires_grad,
        parents=(x,),
    )

    def _backward(upstream: np.ndarray, flow: dict[int, np.ndarray]) -> None:
        restored = _restore_reduced(upstream, axis, keepdims, x.shape)
        accumulate(x, restored, flow)

    out._backward = _backward
    return out


def mean(x, axis=None, keepdims: bool = False) -> Tensor:
    x = _as_tensor(x)
    reduced = np.mean(x.data, axis=axis, keepdims=keepdims)
    count = x.data.size / np.asarray(reduced).size
    out = Tensor(reduced, requires_grad=x.requires_grad, parents=(x,))

    def _backward(upstream: np.ndarray, flow: dict[int, np.ndarray]) -> None:
        restored = _restore_reduced(upstream / count, axis, keepdims, x.shape)
        accumulate(x, restored, flow)

    out._backward = _backward
    return out


def relu(x) -> Tensor:
    x = _as_tensor(x)
    positive = x.data > 0
    out = Tensor(x.data * positive, requires_grad=x.requires_grad, parents=(x,))

    def _backward(upstream: np.ndarray, flow: dict[int, np.ndarray]) -> None:
        # Subgradient at exactly 0 is 0. Gradcheck inputs must stay off the kink.
        accumulate(x, upstream * positive, flow)

    out._backward = _backward
    return out


def tanh(x) -> Tensor:
    x = _as_tensor(x)
    value = np.tanh(x.data)
    out = Tensor(value, requires_grad=x.requires_grad, parents=(x,))

    def _backward(upstream: np.ndarray, flow: dict[int, np.ndarray]) -> None:
        accumulate(x, upstream * (1.0 - np.square(value)), flow)

    out._backward = _backward
    return out
