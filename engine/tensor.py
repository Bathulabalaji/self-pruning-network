"""A NumPy array plus the parents needed to differentiate it.

`backward` walks the tape in reverse topological order and accumulates into
`.grad`. The value that flows along an edge is not read back from `.grad`.
That split is what makes a second `backward()` add another copy of the
gradient instead of re-propagating the already accumulated one.
"""

from __future__ import annotations

import numpy as np


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum `grad` back to `shape`, undoing a NumPy broadcast.

    Leading axes that `shape` does not have are summed away. Axes that were
    size 1 in `shape` and expanded in `grad` are summed with `keepdims` so
    the result lines up with the operand that was broadcast.
    """
    grad = np.asarray(grad, dtype=np.float64)
    shape = tuple(shape)
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    elif extra < 0:
        raise ValueError(f"cannot unbroadcast grad ndim {grad.ndim} to shape {shape}")
    axes = tuple(i for i, size in enumerate(shape) if size == 1 and grad.shape[i] != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return np.reshape(grad, shape)


def accumulate(node: Tensor, contrib: np.ndarray, flow: dict[int, np.ndarray]) -> None:
    """Add `contrib` into a node's stored grad and into this pass's flowing grad.

    Stored `.grad` and the flowing value are separate arrays. Backward reads
    the flowing value, so a later `backward()` call does not replay `.grad`.
    """
    if not node.requires_grad:
        return
    contrib = np.array(contrib, dtype=np.float64, copy=True)
    if node.grad is None:
        node.grad = contrib.copy()
    else:
        node.grad = node.grad + contrib
    key = id(node)
    if key in flow:
        flow[key] = flow[key] + contrib
    else:
        flow[key] = contrib


class Tensor:
    def __init__(
        self,
        data,
        *,
        requires_grad: bool = False,
        parents: tuple[Tensor, ...] = (),
    ) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = bool(requires_grad)
        self.grad: np.ndarray | None = None
        self._parents = tuple(parents)
        self._backward = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    def _topo(self) -> list[Tensor]:
        order: list[Tensor] = []
        seen: set[int] = set()

        def visit(node: Tensor) -> None:
            marker = id(node)
            if marker in seen:
                return
            seen.add(marker)
            for parent in node._parents:
                visit(parent)
            order.append(node)

        visit(self)
        return order

    def zero_grad(self) -> None:
        """Drop stored gradients on this tensor and every tensor it depends on."""
        for node in self._topo():
            node.grad = None

    def backward(self, grad=None) -> None:
        if not self.requires_grad:
            raise RuntimeError("tensor does not require grad")
        if grad is None:
            if self.data.shape != ():
                raise RuntimeError("backward of a non-scalar tensor requires an explicit gradient")
            grad = np.ones((), dtype=np.float64)
        else:
            grad = np.asarray(grad, dtype=np.float64)
            if grad.shape != self.data.shape:
                raise ValueError(f"grad shape {grad.shape} does not match tensor shape {self.data.shape}")

        flowing = np.array(grad, dtype=np.float64, copy=True)
        flow: dict[int, np.ndarray] = {id(self): flowing}
        if self.grad is None:
            self.grad = flowing.copy()
        else:
            self.grad = self.grad + flowing

        for node in reversed(self._topo()):
            upstream = flow.get(id(node))
            if upstream is None or node._backward is None:
                continue
            node._backward(upstream, flow)

    def __add__(self, other):
        from engine.ops import add

        return add(self, other)

    def __radd__(self, other):
        from engine.ops import add

        return add(other, self)

    def __sub__(self, other):
        from engine.ops import sub

        return sub(self, other)

    def __rsub__(self, other):
        from engine.ops import sub

        return sub(other, self)

    def __mul__(self, other):
        from engine.ops import mul

        return mul(self, other)

    def __rmul__(self, other):
        from engine.ops import mul

        return mul(other, self)

    def __truediv__(self, other):
        from engine.ops import div

        return div(self, other)

    def __rtruediv__(self, other):
        from engine.ops import div

        return div(other, self)

    def __matmul__(self, other):
        from engine.ops import matmul

        return matmul(self, other)

    def __rmatmul__(self, other):
        from engine.ops import matmul

        return matmul(other, self)

    def sum(self, axis=None, keepdims: bool = False):
        from engine.ops import sum as tensor_sum

        return tensor_sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims: bool = False):
        from engine.ops import mean as tensor_mean

        return tensor_mean(self, axis=axis, keepdims=keepdims)

    def relu(self):
        from engine.ops import relu

        return relu(self)

    def tanh(self):
        from engine.ops import tanh

        return tanh(self)
