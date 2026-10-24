"""Layers whose parameters the tape can differentiate.

`Linear` multiplies the weight by a constant 0/1 mask inside the graph.
The chain rule then gates the parameter gradient. The gradient of the
masked product, before that gate, is stored on the weight as
``saliency_grad``. Importance scores read that buffer. The optimizer must
read ``weight.grad`` instead.
"""

from __future__ import annotations

import numpy as np

from engine.tensor import Tensor
from nn.init import he_normal


class Linear:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rng: np.random.Generator | None = None,
        weight: np.ndarray | None = None,
        bias: bool = True,
    ) -> None:
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        if weight is None:
            generator = rng if rng is not None else np.random.default_rng(0)
            weight = he_normal(self.in_features, self.out_features, generator)
        weight = np.asarray(weight, dtype=np.float64)
        if weight.shape != (self.in_features, self.out_features):
            raise ValueError(
                f"weight shape {weight.shape} does not match {(self.in_features, self.out_features)}"
            )
        self.weight = Tensor(weight, requires_grad=True)
        self.weight.saliency_grad = None
        self.mask = Tensor(np.ones(weight.shape, dtype=np.float64), requires_grad=False)
        self.bias = Tensor(np.zeros(self.out_features), requires_grad=True) if bias else None

    def parameters(self) -> list[Tensor]:
        if self.bias is None:
            return [self.weight]
        return [self.weight, self.bias]

    def zero_grad(self) -> None:
        """Clear parameter gradients and the pre-mask saliency from the last pass."""
        self.weight.zero_grad()
        self.weight.saliency_grad = None
        if self.bias is not None:
            self.bias.zero_grad()

    def set_mask(self, mask) -> None:
        """Install a new constant mask. Entries must be exactly 0 or 1.

        This replaces the mask tensor. A graph already built keeps the mask
        object it multiplied by, so change the mask between steps, not between
        a forward and its backward.
        """
        values = np.asarray(mask, dtype=np.float64)
        if values.shape != self.weight.shape:
            raise ValueError(f"mask shape {values.shape} does not match weight shape {self.weight.shape}")
        if not np.all((values == 0.0) | (values == 1.0)):
            raise ValueError("mask entries must be exactly 0 or 1")
        self.mask = Tensor(values, requires_grad=False)

    def __call__(self, inputs: Tensor) -> Tensor:
        if not isinstance(inputs, Tensor):
            raise TypeError("Linear inputs must be a Tensor")
        if inputs.ndim != 2 or inputs.shape[1] != self.in_features:
            raise ValueError(f"expected inputs of shape (N, {self.in_features}), got {inputs.shape}")
        masked = self.weight * self.mask
        weight = self.weight
        multiply_backward = masked._backward

        def _backward(upstream: np.ndarray, flow: dict[int, np.ndarray]) -> None:
            # `upstream` is dL/d(W ⊙ M), the matmul's weight gradient before the
            # mask multiplies it. `.grad` on the parameter is that vector times M.
            contrib = np.array(upstream, dtype=np.float64, copy=True)
            if weight.saliency_grad is None:
                weight.saliency_grad = contrib
            else:
                weight.saliency_grad = weight.saliency_grad + contrib
            multiply_backward(upstream, flow)

        masked._backward = _backward
        output = inputs @ masked
        if self.bias is not None:
            output = output + self.bias
        return output
