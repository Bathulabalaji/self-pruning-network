"""A stack of masked linear layers with ReLU on every layer but the last."""

from __future__ import annotations

import numpy as np

from engine.tensor import Tensor
from nn.layers import Linear


class MLP:
    def __init__(self, sizes: list[int], rng: np.random.Generator) -> None:
        if len(sizes) < 2:
            raise ValueError("an MLP needs an input size and an output size")
        self.sizes = [int(size) for size in sizes]
        self.layers = [
            Linear(self.sizes[index], self.sizes[index + 1], rng=rng) for index in range(len(self.sizes) - 1)
        ]

    def parameters(self) -> list[Tensor]:
        parameters: list[Tensor] = []
        for layer in self.layers:
            parameters.extend(layer.parameters())
        return parameters

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def __call__(self, inputs: Tensor) -> Tensor:
        value = inputs
        last = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            value = layer(value)
            if index != last:
                value = value.relu()
        return value
