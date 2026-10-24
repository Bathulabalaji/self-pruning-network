"""Inference that multiplies only the connections the mask kept.

Each output unit stores the input indices that survived and the weights on
those edges. A forward step is a dot product against that short list.
Theoretical matmul FLOPs are ``2 * nnz * batch`` per layer. The dense path
below multiplies the explicit zeros too; it is the baseline the sparse kernel
is timed against, and it is not a sparse speedup.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nn.layers import Linear
from nn.mlp import MLP


@dataclass(frozen=True)
class PackedLinear:
    indices: list[np.ndarray]
    values: list[np.ndarray]
    bias: np.ndarray | None
    fan_in: int
    fan_out: int
    nnz: int


@dataclass(frozen=True)
class PackedMLP:
    layers: tuple[PackedLinear, ...]

    @property
    def nnz(self) -> int:
        return sum(layer.nnz for layer in self.layers)

    @property
    def dense_connections(self) -> int:
        return sum(layer.fan_in * layer.fan_out for layer in self.layers)


def pack_linear(layer: Linear) -> PackedLinear:
    weight = layer.weight.data
    mask = layer.mask.data
    if weight.shape != mask.shape:
        raise ValueError(f"weight shape {weight.shape} does not match mask shape {mask.shape}")
    indices: list[np.ndarray] = []
    values: list[np.ndarray] = []
    nnz = 0
    for column in range(weight.shape[1]):
        kept = np.flatnonzero(mask[:, column] == 1.0)
        indices.append(kept)
        values.append(np.array(weight[kept, column], dtype=np.float64, copy=True))
        nnz += int(kept.size)
    bias = None if layer.bias is None else np.array(layer.bias.data, dtype=np.float64, copy=True)
    return PackedLinear(
        indices=indices,
        values=values,
        bias=bias,
        fan_in=weight.shape[0],
        fan_out=weight.shape[1],
        nnz=nnz,
    )


def pack_mlp(model: MLP) -> PackedMLP:
    return PackedMLP(layers=tuple(pack_linear(layer) for layer in model.layers))


def linear_flops(connections: int, batch: int) -> int:
    """Multiply-adds in a matrix product. One nonzero connection is two flops."""
    if connections < 0 or batch < 0:
        raise ValueError("connections and batch must be non-negative")
    return 2 * int(connections) * int(batch)


def sparse_forward(packed: PackedMLP, inputs: np.ndarray) -> tuple[np.ndarray, int]:
    """Return logits and the theoretical sparse matmul FLOPs for this batch."""
    value = np.asarray(inputs, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"expected inputs of shape (N, D), got {value.shape}")
    flops = 0
    last = len(packed.layers) - 1
    for index, layer in enumerate(packed.layers):
        if value.shape[1] != layer.fan_in:
            raise ValueError(f"layer {index} expected {layer.fan_in} inputs, got {value.shape[1]}")
        output = np.zeros((value.shape[0], layer.fan_out), dtype=np.float64)
        for column, (kept, weights) in enumerate(zip(layer.indices, layer.values)):
            if kept.size:
                output[:, column] = value[:, kept] @ weights
        if layer.bias is not None:
            output += layer.bias
        flops += linear_flops(layer.nnz, value.shape[0])
        value = output if index == last else np.maximum(output, 0.0)
    return value, flops


def dense_times_zero(model: MLP, inputs: np.ndarray) -> tuple[np.ndarray, int]:
    """Dense GEMM with the mask multiplied in. Structural zeros still cost a multiply."""
    value = np.asarray(inputs, dtype=np.float64)
    flops = 0
    last = len(model.layers) - 1
    for index, layer in enumerate(model.layers):
        output = value @ (layer.weight.data * layer.mask.data)
        if layer.bias is not None:
            output = output + layer.bias.data
        flops += linear_flops(layer.weight.data.size, value.shape[0])
        value = output if index == last else np.maximum(output, 0.0)
    return value, flops
