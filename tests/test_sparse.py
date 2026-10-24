"""The sparse forward matches the masked dense network and counts only stored connections."""

import numpy as np

from engine import Tensor
from nn import MLP
from prune.sparse import dense_times_zero, linear_flops, pack_mlp, sparse_forward
from train.seed import set_seed


def _masked_model():
    rng = set_seed(0)
    model = MLP([6, 5, 3], rng)
    mask = np.ones((6, 5))
    mask[0, 0] = 0.0
    mask[2, 4] = 0.0
    mask[5, 1] = 0.0
    model.layers[0].set_mask(mask)
    model.layers[0].weight.data[0, 0] = 1e6
    second = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    model.layers[1].set_mask(second)
    model.layers[1].weight.data[1, 0] = -1e6
    for layer in model.layers:
        layer.weight.data *= layer.mask.data
        layer.weight.data[layer.mask.data == 0.0] = 1e6
    return model


def test_sparse_forward_matches_the_masked_tape_and_ignores_pruned_weights():
    model = _masked_model()
    rng = set_seed(1)
    inputs = rng.normal(size=(4, 6))
    packed = pack_mlp(model)
    sparse, sparse_flops = sparse_forward(packed, inputs)
    dense, dense_flops = dense_times_zero(model, inputs)
    tape = model(Tensor(inputs)).data
    assert np.allclose(sparse, dense, rtol=1e-10, atol=1e-10)
    assert np.allclose(sparse, tape, rtol=1e-10, atol=1e-10)
    batch = inputs.shape[0]
    assert sparse_flops == linear_flops(packed.nnz, batch)
    assert dense_flops == linear_flops(packed.dense_connections, batch)
    assert sparse_flops < dense_flops
    unmasked = inputs @ model.layers[0].weight.data
    masked = inputs @ (model.layers[0].weight.data * model.layers[0].mask.data)
    assert not np.allclose(unmasked, masked)


def test_linear_flops_are_two_per_connection():
    assert linear_flops(10, 4) == 80
    assert linear_flops(0, 4) == 0
