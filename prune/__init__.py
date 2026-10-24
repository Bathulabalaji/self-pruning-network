"""Importance scores, sparsity schedule, and sparse inference."""

from prune.criteria import Magnitude, TaylorSaliency, criterion_from_name, mask_sparsity, prune_to_sparsity
from prune.schedule import sparsity_at
from prune.sparse import dense_times_zero, linear_flops, pack_mlp, sparse_forward

__all__ = [
    "Magnitude",
    "TaylorSaliency",
    "criterion_from_name",
    "dense_times_zero",
    "linear_flops",
    "mask_sparsity",
    "pack_mlp",
    "prune_to_sparsity",
    "sparse_forward",
    "sparsity_at",
]
