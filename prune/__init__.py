"""Importance scores, sparsity schedule, and sparse inference."""

from prune.criteria import Magnitude, TaylorSaliency, criterion_from_name, mask_sparsity, prune_to_sparsity
from prune.schedule import sparsity_at

__all__ = [
    "Magnitude",
    "TaylorSaliency",
    "criterion_from_name",
    "mask_sparsity",
    "prune_to_sparsity",
    "sparsity_at",
]
