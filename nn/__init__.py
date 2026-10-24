"""Layers built on the autodiff engine."""

from nn.init import he_normal
from nn.layers import Linear

__all__ = ["Linear", "he_normal"]
