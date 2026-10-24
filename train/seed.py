"""Seed the generators that training and tests actually draw from.

`set_seed` seeds the Python `random` module and the legacy NumPy global RNG,
then returns a `numpy.random.Generator`. New code should draw from that
generator. The global seeds exist so a library call that still uses
`np.random.*` cannot silently pick a different stream.
"""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
