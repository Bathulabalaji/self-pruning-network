"""Weight initialization.

He normal keeps the variance of a ReLU pre-activation stable. A ReLU zeros
half of a symmetric input on average, so the weight variance has to be twice
the linear-fan-in value: ``2 / fan_in``.
"""

from __future__ import annotations

import numpy as np


def he_normal(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    if fan_in < 1 or fan_out < 1:
        raise ValueError("fan_in and fan_out must be positive")
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0.0, std, size=(fan_in, fan_out))
