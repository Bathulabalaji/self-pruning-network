"""Cubic sparsity ramp from Zhu and Gupta.

Between the start and end steps the kept fraction follows a cubic, so most
connections are removed late, after the network has a signal worth ranking:

    s(t) = s_f + (s_i - s_f) * (1 - (t - t_start) / (t_end - t_start)) ** 3

Before ``t_start`` the target is ``s_i``. At and after ``t_end`` it is ``s_f``
and the mask is left alone so the remaining steps only fine-tune.
"""

from __future__ import annotations


def sparsity_at(step: int, *, initial: float, final: float, start: int, end: int) -> float:
    if not 0.0 <= initial <= 1.0 or not 0.0 <= final <= 1.0:
        raise ValueError("sparsity must lie in [0, 1]")
    if end <= start:
        raise ValueError("the prune window must have positive length")
    if step <= start:
        return float(initial)
    if step >= end:
        return float(final)
    progress = (step - start) / (end - start)
    return float(final + (initial - final) * (1.0 - progress) ** 3)
