"""The seed helper is deterministic and rejects a non-int seed."""

import random

import numpy as np

from train.seed import set_seed


def test_set_seed_repeats_the_generator_stream():
    first = set_seed(7)
    draws_a = first.random(4)
    second = set_seed(7)
    draws_b = second.random(4)
    assert np.array_equal(draws_a, draws_b)


def test_set_seed_also_resets_global_rnss():
    set_seed(3)
    py_a = random.random()
    np_a = np.random.random()
    set_seed(3)
    py_b = random.random()
    np_b = np.random.random()
    assert py_a == py_b
    assert np_a == np_b


def test_set_seed_rejects_non_int():
    try:
        set_seed(1.5)  # type: ignore[arg-type]
    except TypeError:
        return
    raise AssertionError("expected TypeError")
