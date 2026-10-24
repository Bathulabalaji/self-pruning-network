"""Time the sparse forward against a dense GEMM that still multiplies the zeros.

The nonzero counts come from the committed Taylor run at 90% sparsity. The
weights themselves are a fresh He draw with that same mask pattern: FLOPs and
the kernel depend on where the nonzeros sit, not on the trained values. The
script checks that this kernel matches the masked dense product before it
reports a time.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from engine import Tensor
from nn.mlp import MLP
from prune.sparse import dense_times_zero, linear_flops, pack_mlp, sparse_forward
from train.seed import set_seed

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "prune_taylor_s0.9_seed0.json"


def main() -> None:
    trained = json.loads(REPORT.read_text())
    counts = [int(layer["nnz"]) for layer in trained["layers"]]
    rng = set_seed(0)
    model = MLP([784, 128, 64, 10], rng)
    for layer, nnz in zip(model.layers, counts):
        mask = np.zeros(layer.weight.shape, dtype=np.float64)
        chosen = rng.choice(mask.size, size=nnz, replace=False)
        mask.ravel()[chosen] = 1.0
        layer.set_mask(mask)
        layer.weight.data *= mask
    packed = pack_mlp(model)
    if packed.nnz != sum(counts):
        raise SystemExit(f"packed nnz {packed.nnz} does not match the trained report {sum(counts)}")

    batch = rng.normal(size=(128, 784))
    sparse, sparse_flops = sparse_forward(packed, batch)
    dense, dense_flops = dense_times_zero(model, batch)
    tape = model(Tensor(batch)).data
    if not np.allclose(sparse, dense, rtol=1e-8, atol=1e-8):
        raise SystemExit("sparse forward does not match the dense masked product")
    if not np.allclose(sparse, tape, rtol=1e-8, atol=1e-8):
        raise SystemExit("sparse forward does not match the autodiff forward")

    timings = [_time_batch(model, packed, batch_size, rng) for batch_size in (1, 128, 1024)]
    report = {
        "source": REPORT.name,
        "active_parameters": packed.nnz,
        "dense_parameters": packed.dense_connections,
        "achieved_sparsity": 1.0 - packed.nnz / packed.dense_connections,
        "note": (
            "dense_times_zero_ms is a BLAS GEMM that multiplies explicit zeros. "
            "It is not a sparse speedup. sparse_ms is the gather kernel in prune/sparse.py. "
            "tape_ms builds the autodiff graph and is not an inference path."
        ),
        "batches": timings,
        "matches_dense_masked_forward": True,
        "matches_tape_forward": True,
    }
    out = ROOT / "results" / "cost_taylor_s0.9.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    for row in timings:
        print(
            f"batch {row['batch']:4d}  sparse {row['sparse_ms']:.3f} ms  "
            f"dense_times_zero {row['dense_times_zero_ms']:.3f} ms  "
            f"tape {row['tape_ms']:.3f} ms  "
            f"sparse_flops {row['sparse_flops']}  dense_flops {row['dense_flops']}",
            flush=True,
        )
    print(f"wrote {out}")


def _time_batch(model: MLP, packed, batch_size: int, rng: np.random.Generator) -> dict:
    inputs = rng.normal(size=(batch_size, 784))
    sparse_ms = _milliseconds(lambda: sparse_forward(packed, inputs))
    dense_ms = _milliseconds(lambda: dense_times_zero(model, inputs))
    tape_ms = _milliseconds(lambda: model(Tensor(inputs)))
    return {
        "batch": batch_size,
        "sparse_ms": sparse_ms,
        "dense_times_zero_ms": dense_ms,
        "tape_ms": tape_ms,
        "sparse_flops": linear_flops(packed.nnz, batch_size),
        "dense_flops": linear_flops(packed.dense_connections, batch_size),
        "sparse_faster_than_dense_gemm": sparse_ms < dense_ms,
    }


def _milliseconds(fn, warmup: int = 5, repeats: int = 30) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) * 1000.0 / repeats


if __name__ == "__main__":
    main()
