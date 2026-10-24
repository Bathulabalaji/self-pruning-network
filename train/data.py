"""MNIST cached on disk.

scikit-learn is a permitted way to download this dataset and is not used for
training, models, or gradients. ``fetch_openml`` does not complete here:
OpenML responds with HTTP 412, and this machine's Python trust store rejects
the TLS handshake. The loader therefore reads the standard MNIST idx files
(60,000 train, 10,000 test) from the CVDF mirror and caches ``data/mnist.npz``.
Those are the same split and pixel layout an OpenML download would have saved.
"""

from __future__ import annotations

import gzip
import struct
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "mnist.npz"
MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist"
FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)
TRAIN_SPLIT = 60_000


def load_mnist(cache: Path = CACHE) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train and test images in ``[0, 1]`` and integer labels."""
    if not cache.exists():
        _download(cache)
    stored = np.load(cache)
    images = stored["images"]
    labels = stored["labels"]
    return images[:TRAIN_SPLIT], labels[:TRAIN_SPLIT], images[TRAIN_SPLIT:], labels[TRAIN_SPLIT:]


def _download(cache: Path) -> None:
    raw = cache.parent / "mnist_idx"
    raw.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        destination = raw / name
        if destination.exists() and destination.stat().st_size > 0:
            continue
        subprocess.run(
            ["curl", "-fsSL", "--retry", "3", "-o", str(destination), f"{MIRROR}/{name}"],
            check=True,
        )
    images = np.concatenate(
        [
            _read_images(raw / "train-images-idx3-ubyte.gz"),
            _read_images(raw / "t10k-images-idx3-ubyte.gz"),
        ]
    )
    labels = np.concatenate(
        [
            _read_labels(raw / "train-labels-idx1-ubyte.gz"),
            _read_labels(raw / "t10k-labels-idx1-ubyte.gz"),
        ]
    )
    if images.shape != (70_000, 784) or labels.shape != (70_000,):
        raise RuntimeError(f"unexpected MNIST shapes {images.shape} {labels.shape}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, images=images, labels=labels)


def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise RuntimeError(f"{path.name} is not an idx image file (magic {magic})")
        pixels = np.frombuffer(handle.read(), dtype=np.uint8)
    return pixels.reshape(count, rows * cols).astype(np.float64) / 255.0


def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise RuntimeError(f"{path.name} is not an idx label file (magic {magic})")
        labels = np.frombuffer(handle.read(), dtype=np.uint8)
    if labels.shape != (count,):
        raise RuntimeError(f"{path.name} has {labels.shape[0]} labels, header says {count}")
    return labels.astype(np.int64)
