"""Fused log-softmax cross-entropy.

Subtracting the max before the exponential keeps the loss finite for large
logits. The backward below is the derivative of the mean negative log
likelihood, not a softmax op followed by a separate log.

For row `n` and class `c`,

    dL / d z_{n,c} = (softmax(z_n)_c - 1_{c = y_n}) / N

when the loss is averaged over the batch.
"""

from __future__ import annotations

import numpy as np

from engine.tensor import Tensor, accumulate


def cross_entropy(logits: Tensor, targets) -> Tensor:
    if not isinstance(logits, Tensor):
        raise TypeError("logits must be a Tensor")
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape (N, C), got {logits.shape}")
    labels = np.asarray(targets, dtype=np.int64)
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError(f"targets must have shape ({logits.shape[0]},), got {labels.shape}")
    if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise ValueError("targets contain a class index outside the logit width")

    shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
    log_normalizer = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    log_probs = shifted - log_normalizer
    rows = np.arange(logits.shape[0])
    loss = -np.mean(log_probs[rows, labels])
    out = Tensor(loss, requires_grad=logits.requires_grad, parents=(logits,))

    def _backward(upstream: np.ndarray, flow: dict[int, np.ndarray]) -> None:
        probs = np.exp(log_probs)
        grad = probs.copy()
        grad[rows, labels] -= 1.0
        grad /= logits.shape[0]
        accumulate(logits, grad * upstream, flow)

    out._backward = _backward
    return out
