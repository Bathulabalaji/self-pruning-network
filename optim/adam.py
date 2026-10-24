"""Adam with a separate clock for every connection.

A masked connection is not a parameter. Its weight, first moment, second
moment, and step counter stay at zero, and a gradient of zero is not allowed
to decay whatever was stored there before the mask. When the mask opens
again the connection takes a first step, with bias correction ``1 - β^1``,
matching a parameter Adam has never seen. A single global step counter would
make that revived step about ``(1 - β1)`` times too small.
"""

from __future__ import annotations

import numpy as np

from engine.tensor import Tensor


class Adam:
    def __init__(
        self,
        parameters: list[Tensor],
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        if lr < 0:
            raise ValueError("lr must be non-negative")
        beta1, beta2 = betas
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("betas must lie in [0, 1)")
        self.parameters = list(parameters)
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.state: dict[int, dict[str, np.ndarray]] = {}

    def step(self) -> None:
        for param in self.parameters:
            self._step_parameter(param)

    def _step_parameter(self, param: Tensor) -> None:
        active = self._active(param)
        slot = self.state.get(id(param))
        if slot is None:
            slot = {
                "m": np.zeros_like(param.data),
                "v": np.zeros_like(param.data),
                "t": np.zeros(param.shape, dtype=np.int64),
            }
            self.state[id(param)] = slot
        if param.grad is None:
            self._clear_inactive(param, slot, active)
            return

        grad = np.asarray(param.grad, dtype=np.float64)
        if grad.shape != param.shape:
            raise ValueError(f"grad shape {grad.shape} does not match parameter shape {param.shape}")
        moment = slot["m"]
        second = slot["v"]
        step = slot["t"]
        if active is None:
            step = step + 1
            moment = self.beta1 * moment + (1.0 - self.beta1) * grad
            second = self.beta2 * second + (1.0 - self.beta2) * np.square(grad)
            corrected_m = moment / (1.0 - self.beta1**step)
            corrected_v = second / (1.0 - self.beta2**step)
            param.data = param.data - self.lr * corrected_m / (np.sqrt(corrected_v) + self.eps)
        else:
            step = np.where(active, step + 1, 0).astype(np.int64)
            moment = np.where(active, self.beta1 * moment + (1.0 - self.beta1) * grad, 0.0)
            second = np.where(active, self.beta2 * second + (1.0 - self.beta2) * np.square(grad), 0.0)
            first_denom = np.where(active, 1.0 - np.power(self.beta1, step), 1.0)
            second_denom = np.where(active, 1.0 - np.power(self.beta2, step), 1.0)
            corrected_m = moment / first_denom
            corrected_v = second / second_denom
            update = self.lr * corrected_m / (np.sqrt(corrected_v) + self.eps)
            param.data = np.where(active, param.data - update, 0.0)
        slot["m"] = moment
        slot["v"] = second
        slot["t"] = step

    def constrain(self, param: Tensor) -> None:
        """Zero the weight and the Adam state wherever the mask is off.

        Called when a connection is pruned, so the next step does not have to
        decay a moment that should already be gone.
        """
        active = self._active(param)
        if active is None:
            return
        slot = self.state.get(id(param))
        if slot is None:
            param.data = np.where(active, param.data, 0.0)
            return
        self._clear_inactive(param, slot, active)

    def _clear_inactive(self, param: Tensor, slot: dict[str, np.ndarray], active: np.ndarray | None) -> None:
        if active is None:
            return
        slot["m"] = np.where(active, slot["m"], 0.0)
        slot["v"] = np.where(active, slot["v"], 0.0)
        slot["t"] = np.where(active, slot["t"], 0).astype(np.int64)
        param.data = np.where(active, param.data, 0.0)

    def _active(self, param: Tensor) -> np.ndarray | None:
        mask = getattr(param, "mask", None)
        if mask is None:
            return None
        values = mask.data if isinstance(mask, Tensor) else np.asarray(mask, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        if values.shape != param.shape:
            raise ValueError(f"mask shape {values.shape} does not match parameter shape {param.shape}")
        return values == 1.0
