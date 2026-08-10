from numbers import Real
import numpy as np
from numpy.typing import ArrayLike


def _return_scalar_or_array(values: ArrayLike):
    array = np.asarray(values, dtype=float)
    return float(array) if array.ndim == 0 else array


class DeltaToleranceStrategy:
    """Rehedge fully to Black-76 delta when delta deviation exceeds a tolerance."""

    def __init__(self, tolerance: float):
        if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
            raise TypeError("tolerance must be a number.")
        if not np.isfinite(tolerance):
            raise ValueError("tolerance must be finite.")
        if not 0.0 < tolerance < 1.0:
            raise ValueError("tolerance must lie strictly between 0 and 1.")
        self.tolerance = float(tolerance)

    def get_target_position(
        self, step, call_delta, current_position, futures_price, tau, config
    ):
        if step == 0:
            return _return_scalar_or_array(call_delta)
        delta = np.asarray(call_delta, dtype=float)
        current = np.asarray(current_position, dtype=float)
        try:
            delta, current = np.broadcast_arrays(delta, current)
        except ValueError as error:
            raise ValueError(
                "call_delta and current_position must be broadcast-compatible."
            ) from error
        target = np.where(np.abs(delta - current) > self.tolerance, delta, current)
        return _return_scalar_or_array(target)
