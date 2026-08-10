from numbers import Real
import numpy as np
from numpy.typing import ArrayLike


def _return_scalar_or_array(values: ArrayLike):
    array = np.asarray(values, dtype=float)
    return float(array) if array.ndim == 0 else array


class AssetToleranceStrategy:
    """Rehedge fully to Black-76 delta after a sufficiently large futures move."""

    requires_last_hedge_price = True

    def __init__(self, tolerance: float):
        if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
            raise TypeError("tolerance must be a number.")
        if not np.isfinite(tolerance):
            raise ValueError("tolerance must be finite.")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive.")
        self.tolerance = float(tolerance)

    def get_target_position(
        self,
        step,
        call_delta,
        current_position,
        futures_price,
        tau,
        config,
        last_hedge_price=None,
    ):
        if step == 0:
            return _return_scalar_or_array(call_delta)
        if last_hedge_price is None:
            raise ValueError("last_hedge_price is required for AssetToleranceStrategy.")

        delta = np.asarray(call_delta, dtype=float)
        current = np.asarray(current_position, dtype=float)
        price = np.asarray(futures_price, dtype=float)
        reference = np.asarray(last_hedge_price, dtype=float)
        try:
            delta, current, price, reference = np.broadcast_arrays(
                delta, current, price, reference
            )
        except ValueError as error:
            raise ValueError(
                "call_delta, current_position, futures_price and last_hedge_price "
                "must be broadcast-compatible."
            ) from error
        if np.any(~np.isfinite(reference)) or np.any(reference <= 0.0):
            raise ValueError("last_hedge_price must contain positive finite values.")
        move = np.abs((price - reference) / reference)
        target = np.where(move > self.tolerance, delta, current)
        return _return_scalar_or_array(target)
