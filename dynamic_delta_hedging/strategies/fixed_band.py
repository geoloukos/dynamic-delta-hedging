from numbers import Real

import numpy as np
from numpy.typing import ArrayLike


def _return_scalar_or_array(values: ArrayLike, ):
    """
    Return a Python float for scalar calculations and
    a NumPy array for vectorized calculations.
    """

    array = np.asarray(values, dtype=float)

    if array.ndim == 0:
        return float(array)

    return array


class FixedBandStrategy:
    """
    Rebalance the futures hedge only when the existing
    position lies outside a fixed no-transaction region
    around the current call delta.

    The call delta and current hedge position may be
    scalars or broadcast-compatible NumPy arrays.
    """

    def __init__(self, band_width: float, ):
        """
        Create a fixed-band hedging strategy.

        Parameters
        ----------
        band_width
            Positive half-width of the no-transaction
            region around the current call delta.

            The width must be strictly between zero and
            one.
        """

        if (
            isinstance(band_width, bool)
            or not isinstance(band_width, Real)
        ):
            raise TypeError(
                "band_width must be a number."
            )

        if not np.isfinite(band_width):
            raise ValueError(
                "band_width must be finite."
            )

        if band_width <= 0.0:
            raise ValueError(
                "band_width must be positive."
            )

        if band_width >= 1.0:
            raise ValueError(
                "band_width must be smaller than 1."
            )

        self.band_width = float(band_width)

    def get_target_position(
        self,
        step,
        call_delta,
        current_position,
        futures_price,
        tau,
        config,
    ):
        """
        Return the futures position desired by the
        fixed-band strategy.

        At time zero, every path begins from the exact
        delta hedge.

        At later steps:

        - positions below the region move to its lower
          boundary,
        - positions above the region move to its upper
          boundary,
        - positions inside the region remain unchanged.
        """

        if step == 0:
            return _return_scalar_or_array(call_delta)

        call_deltas = np.asarray(call_delta, dtype=float)

        current_positions = np.asarray(current_position, dtype=float)

        try:
            (
                call_deltas,
                current_positions,
            ) = np.broadcast_arrays(call_deltas, current_positions)

        except ValueError as error:
            raise ValueError(
                "call_delta and current_position must "
                "be broadcast-compatible."
            ) from error

        lower_boundaries = np.clip(call_deltas - self.band_width, 0.0, 1.0)

        upper_boundaries = np.clip(call_deltas + self.band_width, 0.0, 1.0)

        target_positions = np.where(
            current_positions
            < lower_boundaries,
            lower_boundaries,
            np.where(
                current_positions
                > upper_boundaries,
                upper_boundaries,
                current_positions,
            ),
        )

        return _return_scalar_or_array(target_positions)