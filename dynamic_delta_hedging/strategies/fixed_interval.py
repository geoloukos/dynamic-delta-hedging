from numbers import Integral

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


class FixedIntervalStrategy:
    """
    Rebalance the futures hedge at fixed time intervals.

    Examples
    --------
    interval=1
        Rebalance at every simulation step.

    interval=5
        Rebalance every five simulation steps.

    The simulation step remains scalar because the
    hedging engine advances all paths through time
    simultaneously.

    The call delta and current hedge position may be
    scalars or NumPy arrays.
    """

    def __init__(self, interval: int, ):
        """
        Create a fixed-interval hedging strategy.

        Parameters
        ----------
        interval
            Positive integer number of simulation steps
            between scheduled rebalancing dates.
        """

        if (
            isinstance(interval, bool)
            or not isinstance(interval, Integral)
            or interval <= 0
        ):
            raise ValueError(
                "interval must be a positive integer."
            )

        self.interval = int(interval)

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
        strategy at the current simulation step.

        On a scheduled rebalancing step, the target is
        the current call delta.

        Between scheduled dates, the existing futures
        position remains unchanged.
        """

        if (
            step
            % self.interval
            == 0
        ):
            return _return_scalar_or_array(call_delta)

        return _return_scalar_or_array(current_position)