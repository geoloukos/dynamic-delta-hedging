from numbers import Real

import numpy as np
from numpy.typing import ArrayLike

from dynamic_delta_hedging.pricing import black76_call_gamma


def _return_scalar_or_array(values: ArrayLike, ):
    """
    Return a Python float for scalar calculations and
    a NumPy array for vectorized calculations.
    """

    array = np.asarray(values, dtype=float)

    if array.ndim == 0:
        return float(array)

    return array


class WWInspiredStrategy:
    """
    Gamma-based dynamic no-transaction band inspired by
    the Whalley-Wilmott asymptotic hedging strategy.

    Important
    ---------
    The original Whalley-Wilmott derivation concerns
    options hedged with the underlying stock.

    This implementation adapts the same functional form
    to options on futures by using:

    - the futures price instead of the spot price,
    - Black-76 delta,
    - Black-76 gamma.

    It should therefore be described as WW-inspired,
    not as an exact futures-specific optimum.
    """

    uses_current_futures_spread = True

    def __init__(self, risk_aversion: float, ):
        """
        Create a WW-inspired hedging strategy.

        Higher risk aversion produces a narrower
        no-transaction region and therefore more
        frequent rebalancing.
        """

        if (
            isinstance(risk_aversion, bool)
            or not isinstance(risk_aversion, Real)
        ):
            raise TypeError(
                "risk_aversion must be a number."
            )

        if not np.isfinite(risk_aversion):
            raise ValueError(
                "risk_aversion must be finite."
            )

        if risk_aversion <= 0.0:
            raise ValueError(
                "risk_aversion must be positive."
            )

        self.risk_aversion = float(risk_aversion)

    def calculate_half_width(self, futures_price, tau, config, futures_spread=None):
        """
        Calculate the dynamic half-width of the
        no-transaction region.

        The futures price and remaining maturity may be
        scalars or broadcast-compatible NumPy arrays.

        The adapted semi-width is:

            [
                3 * c * F * exp(-r * tau)
                --------------------------
                    2 * risk_aversion
            ] ** (1 / 3)

            * abs(Gamma) ** (2 / 3)

        where c is one half of the full bid-ask spread.
        """

        futures_prices = np.asarray(futures_price, dtype=float)

        maturities = np.asarray(tau, dtype=float)

        try:
            (futures_prices, maturities) = np.broadcast_arrays(futures_prices, maturities)

        except ValueError as error:
            raise ValueError(
                "futures_price and tau must be "
                "broadcast-compatible."
            ) from error

        # black76_call_gamma validates the futures
        # prices and maturities.
        call_gammas = np.asarray(
            black76_call_gamma(F=futures_prices, tau=maturities, config=config),
            dtype=float,
        )

        spread = config.futures_spread if futures_spread is None else np.asarray(futures_spread, dtype=float)
        if np.any(~np.isfinite(spread)) or np.any((spread < 0.0) | (spread >= 2.0)):
            raise ValueError("futures_spread must satisfy 0 <= spread < 2 and be finite.")
        proportional_cost_rate = spread / 2.0

        discount_factors = np.exp(-config.r * maturities)

        cube_root_component = np.cbrt(
            (3.0 * proportional_cost_rate * futures_prices * discount_factors)
            / (2.0 * self.risk_aversion)
        )

        gamma_component = np.power(np.abs(call_gammas), 2.0 / 3.0)

        half_widths = (cube_root_component * gamma_component)

        return _return_scalar_or_array(half_widths)

    def get_target_position(
        self,
        step,
        call_delta,
        current_position,
        futures_price,
        tau,
        config,
        current_futures_spread=None,
    ):
        """
        Return the futures position desired by the
        WW-inspired strategy.

        At time zero, every path begins from the exact
        delta hedge.

        At later steps, the strategy trades only to the
        nearest boundary when the current hedge lies
        outside the dynamic no-transaction region.
        """

        if step == 0:
            return _return_scalar_or_array(call_delta)

        half_widths = np.asarray(
            self.calculate_half_width(
                futures_price=futures_price,
                tau=tau,
                config=config,
                futures_spread=current_futures_spread,
            ),
            dtype=float,
        )

        call_deltas = np.asarray(call_delta, dtype=float)

        current_positions = np.asarray(current_position, dtype=float)

        try:
            (
                call_deltas,
                current_positions,
                half_widths,
            ) = np.broadcast_arrays(call_deltas, current_positions, half_widths)

        except ValueError as error:
            raise ValueError(
                "call_delta, current_position, "
                "futures_price and tau must be "
                "broadcast-compatible."
            ) from error

        lower_boundaries = np.clip(call_deltas - half_widths, 0.0, 1.0)

        upper_boundaries = np.clip(call_deltas + half_widths, 0.0, 1.0)

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