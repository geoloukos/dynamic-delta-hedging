from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from dynamic_delta_hedging.config import HedgingConfig


def _validate_mid_price(mid_price: ArrayLike) -> np.ndarray:
    values = np.asarray(mid_price, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("mid_price must contain only finite values.")
    if np.any(values <= 0.0):
        raise ValueError("mid_price must be positive.")
    return values


def _validate_option_value(option_value: ArrayLike) -> np.ndarray:
    values = np.asarray(option_value, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("option_value must contain only finite values.")
    if np.any(values < 0.0):
        raise ValueError("option_value cannot be negative.")
    return values


def _validate_position_change(position_change: ArrayLike) -> np.ndarray:
    values = np.asarray(position_change, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("position_change must contain only finite values.")
    return values


def _return_scalar_or_array(values: np.ndarray):
    return float(values) if values.ndim == 0 else values


def futures_transaction_cost(
    position_change: ArrayLike,
    mid_price: ArrayLike,
    config: HedgingConfig,
    futures_spread: ArrayLike | None = None,
):
    """Contract-level EUR cost of crossing the futures spread.

    ``futures_spread`` is an optional contemporaneous execution spread.  When
    omitted, the historical Research engine uses ``config.futures_spread``
    exactly as before.
    """
    change = _validate_position_change(position_change)
    mid = _validate_mid_price(mid_price)
    spread = config.futures_spread if futures_spread is None else np.asarray(futures_spread, dtype=float)
    if not np.all(np.isfinite(spread)):
        raise ValueError("futures_spread must contain only finite values.")
    if np.any((spread < 0.0) | (spread >= 2.0)):
        raise ValueError("futures_spread must satisfy 0 <= spread < 2.")
    change, mid, spread = np.broadcast_arrays(change, mid, spread)
    cost = (np.abs(change) * mid * spread / 2.0 * config.contract_multiplier)
    return _return_scalar_or_array(cost)


def option_bid_price(theoretical_value: ArrayLike, config: HedgingConfig):
    value = _validate_option_value(theoretical_value)
    bid = value * (1.0 - config.option_spread / 2.0)
    return _return_scalar_or_array(bid)


def option_ask_price(theoretical_value: ArrayLike, config: HedgingConfig):
    value = _validate_option_value(theoretical_value)
    ask = value * (1.0 + config.option_spread / 2.0)
    return _return_scalar_or_array(ask)


@dataclass(frozen=True)
class OptionQuote:
    theoretical_price: float
    bid_price: float
    ask_price: float
    fair_contract_value: float
    sale_proceeds: float
    initial_edge: float


def build_option_quote(theoretical_value: float, config: HedgingConfig) -> OptionQuote:
    value = float(_validate_option_value(theoretical_value))
    bid = float(option_bid_price(value, config))
    ask = float(option_ask_price(value, config))
    multiplier = config.contract_multiplier
    return OptionQuote(
        theoretical_price=value,
        bid_price=bid,
        ask_price=ask,
        fair_contract_value=value * multiplier,
        sale_proceeds=ask * multiplier,
        initial_edge=(ask - value) * multiplier,
    )
