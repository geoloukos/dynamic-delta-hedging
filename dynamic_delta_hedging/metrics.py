from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class HedgingMetrics:
    """
    Store summary statistics for one hedging strategy
    across all Monte Carlo paths.
    """

    mean_pnl: float
    std_pnl: float
    expected_shortfall: float
    mean_transaction_cost: float
    mean_number_of_trades: float
    mean_turnover: float


def _validate_tail_probability(tail_probability) -> float:
    """
    Validate and normalize the lower-tail probability.
    """

    if (
        isinstance(tail_probability, bool)
        or not isinstance(tail_probability, Real)
    ):
        raise TypeError(
            "tail_probability must be a real number."
        )

    probability = float(tail_probability)

    if not np.isfinite(probability):
        raise ValueError(
            "tail_probability must be finite."
        )

    if not (0.0 < probability < 1.0):
        raise ValueError(
            "tail_probability must be strictly "
            "between 0 and 1."
        )

    return probability


def _validate_result_array(
    values: ArrayLike,
    array_name: str,
    *,
    nonnegative: bool = False,
    integer_valued: bool = False,
) -> np.ndarray:
    """
    Convert one path-level result array to float and
    validate its shape and values.

    Every summary calculation requires at least two
    observations because terminal P&L volatility uses
    sample standard deviation with ddof=1.
    """

    try:
        raw_values = np.asarray(values)

        array = np.asarray(values, dtype=float)

    except (TypeError, ValueError) as error:
        raise TypeError(f'{array_name} must contain numeric values.') from error

    if np.issubdtype(raw_values.dtype, np.bool_):
        raise TypeError(
            f"{array_name} must contain numeric values, "
            "not booleans."
        )

    if array.ndim != 1:
        raise ValueError(f'{array_name} must be one-dimensional.')

    if array.size < 2:
        raise ValueError(
            f"{array_name} must contain at least "
            "two observations."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{array_name} must contain only "
            "finite values."
        )

    if (
        nonnegative
        and np.any(array < 0.0)
    ):
        raise ValueError(
            f"{array_name} cannot contain "
            "negative values."
        )

    if (
        integer_valued
        and np.any(array != np.floor(array))
    ):
        raise ValueError(
            f"{array_name} must contain "
            "integer-valued observations."
        )

    return array


def _extract_and_validate_result_arrays(results) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Extract the arrays required by the metrics
    calculation and validate their shared length.
    """

    required_attributes = (
        "terminal_pnls",
        "total_transaction_costs",
        "numbers_of_trades",
        "total_turnovers",
    )

    missing_attributes = [
        attribute
        for attribute in required_attributes
        if not hasattr(results, attribute)
    ]

    if missing_attributes:
        missing_text = ", ".join(missing_attributes)

        raise TypeError(
            "results is missing required attributes: "
            f"{missing_text}."
        )

    terminal_pnls = (
        _validate_result_array(
            values=results.terminal_pnls,
            array_name="terminal_pnls",
        )
    )

    transaction_costs = (
        _validate_result_array(
            values=(results.total_transaction_costs),
            array_name=(
                "total_transaction_costs"
            ),
            nonnegative=True,
        )
    )

    numbers_of_trades = (
        _validate_result_array(
            values=results.numbers_of_trades,
            array_name="numbers_of_trades",
            nonnegative=True,
            integer_valued=True,
        )
    )

    turnovers = _validate_result_array(
        values=results.total_turnovers,
        array_name="total_turnovers",
        nonnegative=True,
    )

    expected_length = (terminal_pnls.size)

    remaining_arrays = (transaction_costs, numbers_of_trades, turnovers)

    if any(
        array.size
        != expected_length
        for array in remaining_arrays
    ):
        raise ValueError(
            "All result arrays must have the same length."
        )

    return (terminal_pnls, transaction_costs, numbers_of_trades, turnovers)


def calculate_hedging_metrics(results, tail_probability: float = 0.05, ) -> HedgingMetrics:
    """
    Calculate summary performance metrics for one
    hedging strategy.

    Expected Shortfall convention
    -----------------------------
    For n observations, the empirical lower tail
    contains:

        ceil(tail_probability * n)

    observations, with at least one observation always
    included.

    Expected Shortfall is the arithmetic mean of those
    worst terminal P&L outcomes.
    """

    probability = (_validate_tail_probability(tail_probability))

    (
        terminal_pnls,
        transaction_costs,
        numbers_of_trades,
        turnovers,
    ) = _extract_and_validate_result_arrays(results)

    number_of_observations = (terminal_pnls.size)

    tail_count = max(1, int(np.ceil(probability * number_of_observations)))

    # Identify the worst lower-tail observations without
    # sorting the complete Monte Carlo sample.
    lower_tail = np.partition(terminal_pnls, tail_count - 1)[:tail_count]

    return HedgingMetrics(
        mean_pnl=float(np.mean(terminal_pnls)),
        std_pnl=float(np.std(terminal_pnls, ddof=1)),
        expected_shortfall=float(np.mean(lower_tail)),
        mean_transaction_cost=float(np.mean(transaction_costs)),
        mean_number_of_trades=float(np.mean(numbers_of_trades)),
        mean_turnover=float(np.mean(turnovers)),
    )
