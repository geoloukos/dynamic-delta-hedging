from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.stats import t

from dynamic_delta_hedging.experiments import MonteCarloHedgingResult


_TIE_ABSOLUTE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PairedComparisonResult:
    """
    Store a paired comparison between two strategies
    evaluated on the same Monte Carlo paths.

    Positive differences favour strategy A:

        difference = terminal_pnl_A - terminal_pnl_B
    """

    strategy_a: str
    strategy_b: str
    n_paths: int
    mean_pnl_difference: float
    median_pnl_difference: float
    std_pnl_difference: float
    standard_error: float
    confidence_level: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    probability_a_outperforms_b: float
    probability_b_outperforms_a: float
    probability_of_tie: float


def _validate_confidence_level(confidence_level) -> float:
    """
    Validate and normalize a confidence level.
    """

    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, Real)
    ):
        raise TypeError(
            "confidence_level must be a real number."
        )

    value = float(confidence_level)

    if not np.isfinite(value):
        raise ValueError(
            "confidence_level must be finite."
        )

    if not (0.0 < value < 1.0):
        raise ValueError(
            "confidence_level must satisfy "
            "0 < confidence_level < 1."
        )

    return value


def _validate_strategy_name(strategy_name, argument_name: str, ) -> str:
    """
    Validate and normalize a strategy label.
    """

    if not isinstance(strategy_name, str):
        raise TypeError(f'{argument_name} must be a string.')

    value = strategy_name.strip()

    if not value:
        raise ValueError(f'{argument_name} cannot be empty.')

    return value


def _extract_terminal_pnls(results: MonteCarloHedgingResult, ) -> np.ndarray:
    """
    Extract and validate terminal Monte Carlo P&L.
    """

    if not isinstance(results, MonteCarloHedgingResult):
        raise TypeError(
            "results must be a "
            "MonteCarloHedgingResult."
        )

    raw_values = np.asarray(results.terminal_pnls)

    if np.issubdtype(raw_values.dtype, np.bool_):
        raise TypeError(
            "terminal_pnls must contain numeric values, "
            "not booleans."
        )

    try:
        terminal_pnls = np.asarray(results.terminal_pnls, dtype=float)

    except (TypeError, ValueError) as error:
        raise TypeError(
            "terminal_pnls must contain numeric values."
        ) from error

    if terminal_pnls.ndim != 1:
        raise ValueError(
            "terminal_pnls must be one-dimensional."
        )

    if terminal_pnls.size < 2:
        raise ValueError(
            "At least two Monte Carlo observations "
            "are required."
        )

    if not np.all(np.isfinite(terminal_pnls)):
        raise ValueError(
            "terminal_pnls must contain only "
            "finite values."
        )

    return terminal_pnls


def _extract_terminal_futures_prices(
    results: MonteCarloHedgingResult,
    expected_size: int,
) -> np.ndarray:
    """
    Extract terminal futures prices for an available
    paired-path alignment check.
    """

    try:
        terminal_prices = np.asarray(results.terminal_futures_prices, dtype=float)

    except (TypeError, ValueError) as error:
        raise TypeError(
            "terminal_futures_prices must contain "
            "numeric values."
        ) from error

    if terminal_prices.ndim != 1:
        raise ValueError(
            "terminal_futures_prices must be "
            "one-dimensional."
        )

    if terminal_prices.size != expected_size:
        raise ValueError(
            "terminal_futures_prices must match the "
            "number of terminal P&L observations."
        )

    if not np.all(np.isfinite(terminal_prices)):
        raise ValueError(
            "terminal_futures_prices must contain only "
            "finite values."
        )

    return terminal_prices


def compare_paired_strategies(
    strategy_a_results: MonteCarloHedgingResult,
    strategy_b_results: MonteCarloHedgingResult,
    strategy_a_name: str,
    strategy_b_name: str,
    confidence_level: float = 0.95,
) -> PairedComparisonResult:
    """
    Compare two strategies path by path.

    Both strategies must use the same simulated paths in
    the same order. Terminal futures prices are checked
    for exact alignment, but terminal values alone cannot
    prove that the complete intrapath trajectories match.

    The confidence interval is a paired t interval for
    the mean terminal P&L difference.
    """

    confidence_level = (_validate_confidence_level(confidence_level))

    strategy_a_name = (
        _validate_strategy_name(
            strategy_a_name,
            "strategy_a_name",
        )
    )

    strategy_b_name = (
        _validate_strategy_name(
            strategy_b_name,
            "strategy_b_name",
        )
    )

    pnl_a = _extract_terminal_pnls(strategy_a_results)

    pnl_b = _extract_terminal_pnls(strategy_b_results)

    if pnl_a.shape != pnl_b.shape:
        raise ValueError(
            "Paired strategy results must contain "
            "the same number of paths."
        )

    prices_a = (
        _extract_terminal_futures_prices(strategy_a_results, expected_size=pnl_a.size)
    )

    prices_b = (
        _extract_terminal_futures_prices(strategy_b_results, expected_size=pnl_b.size)
    )

    if not np.array_equal(prices_a, prices_b):
        raise ValueError(
            "Paired strategy results must originate "
            "from the same simulated paths in the "
            "same order."
        )

    differences = (pnl_a - pnl_b)

    n_paths = int(differences.size)

    mean_difference = float(np.mean(differences))

    median_difference = float(np.median(differences))

    std_difference = float(np.std(differences, ddof=1))

    standard_error = float(std_difference / np.sqrt(n_paths))

    if standard_error == 0.0:
        ci_lower = mean_difference
        ci_upper = mean_difference

    else:
        alpha = (1.0 - confidence_level)

        critical_value = float(t.ppf(1.0 - alpha / 2.0, df=n_paths - 1))

        margin = (critical_value * standard_error)

        ci_lower = (mean_difference - margin)

        ci_upper = (mean_difference + margin)

    ties = np.isclose(differences, 0.0, rtol=0.0, atol=_TIE_ABSOLUTE_TOLERANCE)

    a_outperforms = ((differences > 0.0) & ~ties)

    b_outperforms = ((differences < 0.0) & ~ties)

    return PairedComparisonResult(
        strategy_a=strategy_a_name,
        strategy_b=strategy_b_name,
        n_paths=n_paths,
        mean_pnl_difference=(mean_difference),
        median_pnl_difference=(median_difference),
        std_pnl_difference=(std_difference),
        standard_error=(standard_error),
        confidence_level=(confidence_level),
        confidence_interval_lower=float(ci_lower),
        confidence_interval_upper=float(ci_upper),
        probability_a_outperforms_b=float(np.mean(a_outperforms)),
        probability_b_outperforms_a=float(np.mean(b_outperforms)),
        probability_of_tie=float(np.mean(ties)),
    )
