from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np
import pandas as pd
from scipy.stats import t

from dynamic_delta_hedging.config import HedgingConfig
from dynamic_delta_hedging.experiments import MonteCarloHedgingResult, run_strategy_on_paths
from dynamic_delta_hedging.simulation import simulate_futures_paths
from dynamic_delta_hedging.strategies.fixed_interval import FixedIntervalStrategy


@dataclass(frozen=True)
class FrictionlessConvergenceGridResult:
    """Result of one rebalancing grid in the convergence experiment."""

    config: HedgingConfig
    hedging_results: MonteCarloHedgingResult


def _prepare_n_steps(n_steps_values) -> list[int]:
    """Return sorted, distinct positive rebalancing-step counts."""

    values = list(n_steps_values)

    if len(values) < 2:
        raise ValueError("At least two rebalancing grids are required.")

    if any(
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value <= 0
        for value in values
    ):
        raise ValueError("Rebalancing-step counts must be positive integers.")

    prepared = sorted({int(value) for value in values})

    if len(prepared) < 2:
        raise ValueError("At least two distinct rebalancing grids are required.")

    return prepared


def run_frictionless_convergence(
    base_config: HedgingConfig,
    n_steps_values,
    n_paths: int | None = None,
    seed: int | None = None,
    verbose: bool = True,
) -> dict[int, FrictionlessConvergenceGridResult]:
    """
    Run the delta-hedging engine on progressively finer frictionless grids.

    Transaction costs and futures drift are removed, and the hedge is reset
    to the Black-76 delta at every available time step.
    """

    if not isinstance(base_config, HedgingConfig):
        raise TypeError("base_config must be a HedgingConfig.")

    n_steps_values = _prepare_n_steps(n_steps_values)
    n_paths = base_config.n_paths if n_paths is None else n_paths
    seed = base_config.seed if seed is None else seed

    if isinstance(n_paths, bool) or not isinstance(n_paths, Integral) or n_paths < 2:
        raise ValueError("n_paths must be an integer greater than or equal to 2.")

    strategy = FixedIntervalStrategy(interval=1)
    results = {}

    for n_steps in n_steps_values:
        config = replace(
            base_config,
            futures_spread=0.0,
            option_spread=0.0,
            mu=0.0,
            n_steps=n_steps,
            n_paths=int(n_paths),
            seed=seed,
        )

        if verbose:
            print(f"Running frictionless convergence with {n_steps} steps...")

        futures_paths = simulate_futures_paths(config=config)
        hedging_results = run_strategy_on_paths(
            futures_paths=futures_paths,
            strategy=strategy,
            config=config,
        )

        results[n_steps] = FrictionlessConvergenceGridResult(
            config=config,
            hedging_results=hedging_results,
        )

    return results


def create_convergence_table(
    convergence_results: dict[int, FrictionlessConvergenceGridResult],
) -> pd.DataFrame:
    """Summarize the bias and dispersion of terminal hedging P&L by grid."""

    if not convergence_results:
        raise ValueError("convergence_results cannot be empty.")

    rows = []

    for n_steps, result in sorted(convergence_results.items()):
        terminal_pnls = np.asarray(
            result.hedging_results.terminal_pnls,
            dtype=float,
        )

        n_observations = terminal_pnls.size
        if n_observations < 2:
            raise ValueError("Each convergence grid needs at least two paths.")

        mean_pnl = float(np.mean(terminal_pnls))
        std_pnl = float(np.std(terminal_pnls, ddof=1))
        standard_error = std_pnl / np.sqrt(n_observations)
        critical_value = float(t.ppf(0.975, df=n_observations - 1))
        confidence_margin = critical_value * standard_error

        rows.append(
            {
                "n_steps": int(n_steps),
                "dt": result.config.dt,
                "mean_pnl": mean_pnl,
                "ci_lower_95": mean_pnl - confidence_margin,
                "ci_upper_95": mean_pnl + confidence_margin,
                "std_pnl": std_pnl,
            }
        )

    return pd.DataFrame(rows)