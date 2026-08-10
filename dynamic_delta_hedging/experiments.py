from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from dynamic_delta_hedging.config import HedgingConfig
from dynamic_delta_hedging.hedging_engine import BatchHedgingResult, hedge_short_call_paths
from dynamic_delta_hedging.metrics import calculate_hedging_metrics
from dynamic_delta_hedging.simulation import simulate_futures_paths
from dynamic_delta_hedging.strategy_registry import (
    DEFAULT_STRATEGY_SPECS,
    StrategySpec,
    build_strategy,
)


@dataclass(frozen=True)
class MonteCarloHedgingResult:
    option_premium: float
    terminal_pnls: np.ndarray
    call_payoffs: np.ndarray
    total_hedge_pnls: np.ndarray
    total_transaction_costs: np.ndarray
    numbers_of_trades: np.ndarray
    total_turnovers: np.ndarray
    terminal_futures_prices: np.ndarray
    theoretical_option_price: float = 0.0
    option_bid_price: float = 0.0
    option_ask_price: float = 0.0
    option_sale_proceeds: float = 0.0
    initial_option_edge: float = 0.0


@dataclass(frozen=True)
class FullExperimentResult:
    """Research paths plus one canonical result mapping for the tested strategies."""

    futures_paths: np.ndarray
    results_by_strategy_id: dict[str, MonteCarloHedgingResult]
    strategy_specs: tuple[StrategySpec, ...]


def _validate_experiment_paths(futures_paths, config: HedgingConfig, ) -> np.ndarray:
    paths = np.asarray(futures_paths, dtype=float)
    if paths.ndim != 2:
        raise ValueError("futures_paths must be two-dimensional.")

    expected_shape = (config.n_paths, config.n_steps + 1)
    if paths.shape != expected_shape:
        raise ValueError(f"futures_paths must have shape {expected_shape}.")

    return paths


def _convert_batch_result(batch_result: BatchHedgingResult, ) -> MonteCarloHedgingResult:
    return MonteCarloHedgingResult(
        option_premium=batch_result.option_premium,
        theoretical_option_price=batch_result.theoretical_option_price,
        option_bid_price=batch_result.option_bid_price,
        option_ask_price=batch_result.option_ask_price,
        option_sale_proceeds=batch_result.option_sale_proceeds,
        initial_option_edge=batch_result.initial_option_edge,
        terminal_pnls=batch_result.terminal_pnls,
        call_payoffs=batch_result.call_payoffs,
        total_hedge_pnls=batch_result.total_hedge_pnls,
        total_transaction_costs=batch_result.total_transaction_costs,
        numbers_of_trades=batch_result.numbers_of_trades,
        total_turnovers=batch_result.total_turnovers,
        terminal_futures_prices=batch_result.terminal_futures_prices,
    )


def run_strategy_on_paths(
    futures_paths,
    strategy,
    config: HedgingConfig,
    futures_spread=None,
) -> MonteCarloHedgingResult:
    paths = _validate_experiment_paths(futures_paths, config)
    if futures_spread is None:
        batch_result = hedge_short_call_paths(
            futures_paths=paths,
            strategy=strategy,
            config=config,
        )
    else:
        batch_result = hedge_short_call_paths(
            futures_paths=paths,
            strategy=strategy,
            config=config,
            futures_spread=futures_spread,
        )
    return _convert_batch_result(batch_result)


def _validate_specs(strategy_specs) -> tuple[StrategySpec, ...]:
    try:
        specs = tuple(strategy_specs)
    except TypeError as error:
        raise TypeError(
            "strategy_specs must be an iterable of StrategySpec objects."
        ) from error

    if not specs:
        raise ValueError("strategy_specs cannot be empty.")
    if not all(isinstance(spec, StrategySpec) for spec in specs):
        raise TypeError("strategy_specs must contain only StrategySpec objects.")

    strategy_ids = [spec.strategy_id for spec in specs]
    if len(strategy_ids) != len(set(strategy_ids)):
        raise ValueError("strategy_specs must contain unique strategy IDs.")

    return specs


def run_strategy_specs_on_paths(
    futures_paths,
    strategy_specs,
    config: HedgingConfig,
) -> dict[str, MonteCarloHedgingResult]:
    specs = _validate_specs(strategy_specs)
    return {
        spec.strategy_id: run_strategy_on_paths(
            futures_paths=futures_paths,
            strategy=build_strategy(spec),
            config=config,
        )
        for spec in specs
    }


def run_all_experiments(
    config: HedgingConfig,
    strategy_specs=None,
) -> FullExperimentResult:
    """Run the canonical A-Y universe, or an explicit strategy subset, on common paths."""
    specs = (
        DEFAULT_STRATEGY_SPECS
        if strategy_specs is None
        else _validate_specs(strategy_specs)
    )

    futures_paths = simulate_futures_paths(config=config)
    results_by_strategy_id = run_strategy_specs_on_paths(
        futures_paths=futures_paths,
        strategy_specs=specs,
        config=config,
    )

    return FullExperimentResult(
        futures_paths=futures_paths,
        results_by_strategy_id=results_by_strategy_id,
        strategy_specs=tuple(specs),
    )


def create_strategy_universe_metrics_table(
    experiment: FullExperimentResult,
    tail_probability: float = 0.05,
) -> pd.DataFrame:
    if not isinstance(experiment, FullExperimentResult):
        raise TypeError("experiment must be a FullExperimentResult.")

    rows = []
    for spec in experiment.strategy_specs:
        metrics = calculate_hedging_metrics(
            experiment.results_by_strategy_id[spec.strategy_id],
            tail_probability=tail_probability,
        )
        rows.append(
            {
                "strategy_id": spec.strategy_id,
                "strategy_name": spec.display_name,
                "strategy_family": spec.family,
                "strategy_parameter": spec.parameter_name,
                "strategy_parameter_value": spec.parameter_value,
                **asdict(metrics),
            }
        )

    return pd.DataFrame(rows)
