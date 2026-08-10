from dataclasses import asdict, dataclass, replace
from numbers import Integral

import numpy as np
import pandas as pd

from dynamic_delta_hedging.campaign import REALIZED_VOL_SUPPORT_MULTIPLIERS
from dynamic_delta_hedging.config import HedgingConfig
from dynamic_delta_hedging.experiments import MonteCarloHedgingResult, run_strategy_on_paths
from dynamic_delta_hedging.metrics import HedgingMetrics, calculate_hedging_metrics
from dynamic_delta_hedging.selection import (
    StrategySelectionResult,
    require_mandate_eligible_strategy_ids,
)
from dynamic_delta_hedging.simulation import simulate_futures_paths_from_shocks
from dynamic_delta_hedging.strategy_registry import build_strategy, get_strategy_spec


@dataclass(frozen=True)
class RobustnessScenario:
    name: str
    realized_volatility_multiplier: float
    realized_sigma: float


@dataclass(frozen=True)
class RobustnessExperimentResult:
    scenario: RobustnessScenario
    strategy_id: str
    strategy_name: str
    hedging_result: MonteCarloHedgingResult
    metrics: HedgingMetrics


def build_robustness_scenarios(base_config: HedgingConfig) -> list[RobustnessScenario]:
    """Return exactly the lower/baseline/higher realized-volatility scenarios."""
    if not isinstance(base_config, HedgingConfig):
        raise TypeError("base_config must be a HedgingConfig.")

    low, high = REALIZED_VOL_SUPPORT_MULTIPLIERS
    return [
        RobustnessScenario(
            "lower_realized_volatility",
            low,
            low * base_config.sigma,
        ),
        RobustnessScenario(
            "baseline",
            1.0,
            base_config.sigma,
        ),
        RobustnessScenario(
            "higher_realized_volatility",
            high,
            high * base_config.sigma,
        ),
    ]


def _resolve_n_paths(n_paths, base_n_paths: int) -> int:
    value = base_n_paths if n_paths is None else n_paths
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("n_paths must be an integer.")
    value = int(value)
    if value < 2:
        raise ValueError("n_paths must be greater than or equal to 2.")
    return value


def _resolve_seed(seed, base_seed):
    value = base_seed if seed is None else seed
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("seed must be an integer or None.")
    value = int(value)
    if value < 0:
        raise ValueError("seed must be non-negative.")
    return value


def run_robustness_analysis(
    base_config: HedgingConfig,
    research_selection: StrategySelectionResult,
    n_paths: int | None = None,
    seed: int | None = None,
    tail_probability: float = 0.05,
    verbose: bool = True,
) -> dict[tuple[str, str], RobustnessExperimentResult]:
    """Run paired realized-volatility sensitivity for every mandate survivor.

    Black–76 pricing/Greeks keep ``base_config.sigma`` throughout. Only the
    futures-path generator receives the scenario's realized volatility. One
    standardized shock matrix is shared across every scenario and strategy.
    The futures execution spread remains the fixed value in ``base_config``.
    """
    if not isinstance(base_config, HedgingConfig):
        raise TypeError("base_config must be a HedgingConfig.")
    if not isinstance(research_selection, StrategySelectionResult):
        raise TypeError("research_selection must be a StrategySelectionResult.")
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean.")

    eligible_ids = require_mandate_eligible_strategy_ids(research_selection)
    analysis_config = replace(
        base_config,
        n_paths=_resolve_n_paths(n_paths, base_config.n_paths),
        seed=_resolve_seed(seed, base_config.seed),
    )
    scenarios = build_robustness_scenarios(analysis_config)

    rng = np.random.default_rng(analysis_config.seed)
    common_shocks = rng.standard_normal((analysis_config.n_paths, analysis_config.n_steps))
    paths_by_scenario = {
        scenario.name: simulate_futures_paths_from_shocks(
            analysis_config,
            common_shocks,
            realized_sigma=scenario.realized_sigma,
        )
        for scenario in scenarios
    }

    results: dict[tuple[str, str], RobustnessExperimentResult] = {}
    total_runs = len(eligible_ids) * len(scenarios)
    run_index = 0
    for scenario in scenarios:
        paths = paths_by_scenario[scenario.name]
        for strategy_id in eligible_ids:
            run_index += 1
            spec = get_strategy_spec(strategy_id)
            if verbose:
                print(
                    f"[{run_index}/{total_runs}] Running {strategy_id} under {scenario.name}..."
                )
            hedging_result = run_strategy_on_paths(
                paths,
                build_strategy(spec),
                analysis_config,
            )
            metrics = calculate_hedging_metrics(
                hedging_result,
                tail_probability=tail_probability,
            )
            results[(strategy_id, scenario.name)] = RobustnessExperimentResult(
                scenario=scenario,
                strategy_id=strategy_id,
                strategy_name=spec.display_name,
                hedging_result=hedging_result,
                metrics=metrics,
            )
    return results


def create_robustness_metrics_table(robustness_results) -> pd.DataFrame:
    if not isinstance(robustness_results, dict):
        raise TypeError("robustness_results must be a dictionary.")
    if not robustness_results:
        raise ValueError("robustness_results cannot be empty.")

    rows = []
    for key, result in robustness_results.items():
        if not isinstance(result, RobustnessExperimentResult):
            raise TypeError("Every robustness result must be a RobustnessExperimentResult.")
        spec = get_strategy_spec(result.strategy_id)
        rows.append(
            {
                "scenario": result.scenario.name,
                "realized_volatility_multiplier": result.scenario.realized_volatility_multiplier,
                "realized_sigma": result.scenario.realized_sigma,
                "strategy_id": result.strategy_id,
                "strategy_name": result.strategy_name,
                "strategy_family": spec.family,
                "strategy_parameter": spec.parameter_name,
                "strategy_parameter_value": spec.parameter_value,
                **asdict(result.metrics),
            }
        )
    table = pd.DataFrame(rows)
    scenario_order = {
        "lower_realized_volatility": 0,
        "baseline": 1,
        "higher_realized_volatility": 2,
    }
    table["_scenario_order"] = table["scenario"].map(scenario_order)
    return (
        table.sort_values(["strategy_id", "_scenario_order"])
        .drop(columns="_scenario_order")
        .reset_index(drop=True)
    )
