from itertools import combinations
from numbers import Integral, Real

import numpy as np
import pandas as pd

from dynamic_delta_hedging.experiments import FullExperimentResult
from dynamic_delta_hedging.selection import (
    StrategySelectionResult,
    require_mandate_eligible_strategy_ids,
)
from dynamic_delta_hedging.statistical_analysis import compare_paired_strategies


def _pareto_efficient_mask_2d(
    first_values,
    second_values,
    *,
    first_minimize: bool,
    second_minimize: bool,
):
    """Return a 2D Pareto-efficient mask for explicit objective directions."""
    first = np.asarray(first_values, dtype=float)
    second = np.asarray(second_values, dtype=float)

    if first.shape != second.shape:
        raise ValueError("Pareto objective arrays must have the same shape.")
    if first.ndim != 1:
        raise ValueError("Pareto objective arrays must be one-dimensional.")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("Pareto objective arrays must contain only finite values.")

    first_objective = first if first_minimize else -first
    second_objective = second if second_minimize else -second

    efficient = np.ones(first.size, dtype=bool)
    for i in range(first.size):
        dominated = np.any(
            (first_objective <= first_objective[i])
            & (second_objective <= second_objective[i])
            & (
                (first_objective < first_objective[i])
                | (second_objective < second_objective[i])
            )
        )
        efficient[i] = not dominated

    return efficient


def add_cost_es_frontier_flag(eligible_metrics: pd.DataFrame) -> pd.DataFrame:
    """Add the primary Cost–Tail-Loss 5% efficient-frontier classification.

    Tail Loss 5% is defined as ``-ES_5%(P&L)`` so both axes are minimized and
    the economically preferred direction is lower-left.
    """
    if not isinstance(eligible_metrics, pd.DataFrame):
        raise TypeError("eligible_metrics must be a pandas DataFrame.")

    required = {
        "strategy_id",
        "mean_transaction_cost",
        "expected_shortfall",
    }
    missing = required - set(eligible_metrics.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if eligible_metrics.empty:
        raise ValueError("eligible_metrics cannot be empty.")

    table = eligible_metrics.copy().reset_index(drop=True)
    costs = table["mean_transaction_cost"].to_numpy(dtype=float)
    tail_loss = -table["expected_shortfall"].to_numpy(dtype=float)
    if not np.all(np.isfinite(costs)) or not np.all(np.isfinite(tail_loss)):
        raise ValueError("Cost and Expected Shortfall values must be finite.")

    table["tail_loss_5"] = tail_loss
    table["cost_es_efficient"] = _pareto_efficient_mask_2d(
        costs,
        tail_loss,
        first_minimize=True,
        second_minimize=True,
    )
    return table


def cost_es_frontier_strategy_ids(frontier_table: pd.DataFrame) -> tuple[str, ...]:
    """Return strategy IDs on the primary Cost–ES frontier in table order."""
    if not isinstance(frontier_table, pd.DataFrame):
        raise TypeError("frontier_table must be a pandas DataFrame.")
    required = {"strategy_id", "cost_es_efficient"}
    missing = required - set(frontier_table.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return tuple(
        frontier_table.loc[
            frontier_table["cost_es_efficient"].astype(bool), "strategy_id"
        ].astype(str)
    )


def _validate_tail_probability(value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("tail_probability must be a real number.")
    value = float(value)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("tail_probability must be strictly between 0 and 1.")
    return value


def _validate_n_bootstrap(value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("n_bootstrap must be an integer.")
    value = int(value)
    if value < 2:
        raise ValueError("n_bootstrap must be at least 2.")
    return value


def _validate_seed(seed) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer or None.")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    return seed


def _validate_common_research_paths(
    experiment: FullExperimentResult,
    strategy_ids: tuple[str, ...],
) -> int:
    if not isinstance(experiment, FullExperimentResult):
        raise TypeError("experiment must be a FullExperimentResult.")

    reference_terminal = None
    n_paths = None
    for strategy_id in strategy_ids:
        if strategy_id not in experiment.results_by_strategy_id:
            raise ValueError(f"Research results are missing strategy {strategy_id}.")
        result = experiment.results_by_strategy_id[strategy_id]
        pnls = np.asarray(result.terminal_pnls, dtype=float)
        costs = np.asarray(result.total_transaction_costs, dtype=float)
        terminals = np.asarray(result.terminal_futures_prices, dtype=float)
        if pnls.ndim != 1 or costs.shape != pnls.shape or terminals.shape != pnls.shape:
            raise ValueError("Research path-level arrays are not aligned.")
        if n_paths is None:
            n_paths = int(pnls.size)
            reference_terminal = terminals
        elif pnls.size != n_paths or not np.array_equal(terminals, reference_terminal):
            raise ValueError(
                "Eligible strategies must use the same Research paths in the same order."
            )

    if n_paths is None or n_paths < 2:
        raise ValueError("At least two common Research paths are required.")
    return n_paths


def bootstrap_cost_es_frontier_stability(
    research_selection: StrategySelectionResult,
    experiment: FullExperimentResult,
    *,
    tail_probability: float = 0.05,
    n_bootstrap: int = 2_000,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Estimate paired bootstrap stability of the primary Cost–ES frontier.

    One path-index sample is drawn per bootstrap replication and that exact
    sample is applied to every Desk-Mandate-qualified strategy.
    """
    eligible_ids = require_mandate_eligible_strategy_ids(research_selection)
    probability = _validate_tail_probability(tail_probability)
    n_bootstrap = _validate_n_bootstrap(n_bootstrap)
    seed = _validate_seed(seed)
    n_paths = _validate_common_research_paths(experiment, eligible_ids)

    tail_count = max(1, int(np.ceil(probability * n_paths)))
    pnl_by_id = {
        strategy_id: np.asarray(
            experiment.results_by_strategy_id[strategy_id].terminal_pnls,
            dtype=float,
        )
        for strategy_id in eligible_ids
    }
    cost_by_id = {
        strategy_id: np.asarray(
            experiment.results_by_strategy_id[strategy_id].total_transaction_costs,
            dtype=float,
        )
        for strategy_id in eligible_ids
    }

    frontier_hits = np.zeros(len(eligible_ids), dtype=int)
    rng = np.random.default_rng(seed)

    for _ in range(n_bootstrap):
        sampled_indices = rng.integers(0, n_paths, size=n_paths)
        sampled_costs = np.empty(len(eligible_ids), dtype=float)
        sampled_tail_losses = np.empty(len(eligible_ids), dtype=float)

        for position, strategy_id in enumerate(eligible_ids):
            sampled_pnl = pnl_by_id[strategy_id][sampled_indices]
            sampled_costs[position] = float(
                np.mean(cost_by_id[strategy_id][sampled_indices])
            )
            lower_tail = np.partition(sampled_pnl, tail_count - 1)[:tail_count]
            sampled_tail_losses[position] = -float(np.mean(lower_tail))

        frontier_hits += _pareto_efficient_mask_2d(
            sampled_costs,
            sampled_tail_losses,
            first_minimize=True,
            second_minimize=True,
        )

    point_frontier = add_cost_es_frontier_flag(research_selection.feasible_metrics)
    point_lookup = point_frontier.set_index("strategy_id")

    rows = []
    for position, strategy_id in enumerate(eligible_ids):
        row = point_lookup.loc[strategy_id]
        rows.append(
            {
                "strategy_id": strategy_id,
                "strategy_name": row.get("strategy_name", strategy_id),
                "mean_transaction_cost": float(row["mean_transaction_cost"]),
                "tail_loss_5": float(row["tail_loss_5"]),
                "cost_es_efficient": bool(row["cost_es_efficient"]),
                "frontier_stability": float(frontier_hits[position] / n_bootstrap),
                "n_bootstrap": n_bootstrap,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def _normalize_candidate_ids(
    candidate_strategy_ids,
    eligible_ids: tuple[str, ...],
) -> tuple[str, ...]:
    try:
        raw = tuple(candidate_strategy_ids)
    except TypeError as error:
        raise TypeError("candidate_strategy_ids must be an iterable of strategy IDs.") from error

    normalized = tuple(str(strategy_id).strip().upper() for strategy_id in raw)
    if any(not strategy_id for strategy_id in normalized):
        raise ValueError("candidate_strategy_ids cannot contain empty IDs.")
    if len(normalized) != len(set(normalized)):
        raise ValueError("candidate_strategy_ids must be unique.")
    rejected = tuple(strategy_id for strategy_id in normalized if strategy_id not in eligible_ids)
    if rejected:
        raise ValueError(
            "Focused paired comparisons may use only Desk-Mandate-qualified strategies. "
            f"Rejected IDs: {rejected}."
        )
    return normalized


def create_focused_pairwise_comparisons(
    research_selection: StrategySelectionResult,
    experiment: FullExperimentResult,
    candidate_strategy_ids,
    *,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Compare only explicitly supplied serious candidates on common Research paths."""
    eligible_ids = require_mandate_eligible_strategy_ids(research_selection)
    candidate_ids = _normalize_candidate_ids(candidate_strategy_ids, eligible_ids)
    _validate_common_research_paths(experiment, candidate_ids or eligible_ids[:1])

    columns = [
        "strategy_a_id",
        "strategy_b_id",
        "strategy_a",
        "strategy_b",
        "n_paths",
        "mean_pnl_difference",
        "median_pnl_difference",
        "std_pnl_difference",
        "standard_error",
        "confidence_level",
        "confidence_interval_lower",
        "confidence_interval_upper",
        "probability_a_outperforms_b",
        "probability_b_outperforms_a",
        "probability_of_tie",
        "conclusion",
    ]
    if len(candidate_ids) < 2:
        return pd.DataFrame(columns=columns)

    names = dict(
        zip(
            research_selection.evaluated_metrics["strategy_id"],
            research_selection.evaluated_metrics["strategy_name"],
        )
    )
    rows = []
    for strategy_a, strategy_b in combinations(candidate_ids, 2):
        comparison = compare_paired_strategies(
            experiment.results_by_strategy_id[strategy_a],
            experiment.results_by_strategy_id[strategy_b],
            names.get(strategy_a, strategy_a),
            names.get(strategy_b, strategy_b),
            confidence_level=confidence_level,
        )
        statistically_clear = (
            comparison.confidence_interval_lower > 0.0
            or comparison.confidence_interval_upper < 0.0
        )
        row = {
            "strategy_a_id": strategy_a,
            "strategy_b_id": strategy_b,
            **comparison.__dict__,
            "conclusion": (
                "Statistically distinguishable"
                if statistically_clear
                else "Statistically indistinguishable"
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)
