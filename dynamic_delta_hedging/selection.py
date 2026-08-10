from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd


_FAMILY_PARAMETER_MAP = {
    "Fixed interval": "interval",
    "Fixed band": "band_width",
    "WW-inspired": "risk_aversion",
    "Delta tolerance": "delta_tolerance",
    "Asset tolerance": "asset_tolerance",
}

_REQUIRED_COLUMNS = {
    "strategy_family",
    "strategy_parameter",
    "strategy_parameter_value",
    "std_pnl",
    "expected_shortfall",
    "mean_transaction_cost",
}


@dataclass(frozen=True)
class DeskMandate:
    max_pnl_std: float | None = None
    min_es_5: float | None = None
    max_mean_cost: float | None = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "max_pnl_std",
            _validate_optional_limit(
                self.max_pnl_std,
                "max_pnl_std",
                nonnegative=True,
            ),
        )
        object.__setattr__(
            self,
            "min_es_5",
            _validate_optional_limit(
                self.min_es_5,
                "min_es_5",
                nonnegative=False,
            ),
        )
        object.__setattr__(
            self,
            "max_mean_cost",
            _validate_optional_limit(
                self.max_mean_cost,
                "max_mean_cost",
                nonnegative=True,
            ),
        )


def _validate_percentage(value, name: str) -> float | None:
    """Validate a user-facing percentage expressed in percentage points."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real percentage or None.")

    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if value < 0.0:
        raise ValueError(f"{name} cannot be negative.")
    return value


def create_desk_mandate_from_percentages(
    fair_contract_value,
    *,
    max_pnl_std_pct=None,
    max_tail_loss_5_pct=None,
    max_mean_hedging_cost_pct=None,
) -> DeskMandate:
    """Build the existing euro-based DeskMandate from fair-value percentages.

    Percentage inputs are expressed in percentage points: ``10.0`` means 10%
    of the fair contract value.  The positive Tail Loss limit is converted to
    the internal negative Expected-Shortfall floor used by ``DeskMandate``.
    """
    if isinstance(fair_contract_value, bool) or not isinstance(fair_contract_value, Real):
        raise TypeError("fair_contract_value must be a real number.")
    fair_contract_value = float(fair_contract_value)
    if not np.isfinite(fair_contract_value) or fair_contract_value <= 0.0:
        raise ValueError("fair_contract_value must be positive and finite.")

    max_pnl_std_pct = _validate_percentage(max_pnl_std_pct, "max_pnl_std_pct")
    max_tail_loss_5_pct = _validate_percentage(
        max_tail_loss_5_pct, "max_tail_loss_5_pct"
    )
    max_mean_hedging_cost_pct = _validate_percentage(
        max_mean_hedging_cost_pct, "max_mean_hedging_cost_pct"
    )

    def euro_limit(percentage):
        return None if percentage is None else fair_contract_value * percentage / 100.0

    max_pnl_std = euro_limit(max_pnl_std_pct)
    tail_loss_limit = euro_limit(max_tail_loss_5_pct)
    max_mean_cost = euro_limit(max_mean_hedging_cost_pct)

    return DeskMandate(
        max_pnl_std=max_pnl_std,
        min_es_5=None if tail_loss_limit is None else -tail_loss_limit,
        max_mean_cost=max_mean_cost,
    )


@dataclass(frozen=True)
class StrategySelectionResult:
    """Desk-mandate evaluation and the surviving strategy rows."""

    evaluated_metrics: pd.DataFrame
    feasible_metrics: pd.DataFrame

    @property
    def n_feasible(self) -> int:
        return len(self.feasible_metrics)

    @property
    def status(self) -> str:
        if self.n_feasible == 0:
            return "none"
        if self.n_feasible == 1:
            return "unique"
        return "multiple"

    @property
    def rejected_metrics(self) -> pd.DataFrame:
        return (
            self.evaluated_metrics.loc[~self.evaluated_metrics["is_feasible"]]
            .copy()
            .reset_index(drop=True)
        )

    @property
    def feasible_strategy_ids(self) -> tuple[str, ...]:
        if "strategy_id" not in self.feasible_metrics.columns:
            return ()
        return tuple(self.feasible_metrics["strategy_id"].astype(str))

    @property
    def rejected_strategy_ids(self) -> tuple[str, ...]:
        rejected = self.rejected_metrics
        if "strategy_id" not in rejected.columns:
            return ()
        return tuple(rejected["strategy_id"].astype(str))

    def research_status_for(self, strategy_id: str) -> str:
        row = _lookup_strategy_row(self.evaluated_metrics, strategy_id)
        return str(row["research_status"])

    def failed_constraints_for(self, strategy_id: str) -> tuple[str, ...]:
        row = _lookup_strategy_row(self.evaluated_metrics, strategy_id)
        value = row["failed_constraints"]
        return tuple(value) if isinstance(value, (tuple, list)) else ()


def _lookup_strategy_row(table: pd.DataFrame, strategy_id: str, ) -> pd.Series:
    if "strategy_id" not in table.columns:
        raise ValueError("This selection result does not contain strategy IDs.")
    if not isinstance(strategy_id, str):
        raise TypeError("strategy_id must be a string.")

    strategy_id = strategy_id.strip().upper()
    matches = table.loc[
        table["strategy_id"].astype(str).str.upper() == strategy_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown or non-unique strategy_id: {strategy_id!r}.")

    return matches.iloc[0]


def _validate_optional_limit(
    value,
    name: str,
    *,
    nonnegative: bool,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number or None.")

    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} cannot be negative.")

    return value


def _validate_metrics_table(metrics_table) -> pd.DataFrame:
    if not isinstance(metrics_table, pd.DataFrame):
        raise TypeError("metrics_table must be a pandas DataFrame.")

    missing = _REQUIRED_COLUMNS - set(metrics_table.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    table = metrics_table.copy()
    if table.empty:
        return table

    label_columns = ["strategy_family", "strategy_parameter"]
    if table[label_columns].isna().any().any():
        raise ValueError(
            "Strategy family and parameter labels cannot contain missing values."
        )

    unknown_families = set(table["strategy_family"]) - set(_FAMILY_PARAMETER_MAP)
    if unknown_families:
        raise ValueError(f"Unknown strategy families: {sorted(unknown_families)}")

    for family, parameter in _FAMILY_PARAMETER_MAP.items():
        family_mask = table["strategy_family"] == family
        if family_mask.any() and (
            table.loc[family_mask, "strategy_parameter"] != parameter
        ).any():
            raise ValueError(f"{family} rows must use strategy_parameter='{parameter}'.")

    numeric_columns = (
        "strategy_parameter_value",
        "std_pnl",
        "expected_shortfall",
        "mean_transaction_cost",
    )
    for column in numeric_columns:
        try:
            values = pd.to_numeric(table[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{column} must contain numeric values.") from error

        if not np.all(np.isfinite(values)):
            raise ValueError(f"{column} must contain only finite values.")

    if (table["std_pnl"].to_numpy(float) < 0.0).any():
        raise ValueError("std_pnl cannot contain negative values.")
    if (table["mean_transaction_cost"].to_numpy(float) < 0.0).any():
        raise ValueError("mean_transaction_cost cannot contain negative values.")

    if "strategy_id" in table.columns:
        if table["strategy_id"].isna().any():
            raise ValueError("strategy_id cannot contain missing values.")

        normalized_ids = table["strategy_id"].astype(str).str.strip().str.upper()
        if (normalized_ids == "").any():
            raise ValueError("strategy_id cannot be empty.")
        if normalized_ids.duplicated().any():
            raise ValueError("strategy_id values must be unique.")

        table["strategy_id"] = normalized_ids

    duplicate_columns = [
        "strategy_family",
        "strategy_parameter",
        "strategy_parameter_value",
    ]
    if table.duplicated(subset=duplicate_columns, keep=False).any():
        raise ValueError(
            "Each strategy specification must appear exactly once in metrics_table."
        )

    return table


def apply_desk_mandate(
    metrics_table: pd.DataFrame,
    mandate: DeskMandate,
) -> StrategySelectionResult:
    """Apply hard ex-ante desk limits to the candidate strategy universe."""
    if not isinstance(mandate, DeskMandate):
        raise TypeError("mandate must be a DeskMandate.")

    evaluated = _validate_metrics_table(metrics_table).copy()
    evaluated["passes_std_limit"] = True
    evaluated["passes_es_limit"] = True
    evaluated["passes_cost_limit"] = True

    if mandate.max_pnl_std is not None:
        evaluated["passes_std_limit"] = (
            evaluated["std_pnl"] <= mandate.max_pnl_std
        )
    if mandate.min_es_5 is not None:
        evaluated["passes_es_limit"] = (
            evaluated["expected_shortfall"] >= mandate.min_es_5
        )
    if mandate.max_mean_cost is not None:
        evaluated["passes_cost_limit"] = (
            evaluated["mean_transaction_cost"] <= mandate.max_mean_cost
        )

    evaluated["is_feasible"] = (
        evaluated["passes_std_limit"]
        & evaluated["passes_es_limit"]
        & evaluated["passes_cost_limit"]
    )
    evaluated["research_status"] = np.where(
        evaluated["is_feasible"],
        "QUALIFIED",
        "REJECTED",
    )

    failed_constraints = []
    for row in evaluated.itertuples(index=False):
        reasons = []
        if not row.passes_std_limit:
            reasons.append("P&L Std")
        if not row.passes_es_limit:
            reasons.append("ES 5%")
        if not row.passes_cost_limit:
            reasons.append("Mean Cost")
        failed_constraints.append(tuple(reasons))

    evaluated["failed_constraints"] = failed_constraints

    feasible = (
        evaluated.loc[evaluated["is_feasible"]]
        .copy()
        .reset_index(drop=True)
    )
    fixed_interval_mask = feasible["strategy_family"] == "Fixed interval"
    fixed_interval_values = feasible.loc[
        fixed_interval_mask, "strategy_parameter_value"
    ].to_numpy(dtype=float)
    if fixed_interval_values.size:
        rounded_values = np.rint(fixed_interval_values)
        if not np.all(
            np.isclose(
                fixed_interval_values,
                rounded_values,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError("Fixed-interval parameter values must be integers.")

    return StrategySelectionResult(
        evaluated_metrics=evaluated.reset_index(drop=True),
        feasible_metrics=feasible,
    )


def require_mandate_eligible_strategy_ids(
    research_selection: StrategySelectionResult,
) -> tuple[str, ...]:
    """Return the canonical Desk-Mandate survivor universe or fail explicitly."""
    if not isinstance(research_selection, StrategySelectionResult):
        raise TypeError("research_selection must be a StrategySelectionResult.")

    eligible = tuple(research_selection.feasible_strategy_ids)
    if not eligible:
        raise ValueError(
            "No tested hedging policy satisfies the specified Desk Mandate; "
            "the downstream Research selection pipeline cannot continue."
        )
    return eligible
