from dataclasses import dataclass, field, replace
from numbers import Integral, Real

import numpy as np
import pandas as pd

from dynamic_delta_hedging.config import HedgingConfig
from dynamic_delta_hedging.hedging_engine import (
    DetailedPathHedgingTrace,
    hedge_short_call_path,
    trace_short_call_path,
)
from dynamic_delta_hedging.selection import StrategySelectionResult
from dynamic_delta_hedging.simulation import simulate_futures_paths_from_shocks
from dynamic_delta_hedging.strategy_registry import build_strategy, get_strategy_spec


REALIZED_VOL_SUPPORT_MULTIPLIERS = (0.75, 1.25)
DEFAULT_STARTING_CAPITAL = 25_000.0
DEFAULT_RISK_BUDGET_FRACTION = 0.03
DEFAULT_MAX_DRAWDOWN_LIMIT = 0.20
DEFAULT_N_CAMPAIGN_TRADES = 30
MIN_LOSS_REFERENCE_EUR = 1.0e-9


@dataclass(frozen=True)
class RealizedCampaignEnvironment:
    """Exogenous realized market sequence, generated independently of strategy execution."""

    seed: int | None
    realized_sigmas: np.ndarray = field(repr=False)
    standardized_shocks: np.ndarray = field(repr=False)
    futures_spread: float

    def __post_init__(self):
        sigmas = np.asarray(self.realized_sigmas, dtype=float).copy()
        shocks = np.asarray(self.standardized_shocks, dtype=float).copy()
        if sigmas.ndim != 1 or sigmas.size == 0:
            raise ValueError("realized_sigmas must be a non-empty 1D array.")
        if shocks.ndim != 2 or shocks.shape[0] != sigmas.size or shocks.shape[1] == 0:
            raise ValueError(
                "standardized_shocks must have shape (n_trades, n_steps)."
            )
        if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
            raise ValueError("realized_sigmas must be positive and finite.")
        if not np.all(np.isfinite(shocks)):
            raise ValueError("standardized_shocks must contain only finite values.")
        futures_spread = _nonnegative_real("futures_spread", self.futures_spread)
        seed = _seed_or_none(self.seed)

        sigmas.setflags(write=False)
        shocks.setflags(write=False)
        object.__setattr__(self, "realized_sigmas", sigmas)
        object.__setattr__(self, "standardized_shocks", shocks)
        object.__setattr__(self, "futures_spread", futures_spread)
        object.__setattr__(self, "seed", seed)

    @property
    def n_trades(self) -> int:
        return int(self.realized_sigmas.size)

    @property
    def n_steps(self) -> int:
        return int(self.standardized_shocks.shape[1])


@dataclass(frozen=True)
class RealizedCampaignResult:
    strategy_id: str
    strategy_name: str
    family: str
    research_loss_reference_per_contract: float
    option_spread: float
    futures_spread: float
    strike: float
    starting_capital: float
    risk_budget_fraction: float
    max_drawdown_limit: float
    planned_trades: int
    seed: int | None
    environment: RealizedCampaignEnvironment = field(repr=False)
    trade_table: pd.DataFrame = field(repr=False)
    terminal_capital: float
    terminal_return: float
    max_drawdown: float
    stop_reason: str
    _execution_config: HedgingConfig = field(repr=False, compare=False)


def _positive_real(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return value


def _nonnegative_real(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
    return value


def _probability(name: str, value) -> float:
    value = _positive_real(name, value)
    if value >= 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1.")
    return value


def _positive_integer(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _seed_or_none(seed) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer or None.")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    return seed


def _normalize_strategy_id(strategy_id) -> str:
    return get_strategy_spec(strategy_id).strategy_id


def research_loss_reference_lookup(research_metrics: pd.DataFrame) -> dict[str, float]:
    """Return positive Research downside-loss references defined as -ES 5%."""
    if not isinstance(research_metrics, pd.DataFrame):
        raise TypeError("research_metrics must be a pandas DataFrame.")
    required = {"strategy_id", "expected_shortfall"}
    missing = required - set(research_metrics.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    lookup = {}
    for row in research_metrics.itertuples(index=False):
        strategy_id = _normalize_strategy_id(row.strategy_id)
        expected_shortfall = float(row.expected_shortfall)
        loss_reference = -expected_shortfall
        if not np.isfinite(loss_reference) or loss_reference <= MIN_LOSS_REFERENCE_EUR:
            raise ValueError(
                f"Research ES 5% for strategy {strategy_id} must imply a positive "
                "downside-loss reference for position sizing."
            )
        lookup[strategy_id] = loss_reference
    return lookup


def research_downside_loss_reference_for_strategy(
    research_selection: StrategySelectionResult,
    strategy_id: str,
) -> float:
    """Return -ES 5% for one Research-qualified strategy."""
    normalized = validate_final_strategy_choice(strategy_id, research_selection)
    return research_loss_reference_lookup(
        research_selection.evaluated_metrics
    )[normalized]


def _option_spread_lookup(quote_table: pd.DataFrame) -> dict[str, float]:
    if not isinstance(quote_table, pd.DataFrame):
        raise TypeError("quote_table must be a pandas DataFrame.")
    required = {"strategy_id", "recommended_option_spread"}
    missing = required - set(quote_table.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    lookup = {}
    for row in quote_table.itertuples(index=False):
        strategy_id = _normalize_strategy_id(row.strategy_id)
        spread = float(row.recommended_option_spread)
        if not np.isfinite(spread) or not 0.0 <= spread < 2.0:
            raise ValueError(f"Invalid option spread for strategy {strategy_id}.")
        lookup[strategy_id] = spread
    return lookup


def validate_final_strategy_choice(
    strategy_id: str,
    research_selection: StrategySelectionResult,
) -> str:
    """Require the explicit final choice to have passed the Research Desk Mandate."""
    if not isinstance(research_selection, StrategySelectionResult):
        raise TypeError("research_selection must be a StrategySelectionResult.")
    normalized = _normalize_strategy_id(strategy_id)
    eligible = tuple(research_selection.feasible_strategy_ids)
    if normalized not in eligible:
        raise ValueError(
            "FINAL_SELECTED_STRATEGY must be Research-qualified. "
            f"Eligible strategy IDs: {eligible}."
        )
    return normalized


def _single_trade_position_size(
    capital: float,
    *,
    risk_budget_fraction: float,
    loss_reference: float,
) -> int:
    return max(
        0,
        int(
            np.floor(
                max(capital, 0.0)
                * risk_budget_fraction
                / loss_reference
            )
        ),
    )


def _expiry_status(terminal_futures_price: float, strike: float) -> str:
    """Classify call expiry status with a tight numerical ATM tolerance."""
    terminal = float(terminal_futures_price)
    strike = float(strike)
    tolerance = 1.0e-10 * max(1.0, abs(strike), abs(terminal))
    if abs(terminal - strike) <= tolerance:
        return "ATM"
    return "ITM" if terminal > strike else "OTM"


def generate_realized_campaign_environment(
    research_config: HedgingConfig,
    *,
    n_trades: int,
    seed: int | None,
) -> RealizedCampaignEnvironment:
    """Generate the full exogenous market sequence before any strategy executes."""
    if not isinstance(research_config, HedgingConfig):
        raise TypeError("research_config must be a HedgingConfig.")
    n_trades = _positive_integer("n_trades", n_trades)
    seed = _seed_or_none(seed)

    rng = np.random.default_rng(seed)
    sigma_low = REALIZED_VOL_SUPPORT_MULTIPLIERS[0] * research_config.sigma
    sigma_high = REALIZED_VOL_SUPPORT_MULTIPLIERS[1] * research_config.sigma
    realized_sigmas = rng.triangular(
        sigma_low,
        research_config.sigma,
        sigma_high,
        size=n_trades,
    )
    standardized_shocks = rng.standard_normal((n_trades, research_config.n_steps))

    return RealizedCampaignEnvironment(
        seed=seed,
        realized_sigmas=realized_sigmas,
        standardized_shocks=standardized_shocks,
        futures_spread=float(research_config.futures_spread),
    )


def reconstruct_trade_futures_path(
    config: HedgingConfig,
    environment: RealizedCampaignEnvironment,
    trade_number: int,
) -> np.ndarray:
    """Reconstruct one exogenous futures path from a campaign environment."""
    if not isinstance(config, HedgingConfig):
        raise TypeError("config must be a HedgingConfig.")
    if not isinstance(environment, RealizedCampaignEnvironment):
        raise TypeError("environment must be a RealizedCampaignEnvironment.")
    trade_number = _positive_integer("trade_number", trade_number)
    if trade_number > environment.n_trades:
        raise ValueError(
            f"trade_number must be between 1 and {environment.n_trades}."
        )
    if environment.n_steps != config.n_steps:
        raise ValueError("Campaign environment n_steps does not match config.n_steps.")
    if not np.isclose(
        environment.futures_spread,
        config.futures_spread,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("Campaign environment futures spread does not match config.")

    index = trade_number - 1
    return simulate_futures_paths_from_shocks(
        config,
        environment.standardized_shocks[index : index + 1],
        realized_sigma=float(environment.realized_sigmas[index]),
    )[0]


def validate_trade_to_review(
    realized_campaign: RealizedCampaignResult,
    trade_number: int,
) -> int:
    """Require a requested campaign trade to exist and to have been executed."""
    if not isinstance(realized_campaign, RealizedCampaignResult):
        raise TypeError("realized_campaign must be a RealizedCampaignResult.")
    trade_number = _positive_integer("trade_number", trade_number)
    if trade_number > realized_campaign.planned_trades:
        raise ValueError(
            f"TRADE_TO_REVIEW={trade_number} is outside the planned campaign range "
            f"1..{realized_campaign.planned_trades}."
        )

    matches = realized_campaign.trade_table.loc[
        realized_campaign.trade_table["trade"] == trade_number
    ]
    if matches.empty or int(matches.iloc[0]["contracts"]) <= 0:
        last_executed = realized_campaign.trade_table.loc[
            realized_campaign.trade_table["contracts"] > 0, "trade"
        ]
        last_executed_text = (
            "none" if last_executed.empty else str(int(last_executed.max()))
        )
        raise ValueError(
            f"TRADE_TO_REVIEW={trade_number} was not executed. "
            f"Last executed trade: {last_executed_text}; "
            f"campaign status: {realized_campaign.stop_reason}."
        )
    return trade_number


def build_trade_trace(
    realized_campaign: RealizedCampaignResult,
    trade_number: int,
) -> DetailedPathHedgingTrace:
    """Replay one selected executed trade without storing traces for every trade."""
    trade_number = validate_trade_to_review(realized_campaign, trade_number)
    path = reconstruct_trade_futures_path(
        realized_campaign._execution_config,
        realized_campaign.environment,
        trade_number,
    )
    spec = get_strategy_spec(realized_campaign.strategy_id)
    return trace_short_call_path(
        path,
        build_strategy(spec),
        realized_campaign._execution_config,
    )


def create_trade_summary(
    realized_campaign: RealizedCampaignResult,
    trade_number: int,
    *,
    trace: DetailedPathHedgingTrace | None = None,
) -> dict[str, float | int | str]:
    """Return scale-explicit expiry and P&L facts for an arbitrary selected trade."""
    trade_number = validate_trade_to_review(realized_campaign, trade_number)
    if trace is None:
        trace = build_trade_trace(realized_campaign, trade_number)

    row = realized_campaign.trade_table.loc[
        realized_campaign.trade_table["trade"] == trade_number
    ].iloc[0]
    terminal_futures_price = float(trace.futures_prices[-1])
    strike = float(realized_campaign.strike)
    per_contract_pnl = float(trace.result.terminal_pnl)
    contracts = int(row["contracts"])
    total_trade_pnl = float(row["total_trade_pnl"])

    return {
        "trade": trade_number,
        "realized_sigma": float(row["realized_sigma"]),
        "terminal_futures_price": terminal_futures_price,
        "strike": strike,
        "expiry_status": _expiry_status(terminal_futures_price, strike),
        "contracts": contracts,
        "hedge_trades": int(trace.result.number_of_trades),
        "transaction_cost": float(trace.result.total_transaction_cost),
        "hedge_pnl": float(trace.result.total_hedge_pnl),
        "option_payoff": float(trace.result.call_payoff),
        "per_contract_terminal_pnl": per_contract_pnl,
        "total_trade_pnl": total_trade_pnl,
    }


def run_realized_campaign(
    research_config: HedgingConfig,
    research_selection: StrategySelectionResult,
    quote_table: pd.DataFrame,
    final_strategy_id: str,
    *,
    starting_capital: float = DEFAULT_STARTING_CAPITAL,
    risk_budget_fraction: float = DEFAULT_RISK_BUDGET_FRACTION,
    max_drawdown_limit: float = DEFAULT_MAX_DRAWDOWN_LIMIT,
    n_trades: int = DEFAULT_N_CAMPAIGN_TRADES,
    seed: int | None = None,
) -> RealizedCampaignResult:
    """Run one sequential realized campaign on a pre-generated market environment."""
    if not isinstance(research_config, HedgingConfig):
        raise TypeError("research_config must be a HedgingConfig.")
    final_strategy_id = validate_final_strategy_choice(
        final_strategy_id, research_selection
    )
    spec = get_strategy_spec(final_strategy_id)
    starting_capital = _positive_real("starting_capital", starting_capital)
    risk_budget_fraction = _probability("risk_budget_fraction", risk_budget_fraction)
    max_drawdown_limit = _probability("max_drawdown_limit", max_drawdown_limit)
    n_trades = _positive_integer("n_trades", n_trades)
    seed = _seed_or_none(seed)

    loss_reference = research_downside_loss_reference_for_strategy(
        research_selection,
        final_strategy_id,
    )
    option_spreads = _option_spread_lookup(quote_table)
    if final_strategy_id not in option_spreads:
        raise ValueError(
            f"quote_table does not contain the selected strategy {final_strategy_id}."
        )
    option_spread = option_spreads[final_strategy_id]
    execution_config = replace(
        research_config,
        n_paths=1,
        option_spread=option_spread,
    )

    environment = generate_realized_campaign_environment(
        research_config,
        n_trades=n_trades,
        seed=seed,
    )

    capital = starting_capital
    peak = starting_capital
    max_drawdown = 0.0
    rows = []
    stop_reason = "COMPLETED"

    for trade_index in range(1, n_trades + 1):
        capital_before = capital
        contracts = _single_trade_position_size(
            capital_before,
            risk_budget_fraction=risk_budget_fraction,
            loss_reference=loss_reference,
        )
        if contracts == 0:
            rows.append(
                {
                    "trade": trade_index,
                    "realized_sigma": np.nan,
                    "contracts": 0,
                    "per_contract_pnl": np.nan,
                    "total_trade_pnl": 0.0,
                    "capital_after_trade": capital,
                    "drawdown": 1.0 - capital / peak,
                    "status": "NO NEW TRADE (SIZE 0)",
                }
            )
            stop_reason = "NO NEW TRADE (SIZE 0)"
            break

        path = reconstruct_trade_futures_path(
            execution_config,
            environment,
            trade_index,
        )
        path_result = hedge_short_call_path(
            path,
            build_strategy(spec),
            execution_config,
        )
        per_contract_pnl = float(path_result.terminal_pnl)
        total_trade_pnl = contracts * per_contract_pnl
        capital = capital_before + total_trade_pnl
        peak = max(peak, capital)
        drawdown = 1.0 - capital / peak
        max_drawdown = max(max_drawdown, drawdown)

        breached = drawdown >= max_drawdown_limit
        is_last = trade_index == n_trades
        status = (
            "RISK LIMIT BREACH"
            if breached
            else ("COMPLETED" if is_last else "ACTIVE")
        )
        rows.append(
            {
                "trade": trade_index,
                "realized_sigma": float(environment.realized_sigmas[trade_index - 1]),
                "contracts": contracts,
                "per_contract_pnl": per_contract_pnl,
                "total_trade_pnl": total_trade_pnl,
                "capital_after_trade": capital,
                "drawdown": drawdown,
                "status": status,
            }
        )

        if breached:
            stop_reason = "RISK LIMIT BREACH"
            break

    table = pd.DataFrame(rows)
    return RealizedCampaignResult(
        strategy_id=final_strategy_id,
        strategy_name=spec.display_name,
        family=spec.family,
        research_loss_reference_per_contract=loss_reference,
        option_spread=option_spread,
        futures_spread=float(research_config.futures_spread),
        strike=float(research_config.K),
        starting_capital=starting_capital,
        risk_budget_fraction=risk_budget_fraction,
        max_drawdown_limit=max_drawdown_limit,
        planned_trades=n_trades,
        seed=seed,
        environment=environment,
        trade_table=table,
        terminal_capital=float(capital),
        terminal_return=float(capital / starting_capital - 1.0),
        max_drawdown=float(max_drawdown),
        stop_reason=stop_reason,
        _execution_config=execution_config,
    )
