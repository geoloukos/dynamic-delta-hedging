from dataclasses import dataclass
from math import exp, isfinite
from numbers import Real

import numpy as np
import pandas as pd

from dynamic_delta_hedging.config import HedgingConfig
from dynamic_delta_hedging.pricing import black76_call_price


@dataclass(frozen=True)
class StrategyQuoteRecommendation:
    strategy_id: str
    strategy_name: str
    research_mean_pnl: float
    fair_option_price: float
    fair_contract_value: float
    break_even_ask_price: float
    break_even_ask_eur: float
    recommended_ask_price: float
    recommended_ask_eur: float
    recommended_option_spread: float
    target_expected_terminal_profit_eur: float


def _validate_target_profit(value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("target_expected_terminal_profit_eur must be a real number.")
    value = float(value)
    if not isfinite(value):
        raise ValueError("target_expected_terminal_profit_eur must be finite.")
    if value < 0.0:
        raise ValueError("target_expected_terminal_profit_eur cannot be negative.")
    return value


def _validate_research_mean_pnl(value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("research_mean_pnl must be a real number.")
    value = float(value)
    if not isfinite(value):
        raise ValueError("research_mean_pnl must be finite.")
    return value


def calculate_strategy_quote_recommendation(
    *,
    strategy_id: str,
    strategy_name: str,
    research_mean_pnl: float,
    config: HedgingConfig,
    target_expected_terminal_profit_eur: float,
) -> StrategyQuoteRecommendation:
    """Convert fair-value research P&L into a strategy-specific ask recommendation.

    The Research Monte Carlo is assumed to sell the option at the Black-76
    theoretical value (zero option spread). A higher initial premium enters the
    cash account at t=0 and therefore compounds at the risk-free rate until
    expiry. The recommended ask is chosen so that, on the Research-MC mean,
    the expected terminal P&L equals the requested target profit.
    """
    if not isinstance(config, HedgingConfig):
        raise TypeError("config must be a HedgingConfig.")
    if not np.isclose(config.option_spread, 0.0):
        raise ValueError(
            "Quote recommendations require a Research-MC configuration with option_spread=0."
        )
    if not isinstance(strategy_id, str) or not strategy_id.strip():
        raise TypeError("strategy_id must be a non-empty string.")
    if not isinstance(strategy_name, str) or not strategy_name.strip():
        raise TypeError("strategy_name must be a non-empty string.")

    mean_pnl = _validate_research_mean_pnl(research_mean_pnl)
    target_profit = _validate_target_profit(target_expected_terminal_profit_eur)

    fair_option_price = float(black76_call_price(F=config.F0, tau=config.T, config=config))
    fair_contract_value = fair_option_price * config.contract_multiplier
    terminal_growth = exp(config.r * config.T)

    break_even_initial_increment = -mean_pnl / terminal_growth
    target_initial_increment = (target_profit - mean_pnl) / terminal_growth

    break_even_ask_eur = fair_contract_value + break_even_initial_increment
    recommended_ask_eur = fair_contract_value + target_initial_increment
    break_even_ask_price = break_even_ask_eur / config.contract_multiplier
    recommended_ask_price = recommended_ask_eur / config.contract_multiplier

    recommended_option_spread = 2.0 * (recommended_ask_price / fair_option_price - 1.0)
    if not 0.0 <= recommended_option_spread < 2.0:
        raise ValueError(
            "The recommended ask implies an option spread outside the supported range [0, 2)."
        )

    return StrategyQuoteRecommendation(
        strategy_id=strategy_id.strip().upper(),
        strategy_name=strategy_name.strip(),
        research_mean_pnl=mean_pnl,
        fair_option_price=fair_option_price,
        fair_contract_value=fair_contract_value,
        break_even_ask_price=break_even_ask_price,
        break_even_ask_eur=break_even_ask_eur,
        recommended_ask_price=recommended_ask_price,
        recommended_ask_eur=recommended_ask_eur,
        recommended_option_spread=recommended_option_spread,
        target_expected_terminal_profit_eur=target_profit,
    )


def create_strategy_quote_recommendations(
    metrics_table: pd.DataFrame,
    config: HedgingConfig,
    target_expected_terminal_profit_eur: float,
) -> pd.DataFrame:
    """Create strategy-specific break-even and recommended asks for the supplied Research rows."""
    if not isinstance(metrics_table, pd.DataFrame):
        raise TypeError("metrics_table must be a pandas DataFrame.")
    required = {"strategy_id", "strategy_name", "mean_pnl"}
    missing = required - set(metrics_table.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []
    for row in metrics_table.itertuples(index=False):
        recommendation = calculate_strategy_quote_recommendation(
            strategy_id=row.strategy_id,
            strategy_name=row.strategy_name,
            research_mean_pnl=row.mean_pnl,
            config=config,
            target_expected_terminal_profit_eur=target_expected_terminal_profit_eur,
        )
        rows.append(recommendation.__dict__)
    return pd.DataFrame(rows)
