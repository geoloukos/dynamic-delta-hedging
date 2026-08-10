from dataclasses import dataclass

import numpy as np

from dynamic_delta_hedging.config import HedgingConfig
from dynamic_delta_hedging.market import build_option_quote, futures_transaction_cost
from dynamic_delta_hedging.pricing import black76_call_delta, black76_call_price


@dataclass(frozen=True)
class PathHedgingResult:
    """Contract-level hedging result for one futures path."""

    terminal_pnl: float
    option_premium: float
    call_payoff: float
    total_hedge_pnl: float
    total_transaction_cost: float
    number_of_trades: int
    total_turnover: float
    terminal_futures_price: float
    theoretical_option_price: float = 0.0
    option_bid_price: float = 0.0
    option_ask_price: float = 0.0
    option_sale_proceeds: float = 0.0
    initial_option_edge: float = 0.0


@dataclass(frozen=True)
class BatchHedgingResult:
    """Path-level contract cash results for a batch of futures paths."""

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
class DetailedPathHedgingTrace:
    """Detailed diagnostics for one realized trade without burdening batch Research MC."""

    result: PathHedgingResult
    time_grid: np.ndarray
    futures_prices: np.ndarray
    black76_deltas: np.ndarray
    hedge_positions_after_trade: np.ndarray
    hedge_trade_changes: np.ndarray
    transaction_costs_by_trade_time: np.ndarray
    cumulative_transaction_costs: np.ndarray
    hedge_pnl_contributions: np.ndarray
    cumulative_hedge_pnl: np.ndarray
    interest_accruals: np.ndarray
    cumulative_interest_accrual: np.ndarray
    realized_futures_spread: float

    @property
    def terminal_pnl_decomposition(self) -> dict[str, float]:
        return {
            "option_sale_proceeds": float(self.result.option_sale_proceeds),
            "cash_interest_accrual": float(self.cumulative_interest_accrual[-1]),
            "hedge_pnl": float(self.result.total_hedge_pnl),
            "transaction_cost": -float(self.result.total_transaction_cost),
            "option_payoff_liability": -float(self.result.call_payoff),
            "terminal_pnl": float(self.result.terminal_pnl),
        }


def _validate_single_futures_path(futures_path, config: HedgingConfig) -> np.ndarray:
    path = np.asarray(futures_path, dtype=float)
    if path.ndim != 1:
        raise ValueError("futures_path must be one-dimensional.")
    expected_length = config.n_steps + 1
    if path.size != expected_length:
        raise ValueError(f"futures_path must contain {expected_length} prices.")
    if not np.all(np.isfinite(path)):
        raise ValueError("All futures prices must be finite.")
    if np.any(path <= 0.0):
        raise ValueError("All futures prices must be positive.")
    if not np.isclose(path[0], config.F0):
        raise ValueError("The first path price must equal config.F0.")
    return path


def _validate_futures_paths(futures_paths, config: HedgingConfig) -> np.ndarray:
    paths = np.asarray(futures_paths, dtype=float)
    if paths.ndim != 2:
        raise ValueError("futures_paths must be two-dimensional.")
    if paths.shape[0] == 0:
        raise ValueError("futures_paths must contain at least one path.")
    expected_prices = config.n_steps + 1
    if paths.shape[1] != expected_prices:
        raise ValueError(f'futures_paths must contain {expected_prices} prices per path.')
    if not np.all(np.isfinite(paths)):
        raise ValueError("All futures prices must be finite.")
    if np.any(paths <= 0.0):
        raise ValueError("All futures prices must be positive.")
    if not np.allclose(paths[:, 0], config.F0):
        raise ValueError("Every path must begin at config.F0.")
    return paths


def _validate_execution_spread(futures_spread, expected_shape, config: HedgingConfig):
    if futures_spread is None:
        return None
    spread = np.asarray(futures_spread, dtype=float)
    try:
        spread = np.broadcast_to(spread, expected_shape)
    except ValueError as error:
        raise ValueError(
            "futures_spread must be scalar or broadcast-compatible with the path batch."
        ) from error
    if not np.all(np.isfinite(spread)):
        raise ValueError("futures_spread must contain only finite values.")
    if np.any((spread < 0.0) | (spread >= 2.0)):
        raise ValueError("futures_spread must satisfy 0 <= spread < 2.")
    return np.array(spread, dtype=float, copy=True)


def _strategy_target(
    strategy,
    *,
    step,
    call_delta,
    current_position,
    futures_price,
    tau,
    config,
    last_hedge_price=None,
    current_futures_spread=None,
):
    kwargs = dict(
        step=step,
        call_delta=call_delta,
        current_position=current_position,
        futures_price=futures_price,
        tau=tau,
        config=config,
    )
    if getattr(strategy, "requires_last_hedge_price", False):
        kwargs["last_hedge_price"] = last_hedge_price
    if getattr(strategy, "uses_current_futures_spread", False):
        kwargs["current_futures_spread"] = current_futures_spread
    return strategy.get_target_position(**kwargs)


def _get_target_position(
    strategy,
    step,
    call_delta,
    current_position,
    futures_price,
    tau,
    config,
    last_hedge_price=None,
    current_futures_spread=None,
) -> float:
    target = _strategy_target(
        strategy,
        step=step,
        call_delta=call_delta,
        current_position=current_position,
        futures_price=futures_price,
        tau=tau,
        config=config,
        last_hedge_price=last_hedge_price,
        current_futures_spread=current_futures_spread,
    )
    target_array = np.asarray(target, dtype=float)
    if target_array.ndim != 0:
        raise ValueError("A single-path strategy target must be scalar.")
    if not np.isfinite(target_array):
        raise ValueError("The strategy returned an invalid target position.")
    return float(target_array)


def _get_target_positions(
    strategy,
    step,
    call_delta,
    current_position,
    futures_price,
    tau,
    config,
    expected_shape,
    last_hedge_price=None,
    current_futures_spread=None,
) -> np.ndarray:
    target = _strategy_target(
        strategy,
        step=step,
        call_delta=call_delta,
        current_position=current_position,
        futures_price=futures_price,
        tau=tau,
        config=config,
        last_hedge_price=last_hedge_price,
        current_futures_spread=current_futures_spread,
    )
    try:
        target_array = np.asarray(target, dtype=float)
        target_array = np.broadcast_to(target_array, expected_shape)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "The strategy target positions must be broadcast-compatible "
            "with the path batch."
        ) from error
    if not np.all(np.isfinite(target_array)):
        raise ValueError("The strategy returned invalid target positions.")
    return np.array(target_array, dtype=float, copy=True)


def _apply_batch_trades(
    position_changes: np.ndarray,
    target_positions: np.ndarray,
    mid_prices: np.ndarray,
    current_positions: np.ndarray,
    cash_accounts: np.ndarray,
    total_transaction_costs: np.ndarray,
    total_turnovers: np.ndarray,
    numbers_of_trades: np.ndarray,
    config: HedgingConfig,
    futures_spread=None,
) -> np.ndarray:
    """Execute path-level futures trades and return the executed-trade mask."""
    trade_mask = ~np.isclose(position_changes, 0.0)
    executed_changes = np.where(trade_mask, position_changes, 0.0)
    transaction_costs = futures_transaction_cost(
        position_change=executed_changes,
        mid_price=mid_prices,
        config=config,
        futures_spread=futures_spread,
    )
    cash_accounts -= transaction_costs
    total_transaction_costs += transaction_costs
    total_turnovers += np.abs(executed_changes)
    numbers_of_trades += trade_mask.astype(int)
    np.copyto(current_positions, target_positions, where=trade_mask)
    return trade_mask


def _initial_option_quote(config: HedgingConfig):
    theoretical_price = black76_call_price(F=config.F0, tau=config.T, config=config)
    return build_option_quote(theoretical_price, config)


def hedge_short_call_path(
    futures_path,
    strategy,
    config: HedgingConfig,
    futures_spread=None,
) -> PathHedgingResult:
    """Hedge one short European futures call along one futures path."""
    path = _validate_single_futures_path(futures_path, config)
    execution_spread_array = _validate_execution_spread(futures_spread, (), config)
    execution_spread = None if execution_spread_array is None else float(execution_spread_array)
    quote = _initial_option_quote(config)

    # The call is sold at the quoted ask. All cash amounts are contract-level.
    cash_account = quote.sale_proceeds
    current_position = 0.0
    total_hedge_pnl = 0.0
    total_transaction_cost = 0.0
    total_turnover = 0.0
    number_of_trades = 0
    last_hedge_price = float(path[0])

    initial_delta = black76_call_delta(F=path[0], tau=config.T, config=config)
    target_position = _get_target_position(
        strategy=strategy,
        step=0,
        call_delta=initial_delta,
        current_position=current_position,
        futures_price=path[0],
        tau=config.T,
        config=config,
        last_hedge_price=last_hedge_price,
        current_futures_spread=execution_spread,
    )
    position_change = target_position - current_position
    if not np.isclose(position_change, 0.0):
        cost = futures_transaction_cost(position_change, path[0], config, futures_spread=execution_spread)
        cash_account -= cost
        total_transaction_cost += cost
        total_turnover += abs(position_change)
        number_of_trades += 1
        current_position = target_position
        last_hedge_price = float(path[0])

    interest_growth_factor = np.exp(config.r * config.dt)

    for step in range(config.n_steps):
        current_price = path[step]
        next_price = path[step + 1]
        cash_account *= interest_growth_factor

        hedge_pnl = (
            current_position
            * (next_price - current_price)
            * config.contract_multiplier
        )
        cash_account += hedge_pnl
        total_hedge_pnl += hedge_pnl

        next_step = step + 1
        if next_step < config.n_steps:
            tau = config.T - next_step * config.dt
            call_delta = black76_call_delta(F=next_price, tau=tau, config=config)
            target_position = _get_target_position(
                strategy=strategy,
                step=next_step,
                call_delta=call_delta,
                current_position=current_position,
                futures_price=next_price,
                tau=tau,
                config=config,
                last_hedge_price=last_hedge_price,
                current_futures_spread=execution_spread,
            )
            position_change = target_position - current_position
            if not np.isclose(position_change, 0.0):
                cost = futures_transaction_cost(position_change, next_price, config, futures_spread=execution_spread)
                cash_account -= cost
                total_transaction_cost += cost
                total_turnover += abs(position_change)
                number_of_trades += 1
                current_position = target_position
                last_hedge_price = float(next_price)

    closing_change = -current_position
    if not np.isclose(closing_change, 0.0):
        closing_cost = futures_transaction_cost(closing_change, path[-1], config, futures_spread=execution_spread)
        cash_account -= closing_cost
        total_transaction_cost += closing_cost
        total_turnover += abs(closing_change)
        number_of_trades += 1

    call_payoff = max(path[-1] - config.K, 0.0) * config.contract_multiplier
    cash_account -= call_payoff

    return PathHedgingResult(
        terminal_pnl=float(cash_account),
        option_premium=float(quote.sale_proceeds),
        theoretical_option_price=float(quote.theoretical_price),
        option_bid_price=float(quote.bid_price),
        option_ask_price=float(quote.ask_price),
        option_sale_proceeds=float(quote.sale_proceeds),
        initial_option_edge=float(quote.initial_edge),
        call_payoff=float(call_payoff),
        total_hedge_pnl=float(total_hedge_pnl),
        total_transaction_cost=float(total_transaction_cost),
        number_of_trades=int(number_of_trades),
        total_turnover=float(total_turnover),
        terminal_futures_price=float(path[-1]),
    )


def hedge_short_call_paths(
    futures_paths,
    strategy,
    config: HedgingConfig,
    futures_spread=None,
) -> BatchHedgingResult:
    """Vectorized path-batch engine with one explicit loop over time."""
    paths = _validate_futures_paths(futures_paths, config)
    number_of_paths = paths.shape[0]
    path_shape = (number_of_paths,)
    execution_spreads = _validate_execution_spread(futures_spread, path_shape, config)
    quote = _initial_option_quote(config)

    cash_accounts = np.full(path_shape, quote.sale_proceeds, dtype=float)
    current_positions = np.zeros(path_shape, dtype=float)
    total_hedge_pnls = np.zeros(path_shape, dtype=float)
    total_transaction_costs = np.zeros(path_shape, dtype=float)
    total_turnovers = np.zeros(path_shape, dtype=float)
    numbers_of_trades = np.zeros(path_shape, dtype=int)
    last_hedge_prices = paths[:, 0].copy()

    initial_deltas = black76_call_delta(F=paths[:, 0], tau=config.T, config=config)
    targets = _get_target_positions(
        strategy=strategy,
        step=0,
        call_delta=initial_deltas,
        current_position=current_positions,
        futures_price=paths[:, 0],
        tau=config.T,
        config=config,
        expected_shape=path_shape,
        last_hedge_price=last_hedge_prices,
        current_futures_spread=execution_spreads,
    )
    initial_trade_mask = _apply_batch_trades(
        position_changes=targets - current_positions,
        target_positions=targets,
        mid_prices=paths[:, 0],
        current_positions=current_positions,
        cash_accounts=cash_accounts,
        total_transaction_costs=total_transaction_costs,
        total_turnovers=total_turnovers,
        numbers_of_trades=numbers_of_trades,
        config=config,
        futures_spread=execution_spreads,
    )
    last_hedge_prices[initial_trade_mask] = paths[initial_trade_mask, 0]

    growth = np.exp(config.r * config.dt)
    for step in range(config.n_steps):
        current_prices = paths[:, step]
        next_prices = paths[:, step + 1]
        cash_accounts *= growth

        hedge_pnls = (
            current_positions
            * (next_prices - current_prices)
            * config.contract_multiplier
        )
        cash_accounts += hedge_pnls
        total_hedge_pnls += hedge_pnls

        next_step = step + 1
        if next_step < config.n_steps:
            tau = config.T - next_step * config.dt
            deltas = black76_call_delta(F=next_prices, tau=tau, config=config)
            targets = _get_target_positions(
                strategy=strategy,
                step=next_step,
                call_delta=deltas,
                current_position=current_positions,
                futures_price=next_prices,
                tau=tau,
                config=config,
                expected_shape=path_shape,
                last_hedge_price=last_hedge_prices,
                current_futures_spread=execution_spreads,
            )
            trade_mask = _apply_batch_trades(
                position_changes=targets - current_positions,
                target_positions=targets,
                mid_prices=next_prices,
                current_positions=current_positions,
                cash_accounts=cash_accounts,
                total_transaction_costs=total_transaction_costs,
                total_turnovers=total_turnovers,
                numbers_of_trades=numbers_of_trades,
                config=config,
                futures_spread=execution_spreads,
            )
            last_hedge_prices[trade_mask] = next_prices[trade_mask]

    zero_positions = np.zeros(path_shape, dtype=float)
    _apply_batch_trades(
        position_changes=-current_positions,
        target_positions=zero_positions,
        mid_prices=paths[:, -1],
        current_positions=current_positions,
        cash_accounts=cash_accounts,
        total_transaction_costs=total_transaction_costs,
        total_turnovers=total_turnovers,
        numbers_of_trades=numbers_of_trades,
        config=config,
        futures_spread=execution_spreads,
    )

    terminal_prices = paths[:, -1].copy()
    call_payoffs = (
        np.maximum(terminal_prices - config.K, 0.0)
        * config.contract_multiplier
    )
    cash_accounts -= call_payoffs

    return BatchHedgingResult(
        option_premium=float(quote.sale_proceeds),
        theoretical_option_price=float(quote.theoretical_price),
        option_bid_price=float(quote.bid_price),
        option_ask_price=float(quote.ask_price),
        option_sale_proceeds=float(quote.sale_proceeds),
        initial_option_edge=float(quote.initial_edge),
        terminal_pnls=cash_accounts.copy(),
        call_payoffs=call_payoffs.copy(),
        total_hedge_pnls=total_hedge_pnls.copy(),
        total_transaction_costs=total_transaction_costs.copy(),
        numbers_of_trades=numbers_of_trades.copy(),
        total_turnovers=total_turnovers.copy(),
        terminal_futures_prices=terminal_prices,
    )


def trace_short_call_path(
    futures_path,
    strategy,
    config: HedgingConfig,
    futures_spread=None,
) -> DetailedPathHedgingTrace:
    """Run one trade and retain a detailed time-series diagnostic trace."""
    path = _validate_single_futures_path(futures_path, config)
    spread_array = _validate_execution_spread(futures_spread, (), config)
    execution_spread = None if spread_array is None else float(spread_array)
    displayed_spread = config.futures_spread if execution_spread is None else execution_spread
    quote = _initial_option_quote(config)

    n_prices = config.n_steps + 1
    time_grid = np.linspace(0.0, config.T, n_prices)
    deltas = np.empty(n_prices, dtype=float)
    positions = np.empty(n_prices, dtype=float)
    trade_changes = np.zeros(n_prices, dtype=float)
    transaction_costs = np.zeros(n_prices, dtype=float)
    hedge_pnl_contributions = np.zeros(config.n_steps, dtype=float)
    interest_accruals = np.zeros(config.n_steps, dtype=float)

    cash_account = float(quote.sale_proceeds)
    current_position = 0.0
    total_hedge_pnl = 0.0
    total_transaction_cost = 0.0
    total_turnover = 0.0
    number_of_trades = 0
    last_hedge_price = float(path[0])

    deltas[0] = float(black76_call_delta(F=path[0], tau=config.T, config=config))
    target = _get_target_position(
        strategy=strategy,
        step=0,
        call_delta=deltas[0],
        current_position=current_position,
        futures_price=path[0],
        tau=config.T,
        config=config,
        last_hedge_price=last_hedge_price,
        current_futures_spread=execution_spread,
    )
    change = target - current_position
    if not np.isclose(change, 0.0):
        cost = float(
            futures_transaction_cost(
                change, path[0], config, futures_spread=execution_spread
            )
        )
        cash_account -= cost
        total_transaction_cost += cost
        total_turnover += abs(change)
        number_of_trades += 1
        current_position = target
        last_hedge_price = float(path[0])
        trade_changes[0] = change
        transaction_costs[0] = cost
    positions[0] = current_position

    growth = np.exp(config.r * config.dt)
    for step in range(config.n_steps):
        current_price = float(path[step])
        next_price = float(path[step + 1])

        interest = cash_account * (growth - 1.0)
        cash_account *= growth
        interest_accruals[step] = interest

        hedge_pnl = (
            current_position
            * (next_price - current_price)
            * config.contract_multiplier
        )
        cash_account += hedge_pnl
        total_hedge_pnl += hedge_pnl
        hedge_pnl_contributions[step] = hedge_pnl

        next_step = step + 1
        tau = config.T - next_step * config.dt
        deltas[next_step] = float(
            black76_call_delta(F=next_price, tau=max(tau, 0.0), config=config)
        )

        if next_step < config.n_steps:
            target = _get_target_position(
                strategy=strategy,
                step=next_step,
                call_delta=deltas[next_step],
                current_position=current_position,
                futures_price=next_price,
                tau=tau,
                config=config,
                last_hedge_price=last_hedge_price,
                current_futures_spread=execution_spread,
            )
            change = target - current_position
            if not np.isclose(change, 0.0):
                cost = float(
                    futures_transaction_cost(
                        change,
                        next_price,
                        config,
                        futures_spread=execution_spread,
                    )
                )
                cash_account -= cost
                total_transaction_cost += cost
                total_turnover += abs(change)
                number_of_trades += 1
                current_position = target
                last_hedge_price = next_price
                trade_changes[next_step] = change
                transaction_costs[next_step] = cost
            positions[next_step] = current_position
        else:
            closing_change = -current_position
            if not np.isclose(closing_change, 0.0):
                cost = float(
                    futures_transaction_cost(
                        closing_change,
                        next_price,
                        config,
                        futures_spread=execution_spread,
                    )
                )
                cash_account -= cost
                total_transaction_cost += cost
                total_turnover += abs(closing_change)
                number_of_trades += 1
                trade_changes[next_step] = closing_change
                transaction_costs[next_step] = cost
                current_position = 0.0
            positions[next_step] = current_position

    call_payoff = max(float(path[-1]) - config.K, 0.0) * config.contract_multiplier
    cash_account -= call_payoff

    result = PathHedgingResult(
        terminal_pnl=float(cash_account),
        option_premium=float(quote.sale_proceeds),
        theoretical_option_price=float(quote.theoretical_price),
        option_bid_price=float(quote.bid_price),
        option_ask_price=float(quote.ask_price),
        option_sale_proceeds=float(quote.sale_proceeds),
        initial_option_edge=float(quote.initial_edge),
        call_payoff=float(call_payoff),
        total_hedge_pnl=float(total_hedge_pnl),
        total_transaction_cost=float(total_transaction_cost),
        number_of_trades=int(number_of_trades),
        total_turnover=float(total_turnover),
        terminal_futures_price=float(path[-1]),
    )

    return DetailedPathHedgingTrace(
        result=result,
        time_grid=time_grid,
        futures_prices=path.copy(),
        black76_deltas=deltas,
        hedge_positions_after_trade=positions,
        hedge_trade_changes=trade_changes,
        transaction_costs_by_trade_time=transaction_costs,
        cumulative_transaction_costs=np.cumsum(transaction_costs),
        hedge_pnl_contributions=hedge_pnl_contributions,
        cumulative_hedge_pnl=np.concatenate(
            ([0.0], np.cumsum(hedge_pnl_contributions))
        ),
        interest_accruals=interest_accruals,
        cumulative_interest_accrual=np.concatenate(
            ([0.0], np.cumsum(interest_accruals))
        ),
        realized_futures_spread=float(displayed_spread),
    )
