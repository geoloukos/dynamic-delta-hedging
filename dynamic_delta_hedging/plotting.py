from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _validate_columns(table: pd.DataFrame, required_columns, ) -> None:
    """Verify that a metrics table contains all columns required by a plot."""
    missing_columns = [
        column for column in required_columns if column not in table.columns
    ]
    if missing_columns:
        raise ValueError(f"Metrics table is missing columns: {missing_columns}")


def _finalize_figure(figure, save_path=None) -> None:
    """Apply a clean layout and optionally save the figure to disk."""
    figure.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")


def _find_pareto_efficient_points_2d(
    x_values,
    y_values,
    *,
    x_minimize: bool,
    y_minimize: bool,
) -> np.ndarray:
    """Identify 2D Pareto-efficient observations for arbitrary directions."""
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    if x.shape != y.shape:
        raise ValueError("Pareto objective arrays must have the same shape.")

    x_objective = x if x_minimize else -x
    y_objective = y if y_minimize else -y

    efficient = np.ones(x.size, dtype=bool)
    for point_index in range(x.size):
        dominated = np.any(
            (x_objective <= x_objective[point_index])
            & (y_objective <= y_objective[point_index])
            & (
                (x_objective < x_objective[point_index])
                | (y_objective < y_objective[point_index])
            )
        )
        efficient[point_index] = not dominated

    return efficient


def plot_pareto_frontier_2d(
    metrics_by_family,
    *,
    x_metric: str,
    y_metric: str,
    x_label: str,
    y_label: str,
    title: str,
    x_minimize: bool,
    y_minimize: bool,
    annotation_column: str = "strategy_id",
    save_path=None,
):
    """
    Plot a two-objective Pareto view across all strategy families.

    The objective direction is explicit for each axis, so the same plot
    supports cost/Std minimization and Expected Shortfall maximization
    without transforming the displayed metric values.
    """
    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    all_x = []
    all_y = []

    for family_name, family_data in metrics_by_family.items():
        metrics_table, parameter_name = family_data
        required_columns = [parameter_name, x_metric, y_metric]
        if annotation_column is not None:
            required_columns.append(annotation_column)
        _validate_columns(metrics_table, required_columns)

        sorted_table = metrics_table.sort_values(parameter_name).reset_index(drop=True)
        x_values = sorted_table[x_metric].to_numpy(dtype=float)
        y_values = sorted_table[y_metric].to_numpy(dtype=float)

        all_x.extend(x_values)
        all_y.extend(y_values)

        axis.plot(
            x_values,
            y_values,
            marker="o",
            markersize=6,
            linewidth=1.8,
            label=family_name,
            zorder=3,
        )

        if annotation_column is not None:
            for x_value, y_value, annotation in zip(
                x_values,
                y_values,
                sorted_table[annotation_column],
            ):
                axis.annotate(
                    str(annotation),
                    xy=(x_value, y_value),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=8.5,
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.75,
                    },
                    zorder=5,
                )

    all_x = np.asarray(all_x, dtype=float)
    all_y = np.asarray(all_y, dtype=float)

    if all_x.size:
        efficient_mask = _find_pareto_efficient_points_2d(
            all_x,
            all_y,
            x_minimize=x_minimize,
            y_minimize=y_minimize,
        )
        frontier_x = all_x[efficient_mask]
        frontier_y = all_y[efficient_mask]
        sorting_indices = np.argsort(frontier_x)

        axis.plot(
            frontier_x[sorting_indices],
            frontier_y[sorting_indices],
            color="black",
            linestyle="--",
            linewidth=1.4,
            marker="x",
            markersize=7,
            label="Pareto-efficient frontier",
            zorder=4,
        )

    axis.set_title(title, pad=12)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(visible=True, alpha=0.25)
    axis.margins(x=0.07, y=0.10)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=True,
    )

    _finalize_figure(figure=figure, save_path=save_path)
    return figure, axis


def _trade_review_title(detail: str, trade_number=None) -> str:
    if trade_number is None:
        return f"Selected trade: {detail}"
    return f"Selected trade {int(trade_number)}: {detail}"

def plot_realized_campaign_equity(realized_campaign, save_path=None):
    table = realized_campaign.trade_table
    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    trades = table["trade"].to_numpy(dtype=int)
    capital = table["capital_after_trade"].to_numpy(dtype=float)
    axis.plot(trades, capital, marker="o", linewidth=1.8, label="Capital")
    axis.axhline(
        realized_campaign.starting_capital,
        linestyle="--",
        linewidth=1.2,
        label="Starting capital",
    )
    axis.set_title("Realized campaign equity curve")
    axis.set_xlabel("Trade")
    axis.set_ylabel("Capital (€)")
    axis.grid(visible=True, alpha=0.3)
    axis.legend()
    _finalize_figure(figure, save_path)
    return figure, axis


def plot_realized_campaign_drawdown(realized_campaign, save_path=None):
    table = realized_campaign.trade_table
    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    trades = table["trade"].to_numpy(dtype=int)
    drawdown = 100.0 * table["drawdown"].to_numpy(dtype=float)
    axis.plot(trades, drawdown, marker="o", linewidth=1.8, label="Drawdown")
    axis.axhline(
        100.0 * realized_campaign.max_drawdown_limit,
        linestyle="--",
        linewidth=1.2,
        label="Risk limit",
    )
    axis.set_title("Realized campaign drawdown")
    axis.set_xlabel("Trade")
    axis.set_ylabel("Drawdown (%)")
    axis.grid(visible=True, alpha=0.3)
    axis.legend()
    _finalize_figure(figure, save_path)
    return figure, axis


def plot_realized_campaign_trade_pnls(realized_campaign, save_path=None):
    table = realized_campaign.trade_table
    executed = table["contracts"].to_numpy(dtype=int) > 0
    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    axis.bar(
        table.loc[executed, "trade"].to_numpy(dtype=int),
        table.loc[executed, "total_trade_pnl"].to_numpy(dtype=float),
    )
    axis.axhline(0.0, linewidth=1.0)
    axis.set_title("Realized campaign trade P&Ls")
    axis.set_xlabel("Trade")
    axis.set_ylabel("Total trade P&L (€)")
    axis.grid(visible=True, axis="y", alpha=0.3)
    _finalize_figure(figure, save_path)
    return figure, axis


def plot_trade_futures_path(trace, trade_number=None, save_path=None):
    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    axis.plot(trace.time_grid, trace.futures_prices, linewidth=1.8)
    axis.set_title(_trade_review_title("realized futures path", trade_number))
    axis.set_xlabel("Time (years)")
    axis.set_ylabel("Futures price")
    axis.grid(visible=True, alpha=0.3)
    _finalize_figure(figure, save_path)
    return figure, axis


def plot_trade_hedge(trace, trade_number=None, save_path=None):
    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    axis.plot(trace.time_grid, trace.black76_deltas, linewidth=1.8, label="Black-76 delta")
    axis.plot(
        trace.time_grid,
        trace.hedge_positions_after_trade,
        linewidth=1.8,
        label="Hedge position",
    )
    trade_mask = ~np.isclose(trace.hedge_trade_changes, 0.0)
    axis.scatter(
        trace.time_grid[trade_mask],
        trace.hedge_positions_after_trade[trade_mask],
        marker="o",
        label="Hedge trades",
        zorder=4,
    )
    axis.set_title(_trade_review_title("delta and hedge position", trade_number))
    axis.set_xlabel("Time (years)")
    axis.set_ylabel("Futures contracts per option contract")
    axis.grid(visible=True, alpha=0.3)
    axis.legend()
    _finalize_figure(figure, save_path)
    return figure, axis


def plot_trade_cost_and_hedge_pnl(trace, trade_number=None, save_path=None):
    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    axis.plot(
        trace.time_grid,
        trace.cumulative_transaction_costs,
        linewidth=1.8,
        label="Cumulative transaction cost",
    )
    axis.plot(
        trace.time_grid,
        trace.cumulative_hedge_pnl,
        linewidth=1.8,
        label="Cumulative hedge P&L",
    )
    axis.set_title(_trade_review_title("hedge P&L and transaction costs", trade_number))
    axis.set_xlabel("Time (years)")
    axis.set_ylabel("Contract-level €")
    axis.grid(visible=True, alpha=0.3)
    axis.legend()
    _finalize_figure(figure, save_path)
    return figure, axis
