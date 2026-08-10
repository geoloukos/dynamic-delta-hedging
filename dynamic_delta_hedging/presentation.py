from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display

from dynamic_delta_hedging.campaign import create_trade_summary
from dynamic_delta_hedging.plotting import plot_pareto_frontier_2d


TABLE_BORDER_COLOR = "#b7b7b7"
SEMANTIC_GREEN = "#16A34A"
SEMANTIC_RED = "#DC2626"
SEMANTIC_BLUE = "#2563EB"
SEMANTIC_YELLOW = "#FACC15"
SEMANTIC_ORANGE = "#EA580C"
SEMANTIC_GRAY = "#4B5563"
PARETO_EFFICIENT_COLOR = SEMANTIC_GREEN
PARETO_DOMINATED_COLOR = SEMANTIC_RED
PAIRED_WIN_COLOR = SEMANTIC_GREEN
PAIRED_LOSS_COLOR = SEMANTIC_RED
PAIRED_UNCLEAR_COLOR = SEMANTIC_BLUE

# Semantic thresholds retained for the realized finite-capital campaign table.
FINAL_CAPITAL_STRONG_GAIN = 1.10
FINAL_CAPITAL_MILD_GAIN = 1.00
FINAL_CAPITAL_MILD_LOSS = 0.90


def _with_table_borders(styler):
    return styler.set_table_styles(
        [
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                ],
            },
            {
                "selector": "th",
                "props": [
                    ("border", f"1px solid {TABLE_BORDER_COLOR}"),
                    ("padding", "6px 8px"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("border", f"1px solid {TABLE_BORDER_COLOR}"),
                    ("padding", "6px 8px"),
                ],
            },
        ],
        overwrite=False,
    )


def _neutral_style_frame(table):
    return pd.DataFrame(
        _semantic_cell_style(SEMANTIC_GRAY),
        index=table.index,
        columns=table.columns,
    )


def _style_table(table, hide_index=True):
    styler = table.style
    if hide_index:
        styler = styler.hide(axis="index")
    styles = _neutral_style_frame(table)
    styler = styler.apply(lambda _: styles, axis=None)
    return _with_table_borders(styler)


def _format_signed_eur(value):
    value = float(value)
    sign = "+" if value >= 0.0 else "-"
    return f"{sign}€{abs(value):.2f}"


def _semantic_cell_style(background_color, text_color="white"):
    return (
        f"background-color: {background_color}; "
        f"color: {text_color}; "
        "font-weight: 700;"
    )


def _style_cost_es_frontier_table(table):
    styles = _neutral_style_frame(table)
    for row_index in table.index:
        value = table.at[row_index, "Primary Frontier"]
        color = PARETO_EFFICIENT_COLOR if value == "Efficient" else PARETO_DOMINATED_COLOR
        styles.at[row_index, "Primary Frontier"] = _semantic_cell_style(color)
    styler = table.style.hide(axis="index").apply(lambda _: styles, axis=None)
    return _with_table_borders(styler)


def _style_pairwise_matrix(matrix, status_matrix):
    styles = _neutral_style_frame(matrix)
    for row_index in matrix.index:
        for column in matrix.columns:
            status = status_matrix.at[row_index, column]
            if status == "win":
                styles.at[row_index, column] = _semantic_cell_style(PAIRED_WIN_COLOR)
            elif status == "loss":
                styles.at[row_index, column] = _semantic_cell_style(PAIRED_LOSS_COLOR)
            elif status == "unclear":
                styles.at[row_index, column] = _semantic_cell_style(PAIRED_UNCLEAR_COLOR)
    styler = matrix.style.hide(axis="index").apply(lambda _: styles, axis=None)
    return _with_table_borders(styler)


def _display_pairwise_with_legends(matrix, status_matrix, strategy_legend):
    matrix_html = _style_pairwise_matrix(matrix, status_matrix).to_html()

    strategy_legend_styles = _neutral_style_frame(strategy_legend)
    strategy_legend_style = _with_table_borders(
        strategy_legend.style.hide(axis="index")
        .apply(lambda _: strategy_legend_styles, axis=None)
        .set_caption("Strategy Legend")
    )

    color_legend = pd.DataFrame(
        {
            "Result": ["Win", "Loss", "Inconclusive"],
            "Meaning": [
                "Row strategy statistically outperforms",
                "Row strategy statistically underperforms",
                "95% CI includes zero; statistically indistinguishable",
            ],
        }
    )
    color_styles = pd.DataFrame(
        [
            [_semantic_cell_style(PAIRED_WIN_COLOR), _semantic_cell_style(SEMANTIC_GRAY)],
            [_semantic_cell_style(PAIRED_LOSS_COLOR), _semantic_cell_style(SEMANTIC_GRAY)],
            [_semantic_cell_style(PAIRED_UNCLEAR_COLOR), _semantic_cell_style(SEMANTIC_GRAY)],
        ],
        columns=color_legend.columns,
        index=color_legend.index,
    )
    color_legend_style = _with_table_borders(
        color_legend.style
        .hide(axis="index")
        .apply(lambda _: color_styles, axis=None)
        .set_caption("Color Legend")
    )

    display(
        HTML(
            '<div style="display:flex; align-items:flex-start; gap:24px; flex-wrap:wrap;">'
            f'<div>{matrix_html}</div>'
            '<div style="display:flex; flex-direction:column; gap:14px;">'
            f'{strategy_legend_style.to_html()}'
            f'{color_legend_style.to_html()}'
            '</div>'
            '</div>'
        )
    )


def _format_eur_columns(table, columns):
    displayed = table.copy()
    for column in columns:
        if column in displayed.columns:
            displayed[column] = displayed[column].astype(object)
            displayed[column] = displayed[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.2f}"
            )
    return displayed


def _format_trade_columns(table, columns=("Mean Trades",)):
    displayed = table.copy()
    for column in columns:
        if column in displayed.columns:
            displayed[column] = displayed[column].map(
                lambda value: "" if pd.isna(value) else int(np.floor(float(value) + 0.5))
            )
    return displayed


def _format_turnover_columns(table, columns=("Mean Turnover", "Turnover")):
    displayed = table.copy()
    for column in columns:
        if column in displayed.columns:
            displayed[column] = displayed[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.2f}"
            )
    return displayed


def _create_pairwise_presentation(pairwise_table, evaluated_metrics, candidate_strategy_ids):
    strategy_ids = [str(strategy_id).strip().upper() for strategy_id in candidate_strategy_ids]
    id_to_name = dict(zip(
        evaluated_metrics["strategy_id"],
        evaluated_metrics["strategy_name"],
    ))

    columns = [f"vs {strategy_id}" for strategy_id in strategy_ids]
    matrix = pd.DataFrame("", index=strategy_ids, columns=columns, dtype=object)
    status_matrix = pd.DataFrame("", index=strategy_ids, columns=columns, dtype=object)

    for strategy_id in strategy_ids:
        matrix.loc[strategy_id, f"vs {strategy_id}"] = "—"
        status_matrix.loc[strategy_id, f"vs {strategy_id}"] = "diagonal"

    for row in pairwise_table.itertuples(index=False):
        strategy_a_id = str(row.strategy_a_id).strip().upper()
        strategy_b_id = str(row.strategy_b_id).strip().upper()
        mean_difference = float(row.mean_pnl_difference)
        statistically_clear = (
            row.confidence_interval_lower > 0.0
            or row.confidence_interval_upper < 0.0
        )

        matrix.loc[strategy_a_id, f"vs {strategy_b_id}"] = _format_signed_eur(
            mean_difference
        )
        matrix.loc[strategy_b_id, f"vs {strategy_a_id}"] = _format_signed_eur(
            -mean_difference
        )

        if not statistically_clear:
            status_matrix.loc[strategy_a_id, f"vs {strategy_b_id}"] = "unclear"
            status_matrix.loc[strategy_b_id, f"vs {strategy_a_id}"] = "unclear"
        elif mean_difference > 0.0:
            status_matrix.loc[strategy_a_id, f"vs {strategy_b_id}"] = "win"
            status_matrix.loc[strategy_b_id, f"vs {strategy_a_id}"] = "loss"
        else:
            status_matrix.loc[strategy_a_id, f"vs {strategy_b_id}"] = "loss"
            status_matrix.loc[strategy_b_id, f"vs {strategy_a_id}"] = "win"

    matrix.index.name = "Strategy ID"
    status_matrix.index.name = "Strategy ID"
    matrix = matrix.reset_index()
    status_matrix = status_matrix.reset_index()
    status_matrix["Strategy ID"] = ""

    strategy_legend = pd.DataFrame(
        {
            "ID": strategy_ids,
            "Strategy": [id_to_name[strategy_id] for strategy_id in strategy_ids],
        }
    )
    return matrix, status_matrix, strategy_legend


def display_trade_inception(theoretical_option_price, fair_contract_value, initial_black76_delta):
    table = pd.DataFrame([
        {
            "Black-76 Theoretical Price (price units)": theoretical_option_price,
            "Contract Fair Value (€)": fair_contract_value,
            "Initial Black-76 Delta": initial_black76_delta,
        }
    ])
    displayed = _format_eur_columns(table, ["Contract Fair Value (€)"])
    display(_style_table(displayed))


def display_desk_mandate(max_pnl_std, min_es_5, max_mean_cost):
    table = pd.DataFrame([
        {
            "Max P&L Std (€)": max_pnl_std,
            "Min ES 5% (€)": min_es_5,
            "Max Mean Hedging Cost (€)": max_mean_cost,
        }
    ])
    displayed = _format_eur_columns(
        table,
        ["Max P&L Std (€)", "Min ES 5% (€)", "Max Mean Hedging Cost (€)"],
    )
    display(_style_table(displayed))


def display_feasible_strategies(feasible_with_quotes, research_status, feasible_strategy_ids):
    displayed = feasible_with_quotes[[
        "strategy_id",
        "strategy_name",
        "mean_pnl",
        "std_pnl",
        "expected_shortfall",
        "mean_transaction_cost",
        "mean_number_of_trades",
        "mean_turnover",
        "break_even_ask_eur",
        "recommended_ask_eur",
    ]].rename(columns={
        "strategy_id": "Strategy ID",
        "strategy_name": "Strategy Name",
        "mean_pnl": "Mean P&L (€)",
        "std_pnl": "P&L Std (€)",
        "expected_shortfall": "ES 5% (€)",
        "mean_transaction_cost": "Mean Hedging Cost (€)",
        "mean_number_of_trades": "Mean Trades",
        "mean_turnover": "Mean Turnover",
        "break_even_ask_eur": "Break-even Ask (€)",
        "recommended_ask_eur": "Recommended Ask (€)",
    })
    displayed = _format_eur_columns(
        displayed,
        [
            "Mean P&L (€)",
            "P&L Std (€)",
            "ES 5% (€)",
            "Mean Hedging Cost (€)",
            "Break-even Ask (€)",
            "Recommended Ask (€)",
        ],
    )
    displayed = _format_trade_columns(displayed)
    displayed = _format_turnover_columns(displayed)
    displayed.index = np.arange(1, len(displayed) + 1)

    print(
        f"Research status: {research_status}; "
        f"feasible strategies: {feasible_strategy_ids}"
    )
    display(_style_table(displayed, hide_index=False))


def display_cost_es_frontier(frontier_table, n_eligible, figures_dir="figures"):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(exist_ok=True)

    if n_eligible >= 2:
        metrics_by_family = {}
        for family_name, family_table in frontier_table.groupby("strategy_family", sort=False):
            metrics_by_family[family_name] = (family_table, "strategy_parameter_value")

        plot_pareto_frontier_2d(
            metrics_by_family=metrics_by_family,
            x_metric="mean_transaction_cost",
            y_metric="tail_loss_5",
            x_label="Mean Hedging Cost (€)",
            y_label="Tail Loss 5% (€)",
            title="Primary Cost–ES Efficient Hedging Frontier",
            x_minimize=True,
            y_minimize=True,
            annotation_column="strategy_id",
            save_path=figures_dir / "v2_feasible_cost_vs_es5.png",
        )
        plt.show()
        plt.close()

    displayed = frontier_table[[
        "strategy_id",
        "strategy_name",
        "mean_transaction_cost",
        "tail_loss_5",
        "cost_es_efficient",
    ]].rename(columns={
        "strategy_id": "Strategy ID",
        "strategy_name": "Strategy Name",
        "mean_transaction_cost": "Mean Hedging Cost (€)",
        "tail_loss_5": "Tail Loss 5% (€)",
        "cost_es_efficient": "Primary Frontier",
    })
    displayed["Primary Frontier"] = displayed["Primary Frontier"].map(
        {True: "Efficient", False: "Dominated"}
    )
    displayed = _format_eur_columns(
        displayed,
        ["Mean Hedging Cost (€)", "Tail Loss 5% (€)"],
    )
    display(_style_cost_es_frontier_table(displayed))


def display_pairwise_comparisons(
    pairwise_table,
    evaluated_metrics,
    candidate_strategy_ids,
):
    matrix, status_matrix, strategy_legend = _create_pairwise_presentation(
        pairwise_table,
        evaluated_metrics,
        candidate_strategy_ids,
    )
    _display_pairwise_with_legends(matrix, status_matrix, strategy_legend)


def display_frontier_stability(frontier_stability_table):
    displayed = frontier_stability_table[[
        "strategy_id",
        "strategy_name",
        "cost_es_efficient",
        "frontier_stability",
    ]].rename(columns={
        "strategy_id": "Strategy ID",
        "strategy_name": "Strategy Name",
        "cost_es_efficient": "Primary Frontier",
        "frontier_stability": "Bootstrap Frontier Stability",
    }).copy()
    displayed["Primary Frontier"] = displayed["Primary Frontier"].map(
        {True: "Efficient", False: "Dominated"}
    )
    displayed["Bootstrap Frontier Stability"] = displayed[
        "Bootstrap Frontier Stability"
    ].map(lambda value: f"{100.0 * float(value):.1f}%")

    styles = _neutral_style_frame(displayed)
    for row_index in displayed.index:
        color = (
            PARETO_EFFICIENT_COLOR
            if displayed.at[row_index, "Primary Frontier"] == "Efficient"
            else PARETO_DOMINATED_COLOR
        )
        styles.at[row_index, "Primary Frontier"] = _semantic_cell_style(color)
    display(
        _with_table_borders(
            displayed.style.hide(axis="index").apply(lambda _: styles, axis=None)
        )
    )


def display_selected_quote(selected_strategy, selected_quote, target_expected_terminal_profit_eur):
    table = pd.DataFrame([
        {
            "Strategy ID": selected_strategy,
            "Strategy Name": selected_quote.strategy_name,
            "Black-76 Fair Contract Value (€)": selected_quote.fair_contract_value,
            "Break-even Ask (€)": selected_quote.break_even_ask_eur,
            "Recommended Ask (€)": selected_quote.recommended_ask_eur,
            "Target Expected Terminal Profit (€)": target_expected_terminal_profit_eur,
            "Implied Full Option Spread (%)": 100.0 * float(selected_quote.recommended_option_spread),
        }
    ])
    displayed = _format_eur_columns(
        table,
        [
            "Black-76 Fair Contract Value (€)",
            "Break-even Ask (€)",
            "Recommended Ask (€)",
            "Target Expected Terminal Profit (€)",
        ],
    )
    displayed["Implied Full Option Spread (%)"] = displayed[
        "Implied Full Option Spread (%)"
    ].map(lambda value: f"{float(value):.2f}")
    display(_style_table(displayed))


def display_volatility_robustness_summary(robustness_table):
    required_scenarios = (
        "lower_realized_volatility",
        "baseline",
        "higher_realized_volatility",
    )
    observed = tuple(dict.fromkeys(robustness_table["scenario"]))
    if set(observed) != set(required_scenarios):
        raise ValueError("Volatility robustness requires exactly the 0.75×, 1.00× and 1.25× scenarios.")

    rows = []
    for strategy_id, strategy_rows in robustness_table.groupby("strategy_id", sort=False):
        by_scenario = strategy_rows.set_index("scenario")
        rows.append({
            "Strategy ID": strategy_id,
            "Strategy Name": by_scenario.iloc[0]["strategy_name"],
            "Mean P&L 0.75σ (€)": by_scenario.loc["lower_realized_volatility", "mean_pnl"],
            "Mean P&L 1.00σ (€)": by_scenario.loc["baseline", "mean_pnl"],
            "Mean P&L 1.25σ (€)": by_scenario.loc["higher_realized_volatility", "mean_pnl"],
            "ES 5% 0.75σ (€)": by_scenario.loc["lower_realized_volatility", "expected_shortfall"],
            "ES 5% 1.00σ (€)": by_scenario.loc["baseline", "expected_shortfall"],
            "ES 5% 1.25σ (€)": by_scenario.loc["higher_realized_volatility", "expected_shortfall"],
        })
    raw = pd.DataFrame(rows)
    displayed = _format_eur_columns(
        raw,
        [
            "Mean P&L 0.75σ (€)", "Mean P&L 1.00σ (€)", "Mean P&L 1.25σ (€)",
            "ES 5% 0.75σ (€)", "ES 5% 1.00σ (€)", "ES 5% 1.25σ (€)",
        ],
    )
    styles = _neutral_style_frame(displayed)
    for index in displayed.index:
        baseline_mean = float(raw.at[index, "Mean P&L 1.00σ (€)"])
        baseline_es = float(raw.at[index, "ES 5% 1.00σ (€)"])
        for column in ("Mean P&L 0.75σ (€)", "Mean P&L 1.25σ (€)"):
            value = float(raw.at[index, column])
            if not np.isclose(value, baseline_mean):
                styles.at[index, column] = _semantic_cell_style(
                    SEMANTIC_GREEN if value > baseline_mean else SEMANTIC_RED
                )
        for column in ("ES 5% 0.75σ (€)", "ES 5% 1.25σ (€)"):
            value = float(raw.at[index, column])
            if not np.isclose(value, baseline_es):
                styles.at[index, column] = _semantic_cell_style(
                    SEMANTIC_GREEN if value > baseline_es else SEMANTIC_RED
                )
    display(
        _with_table_borders(
            displayed.style.hide(axis="index").apply(lambda _: styles, axis=None)
        )
    )


def style_convergence_rate_summary(rate_summary):
    styles = _neutral_style_frame(rate_summary)
    return (
        rate_summary.style.hide(axis="index")
        .apply(lambda _: styles, axis=None)
        .format({"Value": "{:.4f}"})
        .set_table_styles(
            [
                {"selector": "table", "props": [("border-collapse", "collapse")]},
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#eeeeee"),
                        ("font-weight", "bold"),
                        ("padding", "7px"),
                        ("border", "1px solid #b7b7b7"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("padding", "6px 10px"),
                        ("border", "1px solid #b7b7b7"),
                    ],
                },
            ]
        )
        .set_caption("Empirical convergence-rate diagnostics")
    )


def _drawdown_style(value, limit):
    ratio = float(value) / float(limit) if float(limit) > 0.0 else np.inf
    if ratio <= 0.25:
        return _semantic_cell_style(SEMANTIC_GREEN)
    if ratio <= 0.50:
        return _semantic_cell_style(SEMANTIC_YELLOW, "black")
    if ratio <= 0.75:
        return _semantic_cell_style(SEMANTIC_ORANGE)
    return _semantic_cell_style(SEMANTIC_RED)


def _final_capital_style(value, starting_capital):
    ratio = float(value) / float(starting_capital)
    if ratio >= FINAL_CAPITAL_STRONG_GAIN:
        return _semantic_cell_style(SEMANTIC_GREEN)
    if ratio > FINAL_CAPITAL_MILD_GAIN:
        return _semantic_cell_style(SEMANTIC_YELLOW, "black")
    if np.isclose(ratio, FINAL_CAPITAL_MILD_GAIN, rtol=0.0, atol=0.005):
        return _semantic_cell_style(SEMANTIC_GRAY)
    if ratio >= FINAL_CAPITAL_MILD_LOSS:
        return _semantic_cell_style(SEMANTIC_ORANGE)
    return _semantic_cell_style(SEMANTIC_RED)


def display_locked_campaign_setup(
    *,
    strategy_id,
    strategy_name,
    recommended_ask_eur,
    recommended_option_spread,
    fixed_futures_spread,
    starting_capital,
    risk_budget_fraction,
    loss_reference_per_contract,
    max_drawdown_limit,
    n_campaign_trades,
    campaign_seed,
):
    """Display the explicit post-decision strategy, quote, sizing rule and limits."""
    seed_display = "Fresh / Random" if campaign_seed is None else str(int(campaign_seed))
    table = pd.DataFrame(
        {
            "Locked Item": [
                "Strategy",
                "Research-derived Option Ask",
                "Research-derived Option Spread",
                "Fixed Futures Execution Spread",
                "Starting Capital",
                "Risk Budget Fraction",
                "Research Downside Loss Reference per Contract",
                "Maximum Drawdown Limit",
                "Planned Campaign Trades",
                "Campaign Seed",
            ],
            "Value": [
                f"{str(strategy_id).strip().upper()} — {strategy_name}",
                f"€{float(recommended_ask_eur):.2f}",
                f"{100.0 * float(recommended_option_spread):.4f}%",
                f"{100.0 * float(fixed_futures_spread):.3f}%",
                f"€{float(starting_capital):,.2f}",
                f"{100.0 * float(risk_budget_fraction):.2f}% of pre-trade capital",
                f"€{float(loss_reference_per_contract):.2f} per contract",
                f"{100.0 * float(max_drawdown_limit):.2f}%",
                int(n_campaign_trades),
                seed_display,
            ],
        }
    )
    display(_style_table(table))


def display_realized_campaign(realized_campaign):
    raw = realized_campaign.trade_table.copy()
    displayed = raw.rename(columns={
        "trade": "Trade",
        "realized_sigma": "Realized σ",
        "contracts": "Contracts",
        "per_contract_pnl": "Per-Contract P&L (€)",
        "total_trade_pnl": "Total Trade P&L (€)",
        "capital_after_trade": "Capital After Trade (€)",
        "drawdown": "Drawdown",
        "status": "Status",
    }).copy()
    displayed["Realized σ"] = displayed["Realized σ"].map(
        lambda v: "—" if pd.isna(v) else f"{100.0 * float(v):.2f}%"
    )
    for column in ["Per-Contract P&L (€)", "Total Trade P&L (€)", "Capital After Trade (€)"]:
        displayed[column] = displayed[column].map(
            lambda v: "—" if pd.isna(v) else f"{float(v):,.2f}"
        )
    displayed["Drawdown"] = displayed["Drawdown"].map(
        lambda v: f"{100.0 * float(v):.2f}%"
    )

    styles = _neutral_style_frame(displayed)
    for index in displayed.index:
        if raw.at[index, "contracts"] > 0:
            for column, raw_column in [
                ("Per-Contract P&L (€)", "per_contract_pnl"),
                ("Total Trade P&L (€)", "total_trade_pnl"),
            ]:
                value = float(raw.at[index, raw_column])
                if value > 0.0:
                    styles.at[index, column] = _semantic_cell_style(SEMANTIC_GREEN)
                elif value < 0.0:
                    styles.at[index, column] = _semantic_cell_style(SEMANTIC_RED)
            styles.at[index, "Capital After Trade (€)"] = _final_capital_style(
                raw.at[index, "capital_after_trade"], realized_campaign.starting_capital
            )
            styles.at[index, "Drawdown"] = _drawdown_style(
                raw.at[index, "drawdown"], realized_campaign.max_drawdown_limit
            )
        status = str(raw.at[index, "status"])
        if status == "RISK LIMIT BREACH":
            styles.at[index, "Status"] = _semantic_cell_style(SEMANTIC_RED)
        elif status == "NO NEW TRADE (SIZE 0)":
            styles.at[index, "Status"] = _semantic_cell_style(SEMANTIC_ORANGE)
        elif status == "COMPLETED":
            styles.at[index, "Status"] = _semantic_cell_style(SEMANTIC_GREEN)
    display(
        _with_table_borders(
            displayed.style.hide(axis="index").apply(lambda _: styles, axis=None)
        )
    )


def display_trade_postmortem(realized_campaign, trade_number, *, trace=None):
    """Display diagnostics for the user-selected realized campaign trade."""
    facts = create_trade_summary(
        realized_campaign,
        trade_number,
        trace=trace,
    )
    if trace is None:
        from dynamic_delta_hedging.campaign import build_trade_trace

        trace = build_trade_trace(realized_campaign, trade_number)

    summary = pd.DataFrame([
        {
            "Selected Trade": facts["trade"],
            "Realized σ": 100.0 * facts["realized_sigma"],
            "Terminal Futures Price": facts["terminal_futures_price"],
            "Strike": facts["strike"],
            "Expiry Status": facts["expiry_status"],
            "Contracts": facts["contracts"],
            "Hedge Trades": facts["hedge_trades"],
            "Transaction Cost (€)": facts["transaction_cost"],
            "Hedge P&L (€)": facts["hedge_pnl"],
            "Option Payoff (€)": facts["option_payoff"],
            "Per-Contract Terminal P&L (€)": facts["per_contract_terminal_pnl"],
            "Total Trade P&L (€)": facts["total_trade_pnl"],
        }
    ])
    displayed = summary.copy()
    displayed["Realized σ"] = displayed["Realized σ"].map(lambda v: f"{float(v):.2f}%")
    for column in [
        "Terminal Futures Price", "Strike", "Transaction Cost (€)",
        "Hedge P&L (€)", "Option Payoff (€)", "Per-Contract Terminal P&L (€)",
        "Total Trade P&L (€)",
    ]:
        displayed[column] = displayed[column].map(lambda v: f"{float(v):,.2f}")
    styles = _neutral_style_frame(displayed)
    styles.at[0, "Expiry Status"] = _semantic_cell_style(SEMANTIC_GRAY)
    for column, value in [
        ("Per-Contract Terminal P&L (€)", facts["per_contract_terminal_pnl"]),
        ("Total Trade P&L (€)", facts["total_trade_pnl"]),
    ]:
        styles.at[0, column] = _semantic_cell_style(
            SEMANTIC_GREEN if value > 0.0 else (SEMANTIC_RED if value < 0.0 else SEMANTIC_GRAY)
        )
    styles.at[0, "Transaction Cost (€)"] = _semantic_cell_style(SEMANTIC_RED)
    display(
        _with_table_borders(
            displayed.style.hide(axis="index").apply(lambda _: styles, axis=None)
        )
    )

    labels = {
        "option_sale_proceeds": "Option Sale Proceeds",
        "cash_interest_accrual": "Cash Interest Accrual",
        "hedge_pnl": "Hedge P&L",
        "transaction_cost": "Transaction Costs",
        "option_payoff_liability": "Option Payoff Liability",
        "terminal_pnl": "Terminal P&L",
    }
    decomposition_values = trace.terminal_pnl_decomposition
    decomposition = pd.DataFrame({
        "Component": [labels[key] for key in decomposition_values],
        "Per-Contract Contribution (€)": list(decomposition_values.values()),
    })
    decomposition["Per-Contract Contribution (€)"] = decomposition[
        "Per-Contract Contribution (€)"
    ].map(lambda v: f"{float(v):,.2f}")
    display(_style_table(decomposition))
