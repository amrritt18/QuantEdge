import pandas as pd

from quantedge.analytics import annualized_volatility
from quantedge.backtest import backtest_long_cash
from quantedge.risk import (
    calculate_cagr,
    calculate_historical_var,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
)
from quantedge.strategies import (
    momentum_strategy,
    moving_average_crossover,
)


def compare_strategies(
    prices: pd.Series,
    short_window: int = 20,
    long_window: int = 50,
    momentum_lookback: int = 20,
    transaction_cost: float = 0.001,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    var_confidence: float = 0.95,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compare two long/cash strategies with buy-and-hold.

    All strategies use the same evaluation dates. Signals are
    shifted by the backtester, so a signal based on a closing
    price cannot earn that same period's return.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series.")

    if len(prices) < 2:
        raise ValueError("At least two prices are required.")

    if not prices.index.is_unique:
        raise ValueError("Price index must not contain duplicates.")

    if not prices.index.is_monotonic_increasing:
        raise ValueError("Prices must be sorted by date.")

    if prices.isna().any() or (prices <= 0).any():
        raise ValueError("Prices must be positive and non-missing.")

    if not 0 <= transaction_cost < 1:
        raise ValueError("transaction_cost must be in [0, 1).")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    if not 0 < var_confidence < 1:
        raise ValueError("var_confidence must be between 0 and 1.")

    signals = {
        "Moving Average": moving_average_crossover(prices, short_window, long_window),
        "Momentum": momentum_strategy(prices, momentum_lookback),
    }

    # Use the same evaluation window for all strategies.
    valid = pd.concat(signals, axis=1).notna().all(axis=1)
    common_prices = prices.loc[valid]

    if len(common_prices) < 3:
        raise ValueError("Not enough common observations after strategy warm-up.")

    results: dict[str, pd.DataFrame] = {}

    for name, signal in signals.items():
        results[name] = backtest_long_cash(
            common_prices,
            signal.loc[valid],
            transaction_cost=transaction_cost,
        )

    # Buy-and-hold is always exposed to the market over the
    # common evaluation window. Use the same backtest convention.
    buy_hold_signal = pd.Series(1.0, index=common_prices.index, name="Buy and Hold")
    results["Buy and Hold"] = backtest_long_cash(
        common_prices,
        buy_hold_signal,
        transaction_cost=transaction_cost,
    )

    summary_rows = []

    for name, result in results.items():
        returns = result["strategy_return"]
        equity = result["strategy_equity"]

        summary_rows.append(
            {
                "Strategy": name,
                "Total Return": float(equity.iloc[-1] - 1),
                "CAGR": calculate_cagr(equity, periods_per_year),
                "Annualized Volatility": annualized_volatility(
                    returns, periods_per_year
                ),
                "Sharpe Ratio": calculate_sharpe_ratio(
                    returns,
                    risk_free_rate=risk_free_rate,
                    periods_per_year=periods_per_year,
                ),
                "Maximum Drawdown": calculate_max_drawdown(equity),
                "Historical VaR": calculate_historical_var(
                    returns, confidence_level=var_confidence
                ),
            }
        )

    summary = pd.DataFrame(summary_rows).set_index("Strategy")

    return summary, results
