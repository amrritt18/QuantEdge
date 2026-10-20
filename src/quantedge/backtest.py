import pandas as pd


def backtest_long_cash(
    prices: pd.Series,
    signals: pd.Series,
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    """
    Backtest a long/cash strategy.

    Signals are generated at time t and applied to the
    return from t+1 onward, preventing look-ahead bias.

    transaction_cost is the cost per unit of position turnover.
    Example: 0.001 = 0.1%.
    """
    if not 0 <= transaction_cost < 1:
        raise ValueError("transaction_cost must be in [0, 1).")

    if prices.empty or prices.isna().any() or (prices <= 0).any():
        raise ValueError("Prices must be positive and non-missing.")

    if not prices.index.equals(signals.index):
        raise ValueError("Prices and signals must have identical indexes.")

    if signals.isna().any():
        raise ValueError(
            "Signals contain unavailable periods. "
            "Drop warm-up periods before backtesting."
        )

    if not signals.isin([0, 1]).all():
        raise ValueError("Signals must contain only 0 or 1.")

    # Price return observed at t.
    market_returns = prices.pct_change().fillna(0.0)

    # Signal at t-1 earns the return from t-1 to t.
    positions = signals.shift(1).fillna(0.0)

    turnover = positions.diff().abs()
    turnover.iloc[0] = abs(positions.iloc[0])

    costs = turnover * transaction_cost

    strategy_returns = positions * market_returns - costs

    benchmark_returns = market_returns.copy()

    result = pd.DataFrame(
        {
            "price": prices,
            "market_return": market_returns,
            "position": positions,
            "turnover": turnover,
            "transaction_cost": costs,
            "strategy_return": strategy_returns,
            "benchmark_return": benchmark_returns,
        }
    )

    result["strategy_equity"] = (1 + result["strategy_return"]).cumprod()
    result["benchmark_equity"] = (1 + result["benchmark_return"]).cumprod()

    return result
