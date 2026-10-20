import pandas as pd


def moving_average_crossover(
    prices: pd.Series,
    short_window: int = 20,
    long_window: int = 50,
) -> pd.Series:
    """Generate long/cash signals using moving-average crossover."""
    if short_window <= 0 or long_window <= short_window:
        raise ValueError("Require 0 < short_window < long_window.")

    if prices.empty or prices.isna().any() or (prices <= 0).any():
        raise ValueError("Prices must be positive and non-missing.")

    short_ma = prices.rolling(
        window=short_window,
        min_periods=short_window,
    ).mean()

    long_ma = prices.rolling(
        window=long_window,
        min_periods=long_window,
    ).mean()

    # 1 means long; 0 means cash.
    signals = (short_ma > long_ma).astype(float)

    # No signal until both averages are available.
    signals[long_ma.isna()] = float("nan")

    return signals


def momentum_strategy(
    prices: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """Hold long when the lookback-period return is positive."""
    if lookback <= 0:
        raise ValueError("lookback must be positive.")

    if prices.empty or prices.isna().any() or (prices <= 0).any():
        raise ValueError("Prices must be positive and non-missing.")

    past_return = prices.pct_change(lookback)

    signals = (past_return > 0).astype(float)
    signals[past_return.isna()] = float("nan")

    return signals
