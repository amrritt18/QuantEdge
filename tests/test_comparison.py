import numpy as np
import pandas as pd
import pytest

from quantedge.comparison import compare_strategies


def sample_prices():
    """Deterministic prices for repeatable tests; not market data."""
    t = np.arange(100)
    values = 100 * np.exp(0.0008 * t + 0.025 * np.sin(t / 4))
    return pd.Series(
        values,
        index=pd.date_range("2025-01-01", periods=100, freq="D"),
        name="Close",
    )


def test_comparison_returns_three_strategies():
    summary, results = compare_strategies(
        sample_prices(),
        short_window=3,
        long_window=7,
        momentum_lookback=5,
    )

    assert set(summary.index) == {
        "Moving Average",
        "Momentum",
        "Buy and Hold",
    }
    assert set(results) == set(summary.index)


def test_all_strategies_use_same_dates():
    _, results = compare_strategies(
        sample_prices(),
        short_window=3,
        long_window=7,
        momentum_lookback=5,
    )

    indexes = [result.index for result in results.values()]

    assert all(index.equals(indexes[0]) for index in indexes[1:])


def test_summary_contains_expected_metrics():
    summary, _ = compare_strategies(
        sample_prices(),
        short_window=3,
        long_window=7,
        momentum_lookback=5,
    )

    expected_columns = {
        "Total Return",
        "CAGR",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Maximum Drawdown",
        "Historical VaR",
    }

    assert expected_columns.issubset(summary.columns)
    assert np.isfinite(summary.to_numpy()).all()


def test_invalid_prices_are_rejected():
    prices = sample_prices()
    prices.iloc[10] = 0.0

    with pytest.raises(ValueError):
        compare_strategies(
            prices,
            short_window=3,
            long_window=7,
            momentum_lookback=5,
        )
