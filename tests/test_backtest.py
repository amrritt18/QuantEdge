import pandas as pd
import pytest

from quantedge.backtest import backtest_long_cash


def test_backtest_shifts_signal():
    prices = pd.Series(
        [100.0, 110.0, 121.0],
        index=pd.date_range("2026-01-01", periods=3),
    )

    signals = pd.Series([1.0, 1.0, 1.0], index=prices.index)

    result = backtest_long_cash(
        prices,
        signals,
        transaction_cost=0.0,
    )

    # The first day's return is not earned because
    # no previous-period signal exists.
    assert result["strategy_return"].iloc[0] == pytest.approx(0.0)

    # The signal from day 1 earns day 2's 10% return.
    assert result["strategy_return"].iloc[1] == pytest.approx(0.10)

    assert result["strategy_equity"].iloc[-1] == pytest.approx(1.21)


def test_transaction_cost_is_applied():
    prices = pd.Series(
        [100.0, 110.0, 110.0],
        index=pd.date_range("2026-01-01", periods=3),
    )

    signals = pd.Series([1.0, 0.0, 0.0], index=prices.index)

    result = backtest_long_cash(
        prices,
        signals,
        transaction_cost=0.001,
    )

    assert result["transaction_cost"].iloc[1] == pytest.approx(0.001)
