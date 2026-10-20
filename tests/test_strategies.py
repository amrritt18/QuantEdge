import pandas as pd
import pytest

from quantedge.strategies import (
    momentum_strategy,
    moving_average_crossover,
)


def test_moving_average_requires_valid_windows():
    prices = pd.Series(range(1, 101), dtype=float)

    with pytest.raises(ValueError):
        moving_average_crossover(prices, 50, 20)


def test_momentum_signal():
    prices = pd.Series(
        [100.0, 101.0, 103.0, 105.0],
        index=pd.date_range("2026-01-01", periods=4),
    )

    signals = momentum_strategy(prices, lookback=2)

    assert pd.isna(signals.iloc[0])
    assert pd.isna(signals.iloc[1])
    assert signals.iloc[2] == 1.0
    assert signals.iloc[3] == 1.0


def test_moving_average_crossover_uptrend():
    prices = pd.Series(
        range(1, 101),
        index=pd.date_range("2026-01-01", periods=100),
        dtype=float,
    )

    signals = moving_average_crossover(
        prices,
        short_window=5,
        long_window=10,
    )

    assert pd.isna(signals.iloc[8])
    assert signals.iloc[9] == 1.0
