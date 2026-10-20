import numpy as np
import pandas as pd
import pytest

from quantedge.analytics import (
    annualized_volatility,
    calculate_returns,
    summarize_returns,
)


def test_calculate_returns():
    prices = pd.Series([100.0, 110.0, 99.0])

    result = calculate_returns(prices)

    np.testing.assert_allclose(
        result.to_numpy(),
        [0.10, -0.10],
    )


def test_return_summary():
    returns = pd.Series([0.01, -0.01, 0.02])

    summary = summarize_returns(returns)

    assert summary["mean_return"] == pytest.approx(0.02 / 3)
    assert summary["positive_return_fraction"] == pytest.approx(2 / 3)
    assert summary["minimum_return"] == pytest.approx(-0.01)
    assert summary["maximum_return"] == pytest.approx(0.02)


def test_annualized_volatility():
    returns = pd.Series([0.01, -0.01, 0.02])

    expected = returns.std(ddof=1) * np.sqrt(252)

    assert annualized_volatility(returns) == pytest.approx(expected)


def test_invalid_prices_are_rejected():
    prices = pd.Series([100.0, 0.0, 110.0])

    with pytest.raises(ValueError):
        calculate_returns(prices)
