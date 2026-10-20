import numpy as np
import pandas as pd
import pytest

from quantedge.risk import (
    calculate_cagr,
    calculate_historical_var,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
)


def test_cagr():
    equity = pd.Series([100.0, 110.0, 121.0])

    # Two periods per year, two observations intervals = one year.
    result = calculate_cagr(equity, periods_per_year=2)

    assert result == pytest.approx(0.21)


def test_cagr_rejects_nonpositive_equity():
    with pytest.raises(ValueError):
        calculate_cagr(pd.Series([100.0, 0.0, 110.0]))


def test_sharpe_ratio():
    returns = pd.Series([0.01, 0.02, 0.03])

    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)

    result = calculate_sharpe_ratio(returns)

    assert result == pytest.approx(expected)


def test_sharpe_rejects_zero_volatility():
    returns = pd.Series([0.01, 0.01, 0.01])

    with pytest.raises(ValueError):
        calculate_sharpe_ratio(returns)


def test_max_drawdown():
    equity = pd.Series([100.0, 120.0, 90.0, 110.0])

    assert calculate_max_drawdown(equity) == pytest.approx(-0.25)


def test_max_drawdown_without_losses():
    equity = pd.Series([100.0, 105.0, 110.0])

    assert calculate_max_drawdown(equity) == pytest.approx(0.0)


def test_historical_var():
    returns = pd.Series([-0.10, -0.02, 0.0, 0.03, 0.04])

    result = calculate_historical_var(
        returns,
        confidence_level=0.80,
    )

    # NumPy's default linear quantile interpolation gives 3.6%.
    assert result == pytest.approx(0.036)


def test_var_rejects_invalid_confidence():
    returns = pd.Series([-0.01, 0.02, 0.03])

    with pytest.raises(ValueError):
        calculate_historical_var(returns, confidence_level=1.0)
