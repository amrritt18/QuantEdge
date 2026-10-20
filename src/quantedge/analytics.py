import numpy as np
import pandas as pd


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate simple period-to-period returns."""
    if prices.empty:
        raise ValueError("Prices cannot be empty.")

    if prices.isna().any() or (prices <= 0).any():
        raise ValueError("Prices must be positive and non-missing.")

    returns = prices.pct_change()
    return returns.dropna()


def summarize_returns(returns: pd.Series) -> dict[str, float]:
    """Calculate descriptive statistics for a return series."""
    if returns.empty:
        raise ValueError("Returns cannot be empty.")

    if returns.isna().any():
        raise ValueError("Returns cannot contain missing values.")

    return {
        "mean_return": float(returns.mean()),
        "median_return": float(returns.median()),
        "daily_volatility": float(returns.std(ddof=1)),
        "minimum_return": float(returns.min()),
        "maximum_return": float(returns.max()),
        "positive_return_fraction": float((returns > 0).mean()),
    }


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Annualize sample volatility for daily trading returns."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    if len(returns) < 2 or returns.isna().any():
        raise ValueError("At least two valid returns are required.")

    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))
