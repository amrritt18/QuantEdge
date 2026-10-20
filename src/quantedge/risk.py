import numpy as np
import pandas as pd


def _validate_series(
    values: pd.Series,
    name: str,
    min_length: int = 1,
) -> None:
    """Validate a numeric series used by risk calculations."""
    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")

    if len(values) < min_length:
        raise ValueError(f"{name} must contain at least {min_length} observations.")

    array = values.to_numpy(dtype=float)

    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values.")


def calculate_cagr(
    equity: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate compound annual growth rate from an equity curve.

    Assumes equally spaced observations and a positive equity curve.
    """
    _validate_series(equity, "equity", min_length=2)

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    values = equity.to_numpy(dtype=float)

    if (values <= 0).any():
        raise ValueError("Equity values must be positive.")

    years = (len(values) - 1) / periods_per_year

    if years <= 0:
        raise ValueError("The observation period must be positive.")

    return float((values[-1] / values[0]) ** (1 / years) - 1)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sharpe ratio.

    risk_free_rate is an annual effective rate, e.g. 0.04 for 4%.
    Returns must be periodic simple returns.
    """
    _validate_series(returns, "returns", min_length=2)

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    if risk_free_rate <= -1:
        raise ValueError("risk_free_rate must be greater than -1.")

    periodic_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    excess_returns = returns.astype(float) - periodic_rf
    volatility = excess_returns.std(ddof=1)

    if volatility == 0:
        raise ValueError(
            "Sharpe ratio is undefined when excess-return volatility is zero."
        )

    return float(excess_returns.mean() / volatility * np.sqrt(periods_per_year))


def calculate_max_drawdown(equity: pd.Series) -> float:
    """
    Return maximum drawdown as a negative fraction.

    Example: -0.20 means a 20% peak-to-trough decline.
    """
    _validate_series(equity, "equity", min_length=1)

    values = equity.to_numpy(dtype=float)

    if (values <= 0).any():
        raise ValueError("Equity values must be positive.")

    running_peak = np.maximum.accumulate(values)
    drawdowns = values / running_peak - 1

    return float(drawdowns.min())


def calculate_historical_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """
    Calculate historical VaR as a loss threshold in return units.

    Example: 0.02 means a 2% loss threshold at the selected
    confidence level. VaR can be negative if the sample quantile
    at the selected tail is positive.
    """
    _validate_series(returns, "returns", min_length=1)

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    lower_quantile = np.quantile(
        returns.to_numpy(dtype=float),
        1 - confidence_level,
    )

    return float(-lower_quantile)
