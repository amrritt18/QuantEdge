import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def download_prices(
    ticker: str = "^NSEI",
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download and validate historical OHLCV prices."""
    if not ticker.strip():
        raise ValueError("Ticker cannot be empty.")

    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No market data returned for ticker: {ticker}")

    # yfinance can return MultiIndex columns for some downloads.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df[REQUIRED_COLUMNS].copy()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if df[["Open", "High", "Low", "Close"]].isna().any().any():
        raise ValueError("Price data contains missing OHLC values.")

    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Prices must be positive.")

    if (df["Volume"] < 0).any():
        raise ValueError("Volume cannot be negative.")

    if (df["High"] < df["Low"]).any():
        raise ValueError("High cannot be lower than Low.")

    return df
