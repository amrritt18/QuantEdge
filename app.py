from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantedge.comparison import compare_strategies
from quantedge.data_loader import download_prices

st.set_page_config(
    page_title="QuantEdge | Trading & Risk Analytics",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(ttl=3600, show_spinner=False)
def load_market_data(period: str) -> pd.DataFrame:
    """Download and cache market data for one hour."""
    return download_prices("^NSEI", period=period)


def format_percent(value: float) -> str:
    return f"{value:.2%}"


# ---------- Header ----------
st.title("QuantEdge")
st.caption("Quantitative Trading & Risk Analytics Simulator")
st.markdown(
    "Explore historical NIFTY 50 data, compare trading strategies, "
    "and evaluate performance and risk."
)

# ---------- Sidebar ----------
st.sidebar.header("Research Settings")

period = st.sidebar.selectbox(
    "Historical period",
    options=["1y", "2y", "5y"],
    index=1,
)

st.sidebar.subheader("Strategy Parameters")

short_window = st.sidebar.slider(
    "Moving-average short window",
    min_value=5,
    max_value=50,
    value=20,
)

long_window = st.sidebar.slider(
    "Moving-average long window",
    min_value=10,
    max_value=150,
    value=50,
)

momentum_lookback = st.sidebar.slider(
    "Momentum lookback",
    min_value=5,
    max_value=60,
    value=20,
)

transaction_cost_pct = st.sidebar.slider(
    "Transaction cost per trade (%)",
    min_value=0.0,
    max_value=0.5,
    value=0.1,
    step=0.05,
)

risk_free_rate_pct = st.sidebar.number_input(
    "Annual risk-free rate (%)",
    min_value=0.0,
    max_value=30.0,
    value=0.0,
    step=0.5,
)

st.sidebar.caption(
    "Transaction costs are applied to position changes. "
    "This is a simplified backtest, not a live trading system."
)

if short_window >= long_window:
    st.error("The short moving-average window must be smaller than the long window.")
    st.stop()

# ---------- Load data ----------
try:
    with st.spinner("Loading NIFTY 50 market data..."):
        prices = load_market_data(period)

    if "Close" not in prices.columns:
        st.error("The downloaded data does not contain a Close column.")
        st.stop()

    close = prices["Close"].dropna().astype(float)

    if close.empty:
        st.error("No valid closing prices were downloaded.")
        st.stop()

    if len(close) < max(long_window, momentum_lookback) + 3:
        st.error(
            "Not enough observations for these settings. "
            "Choose a longer historical period."
        )
        st.stop()

except (ValueError, KeyError, TypeError, RuntimeError) as exc:
    st.error(f"Could not load market data: {exc}")
    st.info(
        "Check your internet connection and try again. "
        "Yahoo Finance data may occasionally be unavailable."
    )
    st.stop()

# ---------- Strategy comparison ----------
try:
    summary, results = compare_strategies(
        prices=close,
        short_window=short_window,
        long_window=long_window,
        momentum_lookback=momentum_lookback,
        transaction_cost=transaction_cost_pct / 100,
        risk_free_rate=risk_free_rate_pct / 100,
    )
except Exception as exc:
    st.error(f"Strategy comparison failed: {exc}")
    st.stop()

# ---------- KPI cards ----------
st.subheader("Market Overview")

latest_close = float(close.iloc[-1])
first_close = float(close.iloc[0])
period_return = latest_close / first_close - 1

col1, col2, col3, col4 = st.columns(4)

col1.metric("Latest NIFTY 50 Close", f"{latest_close:,.2f}")
col2.metric("Period Price Change", format_percent(period_return))
col3.metric("Observations", f"{len(close):,}")
col4.metric(
    "Data Through",
    close.index[-1].strftime("%d %b %Y"),
)

st.caption("Data source: Yahoo Finance (^NSEI). Market data may be delayed or revised.")

# ---------- Tabs ----------
market_tab, strategy_tab, risk_tab, data_tab = st.tabs(
    [
        "Market Analysis",
        "Strategy Comparison",
        "Risk Analysis",
        "Data & Exports",
    ]
)

# ---------- Market Analysis ----------
with market_tab:
    st.subheader("NIFTY 50 Closing Price")

    market_chart = go.Figure()

    market_chart.add_trace(
        go.Scatter(
            x=close.index,
            y=close,
            name="NIFTY 50 Close",
            mode="lines",
        )
    )

    market_chart.add_trace(
        go.Scatter(
            x=close.index,
            y=close.rolling(short_window).mean(),
            name=f"MA {short_window}",
            mode="lines",
        )
    )

    market_chart.add_trace(
        go.Scatter(
            x=close.index,
            y=close.rolling(long_window).mean(),
            name=f"MA {long_window}",
            mode="lines",
        )
    )

    market_chart.update_layout(
        xaxis_title="Date",
        yaxis_title="Index Level",
        hovermode="x unified",
        legend_title="Series",
    )

    st.plotly_chart(market_chart, use_container_width=True)

    st.markdown(
        """
        **How to read this chart**
        - The short moving average reacts more quickly to price changes.
        - The long moving average reacts more slowly.
        - A crossover is a possible trend signal, not a guarantee of profit.
        """
    )

# ---------- Strategy Comparison ----------
with strategy_tab:
    st.subheader("Strategy Performance")

    # Equity curves are normalized to a starting value of 1.
    equity_chart = go.Figure()

    for name, result in results.items():
        equity_chart.add_trace(
            go.Scatter(
                x=result.index,
                y=result["strategy_equity"],
                name=name,
                mode="lines",
            )
        )

    equity_chart.add_hline(
        y=1,
        line_dash="dash",
        annotation_text="Starting equity",
    )

    equity_chart.update_layout(
        xaxis_title="Date",
        yaxis_title="Equity (starting value = 1)",
        hovermode="x unified",
        legend_title="Strategy",
    )

    st.plotly_chart(equity_chart, use_container_width=True)

    st.markdown("### Performance Metrics")

    percent_columns = [
        "Total Return",
        "CAGR",
        "Annualized Volatility",
        "Maximum Drawdown",
        "Historical VaR",
    ]

    display_summary = summary.copy()
    display_summary[percent_columns] *= 100

    st.dataframe(
        display_summary.style.format(
            {
                "Total Return": "{:.2f}%",
                "CAGR": "{:.2f}%",
                "Annualized Volatility": "{:.2f}%",
                "Sharpe Ratio": "{:.3f}",
                "Maximum Drawdown": "{:.2f}%",
                "Historical VaR": "{:.2f}%",
            }
        ),
        use_container_width=True,
    )

    st.caption(
        "Results are historical simulations after modeled transaction "
        "costs. The backtester uses a simplified signal-shift execution "
        "convention and does not model slippage, taxes, or market impact."
    )

# ---------- Risk Analysis ----------
with risk_tab:
    st.subheader("Drawdown Analysis")

    drawdown_chart = go.Figure()

    for name, result in results.items():
        equity = result["strategy_equity"]
        running_peak = equity.cummax()
        drawdown = equity / running_peak - 1

        drawdown_chart.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown * 100,
                name=name,
                mode="lines",
            )
        )

    drawdown_chart.update_layout(
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        legend_title="Strategy",
    )

    st.plotly_chart(drawdown_chart, use_container_width=True)

    st.markdown("### Risk Metric Guide")

    st.markdown(
        """
        - **Annualized volatility:** Measures variability of daily strategy returns.
        - **Sharpe ratio:** Measures excess return relative to return volatility,
          using your selected annual risk-free rate.
        - **Maximum drawdown:** The largest peak-to-trough decline in the
          simulated equity curve.
        - **Historical VaR (95%):** An estimated daily loss threshold based
          on the historical return distribution. It is not a maximum-loss limit.
        """
    )

# ---------- Data & Exports ----------
with data_tab:
    st.subheader("Historical Market Data")
    st.dataframe(prices.tail(20), use_container_width=True)

    st.download_button(
        label="Download market data CSV",
        data=prices.to_csv(index=True).encode("utf-8"),
        file_name="nifty50_prices.csv",
        mime="text/csv",
    )

    st.download_button(
        label="Download strategy metrics CSV",
        data=summary.to_csv(index=True).encode("utf-8"),
        file_name="strategy_comparison.csv",
        mime="text/csv",
    )

    for name, result in results.items():
        filename = name.lower().replace(" ", "_") + "_backtest.csv"

        st.download_button(
            label=f"Download {name} backtest",
            data=result.to_csv(index=True).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
            key=f"download_{name}",
        )

st.divider()
st.caption(
    "QuantEdge is an educational research project, not investment advice "
    "or a recommendation to trade."
)
