# QuantEdge — Quantitative Trading & Risk Analytics

QuantEdge is an educational quantitative-finance project for exploring historical market data, evaluating rule-based trading strategies, and comparing their performance and risk through an interactive Streamlit dashboard.

## Features

- Historical NIFTY 50 data retrieval using Yahoo Finance
- Daily return statistics and annualized volatility
- Moving-average crossover and momentum strategies
- Long/cash backtesting with modeled transaction costs
- Buy-and-hold benchmark comparison
- CAGR, Sharpe ratio, maximum drawdown, and historical Value at Risk (VaR)
- Interactive Plotly charts and a Streamlit dashboard
- CSV exports for market data, performance metrics, and backtest results
- Automated tests and code-quality checks

## Technology Stack

Python 3.12, pandas, NumPy, yfinance, SciPy, Streamlit, Plotly, pytest, Ruff, and uv.

## Project Structure

```text
QuantEdge/
├── app.py
├── pyproject.toml
├── README.md
├── .gitignore
├── src/
│   └── quantedge/
│       ├── __init__.py
│       ├── __main__.py
│       ├── data_loader.py
│       ├── analytics.py
│       ├── strategies.py
│       ├── backtest.py
│       ├── risk.py
│       └── comparison.py
└── tests/
    ├── test_analytics.py
    ├── test_backtest.py
    ├── test_comparison.py
    ├── test_risk.py
    └── test_strategies.py
```

*Adjust the tree to match the actual repository.*

## Setup

Install Python 3.12 and uv, then clone the repository and navigate into it.

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd QuantEdge
uv sync
```

Launch the dashboard:

```powershell
uv run streamlit run app.py
```

Run the command-line analysis:

```powershell
uv run python -m quantedge
```

Run tests and code-quality checks:

```powershell
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
```

## Strategies

**Moving-average crossover:** Holds a long position when the short moving average exceeds the long moving average; otherwise it stays in cash.

**Momentum:** Holds a long position when the trailing return over the selected lookback period is positive; otherwise it stays in cash.

**Buy and hold:** Maintains market exposure over the common strategy-evaluation window.

## Performance and Risk Metrics

- **Total return:** Change in simulated equity over the evaluation period.
- **CAGR:** Annualized compound growth rate based on the equity curve and configured periods per year.
- **Annualized volatility:** Sample standard deviation of returns scaled to an annual frequency.
- **Sharpe ratio:** Annualized excess return divided by return volatility.
- **Maximum drawdown:** Largest peak-to-trough decline in the equity curve.
- **Historical VaR:** A quantile-based estimate of the return-loss threshold at a selected confidence level.

## Assumptions and Limitations

- Historical data is retrieved from Yahoo Finance and may be unavailable, delayed, or revised.
- Backtest execution uses a simplified signal-shift convention.
- Transaction costs are modeled, but slippage, taxes, liquidity constraints, and market impact are not fully simulated.
- Historical performance does not establish future profitability.
- The project is intended for research and learning, not as financial advice or a live trading system.

## Testing

The project includes automated tests for return analytics, trading signals, backtesting, risk metrics, and strategy comparison. Report only the test results and quality checks that you have actually run.

## License

Add a license before distributing the project under an open-source license.
