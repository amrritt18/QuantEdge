from quantedge.comparison import compare_strategies
from quantedge.data_loader import download_prices


def main():
    prices = download_prices("^NSEI", period="2y")

    summary, results = compare_strategies(
        prices=prices["Close"],
        short_window=20,
        long_window=50,
        momentum_lookback=20,
        transaction_cost=0.001,
        risk_free_rate=0.0,
    )

    print("\nRaw strategy metrics:")
    print(summary.to_string())

    print("\nStrategy Comparison")
    display_summary = summary.copy()

    percentage_columns = [
        "Total Return",
        "CAGR",
        "Annualized Volatility",
        "Maximum Drawdown",
        "Historical VaR",
    ]

    display_summary[percentage_columns] *= 100

    print(display_summary.round(3).to_string())

    for name, result in results.items():
        filename = name.lower().replace(" ", "_") + "_backtest.csv"
        result.to_csv(filename)
        print(f"Saved {filename}")


if __name__ == "__main__":
    main()
