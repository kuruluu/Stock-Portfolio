from __future__ import annotations
from pathlib import Path
import pandas as pd
from strategy import generate_target_weights

DATA_PATH = Path("data/monthly_close_2016_2025.parquet")
INITIAL_CAPITAL = 10000.0
NUMBER_OF_ASSETS = 3
TRANSACTION_COST_RATE = 0.0005

def load_monthly_prices(path: Path) -> pd.DataFrame:
    """
    Load adjusted month-end ETF prices from a Parquet file.

    Parameters
    ----------
    path:
        Location of the monthly-price Parquet file.

    Returns
    -------
    pd.DataFrame
        Month-end adjusted closing prices.
    """

    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}. Run data.py first.")
    
    monthly_close = pd.read_parquet(path)

    if monthly_close.empty:
        raise ValueError("The monthly price dataset is empty.")
    
    if not isinstance(monthly_close.index, pd.DatetimeIndex):
        monthly_close.index = pd.to_datetime(monthly_close.index)

    monthly_close = monthly_close.sort_index()

    if monthly_close.columns.duplicated().any():
        raise ValueError("The dataset contains duplicate ticker columns.")

    return monthly_close


def calculate_monthly_returns(monthly_close: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate each ETF's month-to-month simple return.

    Return during month t:
        return[t] = price[t] / price[t - 1] - 1
    """

    monthly_returns = monthly_close.pct_change(fill_method=None)
    return monthly_returns


def calculate_turnover(applied_weights: pd.DataFrame) -> pd.Series:
    """
    Calculate portfolio turnover at each rebalance.
    Turnover is one-half of the total absolute change in weights:
        turnover[t] = 0.5 * sum(|w[t] - w[t - 1]|)

    Examples
    --------
    Replacing one 20% position with another produces:
        0.5 * (20% sold + 20% bought) = 20% turnover
    """

    weight_changes = applied_weights.diff().abs()
    turnover = 0.5 * weight_changes.sum(axis=1)

    if not applied_weights.empty:
        turnover.iloc[0] = 0.5 * applied_weights.iloc[0].abs().sum()
    return turnover


def run_backtest(monthly_close: pd.DataFrame, number_of_assets: int = 5, require_positive_momentum: bool = False,
                use_fixed_slots: bool = True, transaction_cost_rate: float = 0.0, initial_capital: float = 10000.0) -> pd.DataFrame:
    """
    Run the monthly cross-sectional momentum backtest.

    Strategy process
    ----------------
    1. Calculate 12-1 momentum at month-end.
    2. Select the top-ranked ETFs.
    3. Apply those target weights during the following month.
    4. Deduct transaction costs based on turnover.

    Parameters
    ----------
    monthly_close:
        Adjusted month-end ETF prices.

    number_of_assets:
        Maximum number of ETFs selected each month.

    require_positive_momentum:
        When True, exclude ETFs whose momentum is zero or negative.

    use_fixed_slots:
        When True, each qualifying ETF receives
        1 / number_of_assets.

        Unused weight remains in cash.

    transaction_cost_rate:
        Trading cost per unit of turnover.

    initial_capital:
        Starting portfolio value.

    Returns
    -------
    pd.DataFrame
        Monthly weights, returns, costs, and portfolio values.
    """

    if number_of_assets <= 0:
        raise ValueError("number_of_assets must be positive.")
    if not 0 <= transaction_cost_rate < 1:
        raise ValueError("transaction_cost_rate must be between 0 and 1.")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")

    monthly_returns = calculate_monthly_returns(monthly_close)
    target_weights = generate_target_weights(
        monthly_close=monthly_close,
        number_of_assets=number_of_assets,
        require_positive_momentum=require_positive_momentum,
        use_fixed_slots=use_fixed_slots,
    )

    applied_weights = target_weights.shift(1)
    gross_returns = (applied_weights * monthly_returns).sum(axis=1, min_count=1)
    turnover = calculate_turnover(applied_weights.dropna(how="all"))

    turnover = turnover.reindex(monthly_close.index)
    transaction_costs = turnover * transaction_cost_rate
    net_returns = gross_returns - transaction_costs

    valid_months = applied_weights.notna().any(axis=1)
    applied_weights = applied_weights.loc[valid_months]
    gross_returns = gross_returns.loc[valid_months]
    turnover = turnover.loc[valid_months].fillna(0.0)
    transaction_costs = transaction_costs.loc[valid_months].fillna(0.0)
    net_returns = net_returns.loc[valid_months]

    cash_weights = 1.0 - applied_weights.sum(axis=1)
    cash_weights = cash_weights.clip(lower=0.0, upper=1.0)

    gross_portfolio_value = initial_capital * (1 + gross_returns).cumprod()
    net_portfolio_value = initial_capital * (1 + net_returns).cumprod()

    results = pd.DataFrame({
        "gross_return": gross_returns,
        "turnover": turnover,
        "transaction_cost": transaction_costs,
        "net_return": net_returns,
        "cash_weight": cash_weights,
        "gross_portfolio_value": gross_portfolio_value,
        "net_portfolio_value": net_portfolio_value,
    })
    return results


def calculate_spy_benchmark(monthly_close: pd.DataFrame, initial_capital: float = 10000.0) -> pd.DataFrame:
    """
    Calculate a buy-and-hold SPY benchmark.
    """

    if "SPY" not in monthly_close.columns:
        raise ValueError("SPY is missing from the price dataset.")

    spy_returns = monthly_close["SPY"].pct_change(fill_method=None)
    spy_value = initial_capital * (1 + spy_returns).cumprod()

    return pd.DataFrame({"spy_return": spy_returns, "spy_portfolio_value": spy_value})


def calculate_equal_weight_benchmark(monthly_close: pd.DataFrame, initial_capital: float = 10000.0) -> pd.DataFrame:
    """
    Calculate a monthly rebalanced equal-weight benchmark
    containing all ETFs in the dataset.
    """

    monthly_returns = calculate_monthly_returns(monthly_close)
    equal_weight_returns = monthly_returns.mean(axis=1,skipna=True,)
    equal_weight_value = initial_capital * (1 + equal_weight_returns).cumprod()

    return pd.DataFrame({"equal_weight_return": equal_weight_returns, "equal_weight_portfolio_value": equal_weight_value})


def save_results(results: pd.DataFrame, filename: str) -> None:
    """
    Save backtest results as both Parquet and CSV.
    """

    output_directory = Path("results")
    output_directory.mkdir(parents=True, exist_ok=True)

    parquet_path = output_directory / f"{filename}.parquet"
    csv_path = output_directory / f"{filename}.csv"

    results.to_parquet(parquet_path)
    results.to_csv(csv_path)

    print(f"Saved results to {parquet_path}")
    print(f"Saved results to {csv_path}")


def main() -> None:
    monthly_close = load_monthly_prices(DATA_PATH)

    pure_momentum = run_backtest(
        monthly_close=monthly_close,
        number_of_assets=NUMBER_OF_ASSETS,
        require_positive_momentum=False,
        use_fixed_slots=True,
        transaction_cost_rate=TRANSACTION_COST_RATE,
        initial_capital=INITIAL_CAPITAL,
    )

    filtered_momentum = run_backtest(
        monthly_close=monthly_close,
        number_of_assets=NUMBER_OF_ASSETS,
        require_positive_momentum=True,
        use_fixed_slots=True,
        transaction_cost_rate=TRANSACTION_COST_RATE,
        initial_capital=INITIAL_CAPITAL,
    )

    spy_benchmark = calculate_spy_benchmark(
        monthly_close=monthly_close,
        initial_capital=INITIAL_CAPITAL,
    )

    equal_weight_benchmark = calculate_equal_weight_benchmark(
        monthly_close=monthly_close,
        initial_capital=INITIAL_CAPITAL,
    )

    comparison = pure_momentum.join(
        filtered_momentum.add_prefix("filtered_"),
        how="outer",
    )

    comparison = comparison.join(
        spy_benchmark,
        how="left",
    )

    comparison = comparison.join(
        equal_weight_benchmark,
        how="left",
    )

    save_results(
        results=comparison,
        filename="momentum_backtest",
    )

    print("\nBacktest period:")
    print(
        f"{comparison.index.min():%Y-%m} "
        f"to {comparison.index.max():%Y-%m}"
    )

    print("\nFinal portfolio values:")

    print(
        "Pure momentum: "
        f"${pure_momentum['net_portfolio_value'].iloc[-1]:,.2f}"
    )

    print(
        "Filtered momentum: "
        f"${filtered_momentum['net_portfolio_value'].iloc[-1]:,.2f}"
    )

    aligned_spy = spy_benchmark.loc[pure_momentum.index]
    normalized_spy = INITIAL_CAPITAL * (1 + aligned_spy["spy_return"].fillna(0)).cumprod()
    print(
        "SPY buy-and-hold: "
        f"${normalized_spy.iloc[-1]:,.2f}"
    )

    print("\nLast five strategy months:")
    print(pure_momentum.tail().to_string(float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()