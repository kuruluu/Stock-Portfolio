from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


RESULTS_PATH = Path("results/momentum_backtest.parquet")

PERIODS_PER_YEAR = 12
ANNUAL_RISK_FREE_RATE = 0.0


def calculate_equity_curve(
    returns: pd.Series,
    initial_capital: float = 1.0,
) -> pd.Series:
    """
    Convert periodic returns into a portfolio-value series.

    Example
    -------
    Returns:
        5%, -2%, 3%

    Equity curve:
        1.05, 1.029, 1.05987
    """

    returns = returns.dropna().astype(float)

    return initial_capital * (1.0 + returns).cumprod()


def calculate_total_return(
    returns: pd.Series,
) -> float:
    """
    Calculate the strategy's cumulative return.
    """
    equity = calculate_equity_curve(returns)

    if equity.empty:
        return np.nan

    return float(equity.iloc[-1] - 1.0)


def calculate_cagr(
    returns: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """
    Calculate compound annual growth rate.
    """

    returns = returns.dropna().astype(float)

    if returns.empty:
        return np.nan

    years = len(returns) / periods_per_year

    if years <= 0:
        return np.nan

    final_value = (1.0 + returns).prod()

    if final_value <= 0:
        return np.nan

    return float(final_value ** (1.0 / years) - 1.0)


def calculate_annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """
    Calculate annualized standard deviation of returns.
    """

    returns = returns.dropna().astype(float)

    if len(returns) < 2:
        return np.nan

    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def annual_rate_to_periodic_rate(
    annual_rate: float,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """
    Convert an annual compounded rate into a periodic rate.
    """

    if annual_rate <= -1:
        raise ValueError("annual_rate must be greater than -1.")

    return float((1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0)


def calculate_sharpe_ratio(
    returns: pd.Series,
    annual_risk_free_rate: float = ANNUAL_RISK_FREE_RATE,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Sharpe =
        annualized mean excess return
        divided by annualized volatility
    """
    returns = returns.dropna().astype(float)

    if len(returns) < 2:
        return np.nan

    periodic_rf = annual_rate_to_periodic_rate(
        annual_rate=annual_risk_free_rate,
        periods_per_year=periods_per_year,
    )

    excess_returns = returns - periodic_rf
    excess_std = excess_returns.std(ddof=1)

    if excess_std == 0 or np.isnan(excess_std):
        return np.nan

    return float(excess_returns.mean() / excess_std * np.sqrt(periods_per_year))


def calculate_sortino_ratio(
    returns: pd.Series,
    annual_risk_free_rate: float = ANNUAL_RISK_FREE_RATE,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """
    Calculate annualized Sortino ratio.

    Unlike Sharpe, Sortino penalizes only returns below
    the periodic risk-free return.
    """

    returns = returns.dropna().astype(float)

    if returns.empty:
        return np.nan

    periodic_rf = annual_rate_to_periodic_rate(
        annual_rate=annual_risk_free_rate,
        periods_per_year=periods_per_year,
    )

    excess_returns = returns - periodic_rf

    downside_excess = excess_returns[excess_returns < 0]

    if downside_excess.empty:
        return np.nan

    downside_deviation = np.sqrt(np.mean(downside_excess**2))

    if downside_deviation == 0:
        return np.nan

    return float(excess_returns.mean() / downside_deviation * np.sqrt(periods_per_year))


def calculate_drawdown_series(
    returns: pd.Series,
) -> pd.Series:
    """
    Calculate the drawdown at every point in time.

    Drawdown measures the percentage decline from the
    previous portfolio peak.
    """

    equity = calculate_equity_curve(returns)

    running_peak = equity.cummax()

    return equity / running_peak - 1.0


def calculate_max_drawdown(
    returns: pd.Series,
) -> float:
    """
    Calculate the worst peak-to-trough decline.
    """

    drawdowns = calculate_drawdown_series(returns)

    if drawdowns.empty:
        return np.nan

    return float(drawdowns.min())


def calculate_max_drawdown_duration(
    returns: pd.Series,
) -> int:
    """
    Calculate the longest number of consecutive periods
    spent below a previous portfolio peak.

    For monthly returns, the result is measured in months.
    """

    drawdowns = calculate_drawdown_series(returns)

    longest_duration = 0
    current_duration = 0

    for drawdown in drawdowns:
        if drawdown < 0:
            current_duration += 1
            longest_duration = max(longest_duration, current_duration)
        else:
            current_duration = 0

    return longest_duration


def calculate_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """
    Calculate CAGR divided by the absolute maximum drawdown.
    """

    cagr = calculate_cagr(returns=returns, periods_per_year=periods_per_year)
    max_drawdown = calculate_max_drawdown(returns)

    if (np.isnan(cagr) or np.isnan(max_drawdown) or max_drawdown >= 0):
        return np.nan

    return float(cagr / abs(max_drawdown))


def calculate_positive_period_rate(
    returns: pd.Series,
) -> float:
    """
    Calculate the proportion of periods with positive returns.
    """

    returns = returns.dropna().astype(float)

    if returns.empty:
        return np.nan

    return float((returns > 0).mean())


def calculate_turnover_metrics(
    turnover: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> dict[str, float]:
    """
    Calculate average monthly and annualized turnover.
    """

    turnover = turnover.dropna().astype(float)

    if turnover.empty:
        return {
            "average_monthly_turnover": np.nan,
            "annualized_turnover": np.nan,
        }

    average_periodic_turnover = turnover.mean()

    return {
        "average_monthly_turnover": float(average_periodic_turnover),
        "annualized_turnover": float(average_periodic_turnover * periods_per_year)
    }


def calculate_alpha_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    annual_risk_free_rate: float = ANNUAL_RISK_FREE_RATE,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> tuple[float, float]:
    """
    Calculate annualized CAPM-style alpha and beta.

    Beta measures sensitivity to the benchmark.

    Alpha measures average strategy excess return not
    explained by benchmark exposure.
    """

    aligned = pd.concat(
        [
            strategy_returns.rename("strategy"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
    ).dropna()

    if len(aligned) < 2:
        return np.nan, np.nan

    periodic_rf = annual_rate_to_periodic_rate(
        annual_rate=annual_risk_free_rate,
        periods_per_year=periods_per_year,
    )

    strategy_excess = aligned["strategy"] - periodic_rf
    benchmark_excess = aligned["benchmark"] - periodic_rf

    benchmark_variance = benchmark_excess.var(ddof=1)

    if benchmark_variance == 0 or np.isnan(benchmark_variance):
        return np.nan, np.nan

    beta = strategy_excess.cov(benchmark_excess) / benchmark_variance
    periodic_alpha = strategy_excess.mean()- beta * benchmark_excess.mean()
    annualized_alpha = periodic_alpha * periods_per_year

    return float(annualized_alpha), float(beta)


def calculate_information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """
    Calculate annualized information ratio.

    Information ratio =
        average active return / tracking error
    """
    
    aligned = pd.concat(
        [
            strategy_returns.rename("strategy"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
    ).dropna()

    if len(aligned) < 2:
        return np.nan

    active_returns = (
        aligned["strategy"] - aligned["benchmark"]
    )

    tracking_error = active_returns.std(ddof=1)

    if tracking_error == 0 or np.isnan(tracking_error):
        return np.nan

    return float(active_returns.mean() / tracking_error * np.sqrt(periods_per_year))


def calculate_performance_metrics(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    benchmark_returns: pd.Series | None = None,
    annual_risk_free_rate: float = ANNUAL_RISK_FREE_RATE,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> dict[str, float]:
    """
    Calculate a complete strategy-performance summary.
    """
    
    metrics = {
        "total_return": calculate_total_return(returns),
        "cagr": calculate_cagr(returns, periods_per_year),
        "annualized_volatility": (calculate_annualized_volatility(returns, periods_per_year)),
        "sharpe_ratio": calculate_sharpe_ratio(returns, annual_risk_free_rate, periods_per_year),
        "sortino_ratio": calculate_sortino_ratio(returns, annual_risk_free_rate, periods_per_year),
        "maximum_drawdown": calculate_max_drawdown(returns),
        "maximum_drawdown_duration": (calculate_max_drawdown_duration(returns)),
        "calmar_ratio": calculate_calmar_ratio(returns, periods_per_year),
        "positive_month_rate": (calculate_positive_period_rate(returns))
    }

    if turnover is not None:
        metrics.update(calculate_turnover_metrics(
            turnover=turnover,
            periods_per_year=periods_per_year,
        ))

    if benchmark_returns is not None:
        alpha, beta = calculate_alpha_beta(
            strategy_returns=returns,
            benchmark_returns=benchmark_returns,
            annual_risk_free_rate=annual_risk_free_rate,
            periods_per_year=periods_per_year,
        )

        metrics["annualized_alpha"] = alpha
        metrics["beta"] = beta
        metrics["information_ratio"] = (calculate_information_ratio(
            strategy_returns=returns,
            benchmark_returns=benchmark_returns,
            periods_per_year=periods_per_year,
        ))

    return metrics


def format_metrics(metrics: dict[str, float]) -> pd.Series:
    """
    Convert raw metrics into readable strings.
    """

    percentage_metrics = {
        "total_return",
        "cagr",
        "annualized_volatility",
        "maximum_drawdown",
        "positive_month_rate",
        "average_monthly_turnover",
        "annualized_turnover",
        "annualized_alpha",
    }

    formatted: dict[str, str] = {}

    for name, value in metrics.items():
        if pd.isna(value):
            formatted[name] = "N/A"
        elif name in percentage_metrics:
            formatted[name] = f"{value:.2%}"
        elif name == "maximum_drawdown_duration":
            formatted[name] = f"{int(value)} months"
        else:
            formatted[name] = f"{value:.3f}"

    return pd.Series(formatted)


def load_backtest_results(path: Path = RESULTS_PATH) -> pd.DataFrame:
    """
    Load the results created by backtest.py.
    """

    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}. Run backtest.py first.")

    results = pd.read_parquet(path)

    if not isinstance(results.index, pd.DatetimeIndex):
        results.index = pd.to_datetime(results.index)
        
    return results.sort_index()


def main() -> None:
    results = load_backtest_results()

    required_columns = {
        "net_return",
        "turnover",
        "filtered_net_return",
        "filtered_turnover",
        "spy_return",
    }

    missing_columns = required_columns.difference(results.columns)

    if missing_columns:
        raise ValueError("The backtest results are missing columns: "f"{sorted(missing_columns)}")

    pure_metrics = calculate_performance_metrics(
        returns=results["net_return"],
        turnover=results["turnover"],
        benchmark_returns=results["spy_return"],
        annual_risk_free_rate=ANNUAL_RISK_FREE_RATE,
    )

    filtered_metrics = calculate_performance_metrics(
        returns=results["filtered_net_return"],
        turnover=results["filtered_turnover"],
        benchmark_returns=results["spy_return"],
        annual_risk_free_rate=ANNUAL_RISK_FREE_RATE,
    )

    spy_metrics = calculate_performance_metrics(
        returns=results["spy_return"],
        annual_risk_free_rate=ANNUAL_RISK_FREE_RATE,
    )

    summary = pd.DataFrame({
        "Pure momentum": format_metrics(pure_metrics),
        "Filtered momentum": format_metrics(filtered_metrics),
        "SPY": format_metrics(spy_metrics),
    })

    print("\nPerformance summary:\n")
    print(summary.to_string())

    output_directory = Path("results")
    output_directory.mkdir(parents=True, exist_ok=True)

    summary.to_csv(output_directory / "performance_summary.csv")
    print("\nSaved summary to ""results/performance_summary.csv")


if __name__ == "__main__":
    main()