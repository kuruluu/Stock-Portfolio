from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from backtest import run_backtest
from metrics import calculate_performance_metrics


DATA_PATH = Path("data/monthly_close_2016_2025.parquet")
OUTPUT_PATH = Path("results/monte_carlo_results.parquet")

NUMBER_OF_SIMULATIONS = 1000
AVERAGE_BLOCK_LENGTH = 6
INITIAL_PRICE = 100.0
RANDOM_SEED = 2026


@dataclass
class SimulationConfig:
    number_of_simulations: int = NUMBER_OF_SIMULATIONS
    average_block_length: float = AVERAGE_BLOCK_LENGTH
    number_of_assets: int = 3
    require_positive_momentum: bool = False
    use_fixed_slots: bool = True
    transaction_cost_rate: float = 0.0005
    initial_capital: float = 10000.0
    random_seed: int = RANDOM_SEED


def load_monthly_prices(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}.")

    prices = pd.read_parquet(path).sort_index()

    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)

    return prices


def calculate_returns(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    returns = prices.pct_change(fill_method=None).dropna(how="any")

    if returns.empty:
        raise ValueError("No complete monthly return rows are available.")

    return returns


def stationary_bootstrap_indices(
    sample_size: int,
    history_length: int,
    average_block_length: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate indices using the stationary bootstrap.

    At each step:
    - continue from the previous historical observation, or
    - begin a new block at a random historical observation.
    """

    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")

    if history_length <= 0:
        raise ValueError("history_length must be positive.")

    if average_block_length < 1:
        raise ValueError("average_block_length must be at least 1.")

    restart_probability = 1.0 / average_block_length

    indices = np.empty(sample_size, dtype=int)
    indices[0] = rng.integers(history_length)

    for position in range(1, sample_size):
        should_restart = rng.random() < restart_probability

        if should_restart:
            indices[position] = rng.integers(history_length)
        else:
            indices[position] = (indices[position - 1] + 1) % history_length

    return indices


def create_simulated_return_path(
    historical_returns: pd.DataFrame,
    average_block_length: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Bootstrap entire monthly return rows.

    Sampling entire rows preserves contemporaneous correlations
    between ETFs.
    """

    indices = stationary_bootstrap_indices(
        sample_size=len(historical_returns),
        history_length=len(historical_returns),
        average_block_length=average_block_length,
        rng=rng,
    )

    simulated = historical_returns.iloc[indices].copy()
    simulated.index = historical_returns.index
    return simulated


def returns_to_prices(
    returns: pd.DataFrame,
    initial_price: float = INITIAL_PRICE,
) -> pd.DataFrame:
    """
    Convert simulated returns into synthetic price histories.
    """

    if initial_price <= 0:
        raise ValueError("initial_price must be positive.")

    prices = initial_price * (1.0 + returns).cumprod()
    first_date = prices.index[0] - pd.offsets.MonthEnd(1)

    initial_row = pd.DataFrame(
        initial_price,
        index=[first_date],
        columns=prices.columns,
    )

    return pd.concat([initial_row, prices])


def run_single_simulation(
    historical_returns: pd.DataFrame,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    
    simulated_returns = create_simulated_return_path(
        historical_returns=historical_returns,
        average_block_length=config.average_block_length,
        rng=rng,
    )

    simulated_prices = returns_to_prices(simulated_returns)

    backtest_results = run_backtest(
        monthly_close=simulated_prices,
        number_of_assets=config.number_of_assets,
        require_positive_momentum=(
            config.require_positive_momentum
        ),
        use_fixed_slots=config.use_fixed_slots,
        transaction_cost_rate=config.transaction_cost_rate,
        initial_capital=config.initial_capital,
    )

    metrics = calculate_performance_metrics(
        returns=backtest_results["net_return"],
        turnover=backtest_results["turnover"],
    )

    metrics["final_portfolio_value"] = float(
        backtest_results["net_portfolio_value"].iloc[-1]
    )

    metrics["loss_probability_observation"] = float(
        metrics["total_return"] < 0
    )

    return metrics


def run_monte_carlo(
    monthly_prices: pd.DataFrame,
    config: SimulationConfig,
) -> pd.DataFrame:
    historical_returns = calculate_returns(monthly_prices)

    rng = np.random.default_rng(config.random_seed)

    simulation_results: list[dict[str, float]] = []

    for simulation_number in range(
        config.number_of_simulations
    ):
        metrics = run_single_simulation(
            historical_returns=historical_returns,
            config=config,
            rng=rng,
        )

        metrics["simulation"] = simulation_number
        simulation_results.append(metrics)

        if (simulation_number + 1) % 500 == 0:
            print(
                f"Completed {simulation_number + 1:,} / "
                f"{config.number_of_simulations:,}"
            )

    return pd.DataFrame(simulation_results).set_index(
        "simulation"
    )


def summarize_simulations(
    simulations: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "total_return",
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
        "calmar_ratio",
        "final_portfolio_value",
    ]

    summary = simulations[metrics].quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])

    summary.index = [
        "1st percentile",
        "5th percentile",
        "25th percentile",
        "median",
        "75th percentile",
        "95th percentile",
        "99th percentile",
    ]

    return summary


def main() -> None:
    monthly_prices = load_monthly_prices(DATA_PATH)

    config = SimulationConfig(
        number_of_simulations=1000,
        average_block_length=6,
        number_of_assets=3,
        require_positive_momentum=False,
        use_fixed_slots=True,
        transaction_cost_rate=0.0005,
    )

    simulations = run_monte_carlo(
        monthly_prices=monthly_prices,
        config=config,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    simulations.to_parquet(OUTPUT_PATH)

    summary = summarize_simulations(simulations)
    summary.to_csv("results/monte_carlo_summary.csv")

    probability_of_loss = (
        simulations["total_return"] < 0
    ).mean()

    probability_sharpe_positive = (
        simulations["sharpe_ratio"] > 0
    ).mean()

    probability_sharpe_above_one = (
        simulations["sharpe_ratio"] > 1
    ).mean()

    print("\nMonte Carlo summary:\n")
    print(summary)

    print(
        f"\nProbability of losing money: "
        f"{probability_of_loss:.2%}"
    )

    print(
        f"Probability Sharpe > 0: "
        f"{probability_sharpe_positive:.2%}"
    )

    print(
        f"Probability Sharpe > 1: "
        f"{probability_sharpe_above_one:.2%}"
    )


if __name__ == "__main__":
    main()