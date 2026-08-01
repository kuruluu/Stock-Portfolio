from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import calculate_performance_metrics
from monte_carlo import stationary_bootstrap_indices


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "search_monthly_close_2016_2025.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a stationary-bootstrap SPY buy-and-hold simulation."
    )
    parser.add_argument("--simulations", type=int, default=5000)
    parser.add_argument("--block-length", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "strategy_search" / "benchmarks" / "spy_5000",
    )
    return parser.parse_args()


def run_simulations(
    historical_returns: pd.Series,
    simulations: int,
    block_length: float,
    seed: int,
    initial_capital: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, float]] = []
    for simulation in range(simulations):
        indices = stationary_bootstrap_indices(
            sample_size=len(historical_returns),
            history_length=len(historical_returns),
            average_block_length=block_length,
            rng=rng,
        )
        simulated = historical_returns.iloc[indices].reset_index(drop=True)
        metrics = calculate_performance_metrics(simulated)
        metrics["final_portfolio_value"] = float(
            initial_capital * (1.0 + simulated).prod()
        )
        metrics["loss_observation"] = float(metrics["total_return"] < 0.0)
        metrics["simulation"] = simulation
        records.append(metrics)
    return pd.DataFrame(records).set_index("simulation")


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    quantiles = results.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    quantiles.index = [
        "1st percentile",
        "5th percentile",
        "25th percentile",
        "median",
        "75th percentile",
        "95th percentile",
        "99th percentile",
    ]
    return quantiles


def main() -> None:
    args = parse_args()
    if args.simulations <= 0:
        raise ValueError("simulations must be positive")
    if args.block_length < 1:
        raise ValueError("block-length must be at least 1")
    if args.initial_capital <= 0:
        raise ValueError("initial-capital must be positive")

    prices = pd.read_parquet(args.data).sort_index()
    if "SPY" not in prices.columns:
        raise ValueError("SPY is missing from the dataset")
    historical_returns = prices["SPY"].pct_change(fill_method=None).dropna()
    if historical_returns.empty:
        raise ValueError("No SPY monthly returns are available")

    results = run_simulations(
        historical_returns=historical_returns,
        simulations=args.simulations,
        block_length=args.block_length,
        seed=args.seed,
        initial_capital=args.initial_capital,
    )
    summary = summarize(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_parquet(args.output_dir / "simulations.parquet")
    summary.to_csv(args.output_dir / "summary.csv")
    metadata = {
        "strategy": "SPY buy-and-hold",
        "simulations": args.simulations,
        "seed": args.seed,
        "average_block_length_months": args.block_length,
        "initial_capital": args.initial_capital,
        "data_path": str(args.data.resolve()),
        "price_start": str(prices.index.min().date()),
        "price_end": str(prices.index.max().date()),
        "historical_return_observations": len(historical_returns),
        "simulated_months_per_path": len(historical_returns),
        "probability_of_loss": float(results["loss_observation"].mean()),
        "probability_sharpe_above_one": float(
            (results["sharpe_ratio"] > 1.0).mean()
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string())
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
