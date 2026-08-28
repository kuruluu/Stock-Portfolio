from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import pandas as pd

from pit_momentum import (
    PitConfig,
    backtest_point_in_time,
    deflated_sharpe_probability,
    performance_metrics,
    probability_of_backtest_overfitting,
    stationary_bootstrap_metrics,
    walk_forward_summary,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PRICE_PATH = DATA_DIR / "pit_sp500_monthly_close.parquet"
MEMBERSHIP_PATH = DATA_DIR / "pit_sp500_membership.parquet"
LIQUIDITY_PATH = DATA_DIR / "pit_sp500_monthly_liquidity.parquet"
MARKET_PATH = DATA_DIR / "pit_spy_monthly_close.parquet"
DATA_MANIFEST_PATH = DATA_DIR / "pit_sp500_manifest.json"
RESULTS_ROOT = ROOT / "results" / "pit_stock_strategy_search"
STOP_PATH = RESULTS_ROOT / "STOP"
V2_PATH = RESULTS_ROOT / "V2.json"
TEST_YEARS = (2021, 2022, 2023, 2024, 2025)

PARAMETERS: dict[str, tuple[object, ...]] = {
    "strategy_type": ("cross_sectional", "time_series"),
    "lookback_months": (6, 9, 12, 15, 18, 24),
    "skip_months": (1, 2, 3, 6),
    "number_of_assets": (5, 10, 15, 20, 30, 40),
    "universe_size": (100, 200, 300, 500),
    "require_positive_momentum": (False, True),
    "use_fixed_slots": (False, True),
    "rebalance_interval_months": (1, 2, 3, 6),
    "signal_scaling": ("raw", "volatility_adjusted"),
    "weighting": ("equal", "inverse_volatility"),
    "volatility_window_months": (3, 6, 9, 12),
    "market_filter_months": (0, 6, 10, 12),
}

_WORKER_INPUTS: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously search point-in-time walk-forward stock momentum."
    )
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--round-size", type=int, default=250)
    parser.add_argument("--max-rounds", type=int, default=0)
    parser.add_argument("--screen-sims", type=int, default=1000)
    parser.add_argument("--final-sims", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2601)
    parser.add_argument("--block-length", type=float, default=6.0)
    parser.add_argument("--analysis-minutes", type=float, default=60.0)
    return parser.parse_args()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    required = (
        PRICE_PATH,
        MEMBERSHIP_PATH,
        LIQUIDITY_PATH,
        MARKET_PATH,
        DATA_MANIFEST_PATH,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"point-in-time inputs are missing: {missing}")
    manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
    if int(manifest.get("coverage_gaps", -1)) != 0:
        raise ValueError("point-in-time manifest does not certify complete coverage")
    prices = pd.read_parquet(PRICE_PATH).sort_index()
    membership = pd.read_parquet(MEMBERSHIP_PATH).sort_index().astype(bool)
    liquidity = pd.read_parquet(LIQUIDITY_PATH).sort_index()
    market = pd.read_parquet(MARKET_PATH).sort_index()["SPY"]
    if not prices.index.equals(membership.index) or not prices.index.equals(
        liquidity.index
    ):
        raise ValueError("point-in-time input dates do not match")
    if list(prices.columns) != list(membership.columns) or list(
        prices.columns
    ) != list(liquidity.columns):
        raise ValueError("point-in-time input symbols do not match")
    if not prices.index.equals(market.index):
        raise ValueError("market-filter dates do not match stock data")
    return prices, membership, liquidity, market


def initialize_worker() -> None:
    global _WORKER_INPUTS
    _WORKER_INPUTS = load_inputs()


def evaluate_candidate(config_payload: dict[str, object]) -> dict[str, object]:
    global _WORKER_INPUTS
    if _WORKER_INPUTS is None:
        initialize_worker()
    assert _WORKER_INPUTS is not None
    prices, membership, liquidity, market = _WORKER_INPUTS
    config = PitConfig(**config_payload)
    backtest = backtest_point_in_time(
        prices, membership, liquidity, config, market
    )
    walk = walk_forward_summary(backtest, TEST_YEARS)
    oos_returns = walk.pop("oos_returns")
    oos_turnover = walk.pop("oos_turnover")
    holdout = backtest.loc[backtest.index.year == 2026]
    result: dict[str, object] = {
        "config_id": config.config_id,
        "strategy_type": config.strategy_type,
        "config_json": json.dumps(config.to_dict(), sort_keys=True),
        "positive_oos_years": int(walk["positive_oos_years"]),
        "folds_json": json.dumps(walk["folds"], sort_keys=True),
        "oos_returns_json": json.dumps(
            {str(date.date()): float(value) for date, value in oos_returns.items()},
            sort_keys=True,
        ),
        "oos_turnover_json": json.dumps(
            {str(date.date()): float(value) for date, value in oos_turnover.items()},
            sort_keys=True,
        ),
    }
    result.update({f"oos_{key}": value for key, value in walk["oos"].items()})
    for fold in walk["folds"]:
        year = int(fold["test_year"])
        for period in ("train", "test"):
            for key, value in fold[period].items():
                result[f"{period}_{year}_{key}"] = value
    if len(holdout) >= 2:
        holdout_metrics = performance_metrics(
            holdout["net_return"], holdout["turnover"]
        )
        result.update(
            {f"holdout_2026_{key}": value for key, value in holdout_metrics.items()}
        )
    return result


def anchors() -> list[PitConfig]:
    return [
        PitConfig(),
        PitConfig(lookback_months=24, skip_months=3, number_of_assets=5, rebalance_interval_months=2),
        PitConfig(lookback_months=24, skip_months=1, number_of_assets=10, rebalance_interval_months=3),
        PitConfig(lookback_months=12, skip_months=1, number_of_assets=20, market_filter_months=10),
        PitConfig(strategy_type="time_series", number_of_assets=100, universe_size=100),
        PitConfig(strategy_type="time_series", lookback_months=24, skip_months=2, number_of_assets=200, universe_size=200, rebalance_interval_months=1),
        PitConfig(strategy_type="time_series", lookback_months=12, skip_months=1, number_of_assets=200, universe_size=200, weighting="inverse_volatility", volatility_window_months=12, market_filter_months=10),
    ]


def random_config(rng: random.Random) -> PitConfig:
    strategy = str(rng.choice(PARAMETERS["strategy_type"]))
    lookback = int(rng.choice(PARAMETERS["lookback_months"]))
    valid_skips = [int(value) for value in PARAMETERS["skip_months"] if int(value) < lookback]
    skip = rng.choice(valid_skips)
    universe_size = int(rng.choice(PARAMETERS["universe_size"]))
    number_assets = int(rng.choice(PARAMETERS["number_of_assets"]))
    number_assets = min(number_assets, universe_size)
    if strategy == "time_series":
        number_assets = universe_size
    scaling = str(rng.choice(PARAMETERS["signal_scaling"]))
    weighting = str(rng.choice(PARAMETERS["weighting"]))
    volatility_window = int(rng.choice(PARAMETERS["volatility_window_months"]))
    if scaling == "raw" and weighting == "equal":
        volatility_window = 6
    return PitConfig(
        strategy_type=strategy,
        lookback_months=lookback,
        skip_months=skip,
        number_of_assets=number_assets,
        universe_size=universe_size,
        require_positive_momentum=(
            True if strategy == "time_series" else bool(rng.choice(PARAMETERS["require_positive_momentum"]))
        ),
        use_fixed_slots=bool(rng.choice(PARAMETERS["use_fixed_slots"])),
        rebalance_interval_months=int(rng.choice(PARAMETERS["rebalance_interval_months"])),
        signal_scaling=scaling,
        weighting=weighting,
        volatility_window_months=volatility_window,
        market_filter_months=int(rng.choice(PARAMETERS["market_filter_months"])),
    )


def candidate_stream(seed: int):
    rng = random.Random(seed)
    seen: set[str] = set()
    for config in anchors():
        if config.config_id not in seen:
            seen.add(config.config_id)
            yield config
    while True:
        config = random_config(rng)
        if config.config_id not in seen:
            seen.add(config.config_id)
            yield config


def read_records(path: Path) -> list[dict[str, object]]:
    return pd.read_parquet(path).to_dict("records") if path.exists() else []


def write_records(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    frame.to_csv(path.with_suffix(".csv"), index=False)


def series_from_json(payload: object) -> pd.Series:
    values = json.loads(str(payload))
    return pd.Series(
        list(values.values()),
        dtype=float,
        index=pd.to_datetime(list(values.keys())),
    ).sort_index()


def walk_forward_policy(records: list[dict[str, object]]) -> dict[str, object]:
    chosen: list[dict[str, object]] = []
    returns: list[pd.Series] = []
    turnover: list[pd.Series] = []
    for year in TEST_YEARS:
        eligible = [
            record
            for record in records
            if math.isfinite(float(record.get(f"train_{year}_cagr", math.nan)))
        ]
        winner = max(
            eligible,
            key=lambda record: (
                float(record[f"train_{year}_cagr"]),
                -float(record[f"train_{year}_annualized_volatility"]),
                float(record[f"train_{year}_maximum_drawdown"]),
                -float(record[f"train_{year}_annualized_turnover"]),
            ),
        )
        chosen.append(
            {
                "test_year": year,
                "config_id": winner["config_id"],
                "train_cagr": winner[f"train_{year}_cagr"],
                "test_cagr": winner[f"test_{year}_cagr"],
            }
        )
        candidate_returns = series_from_json(winner["oos_returns_json"])
        candidate_turnover = series_from_json(winner["oos_turnover_json"])
        returns.append(candidate_returns.loc[candidate_returns.index.year == year])
        turnover.append(candidate_turnover.loc[candidate_turnover.index.year == year])
    joined_returns = pd.concat(returns).sort_index()
    joined_turnover = pd.concat(turnover).sort_index()
    return {
        "selection_rule": "highest expanding-window train CAGR; volatility, drawdown, and turnover are tie-breakers",
        "fold_selections": chosen,
        "stitched_oos": performance_metrics(joined_returns, joined_turnover),
    }


def parameter_patterns(records: list[dict[str, object]]) -> dict[str, object]:
    frame = pd.DataFrame(records)
    cutoff = float(frame["oos_cagr"].quantile(0.80))
    top_ids = set(frame.loc[frame["oos_cagr"] >= cutoff, "config_id"])
    configs = []
    for record in records:
        config = json.loads(str(record["config_json"]))
        config["config_id"] = record["config_id"]
        config["oos_cagr"] = record["oos_cagr"]
        configs.append(config)
    config_frame = pd.DataFrame(configs)
    output: dict[str, object] = {"top_quintile_cutoff_oos_cagr": cutoff, "parameters": {}}
    for parameter in PARAMETERS:
        rows = []
        for value, group in config_frame.groupby(parameter):
            rows.append(
                {
                    "value": value,
                    "count": len(group),
                    "top_quintile_count": int(group["config_id"].isin(top_ids).sum()),
                    "median_oos_cagr": float(group["oos_cagr"].median()),
                }
            )
        output["parameters"][parameter] = rows
    return output


def analyze_and_validate(
    records: list[dict[str, object]],
    args: argparse.Namespace,
) -> dict[str, object]:
    valid = [record for record in records if "oos_returns_json" in record]
    return_series = {
        str(record["config_id"]): series_from_json(record["oos_returns_json"])
        for record in valid
    }
    turnover_series = {
        str(record["config_id"]): series_from_json(record["oos_turnover_json"])
        for record in valid
    }
    matrix = pd.DataFrame(return_series).sort_index()
    trial_sharpes = matrix.apply(
        lambda column: column.mean() / column.std(ddof=1)
    ).to_numpy(dtype=float)
    pbo = probability_of_backtest_overfitting(matrix, number_of_blocks=10)

    candidates: list[dict[str, object]] = []
    for record in valid:
        config_id = str(record["config_id"])
        dsr = deflated_sharpe_probability(
            return_series[config_id], trial_sharpes, len(valid)
        )
        candidate = dict(record)
        candidate.update(dsr)
        candidates.append(candidate)
    eligible_candidates = [
        row for row in candidates if int(row["positive_oos_years"]) >= 3
    ] or candidates
    eligible_candidates.sort(
        key=lambda row: (
            float(row["oos_cagr"]),
            float(row["dsr_probability"]),
            int(row["positive_oos_years"]),
            -float(row["oos_annualized_volatility"]),
            float(row["oos_maximum_drawdown"]),
            -float(row["oos_annualized_turnover"]),
        ),
        reverse=True,
    )

    preliminary: list[dict[str, object]] = []
    for candidate in eligible_candidates[: min(24, len(eligible_candidates))]:
        config_id = str(candidate["config_id"])
        mc = stationary_bootstrap_metrics(
            return_series[config_id],
            turnover_series[config_id],
            args.screen_sims,
            args.seed + int(config_id[:8], 16),
            args.block_length,
        )
        preliminary.append({**candidate, "monte_carlo_1000": mc})
    preliminary.sort(
        key=lambda row: (
            float(row["monte_carlo_1000"]["cagr_median"]),
            -float(row["monte_carlo_1000"]["annualized_volatility_median"]),
            float(row["monte_carlo_1000"]["maximum_drawdown_q05"]),
            -float(row["monte_carlo_1000"]["annualized_turnover_median"]),
        ),
        reverse=True,
    )

    finalists: list[dict[str, object]] = []
    for candidate in preliminary[: min(4, len(preliminary))]:
        config_id = str(candidate["config_id"])
        mc = stationary_bootstrap_metrics(
            return_series[config_id],
            turnover_series[config_id],
            args.final_sims,
            args.seed + 10_000_019 + int(config_id[:8], 16),
            args.block_length,
        )
        finalists.append({**candidate, "monte_carlo_10000": mc})
    finalists.sort(
        key=lambda row: (
            float(row["monte_carlo_10000"]["cagr_median"]),
            -float(row["monte_carlo_10000"]["annualized_volatility_median"]),
            float(row["monte_carlo_10000"]["maximum_drawdown_q05"]),
            -float(row["monte_carlo_10000"]["annualized_turnover_median"]),
        ),
        reverse=True,
    )
    return {
        "updated": datetime.now().astimezone().isoformat(),
        "version": "V2",
        "objective": "10,000-path median OOS CAGR, then volatility, drawdown, and turnover",
        "completed_strategy_trials": len(valid),
        "walk_forward_test_years": list(TEST_YEARS),
        "point_in_time_universe": "S&P 500 historical membership with trailing monthly liquidity subsets",
        "pbo": pbo,
        "walk_forward_policy": walk_forward_policy(valid),
        "preliminary_1000_path_count": len(preliminary),
        "leaders": finalists,
        "selection_warning": (
            "2021-2025 OOS results are reused for finalist comparison and are therefore validation data; "
            "DSR and PBO quantify but do not eliminate this selection risk. 2026 YTD is reported but not ranked."
        ),
    }


def write_report(run_dir: Path, analysis: dict[str, object]) -> None:
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "V2.json").write_text(
        json.dumps(analysis, indent=2, default=str), encoding="utf-8"
    )
    V2_PATH.parent.mkdir(parents=True, exist_ok=True)
    V2_PATH.write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")
    lines = [
        "# V2 Point-in-Time Stock Momentum Leaders",
        "",
        f"Updated: {analysis['updated']}",
        f"Completed trials: {analysis['completed_strategy_trials']}",
        f"PBO: {float(analysis['pbo']['pbo']):.2%}",
        "",
    ]
    for index, leader in enumerate(analysis["leaders"], 1):
        config = json.loads(str(leader["config_json"]))
        mc = leader["monte_carlo_10000"]
        lines.extend(
            [
                f"## {index}. {leader['config_id']} ({leader['strategy_type']})",
                "",
                f"- Configuration: `{json.dumps(config, sort_keys=True)}`",
                f"- DSR probability: {float(leader['dsr_probability']):.2%}",
                f"- Positive OOS years: {int(leader['positive_oos_years'])}/5",
                f"- Median CAGR: {float(mc['cagr_median']):.2%}",
                f"- Median volatility: {float(mc['annualized_volatility_median']):.2%}",
                f"- 5th-percentile drawdown: {float(mc['maximum_drawdown_q05']):.2%}",
                f"- Median annualized turnover: {float(mc['annualized_turnover_median']):.2%}",
                "",
            ]
        )
    (report_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def snapshot_inputs(run_dir: Path) -> None:
    baseline = run_dir / "baseline"
    if baseline.exists():
        return
    for source in (
        ROOT / "src" / "pit_momentum.py",
        ROOT / "src" / "pit_strategy_search.py",
        ROOT / "src" / "prepare_pit_data.py",
        DATA_MANIFEST_PATH,
    ):
        target = baseline / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.round_size < 1:
        raise ValueError("workers and round size must be positive")
    load_inputs()
    run_dir = RESULTS_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    STOP_PATH.unlink(missing_ok=True)
    snapshot_inputs(run_dir)
    manifest = {
        **vars(args),
        "run_dir": str(run_dir),
        "pid": os.getpid(),
        "started": datetime.now().astimezone().isoformat(),
        "data_manifest": str(DATA_MANIFEST_PATH),
        "objective_priority": [
            "10,000-path median OOS CAGR",
            "median volatility",
            "5th-percentile maximum drawdown",
            "median annualized turnover",
        ],
        "validations": ["point-in-time membership", "walk-forward", "DSR", "PBO", "stationary bootstrap"],
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "active_run.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    results_path = run_dir / "screening" / "results.parquet"
    records = read_records(results_path)
    completed_ids = {str(record["config_id"]) for record in records}
    stream = candidate_stream(args.seed)
    round_number = 0
    last_hourly = time.monotonic()
    print(f"V2 PIT search started in {run_dir}", flush=True)
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=initialize_worker
    ) as executor:
        while not STOP_PATH.exists():
            if args.max_rounds and round_number >= args.max_rounds:
                break
            batch: list[PitConfig] = []
            while len(batch) < args.round_size:
                config = next(stream)
                if config.config_id in completed_ids:
                    continue
                completed_ids.add(config.config_id)
                batch.append(config)
            payloads = [config.to_dict() for config in batch]
            for config, outcome in zip(batch, executor.map(evaluate_candidate, payloads, chunksize=1)):
                records.append(outcome)
                print(
                    f"screen {config.config_id} {config.strategy_type} OOS CAGR "
                    f"{float(outcome['oos_cagr']):.2%}",
                    flush=True,
                )
            round_number += 1
            write_records(records, results_path)
            analysis = analyze_and_validate(records, args)
            write_report(run_dir, analysis)
            state = {
                "updated": datetime.now().astimezone().isoformat(),
                "status": "running",
                "round": round_number,
                "completed_strategy_trials": len(records),
                "leaders": [leader["config_id"] for leader in analysis["leaders"]],
                "pbo": analysis["pbo"]["pbo"],
                "stop_requested": STOP_PATH.exists(),
            }
            checkpoint = run_dir / "checkpoint" / "state.json"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps(state, indent=2), encoding="utf-8")
            if time.monotonic() - last_hourly >= args.analysis_minutes * 60.0:
                hourly = {
                    "created": datetime.now().astimezone().isoformat(),
                    "completed_strategy_trials": len(records),
                    "patterns": parameter_patterns(records),
                    "leaders": analysis["leaders"],
                    "pbo": analysis["pbo"],
                }
                hourly_dir = run_dir / "hourly_analysis"
                hourly_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                (hourly_dir / f"analysis_{stamp}.json").write_text(
                    json.dumps(hourly, indent=2, default=str), encoding="utf-8"
                )
                (hourly_dir / "latest.json").write_text(
                    json.dumps(hourly, indent=2, default=str), encoding="utf-8"
                )
                last_hourly = time.monotonic()
    final_state = {
        "updated": datetime.now().astimezone().isoformat(),
        "status": "finished",
        "round": round_number,
        "completed_strategy_trials": len(records),
        "stop_requested": STOP_PATH.exists(),
    }
    checkpoint = run_dir / "checkpoint" / "state.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps(final_state, indent=2), encoding="utf-8")
    print("V2 PIT search stopped cleanly", flush=True)


if __name__ == "__main__":
    main()
