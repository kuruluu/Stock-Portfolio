from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import pandas as pd

from search_engine import (
    StrategyConfig,
    TimeSeriesConfig,
    TrialConfig,
    evaluate_config,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "search_monthly_close_2016_2025.parquet"
FALLBACK_DATA_PATH = ROOT / "data" / "monthly_close_2016_2025.parquet"
RESULTS_ROOT = ROOT / "results" / "strategy_search"
STOP_PATH = RESULTS_ROOT / "STOP"
SPY_BENCHMARK_DIR = RESULTS_ROOT / "benchmarks" / "spy_5000"
BASE_SYMBOLS = (
    "SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP",
    "XLY", "TLT", "GLD", "MDY", "EFA", "EEM", "VNQ", "SHY", "IEF",
    "TIP", "LQD", "DBC",
)
DEFENSIVE_SYMBOLS = {
    "TLT", "GLD", "SHY", "IEF", "TIP", "LQD", "DBC", "AGG", "HYG",
    "EMB", "BIL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously search momentum-strategy configurations."
    )
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--screen-sims", type=int, default=1000)
    parser.add_argument("--final-sims", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--block-length", type=float, default=6.0)
    parser.add_argument("--hours", type=float, default=0.0)
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_inputs(run_dir: Path, data_path: Path) -> None:
    destination = run_dir / "baseline"
    destination.mkdir(parents=True, exist_ok=True)
    candidates = [
        ROOT / "data.py",
        ROOT / "requirements.txt",
        ROOT / "src" / "strategy.py",
        ROOT / "src" / "backtest.py",
        ROOT / "src" / "metrics.py",
        ROOT / "src" / "monte_carlo.py",
        ROOT / "results" / "monte_carlo_results.parquet",
        ROOT / "results" / "monte_carlo_summary.csv",
        data_path,
    ]
    hashes: list[dict[str, str]] = []
    for source in candidates:
        if not source.exists():
            continue
        relative = source.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes.append({"path": str(relative), "sha256": sha256(source)})
    pd.DataFrame(hashes).to_csv(destination / "sha256.csv", index=False)


def load_prices(path: Path) -> pd.DataFrame:
    if not path.exists():
        if path == DEFAULT_DATA_PATH and FALLBACK_DATA_PATH.exists():
            path = FALLBACK_DATA_PATH
        else:
            raise FileNotFoundError(path)
    prices = pd.read_parquet(path).sort_index()
    prices.index = pd.to_datetime(prices.index)
    expected = pd.date_range("2016-01-31", "2025-12-31", freq="ME")
    if not prices.index.equals(expected):
        raise ValueError("Dataset must contain every month from Jan 2016 through Dec 2025.")
    incomplete = prices.columns[prices.isna().any()].tolist()
    if incomplete:
        raise ValueError(f"Incomplete ETF coverage: {incomplete}")
    if "SPY" not in prices.columns:
        raise ValueError("SPY is required as the benchmark.")
    return prices


def baseline_config(universe: tuple[str, ...]) -> TrialConfig:
    symbols = tuple(symbol for symbol in BASE_SYMBOLS if symbol in universe)
    return TrialConfig(symbols=symbols)


def time_series_baseline_config(
    universe: tuple[str, ...],
) -> TimeSeriesConfig:
    return TimeSeriesConfig(
        symbols=universe,
        lookback_months=12,
        skip_months=1,
        rebalance_interval_months=1,
        cash_policy="fixed_budget",
        weighting="equal",
    )


def candidate_stream(universe: tuple[str, ...], seed: int):
    rng = random.Random(seed)
    seen: set[str] = set()

    def emit(config: TrialConfig):
        if config.config_id not in seen:
            seen.add(config.config_id)
            return config
        return None

    anchors = [
        baseline_config(universe),
        TrialConfig(symbols=universe, number_of_assets=min(3, len(universe))),
        TrialConfig(
            symbols=universe,
            lookback_months=12,
            skip_months=1,
            number_of_assets=min(5, len(universe)),
            rank_buffer=1,
        ),
        TrialConfig(
            symbols=universe,
            lookback_months=9,
            skip_months=1,
            number_of_assets=min(3, len(universe)),
            require_positive_momentum=True,
        ),
    ]
    for config in anchors:
        emitted = emit(config)
        if emitted:
            yield emitted

    time_series_anchors: list[TimeSeriesConfig] = []
    for lookback in (6, 9, 12, 15, 18):
        time_series_anchors.append(
            TimeSeriesConfig(
                symbols=universe,
                lookback_months=lookback,
                skip_months=1,
                rebalance_interval_months=1,
                cash_policy="fixed_budget",
                weighting="equal",
            )
        )
    time_series_anchors.extend([
        TimeSeriesConfig(
            symbols=universe,
            lookback_months=12,
            skip_months=1,
            rebalance_interval_months=1,
            cash_policy="fully_invested",
            weighting="equal",
        ),
        TimeSeriesConfig(
            symbols=universe,
            lookback_months=12,
            skip_months=1,
            rebalance_interval_months=1,
            cash_policy="fixed_budget",
            weighting="inverse_volatility",
        ),
        TimeSeriesConfig(
            symbols=universe,
            lookback_months=12,
            skip_months=1,
            rebalance_interval_months=3,
            cash_policy="fixed_budget",
            weighting="equal",
        ),
    ])
    for config in time_series_anchors:
        emitted = emit(config)
        if emitted:
            yield emitted

    equities = [symbol for symbol in universe if symbol not in DEFENSIVE_SYMBOLS]
    defensive = [symbol for symbol in universe if symbol in DEFENSIVE_SYMBOLS]
    sizes = [size for size in (8, 10, 12, 15, 18, 21, 24, 28, 32, len(universe)) if size <= len(universe)]

    while True:
        size = rng.choice(sizes)
        for _ in range(50):
            symbols = tuple(sorted(rng.sample(list(universe), size)))
            equity_count = sum(symbol in equities for symbol in symbols)
            defensive_count = sum(symbol in defensive for symbol in symbols)
            if equity_count >= min(4, size - 1) and defensive_count >= 1:
                break
        lookback = rng.choice((6, 9, 12, 15, 18))
        skip = rng.choice((1, 2, 3))
        if rng.random() < 0.5:
            config = TimeSeriesConfig(
                symbols=symbols,
                lookback_months=lookback,
                skip_months=skip,
                rebalance_interval_months=rng.choice((1, 1, 1, 2, 3)),
                cash_policy=rng.choice(
                    ("fixed_budget", "fixed_budget", "fully_invested")
                ),
                weighting=rng.choice(
                    ("equal", "equal", "inverse_volatility")
                ),
                volatility_window_months=rng.choice((3, 6, 9, 12)),
                transaction_cost_rate=0.0005,
            )
        else:
            config = TrialConfig(
                symbols=symbols,
                lookback_months=lookback,
                skip_months=skip,
                number_of_assets=min(rng.choice((1, 2, 3, 4, 5, 6)), size),
                require_positive_momentum=rng.choice((False, False, True)),
                use_fixed_slots=rng.choice((True, True, False)),
                rebalance_interval_months=rng.choice((1, 1, 1, 2, 3)),
                rank_buffer=rng.choice((0, 0, 1, 2)),
                signal_scaling=rng.choice(("raw", "raw", "volatility_adjusted")),
                volatility_window_months=rng.choice((3, 6, 9, 12)),
                transaction_cost_rate=0.0005,
            )
        if config.lookback_months <= config.skip_months:
            continue
        emitted = emit(config)
        if emitted:
            yield emitted


def read_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return pd.read_parquet(path).to_dict("records")


def config_from_record(record: dict[str, object]) -> StrategyConfig:
    values = json.loads(str(record["config_json"]))
    values["symbols"] = tuple(values["symbols"])
    if values.get("strategy_type") == "time_series":
        values.pop("strategy_type")
        return TimeSeriesConfig(**values)
    return TrialConfig(**values)


def record_strategy_type(record: dict[str, object]) -> str:
    strategy_type = record.get("strategy_type")
    if strategy_type and not pd.isna(strategy_type):
        return str(strategy_type)
    values = json.loads(str(record["config_json"]))
    return str(values.get("strategy_type", "cross_sectional"))


def write_records(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)
    frame.to_csv(path.with_suffix(".csv"), index=False)


def is_complete(record: dict[str, object]) -> bool:
    requested = int(record.get("requested_simulations", 0))
    completed = int(record.get("simulation_count", 0))
    valid = int(record.get("valid_simulations", 0))
    return requested > 0 and completed == requested and valid / requested >= 0.99


def needs_final_validation(
    record: dict[str, object],
    baseline_median: float | None,
    prior_best_screen: float,
    is_baseline: bool = False,
) -> bool:
    if not is_complete(record):
        return False
    if baseline_median is None:
        return False
    median = float(record.get("cagr_median", float("-inf")))
    return (
        is_baseline
        or median >= baseline_median + 0.02
        or (median > baseline_median and median >= prior_best_screen)
    )


def write_report(
    run_dir: Path,
    screening: list[dict[str, object]],
    finalists: list[dict[str, object]],
) -> None:
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    valid_finalists = [row for row in finalists if is_complete(row)]
    best_cagr = max(valid_finalists, key=lambda row: float(row["cagr_median"]), default=None)
    best_by_strategy: dict[str, dict[str, object]] = {}
    for strategy_type in ("cross_sectional", "time_series"):
        candidates = [
            row
            for row in valid_finalists
            if record_strategy_type(row) == strategy_type
        ]
        if candidates:
            best_by_strategy[strategy_type] = max(
                candidates, key=lambda row: float(row["cagr_median"])
            )
    risk_sane = [
        row for row in valid_finalists
        if float(row.get("cagr_q05", -1.0)) > 0.0
        and float(row.get("maximum_drawdown_q05", -1.0)) >= -0.35
        and float(row.get("annualized_volatility_median", 99.0))
        <= 1.25 * float(row.get("spy_annualized_volatility_median", 0.0))
        and float(row.get("sharpe_ratio_median", -99.0))
        >= float(row.get("spy_sharpe_ratio_median", 99.0))
    ]
    best_overall = max(risk_sane, key=lambda row: float(row["cagr_median"]), default=None)

    def without_spy_metrics(
        record: dict[str, object] | None,
    ) -> dict[str, object] | None:
        if record is None:
            return None
        return {
            key: value
            for key, value in record.items()
            if not key.startswith("spy_")
        }

    best_cagr_payload = {
        "updated": datetime.now().astimezone().isoformat(),
        "objective": "highest median Monte Carlo CAGR",
        "best_overall": without_spy_metrics(best_cagr),
        "best_by_strategy": {
            strategy_type: without_spy_metrics(record)
            for strategy_type, record in best_by_strategy.items()
        },
    }
    (report_dir / "best_cagr.json").write_text(
        json.dumps(best_cagr_payload, indent=2, default=str), encoding="utf-8"
    )
    (report_dir / "best_overall.json").write_text(
        json.dumps(best_overall, indent=2, default=str), encoding="utf-8"
    )
    write_spy_data(report_dir)
    lines = [
        "# Momentum Strategy Search",
        "",
        f"Updated: {datetime.now().astimezone().isoformat()}",
        f"Screened configurations: {len(screening)}",
        f"Completed 5,000-path finalists: {len(valid_finalists)}",
        "",
    ]

    def append_record(title: str, record: dict[str, object] | None) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if record is None:
            lines.append("No qualifying completed finalist yet.")
        else:
            config = json.loads(str(record["config_json"]))
            lines.extend([
                f"- Config ID: `{record['config_id']}`",
                f"- Strategy type: {record_strategy_type(record)}",
                f"- Median CAGR: {float(record['cagr_median']):.2%}",
                f"- 5th-percentile CAGR: {float(record['cagr_q05']):.2%}",
                f"- Median volatility: {float(record['annualized_volatility_median']):.2%}",
                f"- 5th-percentile maximum drawdown: {float(record['maximum_drawdown_q05']):.2%}",
                f"- Median Sharpe: {float(record['sharpe_ratio_median']):.3f}",
                f"- Parameters: `{json.dumps(config, sort_keys=True)}`",
            ])
        lines.append("")

    append_record("Highest median CAGR", best_cagr)
    append_record("Best overall", best_overall)
    append_record(
        "Best cross-sectional momentum",
        best_by_strategy.get("cross_sectional"),
    )
    append_record(
        "Best time-series momentum",
        best_by_strategy.get("time_series"),
    )
    (report_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_spy_data(report_dir: Path) -> None:
    summary_path = SPY_BENCHMARK_DIR / "summary.csv"
    metadata_path = SPY_BENCHMARK_DIR / "metadata.json"
    if not summary_path.exists() or not metadata_path.exists():
        return

    summary = pd.read_csv(summary_path, index_col=0)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    percentile_names = {
        "1st percentile": "p01",
        "5th percentile": "p05",
        "25th percentile": "p25",
        "median": "median",
        "75th percentile": "p75",
        "95th percentile": "p95",
        "99th percentile": "p99",
    }
    metrics: dict[str, dict[str, float]] = {}
    for column in summary.columns:
        metrics[column] = {
            percentile_names[str(index)]: float(value)
            for index, value in summary[column].items()
            if str(index) in percentile_names and pd.notna(value)
        }

    payload = {
        "updated": datetime.now().astimezone().isoformat(),
        "benchmark": metadata["strategy"],
        "method": {
            "simulations": metadata["simulations"],
            "seed": metadata["seed"],
            "average_block_length_months": metadata[
                "average_block_length_months"
            ],
            "initial_capital": metadata["initial_capital"],
            "price_start": metadata["price_start"],
            "price_end": metadata["price_end"],
            "historical_return_observations": metadata[
                "historical_return_observations"
            ],
            "simulated_months_per_path": metadata[
                "simulated_months_per_path"
            ],
        },
        "probabilities": {
            "loss": metadata["probability_of_loss"],
            "sharpe_above_one": metadata[
                "probability_sharpe_above_one"
            ],
        },
        "metrics": metrics,
        "source_files": {
            "summary": str(summary_path),
            "metadata": str(metadata_path),
            "simulations": str(SPY_BENCHMARK_DIR / "simulations.parquet"),
        },
    }
    (report_dir / "spy_data.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def write_state(run_dir: Path, state: dict[str, object]) -> None:
    target = run_dir / "checkpoint" / "state.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.screen_sims < 1 or args.final_sims < 1:
        raise ValueError("workers and simulation counts must be positive")
    run_dir = RESULTS_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    STOP_PATH.unlink(missing_ok=True)
    prices = load_prices(args.data)
    data_path = args.data if args.data.exists() else FALLBACK_DATA_PATH
    snapshot_inputs(run_dir, data_path)
    universe = tuple(prices.columns)
    manifest = {
        **vars(args),
        "data": str(data_path),
        "run_dir": str(run_dir),
        "pid": os.getpid(),
        "started": datetime.now().astimezone().isoformat(),
        "universe": list(universe),
        "coverage_rows": len(prices),
        "coverage_start": str(prices.index.min().date()),
        "coverage_end": str(prices.index.max().date()),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "active_run.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    screening_path = run_dir / "screening" / "results.parquet"
    finalists_path = run_dir / "finalists" / "results.parquet"
    screening = read_records(screening_path)
    finalists = read_records(finalists_path)
    screened_ids = {str(row["config_id"]) for row in screening}
    finalist_ids = {
        str(row["config_id"]) for row in finalists if is_complete(row)
    }
    baseline_ids = {
        "cross_sectional": baseline_config(universe).config_id,
        "time_series": time_series_baseline_config(universe).config_id,
    }
    baseline_medians: dict[str, float | None] = {
        strategy_type: next(
            (
                float(row["cagr_median"])
                for row in screening
                if row["config_id"] == config_id and is_complete(row)
            ),
            None,
        )
        for strategy_type, config_id in baseline_ids.items()
    }
    best_screen = max(
        (float(row["cagr_median"]) for row in screening if is_complete(row)),
        default=float("-inf"),
    )
    best_screen_by_strategy = {
        strategy_type: max(
            (
                float(row["cagr_median"])
                for row in screening
                if is_complete(row)
                and record_strategy_type(row) == strategy_type
            ),
            default=float("-inf"),
        )
        for strategy_type in baseline_ids
    }
    queued_finalists: deque[StrategyConfig] = deque()
    completed_screens = [row for row in screening if is_complete(row)]
    for strategy_type, baseline_median in baseline_medians.items():
        if baseline_median is None:
            continue
        for record in completed_screens:
            if record_strategy_type(record) != strategy_type:
                continue
            record_id = str(record["config_id"])
            median = float(record["cagr_median"])
            qualifies = (
                record_id == baseline_ids[strategy_type]
                or median >= baseline_median + 0.02
                or (
                    median > baseline_median
                    and median >= best_screen_by_strategy[strategy_type]
                )
            )
            if qualifies and record_id not in finalist_ids:
                finalist_ids.add(record_id)
                queued_finalists.append(config_from_record(record))
    stream = candidate_stream(universe, args.seed)
    deadline = time.monotonic() + args.hours * 3600 if args.hours > 0 else None
    submitted_screens = len(screening)
    pending: dict[object, tuple[str, StrategyConfig]] = {}

    def submit(
        executor: ProcessPoolExecutor,
        stage: str,
        config: StrategyConfig,
    ) -> None:
        simulations = args.final_sims if stage == "final" else args.screen_sims
        future = executor.submit(
            evaluate_config,
            prices,
            config,
            simulations,
            args.seed,
            args.block_length,
            str(STOP_PATH),
        )
        pending[future] = (stage, config)

    print(f"Search started in {run_dir} with {args.workers} workers", flush=True)
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            while True:
                should_stop = STOP_PATH.exists() or (deadline is not None and time.monotonic() >= deadline)
                while not should_stop and len(pending) < args.workers:
                    if queued_finalists:
                        config = queued_finalists.popleft()
                        submit(executor, "final", config)
                        continue
                    if args.max_configs and submitted_screens >= args.max_configs:
                        break
                    config = next(stream)
                    if config.config_id in screened_ids:
                        continue
                    screened_ids.add(config.config_id)
                    submitted_screens += 1
                    submit(executor, "screen", config)

                if not pending:
                    break
                done, _ = wait(pending, timeout=5.0, return_when=FIRST_COMPLETED)
                for future in done:
                    stage, config = pending.pop(future)
                    try:
                        record = future.result()
                    except Exception as error:
                        record = {
                            "config_id": config.config_id,
                            "config_json": json.dumps(config.to_dict(), sort_keys=True),
                            "stage_error": repr(error),
                            "requested_simulations": args.final_sims if stage == "final" else args.screen_sims,
                            "simulation_count": 0,
                            "valid_simulations": 0,
                            "invalid_simulations": 0,
                            "stopped_early": False,
                        }
                    if stage == "screen":
                        screening.append(record)
                        write_records(screening, screening_path)
                        strategy_type = record_strategy_type(record)
                        strategy_baseline_id = baseline_ids[strategy_type]
                        if (
                            config.config_id == strategy_baseline_id
                            and is_complete(record)
                        ):
                            baseline_medians[strategy_type] = float(
                                record["cagr_median"]
                            )
                            earlier = [
                                row
                                for row in screening
                                if is_complete(row)
                                and record_strategy_type(row) == strategy_type
                            ]
                            earlier_best = max(
                                (float(row["cagr_median"]) for row in earlier),
                                default=baseline_medians[strategy_type],
                            )
                            for prior_record in earlier:
                                prior_id = str(prior_record["config_id"])
                                prior_median = float(prior_record["cagr_median"])
                                qualifies = (
                                    prior_median
                                    >= float(baseline_medians[strategy_type]) + 0.02
                                    or (
                                        prior_median
                                        > float(baseline_medians[strategy_type])
                                        and prior_median >= earlier_best
                                    )
                                )
                                if qualifies and prior_id not in finalist_ids:
                                    finalist_ids.add(prior_id)
                                    queued_finalists.append(
                                        config_from_record(prior_record)
                                    )
                        prior_best = best_screen_by_strategy[strategy_type]
                        if is_complete(record):
                            best_screen = max(best_screen, float(record["cagr_median"]))
                            best_screen_by_strategy[strategy_type] = max(
                                best_screen_by_strategy[strategy_type],
                                float(record["cagr_median"]),
                            )
                        if needs_final_validation(
                            record,
                            baseline_medians[strategy_type],
                            prior_best,
                            is_baseline=(
                                config.config_id == strategy_baseline_id
                            ),
                        ) and config.config_id not in finalist_ids:
                            finalist_ids.add(config.config_id)
                            queued_finalists.append(config)
                    else:
                        finalists.append(record)
                        write_records(finalists, finalists_path)
                    write_report(run_dir, screening, finalists)
                    write_state(run_dir, {
                        "updated": datetime.now().astimezone().isoformat(),
                        "screened": len(screening),
                        "finalists": len(finalists),
                        "pending": len(pending),
                        "best_screen_median_cagr": best_screen,
                        "baseline_median_cagr": baseline_medians[
                            "cross_sectional"
                        ],
                        "time_series_baseline_median_cagr": baseline_medians[
                            "time_series"
                        ],
                        "best_screen_by_strategy": best_screen_by_strategy,
                        "stop_requested": STOP_PATH.exists(),
                    })
                    metric = record.get("cagr_median")
                    metric_text = f"{float(metric):.2%}" if metric is not None else "error"
                    print(f"{stage}: {config.config_id} median CAGR {metric_text}", flush=True)
                if STOP_PATH.exists() or (deadline is not None and time.monotonic() >= deadline):
                    for future in pending:
                        future.cancel()
                    if all(future.done() for future in pending):
                        break
                if args.max_configs and submitted_screens >= args.max_configs and not queued_finalists and not pending:
                    break
    finally:
        write_report(run_dir, screening, finalists)
        write_state(run_dir, {
            "updated": datetime.now().astimezone().isoformat(),
            "screened": len(screening),
            "finalists": len(finalists),
            "pending": 0,
            "best_screen_median_cagr": best_screen,
            "baseline_median_cagr": baseline_medians["cross_sectional"],
            "time_series_baseline_median_cagr": baseline_medians[
                "time_series"
            ],
            "best_screen_by_strategy": best_screen_by_strategy,
            "finished": True,
        })
        print("Search stopped cleanly", flush=True)


if __name__ == "__main__":
    main()
