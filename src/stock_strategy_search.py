from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import pandas as pd

from stock_search_engine import StockConfig, evaluate_stock_config


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "stock_monthly_close_2016_2025.parquet"
UNIVERSE_PATH = ROOT / "data" / "stock_universe_200.csv"
MANIFEST_PATH = ROOT / "data" / "stock_universe_manifest.json"
RESULTS_ROOT = ROOT / "results" / "stock_strategy_search"
STOP_PATH = RESULTS_ROOT / "STOP"

PARAMETER_VALUES: dict[str, tuple[object, ...]] = {
    "strategy_type": ("cross_sectional", "sector_neutral", "time_series"),
    "lookback_months": (6, 9, 12, 15, 18, 24),
    "skip_months": (1, 2, 3, 6),
    "number_of_assets": (5, 10, 15, 20, 30, 40),
    "assets_per_sector": (1, 2, 3),
    "require_positive_momentum": (False, True),
    "use_fixed_slots": (False, True),
    "rebalance_interval_months": (1, 2, 3, 6),
    "signal_scaling": ("raw", "volatility_adjusted"),
    "weighting": ("equal", "inverse_volatility"),
    "volatility_window_months": (3, 6, 9, 12),
    "universe_size": (50, 100, 150, 200),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously search adaptive stock-momentum strategies."
    )
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--screen-sims", type=int, default=1000)
    parser.add_argument("--final-sims", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--block-length", type=float, default=6.0)
    parser.add_argument("--analysis-minutes", type=float, default=60.0)
    parser.add_argument("--hours", type=float, default=0.0)
    parser.add_argument("--max-configs", type=int, default=0)
    return parser.parse_args()


def encode_value(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def default_sampler_weights() -> dict[str, dict[str, float]]:
    return {
        parameter: {encode_value(value): 1.0 for value in values}
        for parameter, values in PARAMETER_VALUES.items()
    }


def weighted_choice(
    rng: random.Random,
    parameter: str,
    sampler_weights: dict[str, dict[str, float]],
) -> object:
    values = PARAMETER_VALUES[parameter]
    weights = [
        sampler_weights[parameter].get(encode_value(value), 1.0)
        for value in values
    ]
    return rng.choices(values, weights=weights, k=1)[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_inputs(run_dir: Path) -> None:
    destination = run_dir / "baseline"
    destination.mkdir(parents=True, exist_ok=True)
    hash_manifest = destination / "sha256.csv"
    if hash_manifest.exists():
        return
    candidates = [
        ROOT / "requirements.txt",
        ROOT / "src" / "stock_search_engine.py",
        ROOT / "src" / "stock_strategy_search.py",
        ROOT / "src" / "prepare_stock_data.py",
        DATA_PATH,
        UNIVERSE_PATH,
        MANIFEST_PATH,
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
    pd.DataFrame(hashes).to_csv(hash_manifest, index=False)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    prices = pd.read_parquet(DATA_PATH).sort_index()
    prices.index = pd.to_datetime(prices.index)
    expected = pd.date_range("2016-01-31", "2025-12-31", freq="ME")
    if prices.shape != (120, 200) or not prices.index.equals(expected):
        raise ValueError("Stock price data must be a complete 120 x 200 matrix")
    if prices.isna().any().any():
        raise ValueError("Stock price data contains missing values")
    universe = pd.read_csv(UNIVERSE_PATH)
    if set(universe["symbol"]) != set(prices.columns):
        raise ValueError("Universe metadata does not match stock price columns")
    universe = universe.set_index("symbol").loc[list(prices.columns)].reset_index()
    sector_by_symbol = dict(zip(universe["symbol"], universe["sector"]))
    return prices, universe, sector_by_symbol


def allocate_balanced_counts(
    pools: dict[str, list[str]], size: int
) -> dict[str, int]:
    counts = {sector: 1 for sector in pools}
    if size < len(counts):
        raise ValueError("subset is too small to contain every sector")
    total_available = sum(len(values) for values in pools.values())
    while sum(counts.values()) < size:
        candidates = [
            sector
            for sector, values in pools.items()
            if counts[sector] < len(values)
        ]
        sector = max(
            candidates,
            key=lambda value: (
                size * len(pools[value]) / total_available - counts[value],
                len(pools[value]) - counts[value],
                value,
            ),
        )
        counts[sector] += 1
    return counts


def balanced_subset(
    universe: pd.DataFrame,
    size: int,
    rng: random.Random,
) -> tuple[str, ...]:
    pools = {
        sector: group["symbol"].tolist()
        for sector, group in universe.groupby("sector")
    }
    counts = allocate_balanced_counts(pools, size)
    selected: list[str] = []
    for sector in sorted(pools):
        selected.extend(rng.sample(pools[sector], counts[sector]))
    return tuple(sorted(selected))


def anchor_configs(symbols: tuple[str, ...]) -> list[StockConfig]:
    return [
        StockConfig(symbols=symbols, strategy_type="cross_sectional"),
        StockConfig(
            symbols=symbols,
            strategy_type="sector_neutral",
            assets_per_sector=2,
            require_positive_momentum=False,
        ),
        StockConfig(
            symbols=symbols,
            strategy_type="time_series",
            require_positive_momentum=True,
            use_fixed_slots=False,
        ),
        StockConfig(
            symbols=symbols,
            strategy_type="cross_sectional",
            lookback_months=15,
            skip_months=3,
            number_of_assets=10,
            rebalance_interval_months=3,
        ),
        StockConfig(
            symbols=symbols,
            strategy_type="cross_sectional",
            lookback_months=12,
            skip_months=1,
            number_of_assets=30,
            require_positive_momentum=True,
        ),
        StockConfig(
            symbols=symbols,
            strategy_type="sector_neutral",
            lookback_months=15,
            skip_months=3,
            assets_per_sector=1,
            rebalance_interval_months=3,
        ),
        StockConfig(
            symbols=symbols,
            strategy_type="time_series",
            lookback_months=15,
            skip_months=3,
            weighting="inverse_volatility",
            volatility_window_months=9,
            use_fixed_slots=True,
        ),
    ]


def candidate_stream(
    universe: pd.DataFrame,
    seed: int,
    sampler_weights: dict[str, dict[str, float]],
):
    rng = random.Random(seed)
    full_symbols = tuple(sorted(universe["symbol"]))
    seen: set[str] = set()
    for config in anchor_configs(full_symbols):
        if config.config_id not in seen:
            seen.add(config.config_id)
            yield config

    while True:
        size = int(weighted_choice(rng, "universe_size", sampler_weights))
        symbols = balanced_subset(universe, size, rng)
        strategy_type = str(
            weighted_choice(rng, "strategy_type", sampler_weights)
        )
        lookback = int(
            weighted_choice(rng, "lookback_months", sampler_weights)
        )
        valid_skips = [
            value
            for value in PARAMETER_VALUES["skip_months"]
            if int(value) < lookback
        ]
        skip_weights = [
            sampler_weights["skip_months"].get(encode_value(value), 1.0)
            for value in valid_skips
        ]
        skip = int(rng.choices(valid_skips, weights=skip_weights, k=1)[0])
        number_assets = min(
            int(weighted_choice(rng, "number_of_assets", sampler_weights)),
            size,
        )
        assets_per_sector = int(
            weighted_choice(rng, "assets_per_sector", sampler_weights)
        )
        require_positive = bool(
            weighted_choice(rng, "require_positive_momentum", sampler_weights)
        )
        signal_scaling = str(
            weighted_choice(rng, "signal_scaling", sampler_weights)
        )
        weighting = str(weighted_choice(rng, "weighting", sampler_weights))
        volatility_window = int(
            weighted_choice(
                rng, "volatility_window_months", sampler_weights
            )
        )
        if strategy_type == "time_series":
            number_assets = min(20, size)
            assets_per_sector = 2
            require_positive = True
            signal_scaling = "raw"
        elif strategy_type == "sector_neutral":
            number_assets = min(20, size)
        else:
            assets_per_sector = 2
        if signal_scaling == "raw" and weighting == "equal":
            volatility_window = 6
        config = StockConfig(
            symbols=symbols,
            strategy_type=strategy_type,
            lookback_months=lookback,
            skip_months=skip,
            number_of_assets=number_assets,
            assets_per_sector=assets_per_sector,
            require_positive_momentum=require_positive,
            use_fixed_slots=bool(
                weighted_choice(rng, "use_fixed_slots", sampler_weights)
            ),
            rebalance_interval_months=int(
                weighted_choice(
                    rng, "rebalance_interval_months", sampler_weights
                )
            ),
            signal_scaling=signal_scaling,
            weighting=weighting,
            volatility_window_months=volatility_window,
            transaction_cost_rate=0.0005,
        )
        if config.config_id not in seen:
            seen.add(config.config_id)
            yield config


def config_from_record(record: dict[str, object]) -> StockConfig:
    values = json.loads(str(record["config_json"]))
    values["symbols"] = tuple(values["symbols"])
    return StockConfig(**values)


def is_complete(record: dict[str, object]) -> bool:
    requested = int(record.get("requested_simulations", 0))
    completed = int(record.get("simulation_count", 0))
    valid = int(record.get("valid_simulations", 0))
    return requested > 0 and completed == requested and valid / requested >= 0.99


def read_records(path: Path) -> list[dict[str, object]]:
    return pd.read_parquet(path).to_dict("records") if path.exists() else []


def write_records(records: list[dict[str, object]], path: Path) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)
    frame.to_csv(path.with_suffix(".csv"), index=False)


def parameter_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in records:
        if not is_complete(record) or "cagr_median" not in record:
            continue
        config = json.loads(str(record["config_json"]))
        rows.append(
            {
                "config_id": record["config_id"],
                "cagr_median": float(record["cagr_median"]),
                "volatility_median": float(
                    record["annualized_volatility_median"]
                ),
                "drawdown_q05": float(record["maximum_drawdown_q05"]),
                "turnover_median": float(record["annualized_turnover_median"]),
                "strategy_type": config["strategy_type"],
                "lookback_months": config["lookback_months"],
                "skip_months": config["skip_months"],
                "number_of_assets": config["number_of_assets"],
                "assets_per_sector": config["assets_per_sector"],
                "require_positive_momentum": config[
                    "require_positive_momentum"
                ],
                "use_fixed_slots": config["use_fixed_slots"],
                "rebalance_interval_months": config[
                    "rebalance_interval_months"
                ],
                "signal_scaling": config["signal_scaling"],
                "weighting": config["weighting"],
                "volatility_window_months": config[
                    "volatility_window_months"
                ],
                "universe_size": len(config["symbols"]),
            }
        )
    return pd.DataFrame(rows)


def relevant_parameter_rows(
    frame: pd.DataFrame,
    parameter: str,
) -> pd.DataFrame:
    if parameter == "number_of_assets":
        return frame.loc[frame["strategy_type"] == "cross_sectional"]
    if parameter == "assets_per_sector":
        return frame.loc[frame["strategy_type"] == "sector_neutral"]
    if parameter in {"require_positive_momentum", "signal_scaling"}:
        return frame.loc[frame["strategy_type"] != "time_series"]
    if parameter == "volatility_window_months":
        return frame.loc[
            (frame["signal_scaling"] == "volatility_adjusted")
            | (frame["weighting"] == "inverse_volatility")
        ]
    return frame


def analyze_and_adapt(
    screening: list[dict[str, object]],
    sampler_weights: dict[str, dict[str, float]],
    analysis_dir: Path,
) -> tuple[dict[str, object], list[StockConfig]]:
    frame = parameter_frame(screening)
    if len(frame) < 20:
        return {"status": "insufficient completed screens", "count": len(frame)}, []
    cutoff = float(frame["cagr_median"].quantile(0.80))
    top = frame.loc[frame["cagr_median"] >= cutoff]
    analysis: dict[str, object] = {
        "created": datetime.now().astimezone().isoformat(),
        "completed_screens": len(frame),
        "top_quintile_cutoff_cagr": cutoff,
        "top_quintile_count": len(top),
        "parameters": {},
    }
    for parameter, values in PARAMETER_VALUES.items():
        parameter_data = relevant_parameter_rows(frame, parameter)
        groups: dict[str, object] = {}
        parameter_cutoff = float(parameter_data["cagr_median"].quantile(0.80))
        parameter_top = parameter_data.loc[
            parameter_data["cagr_median"] >= parameter_cutoff
        ]
        global_top_rate = len(parameter_top) / len(parameter_data)
        for value in values:
            subset = parameter_data.loc[parameter_data[parameter] == value]
            if subset.empty:
                groups[encode_value(value)] = {"count": 0, "weight": 1.0}
                sampler_weights[parameter][encode_value(value)] = 1.0
                continue
            subset_top = subset.loc[
                subset["cagr_median"] >= parameter_cutoff
            ]
            smoothed_top_rate = (len(subset_top) + 2.0) / (len(subset) + 10.0)
            relative_success = smoothed_top_rate / global_top_rate
            cagr_median = float(subset["cagr_median"].median())
            near_best = cagr_median >= float(
                parameter_data.groupby(parameter)["cagr_median"].median().max()
            ) - 0.005
            risk_bonus = 1.0
            if near_best:
                risk_values = (
                    subset["volatility_median"].rank(pct=True).median()
                    + subset["drawdown_q05"].abs().rank(pct=True).median()
                    + subset["turnover_median"].rank(pct=True).median()
                ) / 3.0
                risk_bonus = 1.25 - 0.25 * float(risk_values)
            new_weight = float(
                np.clip((0.5 + 1.75 * relative_success) * risk_bonus, 0.25, 5.0)
            )
            sampler_weights[parameter][encode_value(value)] = new_weight
            groups[encode_value(value)] = {
                "count": len(subset),
                "top_quintile_count": len(subset_top),
                "relevant_family_count": len(parameter_data),
                "relevant_top_quintile_cutoff_cagr": parameter_cutoff,
                "median_cagr": cagr_median,
                "median_volatility": float(
                    subset["volatility_median"].median()
                ),
                "median_q05_drawdown": float(subset["drawdown_q05"].median()),
                "median_annualized_turnover": float(
                    subset["turnover_median"].median()
                ),
                "new_sampling_weight": new_weight,
            }
        analysis["parameters"][parameter] = groups
    leaders = (
        frame.sort_values("cagr_median", ascending=False)
        .drop_duplicates("strategy_type")
        .to_dict("records")
    )
    analysis["strategy_leaders"] = leaders
    analysis["sampler_weights"] = sampler_weights
    analysis_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (analysis_dir / f"analysis_{timestamp}.json").write_text(
        json.dumps(analysis, indent=2, default=str), encoding="utf-8"
    )
    (analysis_dir / "latest.json").write_text(
        json.dumps(analysis, indent=2, default=str), encoding="utf-8"
    )
    record_by_id = {str(record["config_id"]): record for record in screening}
    validation_configs = [
        config_from_record(record_by_id[str(row["config_id"])])
        for row in leaders
    ]
    return analysis, validation_configs


def choose_balanced_winner(
    finalists: list[dict[str, object]],
) -> tuple[dict[str, object] | None, dict[str, object]]:
    valid = [record for record in finalists if is_complete(record)]
    if not valid:
        return None, {"rule": "no completed finalists"}
    frame = pd.DataFrame(valid)
    best_cagr = float(frame["cagr_median"].max())
    near = frame.loc[frame["cagr_median"] >= best_cagr - 0.01].copy()
    near["volatility_rank"] = near["annualized_volatility_median"].rank(pct=True)
    near["drawdown_rank"] = near["maximum_drawdown_q05"].abs().rank(pct=True)
    near["turnover_rank"] = near["annualized_turnover_median"].rank(pct=True)
    near["secondary_risk_score"] = near[
        ["volatility_rank", "drawdown_rank", "turnover_rank"]
    ].mean(axis=1)
    winner_row = near.sort_values(
        ["secondary_risk_score", "cagr_median"],
        ascending=[True, False],
    ).iloc[0]
    winner = next(
        record
        for record in valid
        if record["config_id"] == winner_row["config_id"]
    )
    rule = {
        "primary": "median CAGR within 1 percentage point of the highest finalist",
        "secondary": "lowest equal-weight percentile rank across median volatility, absolute 5th-percentile drawdown, and median annualized turnover",
        "highest_median_cagr": best_cagr,
        "eligible_finalists": len(near),
        "winner_secondary_risk_score": float(
            winner_row["secondary_risk_score"]
        ),
    }
    return winner, rule


def write_reports(
    run_dir: Path,
    screening: list[dict[str, object]],
    finalists: list[dict[str, object]],
) -> None:
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    valid_screens = [record for record in screening if is_complete(record)]
    valid_finals = [record for record in finalists if is_complete(record)]
    screen_leader = max(
        valid_screens,
        key=lambda record: float(record.get("cagr_median", -math.inf)),
        default=None,
    )
    best_cagr = max(
        valid_finals,
        key=lambda record: float(record.get("cagr_median", -math.inf)),
        default=None,
    )
    best_by_strategy: dict[str, dict[str, object]] = {}
    for strategy_type in PARAMETER_VALUES["strategy_type"]:
        candidates = [
            record
            for record in valid_finals
            if record.get("strategy_type") == strategy_type
        ]
        if candidates:
            best_by_strategy[str(strategy_type)] = max(
                candidates, key=lambda record: float(record["cagr_median"])
            )
    balanced, balanced_rule = choose_balanced_winner(valid_finals)
    (report_dir / "best_cagr.json").write_text(
        json.dumps(
            {
                "updated": datetime.now().astimezone().isoformat(),
                "objective": "highest median Monte Carlo CAGR",
                "finalist_simulations": 10000,
                "best_overall": best_cagr,
                "best_by_strategy": best_by_strategy,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    (report_dir / "best_balanced.json").write_text(
        json.dumps(
            {
                "updated": datetime.now().astimezone().isoformat(),
                "selection_rule": balanced_rule,
                "winner": balanced,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    (report_dir / "screening_leader.json").write_text(
        json.dumps(screen_leader, indent=2, default=str), encoding="utf-8"
    )
    lines = [
        "# Adaptive Stock Momentum Search",
        "",
        f"Updated: {datetime.now().astimezone().isoformat()}",
        f"Completed 1,000-path screens: {len(valid_screens)}",
        f"Completed 10,000-path finalists: {len(valid_finals)}",
        "",
    ]
    for title, record in (
        ("Highest median CAGR finalist", best_cagr),
        ("Best near-peak risk/turnover balance", balanced),
        ("Current screening leader", screen_leader),
    ):
        lines.extend([f"## {title}", ""])
        if record is None:
            lines.append("No qualifying result yet.")
        else:
            lines.extend(
                [
                    f"- Config ID: `{record['config_id']}`",
                    f"- Strategy: {record['strategy_type']}",
                    f"- Median CAGR: {float(record['cagr_median']):.2%}",
                    f"- Median volatility: {float(record['annualized_volatility_median']):.2%}",
                    f"- 5th-percentile drawdown: {float(record['maximum_drawdown_q05']):.2%}",
                    f"- Median annualized turnover: {float(record['annualized_turnover_median']):.2%}",
                ]
            )
        lines.append("")
    (report_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_state(run_dir: Path, state: dict[str, object]) -> None:
    path = run_dir / "checkpoint" / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def next_analysis_time(
    run_dir: Path,
    interval_minutes: float,
    now: datetime | None = None,
) -> datetime:
    current = now or datetime.now().astimezone()
    latest_path = run_dir / "hourly_analysis" / "latest.json"
    if not latest_path.exists():
        return current + timedelta(minutes=interval_minutes)
    try:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        last_created = datetime.fromisoformat(str(latest["created"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return current + timedelta(minutes=interval_minutes)
    scheduled = last_created + timedelta(minutes=interval_minutes)
    return max(current, scheduled)


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.screen_sims < 1 or args.final_sims < 1:
        raise ValueError("workers and simulation counts must be positive")
    prices, universe, sector_by_symbol = load_inputs()
    run_dir = RESULTS_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    STOP_PATH.unlink(missing_ok=True)
    snapshot_inputs(run_dir)
    manifest = {
        **vars(args),
        "run_dir": str(run_dir),
        "pid": os.getpid(),
        "started": datetime.now().astimezone().isoformat(),
        "universe_count": len(prices.columns),
        "sectors": universe["sector"].value_counts().sort_index().to_dict(),
        "coverage_start": str(prices.index.min().date()),
        "coverage_end": str(prices.index.max().date()),
        "objective_priority": [
            "median CAGR",
            "median volatility",
            "5th-percentile maximum drawdown",
            "median annualized turnover",
        ],
        "known_limitation": "fixed current-constituent universe creates survivorship and membership-selection bias",
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
    screened_ids = {str(record["config_id"]) for record in screening}
    finalist_ids = {
        str(record["config_id"]) for record in finalists if is_complete(record)
    }
    sampler_path = run_dir / "checkpoint" / "sampler_weights.json"
    sampler_weights = (
        json.loads(sampler_path.read_text(encoding="utf-8"))
        if sampler_path.exists()
        else default_sampler_weights()
    )
    stream = candidate_stream(universe, args.seed, sampler_weights)
    full_symbols = tuple(sorted(prices.columns))
    baseline_configs = {
        config.strategy_type: config
        for config in anchor_configs(full_symbols)[:3]
    }
    baseline_ids = {
        strategy_type: config.config_id
        for strategy_type, config in baseline_configs.items()
    }
    best_screen_by_strategy = {
        str(strategy_type): max(
            (
                float(record["cagr_median"])
                for record in screening
                if is_complete(record)
                and record.get("strategy_type") == strategy_type
            ),
            default=-math.inf,
        )
        for strategy_type in PARAMETER_VALUES["strategy_type"]
    }
    queued_finalists: deque[StockConfig] = deque()
    for config in baseline_configs.values():
        if config.config_id in screened_ids and config.config_id not in finalist_ids:
            record = next(
                row for row in screening if row["config_id"] == config.config_id
            )
            queued_finalists.append(config_from_record(record))
            finalist_ids.add(config.config_id)

    pending: dict[object, tuple[str, StockConfig]] = {}
    submitted_screens = len(screening)
    completed_since_checkpoint = 0
    last_checkpoint = time.monotonic()
    next_analysis = next_analysis_time(
        run_dir,
        args.analysis_minutes,
    )
    deadline = time.monotonic() + args.hours * 3600 if args.hours > 0 else None

    def submit(
        executor: ProcessPoolExecutor,
        stage: str,
        config: StockConfig,
    ) -> None:
        simulations = args.final_sims if stage == "final" else args.screen_sims
        future = executor.submit(
            evaluate_stock_config,
            prices,
            sector_by_symbol,
            config,
            simulations,
            args.seed,
            args.block_length,
            str(STOP_PATH),
            250,
        )
        pending[future] = (stage, config)

    print(
        f"Stock search started in {run_dir} with {args.workers} workers",
        flush=True,
    )
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            while True:
                should_stop = STOP_PATH.exists() or (
                    deadline is not None and time.monotonic() >= deadline
                )
                while not should_stop and len(pending) < args.workers:
                    if queued_finalists:
                        submit(executor, "final", queued_finalists.popleft())
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

                done, _ = wait(pending, timeout=2.0, return_when=FIRST_COMPLETED)
                for future in done:
                    stage, config = pending.pop(future)
                    try:
                        record = future.result()
                    except Exception as error:
                        record = {
                            "config_id": config.config_id,
                            "strategy_type": config.strategy_type,
                            "config_json": json.dumps(
                                config.to_dict(), sort_keys=True
                            ),
                            "requested_simulations": (
                                args.final_sims if stage == "final" else args.screen_sims
                            ),
                            "simulation_count": 0,
                            "valid_simulations": 0,
                            "invalid_simulations": 0,
                            "stopped_early": False,
                            "stage_error": repr(error),
                        }
                    if stage == "screen":
                        screening.append(record)
                        if is_complete(record):
                            strategy_type = config.strategy_type
                            median = float(record["cagr_median"])
                            prior_best = best_screen_by_strategy[strategy_type]
                            best_screen_by_strategy[strategy_type] = max(
                                prior_best, median
                            )
                            is_baseline = config.config_id == baseline_ids[strategy_type]
                            is_new_leader = median >= prior_best + 0.001
                            if (
                                (is_baseline or is_new_leader)
                                and config.config_id not in finalist_ids
                            ):
                                finalist_ids.add(config.config_id)
                                queued_finalists.append(config)
                    else:
                        finalists.append(record)
                    completed_since_checkpoint += 1
                    metric = record.get("cagr_median")
                    metric_text = (
                        f"{float(metric):.2%}" if metric is not None else "error"
                    )
                    print(
                        f"{stage}: {config.strategy_type} {config.config_id} "
                        f"median CAGR {metric_text}",
                        flush=True,
                    )

                now = datetime.now().astimezone()
                if now >= next_analysis:
                    analysis, validation_configs = analyze_and_adapt(
                        screening,
                        sampler_weights,
                        run_dir / "hourly_analysis",
                    )
                    for config in validation_configs:
                        if config.config_id not in finalist_ids:
                            finalist_ids.add(config.config_id)
                            queued_finalists.append(config)
                    sampler_path.parent.mkdir(parents=True, exist_ok=True)
                    sampler_path.write_text(
                        json.dumps(sampler_weights, indent=2), encoding="utf-8"
                    )
                    print(
                        f"hourly adaptation: {analysis.get('completed_screens', 0)} "
                        "screens analyzed",
                        flush=True,
                    )
                    next_analysis = now + timedelta(minutes=args.analysis_minutes)

                if (
                    completed_since_checkpoint >= 10
                    or time.monotonic() - last_checkpoint >= 10.0
                ):
                    write_records(screening, screening_path)
                    write_records(finalists, finalists_path)
                    write_reports(run_dir, screening, finalists)
                    write_state(
                        run_dir,
                        {
                            "updated": now.isoformat(),
                            "status": "running",
                            "screened": len(screening),
                            "finalists": len(finalists),
                            "pending": len(pending),
                            "queued_finalists": len(queued_finalists),
                            "best_screen_by_strategy": best_screen_by_strategy,
                            "next_hourly_analysis": next_analysis.isoformat(),
                            "stop_requested": STOP_PATH.exists(),
                        },
                    )
                    completed_since_checkpoint = 0
                    last_checkpoint = time.monotonic()

                if should_stop or STOP_PATH.exists():
                    for future in pending:
                        future.cancel()
                    if all(future.done() for future in pending):
                        break
                if (
                    args.max_configs
                    and submitted_screens >= args.max_configs
                    and not queued_finalists
                    and not pending
                ):
                    break
    finally:
        write_records(screening, screening_path)
        write_records(finalists, finalists_path)
        write_reports(run_dir, screening, finalists)
        write_state(
            run_dir,
            {
                "updated": datetime.now().astimezone().isoformat(),
                "status": "finished",
                "screened": len(screening),
                "finalists": len(finalists),
                "pending": 0,
                "best_screen_by_strategy": best_screen_by_strategy,
            },
        )
        print("Stock search stopped cleanly", flush=True)


if __name__ == "__main__":
    main()
