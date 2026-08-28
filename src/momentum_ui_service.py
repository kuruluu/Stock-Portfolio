"""Read-only service boundary for the standalone ETF momentum workbench."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np
import pandas as pd

from momentum_engine import MomentumConfig, backtest_momentum, validate_etf_inputs
from momentum_metrics import calculate_momentum_metrics


ROOT = Path(__file__).resolve().parents[1]
WORKBENCH_PATH = ROOT / "momentum_workbench.json"
ETF_DISCLOSURE = (
    "Exploratory ETF data: point-in-time lifecycle history is unavailable, so all "
    "36 complete-history symbols are treated as eligible in every month."
)


class MomentumUIError(RuntimeError):
    """Base class for safe, user-facing workbench failures."""


class DataUnavailableError(MomentumUIError):
    """Raised when a required development input is unavailable."""


class DataIdentityError(MomentumUIError):
    """Raised when a development input no longer matches its manifest."""


class BacktestExecutionError(MomentumUIError):
    """Raised when the engine rejects an exploratory run."""


ALLOWED_LOOKBACKS = (6, 9, 12)
MIN_HOLDINGS = 1
MAX_HOLDINGS = 36
ALLOWED_REBALANCE_INTERVALS = (1, 3)
ALLOWED_WEIGHTINGS = ("equal", "inverse_volatility")
ALLOWED_TRANSACTION_COSTS = (0.0, 0.001, 0.0025, 0.005)
CHART_VIEWS = ("1Y", "3Y", "Max")


def read_json(path: Path, label: str = "workbench manifest") -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataIdentityError(f"The {label} is invalid: {exc}") from exc
    if not isinstance(value, dict):
        raise DataIdentityError(f"The {label} must be an object.")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class StrategyInputs:
    lookback_months: int = 12
    number_of_assets: int = 20
    rebalance_interval_months: int = 1
    weighting: str = "equal"
    require_positive_momentum: bool = True
    transaction_cost_rate: float = 0.001

    def __post_init__(self) -> None:
        if type(self.number_of_assets) is not int or not (
            MIN_HOLDINGS <= self.number_of_assets <= MAX_HOLDINGS
        ):
            raise ValueError(
                f"number_of_assets must be an integer from {MIN_HOLDINGS} to {MAX_HOLDINGS}"
            )
        allowed = {
            "lookback_months": (self.lookback_months, ALLOWED_LOOKBACKS),
            "rebalance_interval_months": (
                self.rebalance_interval_months,
                ALLOWED_REBALANCE_INTERVALS,
            ),
            "weighting": (self.weighting, ALLOWED_WEIGHTINGS),
            "transaction_cost_rate": (
                self.transaction_cost_rate,
                ALLOWED_TRANSACTION_COSTS,
            ),
        }
        invalid = [name for name, (value, choices) in allowed.items() if value not in choices]
        if invalid:
            raise ValueError(f"Unsupported strategy setting: {', '.join(invalid)}")
        if type(self.require_positive_momentum) is not bool:
            raise ValueError("require_positive_momentum must be true or false")

    def to_engine_config(self) -> MomentumConfig:
        identity = canonical_json_hash(asdict(self))[:12]
        return MomentumConfig(
            config_id=f"exploratory-ui-{identity}",
            family_id="EXPLORATORY_ETF_CROSS_SECTIONAL",
            strategy_type="cross_sectional",
            lookback_months=self.lookback_months,
            skip_months=1,
            number_of_assets=self.number_of_assets,
            require_positive_momentum=self.require_positive_momentum,
            use_fixed_slots=True,
            rebalance_interval_months=self.rebalance_interval_months,
            weighting=self.weighting,
            volatility_window_months=6,
            market_filter_months=0,
            maximum_leverage=1.0,
            transaction_cost_rate=self.transaction_cost_rate,
        )

    def display_items(self) -> tuple[tuple[str, str], ...]:
        return (
            ("Lookback", f"{self.lookback_months} months"),
            ("Holdings", str(self.number_of_assets)),
            ("Rebalance", "Monthly" if self.rebalance_interval_months == 1 else "Quarterly"),
            ("Weighting", "Equal" if self.weighting == "equal" else "Inverse volatility"),
            ("Positive momentum", "Required" if self.require_positive_momentum else "Not required"),
            ("Transaction cost", f"{self.transaction_cost_rate * 10_000:g} bps"),
        )


@dataclass(frozen=True)
class DevelopmentFrames:
    prices: pd.DataFrame
    eligibility: pd.DataFrame
    benchmark: pd.Series
    cost_stress_rate: float


@dataclass(frozen=True)
class ChartData:
    background: pd.DataFrame
    segments: pd.DataFrame
    quarter_guides: pd.DataFrame


@dataclass(frozen=True)
class BacktestSummary:
    net_cagr: float | None
    sharpe_ratio: float | None
    maximum_drawdown: float | None
    annualized_volatility: float | None
    annualized_turnover: float | None
    cost_stress_cagr: float | None
    observations: int
    period_start: str
    period_end: str
    elapsed_seconds: float
    configuration: StrategyInputs

    def metric_items(self) -> tuple[tuple[str, float | None, str], ...]:
        return (
            ("Net CAGR", self.net_cagr, "percent"),
            ("Sharpe ratio", self.sharpe_ratio, "ratio"),
            ("Maximum drawdown", self.maximum_drawdown, "percent"),
            ("Annualized volatility", self.annualized_volatility, "percent"),
            ("Annualized turnover", self.annualized_turnover, "percent"),
            ("Cost-stressed CAGR", self.cost_stress_cagr, "percent"),
        )


@dataclass(frozen=True)
class WorkbenchResult:
    summary: BacktestSummary
    chart: ChartData
    disclosure: str


def format_metric(value: float | None, kind: str) -> str:
    if value is None or not math.isfinite(value):
        return "Not available"
    if kind == "percent":
        return f"{value:.1%}"
    if kind == "ratio":
        return f"{value:.2f}"
    raise ValueError(f"Unsupported metric kind: {kind}")


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DataIdentityError(f"The recorded {label} is invalid.")
    return dict(value)


def _resolve_data_path(root: Path, value: object, label: str) -> Path:
    relative = Path(str(value))
    resolved = (relative if relative.is_absolute() else root / relative).resolve()
    data_root = (root / "data").resolve()
    if not resolved.is_relative_to(data_root):
        raise DataIdentityError(f"The {label} path is outside the approved data area.")
    return resolved


def load_development_frames(
    root: Path = ROOT,
    workbench_path: Path = WORKBENCH_PATH,
) -> DevelopmentFrames:
    root = root.resolve()
    if not workbench_path.is_file():
        raise DataUnavailableError("The momentum workbench manifest is unavailable.")
    manifest = read_json(workbench_path)
    if manifest.get("schema_revision") != 2:
        raise DataIdentityError("The workbench manifest schema is unsupported.")
    if manifest.get("market_id") != "US_ETF_EXPLORATORY":
        raise DataIdentityError("The configured market is not the exploratory US ETF market.")
    if manifest.get("benchmark") != "SPY":
        raise DataIdentityError("The configured benchmark is not the approved SPY series.")
    if manifest.get("eligibility_policy") != "all_complete_symbols":
        raise DataIdentityError("The ETF eligibility policy is unsupported.")
    if manifest.get("research_only") is not True or manifest.get("read_only") is not True:
        raise DataIdentityError("The workbench must remain research-only and read-only.")

    inputs = _mapping(manifest.get("inputs"), "input paths")
    expected_hashes = _mapping(manifest.get("sha256"), "input identities")
    expected_names = {"prices", "manifest"}
    if set(inputs) != expected_names or set(expected_hashes) != expected_names:
        raise DataIdentityError("The workbench input contract is incomplete.")
    sources = {
        name: _resolve_data_path(root, inputs[name], name) for name in sorted(expected_names)
    }
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise DataUnavailableError(
            f"Required ETF development data is missing: {', '.join(sorted(missing))}."
        )
    current_hashes = {name: file_sha256(path) for name, path in sources.items()}
    changed = sorted(
        name for name, digest in current_hashes.items() if expected_hashes.get(name) != digest
    )
    if changed:
        raise DataIdentityError(f"The audited data identity changed for: {', '.join(changed)}.")

    prices = pd.read_parquet(sources["prices"]).sort_index()
    data_manifest = read_json(sources["manifest"], "ETF universe manifest")
    accepted = data_manifest.get("accepted")
    if not isinstance(accepted, list) or [str(value) for value in accepted] != list(
        prices.columns
    ):
        raise DataIdentityError("The ETF manifest symbols do not match the price columns.")
    if int(data_manifest.get("rows", -1)) != len(prices):
        raise DataIdentityError("The ETF manifest row count does not match the prices.")
    if str(data_manifest.get("start")) != str(prices.index.min().date()) or str(
        data_manifest.get("end")
    ) != str(prices.index.max().date()):
        raise DataIdentityError("The ETF manifest date range does not match the prices.")
    if not np.isfinite(prices.astype(float).to_numpy()).all():
        raise DataIdentityError("The ETF prices contain non-finite values.")
    if (prices.astype(float) <= 0.0).to_numpy().any():
        raise DataIdentityError("The ETF prices must be positive.")

    benchmark_symbol = str(manifest["benchmark"])
    if benchmark_symbol not in prices.columns:
        raise DataIdentityError("The ETF prices do not contain the benchmark.")
    eligibility = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    benchmark = prices[benchmark_symbol].rename(benchmark_symbol)
    try:
        validate_etf_inputs(prices, eligibility, benchmark)
    except ValueError as exc:
        raise DataIdentityError(f"The ETF data is not aligned: {exc}") from exc

    cost_stress_rate = float(manifest.get("cost_stress_rate", -1.0))
    if not 0.0 <= cost_stress_rate < 1.0:
        raise DataIdentityError("The recorded cost-stress rate is invalid.")
    return DevelopmentFrames(prices, eligibility, benchmark, cost_stress_rate)


def normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        raise DataIdentityError("ETF chart prices cannot be empty.")
    numeric = prices.astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise DataIdentityError("ETF chart prices contain non-finite values.")
    if (numeric <= 0.0).to_numpy().any():
        raise DataIdentityError("ETF chart prices must be positive.")
    return numeric.divide(numeric.iloc[0], axis="columns") * 100.0


def build_chart_data(prices: pd.DataFrame, result: pd.DataFrame) -> ChartData:
    if result.empty or "held_symbols" not in result.columns:
        raise DataIdentityError("The ETF result does not contain holding history.")
    if not result.index.isin(prices.index).all():
        raise DataIdentityError("The ETF result dates do not match the chart prices.")
    first_position = prices.index.get_loc(result.index.min())
    if not isinstance(first_position, int) or first_position < 1:
        raise DataIdentityError("The ETF chart cannot locate the first holding interval.")
    last_position = prices.index.get_loc(result.index.max())
    if not isinstance(last_position, int):
        raise DataIdentityError("The ETF chart cannot locate the final holding interval.")

    window = prices.iloc[first_position - 1 : last_position + 1].copy()
    normalized = normalize_prices(window)
    background = window.rename_axis("date").reset_index().melt(
        id_vars="date", var_name="symbol", value_name="adjusted_close"
    )
    normalized_long = normalized.rename_axis("date").reset_index().melt(
        id_vars="date", var_name="symbol", value_name="normalized_value"
    )
    background = background.merge(normalized_long, on=["date", "symbol"], validate="one_to_one")
    held_pairs = {
        (pd.Timestamp(date), str(symbol))
        for date, symbols in result["held_symbols"].items()
        for symbol in symbols
    }
    background["held"] = [
        "Held" if (pd.Timestamp(date), str(symbol)) in held_pairs else "Not held"
        for date, symbol in zip(background["date"], background["symbol"], strict=True)
    ]

    segment_rows: list[dict[str, object]] = []
    for date, symbols in result["held_symbols"].items():
        end = pd.Timestamp(date)
        end_position = window.index.get_loc(end)
        if not isinstance(end_position, int) or end_position < 1:
            raise DataIdentityError("The ETF holding interval is not chronological.")
        start = pd.Timestamp(window.index[end_position - 1])
        for raw_symbol in symbols:
            symbol = str(raw_symbol)
            if symbol not in window.columns:
                raise DataIdentityError(f"The held ETF is absent from chart prices: {symbol}.")
            segment_id = f"{symbol}|{end.date().isoformat()}"
            for point in (start, end):
                segment_rows.append(
                    {
                        "segment_id": segment_id,
                        "date": point,
                        "symbol": symbol,
                        "adjusted_close": float(window.at[point, symbol]),
                        "normalized_value": float(normalized.at[point, symbol]),
                        "held": "Held",
                    }
                )
    segments = pd.DataFrame(
        segment_rows,
        columns=(
            "segment_id",
            "date",
            "symbol",
            "adjusted_close",
            "normalized_value",
            "held",
        ),
    )
    quarter_guides = pd.DataFrame(
        {
            "date": sorted(
                pd.Timestamp(value)
                for value in background["date"].unique()
                if pd.Timestamp(value).month in (3, 6, 9, 12)
            )
        }
    )
    return ChartData(
        background=background,
        segments=segments,
        quarter_guides=quarter_guides,
    )


def select_chart_view(chart_data: ChartData, view: str) -> ChartData:
    if view not in CHART_VIEWS:
        raise ValueError(f"Unsupported chart view: {view}")
    if chart_data.background.empty:
        raise DataIdentityError("ETF chart data cannot be empty.")

    background = chart_data.background.copy()
    background["date"] = pd.to_datetime(background["date"])
    final_date = pd.Timestamp(background["date"].max())
    years = {"1Y": 1, "3Y": 3, "Max": None}[view]
    if years is not None:
        cutoff = final_date - pd.DateOffset(years=years)
        background = background[background["date"] >= cutoff].copy()

    background = background.sort_values(["symbol", "date"]).reset_index(drop=True)
    starts = background.groupby("symbol", sort=False)["adjusted_close"].transform("first")
    background["normalized_value"] = background["adjusted_close"].divide(starts) * 100.0

    first_visible = pd.Timestamp(background["date"].min())
    segments = chart_data.segments.copy()
    segments["date"] = pd.to_datetime(segments["date"])
    segments = segments[segments["date"] >= first_visible].copy()
    complete = segments.groupby("segment_id")["date"].transform("size") == 2
    segments = segments[complete].drop(columns=["adjusted_close", "normalized_value"])
    segments = segments.merge(
        background[["date", "symbol", "adjusted_close", "normalized_value"]],
        on=["date", "symbol"],
        how="inner",
        validate="many_to_one",
    ).sort_values(["segment_id", "date"])

    quarter_guides = pd.DataFrame(
        {
            "date": sorted(
                pd.Timestamp(value)
                for value in background["date"].unique()
                if pd.Timestamp(value).month in (3, 6, 9, 12)
            )
        }
    )
    return ChartData(
        background=background,
        segments=segments.reset_index(drop=True),
        quarter_guides=quarter_guides,
    )


def run_exploratory_backtest(inputs: StrategyInputs) -> WorkbenchResult:
    started = perf_counter()
    frames = load_development_frames()
    try:
        result = backtest_momentum(
            frames.prices,
            frames.eligibility,
            frames.benchmark,
            inputs.to_engine_config(),
        )
        metrics = calculate_momentum_metrics(result, frames.cost_stress_rate)
        chart = build_chart_data(frames.prices, result)
    except ValueError as exc:
        raise BacktestExecutionError(str(exc)) from exc
    values = metrics.to_dict()
    summary = BacktestSummary(
        net_cagr=values["net_cagr"],
        sharpe_ratio=values["sharpe_ratio"],
        maximum_drawdown=values["maximum_drawdown"],
        annualized_volatility=values["annualized_volatility"],
        annualized_turnover=values["annualized_turnover"],
        cost_stress_cagr=values["cost_stress_cagr"],
        observations=int(values["observations"]),
        period_start=result.index.min().date().isoformat(),
        period_end=result.index.max().date().isoformat(),
        elapsed_seconds=perf_counter() - started,
        configuration=inputs,
    )
    return WorkbenchResult(summary=summary, chart=chart, disclosure=ETF_DISCLOSURE)
