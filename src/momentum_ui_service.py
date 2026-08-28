"""Read-only service boundary for the standalone momentum workbench."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Mapping

import pandas as pd

from momentum_engine import MomentumConfig, backtest_momentum, validate_stock_inputs
from momentum_metrics import calculate_momentum_metrics


ROOT = Path(__file__).resolve().parents[1]
WORKBENCH_PATH = ROOT / "momentum_workbench.json"


class MomentumUIError(RuntimeError):
    """Base class for safe, user-facing workbench failures."""


class DataUnavailableError(MomentumUIError):
    """Raised when a required development input is unavailable."""


class DataIdentityError(MomentumUIError):
    """Raised when a development input no longer matches its manifest."""


class BacktestExecutionError(MomentumUIError):
    """Raised when the engine rejects an exploratory run."""


ALLOWED_LOOKBACKS = (6, 9, 12)
ALLOWED_HOLDINGS = (10, 20, 30)
ALLOWED_REBALANCE_INTERVALS = (1, 3)
ALLOWED_WEIGHTINGS = ("equal", "inverse_volatility")
ALLOWED_TRANSACTION_COSTS = (0.0, 0.001, 0.0025, 0.005)


def read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataIdentityError(f"The workbench manifest is invalid: {exc}") from exc
    if not isinstance(value, dict):
        raise DataIdentityError("The workbench manifest must be an object.")
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
        allowed = {
            "lookback_months": (self.lookback_months, ALLOWED_LOOKBACKS),
            "number_of_assets": (self.number_of_assets, ALLOWED_HOLDINGS),
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
            family_id="EXPLORATORY_UI_CROSS_SECTIONAL",
            strategy_type="cross_sectional",
            lookback_months=self.lookback_months,
            skip_months=1,
            universe_size=300,
            number_of_assets=self.number_of_assets,
            require_positive_momentum=self.require_positive_momentum,
            use_fixed_slots=True,
            rebalance_interval_months=self.rebalance_interval_months,
            weighting=self.weighting,
            volatility_window_months=6,
            market_filter_months=0,
            target_volatility=None,
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
    membership: pd.DataFrame
    liquidity: pd.DataFrame
    benchmark: pd.Series
    cost_stress_rate: float


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
    if manifest.get("schema_revision") != 1:
        raise DataIdentityError("The workbench manifest schema is unsupported.")
    if manifest.get("market_id") != "US_SP500_PIT":
        raise DataIdentityError("The configured market is not the approved US stock market.")
    if manifest.get("benchmark") != "SPY":
        raise DataIdentityError("The configured benchmark is not the approved SPY series.")
    if manifest.get("research_only") is not True or manifest.get("read_only") is not True:
        raise DataIdentityError("The workbench must remain research-only and read-only.")

    inputs = _mapping(manifest.get("inputs"), "input paths")
    expected_hashes = _mapping(manifest.get("sha256"), "input identities")
    expected_names = {"prices", "membership", "liquidity", "benchmark", "manifest"}
    if set(inputs) != expected_names or set(expected_hashes) != expected_names:
        raise DataIdentityError("The workbench input contract is incomplete.")
    sources = {
        name: _resolve_data_path(root, inputs[name], name) for name in sorted(expected_names)
    }
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise DataUnavailableError(
            f"Required US development data is missing: {', '.join(sorted(missing))}."
        )
    current_hashes = {name: file_sha256(path) for name, path in sources.items()}
    changed = sorted(
        name for name, digest in current_hashes.items() if expected_hashes.get(name) != digest
    )
    if changed:
        raise DataIdentityError(f"The audited data identity changed for: {', '.join(changed)}.")

    prices = pd.read_parquet(sources["prices"]).sort_index()
    membership = pd.read_parquet(sources["membership"]).sort_index()
    liquidity = pd.read_parquet(sources["liquidity"]).sort_index()
    benchmark_frame = pd.read_parquet(sources["benchmark"]).sort_index()
    if isinstance(benchmark_frame, pd.Series):
        benchmark = benchmark_frame
    elif benchmark_frame.shape[1] == 1:
        benchmark = benchmark_frame.iloc[:, 0]
    else:
        raise DataIdentityError("The audited benchmark must contain exactly one series.")
    benchmark = benchmark.rename("SPY").reindex(prices.index)
    try:
        validate_stock_inputs(prices, membership, liquidity, benchmark)
    except ValueError as exc:
        raise DataIdentityError(f"The audited data is not aligned: {exc}") from exc

    cost_stress_rate = float(manifest.get("cost_stress_rate", -1.0))
    if not 0.0 <= cost_stress_rate < 1.0:
        raise DataIdentityError("The recorded cost-stress rate is invalid.")
    return DevelopmentFrames(prices, membership, liquidity, benchmark, cost_stress_rate)


def run_exploratory_backtest(inputs: StrategyInputs) -> BacktestSummary:
    started = perf_counter()
    frames = load_development_frames()
    try:
        result = backtest_momentum(
            frames.prices,
            frames.membership,
            frames.liquidity,
            frames.benchmark,
            inputs.to_engine_config(),
        )
        metrics = calculate_momentum_metrics(result, frames.cost_stress_rate)
    except ValueError as exc:
        raise BacktestExecutionError(str(exc)) from exc
    values = metrics.to_dict()
    return BacktestSummary(
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
