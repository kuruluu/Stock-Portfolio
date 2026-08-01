from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import calculate_performance_metrics


@dataclass(frozen=True)
class TrialConfig:
    symbols: tuple[str, ...]
    lookback_months: int = 12
    skip_months: int = 1
    number_of_assets: int = 3
    require_positive_momentum: bool = False
    use_fixed_slots: bool = True
    rebalance_interval_months: int = 1
    rank_buffer: int = 0
    signal_scaling: str = "raw"
    volatility_window_months: int = 6
    transaction_cost_rate: float = 0.0005

    def validate(self) -> None:
        if not self.symbols:
            raise ValueError("symbols cannot be empty")
        if self.lookback_months <= self.skip_months:
            raise ValueError("lookback_months must exceed skip_months")
        if self.number_of_assets <= 0:
            raise ValueError("number_of_assets must be positive")
        if self.number_of_assets > len(self.symbols):
            raise ValueError("number_of_assets cannot exceed the universe size")
        if self.rebalance_interval_months <= 0:
            raise ValueError("rebalance_interval_months must be positive")
        if self.rank_buffer < 0:
            raise ValueError("rank_buffer cannot be negative")
        if self.signal_scaling not in {"raw", "volatility_adjusted"}:
            raise ValueError("unsupported signal_scaling")
        if self.volatility_window_months < 2:
            raise ValueError("volatility_window_months must be at least 2")
        if not 0 <= self.transaction_cost_rate < 1:
            raise ValueError("transaction_cost_rate must be between 0 and 1")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["symbols"] = list(self.symbols)
        return value

    @property
    def config_id(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True)
class TimeSeriesConfig:
    symbols: tuple[str, ...]
    lookback_months: int = 12
    skip_months: int = 1
    rebalance_interval_months: int = 1
    cash_policy: str = "fixed_budget"
    weighting: str = "equal"
    volatility_window_months: int = 6
    transaction_cost_rate: float = 0.0005
    strategy_type: str = field(default="time_series", init=False)

    def validate(self) -> None:
        if not self.symbols:
            raise ValueError("symbols cannot be empty")
        if self.lookback_months <= self.skip_months:
            raise ValueError("lookback_months must exceed skip_months")
        if self.rebalance_interval_months <= 0:
            raise ValueError("rebalance_interval_months must be positive")
        if self.cash_policy not in {"fixed_budget", "fully_invested"}:
            raise ValueError("unsupported cash_policy")
        if self.weighting not in {"equal", "inverse_volatility"}:
            raise ValueError("unsupported weighting")
        if self.volatility_window_months < 2:
            raise ValueError("volatility_window_months must be at least 2")
        if not 0 <= self.transaction_cost_rate < 1:
            raise ValueError("transaction_cost_rate must be between 0 and 1")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["symbols"] = list(self.symbols)
        return value

    @property
    def config_id(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


StrategyConfig = TrialConfig | TimeSeriesConfig


def calculate_signal(prices: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    config.validate()
    signal = (
        prices.shift(config.skip_months)
        / prices.shift(config.lookback_months)
        - 1.0
    )

    if getattr(config, "signal_scaling", "raw") == "volatility_adjusted":
        returns = prices.pct_change(fill_method=None)
        volatility = (
            returns.rolling(config.volatility_window_months)
            .std(ddof=1)
            .shift(config.skip_months)
        )
        signal = signal / volatility.replace(0.0, np.nan)

    return signal


def generate_time_series_target_weights(
    prices: pd.DataFrame,
    config: TimeSeriesConfig,
) -> pd.DataFrame:
    signal = calculate_signal(prices, config)
    active = signal.gt(0.0).astype(float)
    valid_signal = signal.notna().any(axis=1)
    active = active.where(valid_signal, np.nan)

    if config.weighting == "inverse_volatility":
        volatility = (
            prices.pct_change(fill_method=None)
            .rolling(config.volatility_window_months)
            .std(ddof=1)
            .shift(config.skip_months)
        )
        raw_weights = active / volatility.replace(0.0, np.nan)
    else:
        raw_weights = active

    row_total = raw_weights.sum(axis=1, min_count=1).replace(0.0, np.nan)
    normalized = raw_weights.div(row_total, axis=0).fillna(0.0)
    normalized = normalized.where(valid_signal, np.nan)

    if config.cash_policy == "fixed_budget":
        exposure = active.sum(axis=1, min_count=1) / len(config.symbols)
        desired = normalized.mul(exposure, axis=0)
    else:
        desired = normalized

    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    valid_positions = np.flatnonzero(valid_signal.to_numpy())
    if len(valid_positions) == 0:
        return weights
    first_position = int(valid_positions[0])
    rebalance_positions = np.arange(
        first_position,
        len(prices),
        config.rebalance_interval_months,
    )
    weights.iloc[rebalance_positions] = desired.iloc[rebalance_positions]
    return weights.ffill()


def generate_target_weights(
    prices: pd.DataFrame,
    config: StrategyConfig,
) -> pd.DataFrame:
    if isinstance(config, TimeSeriesConfig):
        return generate_time_series_target_weights(prices, config)
    signal = calculate_signal(prices, config)
    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    current_holdings: list[str] = []
    first_signal_position: int | None = None

    for position, (_, row) in enumerate(signal.iterrows()):
        available = row.dropna().sort_values(ascending=False)
        if available.empty:
            continue

        if first_signal_position is None:
            first_signal_position = position

        should_rebalance = (
            (position - first_signal_position)
            % config.rebalance_interval_months
            == 0
        )

        eligible = available
        if config.require_positive_momentum:
            eligible = eligible[eligible > 0]

        if should_rebalance:
            rank_limit = config.number_of_assets + config.rank_buffer
            retainable = set(eligible.iloc[:rank_limit].index)
            selected = [
                symbol
                for symbol in current_holdings
                if symbol in retainable
            ]
            for symbol in eligible.index:
                if symbol not in selected:
                    selected.append(symbol)
                if len(selected) == config.number_of_assets:
                    break
            current_holdings = selected[: config.number_of_assets]

        row_weights = pd.Series(0.0, index=prices.columns, dtype=float)
        if current_holdings:
            denominator = (
                config.number_of_assets
                if config.use_fixed_slots
                else len(current_holdings)
            )
            row_weights.loc[current_holdings] = 1.0 / denominator
        weights.iloc[position] = row_weights

    return weights


def calculate_turnover(applied_weights: pd.DataFrame) -> pd.Series:
    changes = applied_weights.diff().abs()
    turnover = 0.5 * changes.sum(axis=1)
    if not applied_weights.empty:
        turnover.iloc[0] = 0.5 * applied_weights.iloc[0].abs().sum()
    return turnover


def run_trial_backtest(
    prices: pd.DataFrame,
    config: StrategyConfig,
    initial_capital: float = 10_000.0,
) -> pd.DataFrame:
    config.validate()
    selected_prices = prices.loc[:, list(config.symbols)]
    monthly_returns = selected_prices.pct_change(fill_method=None)
    target_weights = generate_target_weights(selected_prices, config)
    applied_weights = target_weights.shift(1)
    valid = applied_weights.notna().any(axis=1)

    applied_weights = applied_weights.loc[valid]
    monthly_returns = monthly_returns.loc[valid]
    gross_returns = (applied_weights * monthly_returns).sum(
        axis=1, min_count=1
    )
    turnover = calculate_turnover(applied_weights).fillna(0.0)
    transaction_cost = turnover * config.transaction_cost_rate
    net_returns = gross_returns - transaction_cost
    portfolio_value = initial_capital * (1.0 + net_returns).cumprod()

    return pd.DataFrame(
        {
            "gross_return": gross_returns,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "net_return": net_returns,
            "net_portfolio_value": portfolio_value,
        }
    )


def stationary_bootstrap_indices(
    sample_size: int,
    history_length: int,
    average_block_length: float,
    rng: np.random.Generator,
) -> np.ndarray:
    restart_probability = 1.0 / average_block_length
    indices = np.empty(sample_size, dtype=int)
    indices[0] = rng.integers(history_length)
    for position in range(1, sample_size):
        if rng.random() < restart_probability:
            indices[position] = rng.integers(history_length)
        else:
            indices[position] = (indices[position - 1] + 1) % history_length
    return indices


def returns_to_prices(returns: pd.DataFrame) -> pd.DataFrame:
    prices = 100.0 * (1.0 + returns).cumprod()
    first_date = prices.index[0] - pd.offsets.MonthEnd(1)
    initial_row = pd.DataFrame(
        100.0, index=[first_date], columns=prices.columns
    )
    return pd.concat([initial_row, prices])


def summarize_simulations(simulations: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    quantiles = {"q05": 0.05, "median": 0.50, "q95": 0.95}
    for column in simulations.columns:
        if pd.api.types.is_numeric_dtype(simulations[column]):
            values = simulations[column].dropna()
            if values.empty:
                continue
            for label, quantile in quantiles.items():
                result[f"{column}_{label}"] = float(values.quantile(quantile))
            result[f"{column}_mean"] = float(values.mean())
    return result


def evaluate_config(
    full_prices: pd.DataFrame,
    config: StrategyConfig,
    number_of_simulations: int,
    random_seed: int = 2026,
    average_block_length: float = 6.0,
    stop_file: str | None = None,
) -> dict[str, object]:
    config.validate()
    full_returns = full_prices.pct_change(fill_method=None).dropna(how="any")
    rng = np.random.default_rng(random_seed)
    records: list[dict[str, float]] = []
    invalid_paths = 0

    stopped_early = False
    for simulation_number in range(number_of_simulations):
        if (
            stop_file
            and simulation_number % 10 == 0
            and Path(stop_file).exists()
        ):
            stopped_early = True
            break
        indices = stationary_bootstrap_indices(
            sample_size=len(full_returns),
            history_length=len(full_returns),
            average_block_length=average_block_length,
            rng=rng,
        )
        simulated_returns = full_returns.iloc[indices].copy()
        simulated_returns.index = full_returns.index
        simulated_prices = returns_to_prices(simulated_returns)

        try:
            backtest = run_trial_backtest(simulated_prices, config)
            spy_returns = simulated_prices["SPY"].pct_change(
                fill_method=None
            ).reindex(backtest.index)
            metrics = calculate_performance_metrics(
                returns=backtest["net_return"],
                turnover=backtest["turnover"],
                benchmark_returns=spy_returns,
            )
            spy_metrics = calculate_performance_metrics(returns=spy_returns)
            for name, value in spy_metrics.items():
                metrics[f"spy_{name}"] = value
            metrics["final_portfolio_value"] = float(
                backtest["net_portfolio_value"].iloc[-1]
            )
            metrics["loss_observation"] = float(metrics["total_return"] < 0)
            records.append(metrics)
        except (ValueError, IndexError, ZeroDivisionError, FloatingPointError):
            invalid_paths += 1

    simulations = pd.DataFrame(records)
    result: dict[str, object] = {
        "config_id": config.config_id,
        "strategy_type": getattr(
            config, "strategy_type", "cross_sectional"
        ),
        "config_json": json.dumps(config.to_dict(), sort_keys=True),
        "requested_simulations": number_of_simulations,
        "simulation_count": len(records) + invalid_paths,
        "valid_simulations": len(simulations),
        "invalid_simulations": invalid_paths,
        "stopped_early": stopped_early,
    }
    result.update(summarize_simulations(simulations))
    return result
