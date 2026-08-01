from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StockConfig:
    symbols: tuple[str, ...]
    strategy_type: str = "cross_sectional"
    lookback_months: int = 12
    skip_months: int = 1
    number_of_assets: int = 20
    assets_per_sector: int = 2
    require_positive_momentum: bool = False
    use_fixed_slots: bool = True
    rebalance_interval_months: int = 1
    signal_scaling: str = "raw"
    weighting: str = "equal"
    volatility_window_months: int = 6
    transaction_cost_rate: float = 0.0005

    def validate(self) -> None:
        if not self.symbols:
            raise ValueError("symbols cannot be empty")
        if self.strategy_type not in {
            "cross_sectional",
            "sector_neutral",
            "time_series",
        }:
            raise ValueError("unsupported strategy_type")
        if self.lookback_months <= self.skip_months:
            raise ValueError("lookback_months must exceed skip_months")
        if not 0 < self.number_of_assets <= len(self.symbols):
            raise ValueError("number_of_assets is outside the universe")
        if self.assets_per_sector <= 0:
            raise ValueError("assets_per_sector must be positive")
        if self.rebalance_interval_months <= 0:
            raise ValueError("rebalance_interval_months must be positive")
        if self.signal_scaling not in {"raw", "volatility_adjusted"}:
            raise ValueError("unsupported signal_scaling")
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


def bootstrap_indices(
    number_of_simulations: int,
    sample_size: int,
    history_length: int,
    average_block_length: float,
    random_seed: int,
) -> np.ndarray:
    if number_of_simulations <= 0 or sample_size <= 0 or history_length <= 0:
        raise ValueError("simulation and history dimensions must be positive")
    if average_block_length < 1:
        raise ValueError("average_block_length must be at least 1")
    rng = np.random.default_rng(random_seed)
    random_locations = rng.integers(
        history_length, size=(number_of_simulations, sample_size)
    )
    restart = rng.random((number_of_simulations, sample_size - 1)) < (
        1.0 / average_block_length
    )
    indices = np.empty_like(random_locations)
    indices[:, 0] = random_locations[:, 0]
    for position in range(1, sample_size):
        indices[:, position] = np.where(
            restart[:, position - 1],
            random_locations[:, position],
            (indices[:, position - 1] + 1) % history_length,
        )
    return indices


def _top_mask(scores: np.ndarray, count: int) -> np.ndarray:
    rows, columns = scores.shape
    count = min(count, columns)
    candidates = np.where(np.isfinite(scores), scores, -np.inf)
    locations = np.argpartition(candidates, columns - count, axis=1)[
        :, columns - count :
    ]
    selected = np.zeros((rows, columns), dtype=bool)
    np.put_along_axis(selected, locations, True, axis=1)
    selected &= np.isfinite(scores)
    return selected


def _select_assets(
    scores: np.ndarray,
    raw_signal: np.ndarray,
    config: StockConfig,
    sector_locations: dict[str, np.ndarray],
) -> tuple[np.ndarray, int]:
    if config.strategy_type == "time_series":
        selected = np.isfinite(raw_signal) & (raw_signal > 0.0)
        return selected, len(config.symbols)

    eligible_scores = scores.copy()
    if config.require_positive_momentum:
        eligible_scores = np.where(raw_signal > 0.0, eligible_scores, np.nan)

    if config.strategy_type == "cross_sectional":
        return _top_mask(eligible_scores, config.number_of_assets), config.number_of_assets

    selected = np.zeros_like(eligible_scores, dtype=bool)
    for locations in sector_locations.values():
        local_mask = _top_mask(
            eligible_scores[:, locations],
            min(config.assets_per_sector, len(locations)),
        )
        selected[:, locations] = local_mask
    target_slots = sum(
        min(config.assets_per_sector, len(locations))
        for locations in sector_locations.values()
    )
    return selected, target_slots


def _weights_from_selection(
    selected: np.ndarray,
    volatility: np.ndarray,
    target_slots: int,
    config: StockConfig,
) -> np.ndarray:
    if config.weighting == "inverse_volatility":
        eligible = selected & np.isfinite(volatility) & (volatility > 0.0)
        base = np.divide(
            1.0,
            volatility,
            out=np.zeros_like(volatility),
            where=eligible,
        )
    else:
        base = selected.astype(float)
    counts = selected.sum(axis=1)
    totals = base.sum(axis=1)
    normalized = np.divide(
        base,
        totals[:, None],
        out=np.zeros_like(base),
        where=totals[:, None] > 0.0,
    )
    if config.use_fixed_slots:
        exposure = counts / target_slots
        normalized *= exposure[:, None]
    return normalized


def _calculate_path_metrics(
    net_returns: np.ndarray,
    turnover: np.ndarray,
    initial_capital: float,
) -> pd.DataFrame:
    periods = net_returns.shape[1]
    growth = np.prod(1.0 + net_returns, axis=1)
    total_return = growth - 1.0
    cagr = np.where(growth > 0.0, growth ** (12.0 / periods) - 1.0, np.nan)
    volatility = net_returns.std(axis=1, ddof=1) * np.sqrt(12.0)
    standard_deviation = net_returns.std(axis=1, ddof=1)
    sharpe = np.divide(
        net_returns.mean(axis=1) * np.sqrt(12.0),
        standard_deviation,
        out=np.full(len(net_returns), np.nan),
        where=standard_deviation > 0.0,
    )
    downside_count = (net_returns < 0.0).sum(axis=1)
    downside_deviation = np.sqrt(
        np.divide(
            np.where(net_returns < 0.0, net_returns**2, 0.0).sum(axis=1),
            downside_count,
            out=np.full(len(net_returns), np.nan),
            where=downside_count > 0,
        )
    )
    sortino = np.divide(
        net_returns.mean(axis=1) * np.sqrt(12.0),
        downside_deviation,
        out=np.full(len(net_returns), np.nan),
        where=downside_deviation > 0.0,
    )
    equity = np.cumprod(1.0 + net_returns, axis=1)
    drawdown = equity / np.maximum.accumulate(equity, axis=1) - 1.0
    maximum_drawdown = drawdown.min(axis=1)
    current_duration = np.zeros(len(net_returns), dtype=int)
    maximum_duration = np.zeros(len(net_returns), dtype=int)
    for position in range(periods):
        current_duration = np.where(
            drawdown[:, position] < 0.0,
            current_duration + 1,
            0,
        )
        maximum_duration = np.maximum(maximum_duration, current_duration)
    calmar = np.divide(
        cagr,
        np.abs(maximum_drawdown),
        out=np.full(len(net_returns), np.nan),
        where=maximum_drawdown < 0.0,
    )
    return pd.DataFrame(
        {
            "total_return": total_return,
            "cagr": cagr,
            "annualized_volatility": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "maximum_drawdown": maximum_drawdown,
            "maximum_drawdown_duration": maximum_duration.astype(float),
            "calmar_ratio": calmar,
            "positive_month_rate": (net_returns > 0.0).mean(axis=1),
            "average_monthly_turnover": turnover.mean(axis=1),
            "annualized_turnover": turnover.mean(axis=1) * 12.0,
            "final_portfolio_value": initial_capital * growth,
            "loss_observation": (total_return < 0.0).astype(float),
        }
    )


def simulate_paths(
    historical_returns: np.ndarray,
    indices: np.ndarray,
    config: StockConfig,
    sectors: tuple[str, ...],
    initial_capital: float = 10_000.0,
) -> pd.DataFrame:
    config.validate()
    path_returns = historical_returns[indices]
    prices = np.concatenate(
        [
            np.ones((len(indices), 1, historical_returns.shape[1])),
            np.cumprod(1.0 + path_returns, axis=1),
        ],
        axis=1,
    )
    needs_volatility = (
        config.signal_scaling == "volatility_adjusted"
        or config.weighting == "inverse_volatility"
    )
    start_position = config.lookback_months
    if needs_volatility:
        start_position = max(
            start_position,
            config.skip_months + config.volatility_window_months,
        )
    evaluation_periods = path_returns.shape[1] - start_position
    if evaluation_periods < 2:
        raise ValueError("configuration leaves too few evaluation periods")

    sector_locations = {
        sector: np.flatnonzero(np.asarray(sectors) == sector)
        for sector in sorted(set(sectors))
    }
    current_weights = np.zeros_like(historical_returns[indices[:, 0]], dtype=float)
    net_returns = np.zeros((len(indices), evaluation_periods), dtype=float)
    turnover = np.zeros_like(net_returns)

    for output_position, formation_position in enumerate(
        range(start_position, path_returns.shape[1])
    ):
        should_rebalance = (
            (formation_position - start_position)
            % config.rebalance_interval_months
            == 0
        )
        if should_rebalance:
            raw_signal = (
                prices[:, formation_position - config.skip_months, :]
                / prices[:, formation_position - config.lookback_months, :]
                - 1.0
            )
            volatility_end = formation_position - config.skip_months
            volatility_start = (
                volatility_end - config.volatility_window_months
            )
            if volatility_start >= 0:
                volatility = path_returns[
                    :, volatility_start:volatility_end, :
                ].std(axis=1, ddof=1)
            else:
                volatility = np.full_like(raw_signal, np.nan)
            if config.signal_scaling == "volatility_adjusted":
                scores = np.divide(
                    raw_signal,
                    volatility,
                    out=np.full_like(raw_signal, np.nan),
                    where=volatility > 0.0,
                )
            else:
                scores = raw_signal
            selected, target_slots = _select_assets(
                scores, raw_signal, config, sector_locations
            )
            new_weights = _weights_from_selection(
                selected, volatility, target_slots, config
            )
            turnover[:, output_position] = 0.5 * np.abs(
                new_weights - current_weights
            ).sum(axis=1)
            current_weights = new_weights
        gross_return = (
            current_weights * path_returns[:, formation_position, :]
        ).sum(axis=1)
        net_returns[:, output_position] = (
            gross_return
            - turnover[:, output_position] * config.transaction_cost_rate
        )

    metrics = _calculate_path_metrics(net_returns, turnover, initial_capital)
    metrics["evaluation_months"] = float(evaluation_periods)
    return metrics


def summarize_simulations(simulations: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for column in simulations.columns:
        values = simulations[column].dropna()
        if values.empty:
            continue
        result[f"{column}_q05"] = float(values.quantile(0.05))
        result[f"{column}_median"] = float(values.quantile(0.50))
        result[f"{column}_q95"] = float(values.quantile(0.95))
        result[f"{column}_mean"] = float(values.mean())
    return result


def evaluate_stock_config(
    full_prices: pd.DataFrame,
    sector_by_symbol: dict[str, str],
    config: StockConfig,
    number_of_simulations: int,
    random_seed: int = 2026,
    average_block_length: float = 6.0,
    stop_file: str | None = None,
    batch_size: int = 250,
) -> dict[str, object]:
    config.validate()
    missing = sorted(set(config.symbols).difference(full_prices.columns))
    if missing:
        raise ValueError(f"configuration symbols are missing: {missing}")
    selected_prices = full_prices.loc[:, list(config.symbols)]
    historical_returns = (
        selected_prices.pct_change(fill_method=None).dropna(how="any").to_numpy()
    )
    sectors = tuple(sector_by_symbol[symbol] for symbol in config.symbols)
    all_indices = bootstrap_indices(
        number_of_simulations=number_of_simulations,
        sample_size=len(historical_returns),
        history_length=len(historical_returns),
        average_block_length=average_block_length,
        random_seed=random_seed,
    )
    records: list[pd.DataFrame] = []
    stopped_early = False
    for offset in range(0, number_of_simulations, batch_size):
        if stop_file and Path(stop_file).exists():
            stopped_early = True
            break
        batch_indices = all_indices[offset : offset + batch_size]
        records.append(
            simulate_paths(
                historical_returns,
                batch_indices,
                config,
                sectors,
            )
        )
    simulations = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    valid = simulations["cagr"].notna().sum() if not simulations.empty else 0
    result: dict[str, object] = {
        "config_id": config.config_id,
        "strategy_type": config.strategy_type,
        "config_json": json.dumps(config.to_dict(), sort_keys=True),
        "requested_simulations": number_of_simulations,
        "simulation_count": len(simulations),
        "valid_simulations": int(valid),
        "invalid_simulations": int(len(simulations) - valid),
        "stopped_early": stopped_early,
    }
    if not simulations.empty:
        result.update(summarize_simulations(simulations))
    return result
