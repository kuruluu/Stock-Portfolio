from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MomentumConfig:
    config_id: str
    family_id: str
    strategy_type: str
    lookback_months: int
    skip_months: int
    universe_size: int
    number_of_assets: int
    require_positive_momentum: bool
    use_fixed_slots: bool
    rebalance_interval_months: int
    weighting: str
    volatility_window_months: int
    market_filter_months: int
    target_volatility: float | None
    maximum_leverage: float
    transaction_cost_rate: float

    def validate(self) -> None:
        if self.strategy_type not in {"cross_sectional", "time_series"}:
            raise ValueError("unsupported stock strategy type")
        if self.weighting not in {"equal", "inverse_volatility"}:
            raise ValueError("unsupported stock weighting")
        if self.lookback_months <= self.skip_months or self.skip_months < 1:
            raise ValueError("lookback must exceed a positive skip")
        if self.universe_size < 1 or not 1 <= self.number_of_assets <= self.universe_size:
            raise ValueError("invalid stock breadth")
        if self.strategy_type == "time_series" and self.number_of_assets != self.universe_size:
            raise ValueError("time-series stock configurations must use every universe slot")
        if self.rebalance_interval_months < 1 or self.volatility_window_months < 2:
            raise ValueError("invalid stock timing")
        if self.market_filter_months < 0:
            raise ValueError("market filter cannot be negative")
        if self.target_volatility is not None and self.target_volatility <= 0.0:
            raise ValueError("target volatility must be positive")
        if not 0.0 < self.maximum_leverage <= 1.0:
            raise ValueError("maximum leverage must be in (0, 1]")
        if not 0.0 <= self.transaction_cost_rate < 1.0:
            raise ValueError("transaction cost rate must be in [0, 1)")


def validate_stock_inputs(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
    liquidity: pd.DataFrame,
    benchmark: pd.Series,
) -> None:
    if prices.empty or membership.empty or liquidity.empty or benchmark.empty:
        raise ValueError("stock inputs cannot be empty")
    if not prices.index.equals(membership.index) or not prices.index.equals(liquidity.index):
        raise ValueError("stock input dates must match")
    if not prices.index.equals(benchmark.index):
        raise ValueError("SPY benchmark dates must match stock inputs")
    if list(prices.columns) != list(membership.columns) or list(prices.columns) != list(
        liquidity.columns
    ):
        raise ValueError("stock input symbols must match")
    if membership.isna().to_numpy().any():
        raise ValueError("stock membership cannot contain missing values")
    if benchmark.isna().any():
        raise ValueError("SPY benchmark cannot contain missing values")


def momentum_signal(
    prices: pd.DataFrame,
    formation: int,
    lookback_months: int,
    skip_months: int,
) -> pd.Series:
    signal_end = formation - skip_months
    signal_start = signal_end - lookback_months
    if signal_start < 0:
        raise ValueError("insufficient history for momentum signal")
    return prices.iloc[signal_end].divide(prices.iloc[signal_start]) - 1.0


def _base_weights(
    signal: pd.Series,
    volatility: pd.Series,
    eligible: pd.Series,
    liquidity: pd.Series,
    config: MomentumConfig,
) -> pd.Series:
    candidates = eligible & signal.notna() & liquidity.notna()
    liquid = liquidity.where(candidates).dropna().nlargest(config.universe_size)
    scores = signal.reindex(liquid.index).dropna().sort_values(ascending=False)
    if config.require_positive_momentum:
        scores = scores[scores > 0.0]
    if config.strategy_type == "cross_sectional":
        scores = scores.iloc[: config.number_of_assets]
    if scores.empty:
        return pd.Series(0.0, index=signal.index)

    selected = scores.index
    slot_count = config.number_of_assets if config.use_fixed_slots else len(selected)
    gross_budget = min(1.0, len(selected) / slot_count)
    weights = pd.Series(0.0, index=signal.index)
    if config.weighting == "inverse_volatility":
        inverse = volatility.reindex(selected).where(lambda row: row > 0.0).rdiv(1.0)
        inverse = inverse.replace([np.inf, -np.inf], np.nan).dropna()
        if not inverse.empty:
            weights.loc[inverse.index] = inverse / inverse.sum() * gross_budget
    else:
        weights.loc[selected] = 1.0 / slot_count
    return weights


def _apply_volatility_target(
    weights: pd.Series,
    trailing_returns: pd.DataFrame,
    target: float | None,
    maximum_leverage: float,
) -> pd.Series:
    gross = float(weights.abs().sum())
    if target is None or gross <= 0.0:
        return weights
    selected = weights.index[weights.abs() > 1e-14]
    covariance = trailing_returns[selected].cov(min_periods=2).fillna(0.0)
    vector = weights.reindex(selected).to_numpy(dtype=float)
    annual_variance = float(vector @ covariance.to_numpy(dtype=float) @ vector) * 12.0
    if not math.isfinite(annual_variance) or annual_variance <= 0.0:
        return pd.Series(0.0, index=weights.index)
    estimated = math.sqrt(annual_variance)
    scale = min(target / estimated, maximum_leverage / gross)
    return weights * max(0.0, scale)


def backtest_momentum(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
    liquidity: pd.DataFrame,
    benchmark: pd.Series,
    config: MomentumConfig,
) -> pd.DataFrame:
    config.validate()
    validate_stock_inputs(prices, membership, liquidity, benchmark)
    returns = prices.pct_change(fill_method=None)
    start = max(
        config.lookback_months + config.skip_months,
        config.volatility_window_months + config.skip_months,
        config.market_filter_months + config.skip_months,
    )
    if len(prices) - start < 10:
        raise ValueError("stock configuration leaves too few evaluation months")

    current_weights = pd.Series(0.0, index=prices.columns)
    rows: list[dict[str, object]] = []
    for formation in range(start, len(prices) - 1):
        eligible = membership.iloc[formation].astype(bool) & prices.iloc[formation].notna()
        investable = current_weights.where(eligible, 0.0)
        forced_turnover = 0.5 * float((investable - current_weights).abs().sum())
        scheduled = (formation - start) % config.rebalance_interval_months == 0
        signal_end = formation - config.skip_months
        market_ok = True
        if config.market_filter_months:
            filter_start = signal_end - config.market_filter_months
            market_ok = bool(benchmark.iloc[signal_end] > benchmark.iloc[filter_start])

        rebalance_turnover = 0.0
        if scheduled:
            signal = momentum_signal(
                prices,
                formation,
                config.lookback_months,
                config.skip_months,
            )
            trailing = returns.iloc[
                signal_end - config.volatility_window_months + 1 : signal_end + 1
            ]
            volatility = trailing.std(ddof=1)
            target = (
                _base_weights(
                    signal,
                    volatility,
                    eligible,
                    liquidity.iloc[formation],
                    config,
                )
                if market_ok
                else pd.Series(0.0, index=prices.columns)
            )
            target = _apply_volatility_target(
                target,
                trailing,
                config.target_volatility,
                config.maximum_leverage,
            )
            rebalance_turnover = 0.5 * float((target - investable).abs().sum())
            investable = target

        turnover = forced_turnover + rebalance_turnover
        next_returns = returns.iloc[formation + 1]
        held = investable.abs() > 1e-14
        missing = list(next_returns.index[held & next_returns.isna()])
        if missing:
            raise ValueError(
                f"unexplained holding-period return gap at "
                f"{prices.index[formation + 1].date()}: {missing[:8]}"
            )
        gross_return = float((investable * next_returns.fillna(0.0)).sum())
        net_return = gross_return - turnover * config.transaction_cost_rate
        rows.append(
            {
                "date": prices.index[formation + 1],
                "config_id": config.config_id,
                "gross_return": gross_return,
                "net_return": net_return,
                "turnover": turnover,
                "invested_fraction": float(investable.abs().sum()),
                "cash_fraction": 1.0 - float(investable.abs().sum()),
                "positions": int(held.sum()),
                "market_filter_passed": market_ok,
            }
        )
        end_values = investable * (1.0 + next_returns.fillna(0.0))
        gross_growth = 1.0 + gross_return
        current_weights = (
            end_values / gross_growth
            if gross_growth > 0.0
            else pd.Series(0.0, index=prices.columns)
        )

    result = pd.DataFrame(rows).set_index("date")
    result.index = pd.DatetimeIndex(result.index, name="date")
    return result
