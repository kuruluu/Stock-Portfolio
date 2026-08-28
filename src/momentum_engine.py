from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MomentumConfig:
    config_id: str
    family_id: str
    strategy_type: str
    lookback_months: int
    skip_months: int
    number_of_assets: int
    require_positive_momentum: bool
    use_fixed_slots: bool
    rebalance_interval_months: int
    weighting: str
    volatility_window_months: int
    market_filter_months: int
    maximum_leverage: float
    transaction_cost_rate: float

    def validate(self) -> None:
        if self.strategy_type != "cross_sectional":
            raise ValueError("unsupported ETF strategy type")
        if self.weighting not in {"equal", "inverse_volatility"}:
            raise ValueError("unsupported ETF weighting")
        if self.lookback_months <= self.skip_months or self.skip_months < 1:
            raise ValueError("ETF lookback must exceed a positive skip")
        if self.number_of_assets < 1:
            raise ValueError("ETF breadth must be positive")
        if self.rebalance_interval_months < 1 or self.volatility_window_months < 2:
            raise ValueError("invalid ETF timing")
        if self.market_filter_months < 0:
            raise ValueError("ETF market filter cannot be negative")
        if not 0.0 < self.maximum_leverage <= 1.0:
            raise ValueError("maximum leverage must be in (0, 1]")
        if not 0.0 <= self.transaction_cost_rate < 1.0:
            raise ValueError("transaction cost rate must be in [0, 1)")


def validate_etf_inputs(
    prices: pd.DataFrame,
    eligibility: pd.DataFrame,
    benchmark: pd.Series,
) -> None:
    if prices.empty or eligibility.empty or benchmark.empty:
        raise ValueError("ETF inputs cannot be empty")
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("ETF dates must be unique and chronological")
    if prices.columns.has_duplicates:
        raise ValueError("ETF symbols must be unique")
    if not prices.index.equals(eligibility.index) or not prices.index.equals(
        benchmark.index
    ):
        raise ValueError("ETF input dates must match")
    if list(prices.columns) != list(eligibility.columns):
        raise ValueError("ETF eligibility symbols must match price symbols")
    if eligibility.isna().to_numpy().any():
        raise ValueError("ETF eligibility cannot contain missing values")
    if benchmark.isna().any():
        raise ValueError("ETF benchmark cannot contain missing values")
    if (eligibility.astype(bool) & prices.isna()).to_numpy().any():
        raise ValueError("eligible ETF periods cannot have missing prices")


def momentum_signal(
    prices: pd.DataFrame,
    formation: int,
    lookback_months: int,
    skip_months: int,
) -> pd.Series:
    signal_end = formation - skip_months
    signal_start = signal_end - lookback_months
    if signal_start < 0:
        raise ValueError("insufficient history for ETF momentum signal")
    return prices.iloc[signal_end].divide(prices.iloc[signal_start]) - 1.0


def _target_weights(
    signal: pd.Series,
    volatility: pd.Series,
    eligible: pd.Series,
    config: MomentumConfig,
) -> pd.Series:
    scores = signal.where(eligible & signal.notna()).dropna().sort_values(ascending=False)
    if config.require_positive_momentum:
        scores = scores[scores > 0.0]
    scores = scores.iloc[: config.number_of_assets]
    weights = pd.Series(0.0, index=signal.index)
    if scores.empty:
        return weights

    slot_count = config.number_of_assets if config.use_fixed_slots else len(scores)
    gross_budget = min(config.maximum_leverage, len(scores) / slot_count)
    if config.weighting == "inverse_volatility":
        inverse = volatility.reindex(scores.index).where(lambda row: row > 0.0).rdiv(1.0)
        inverse = inverse.replace([np.inf, -np.inf], np.nan).dropna()
        if not inverse.empty:
            weights.loc[inverse.index] = inverse / inverse.sum() * gross_budget
    else:
        weights.loc[scores.index] = gross_budget / len(scores)
    return weights


def backtest_momentum(
    prices: pd.DataFrame,
    eligibility: pd.DataFrame,
    benchmark: pd.Series,
    config: MomentumConfig,
) -> pd.DataFrame:
    config.validate()
    validate_etf_inputs(prices, eligibility, benchmark)
    if config.number_of_assets > len(prices.columns):
        raise ValueError("ETF breadth exceeds the available universe")

    returns = prices.pct_change(fill_method=None)
    start = max(
        config.lookback_months + config.skip_months,
        config.volatility_window_months + config.skip_months,
        config.market_filter_months + config.skip_months,
    )
    if len(prices) - start < 10:
        raise ValueError("ETF configuration leaves too few evaluation months")

    current_weights = pd.Series(0.0, index=prices.columns)
    rows: list[dict[str, object]] = []
    for formation in range(start, len(prices) - 1):
        eligible = eligibility.iloc[formation].astype(bool) & prices.iloc[formation].notna()
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
            target = (
                _target_weights(signal, trailing.std(ddof=1), eligible, config)
                if market_ok
                else pd.Series(0.0, index=prices.columns)
            )
            rebalance_turnover = 0.5 * float((target - investable).abs().sum())
            investable = target

        turnover = forced_turnover + rebalance_turnover
        next_returns = returns.iloc[formation + 1]
        held = investable.abs() > 1e-14
        missing = list(next_returns.index[held & next_returns.isna()])
        if missing:
            raise ValueError(
                f"unexplained ETF holding-period return gap at "
                f"{prices.index[formation + 1].date()}: {missing[:8]}"
            )
        gross_return = float((investable * next_returns.fillna(0.0)).sum())
        net_return = gross_return - turnover * config.transaction_cost_rate
        held_symbols = tuple(str(symbol) for symbol in prices.columns[held])
        rows.append(
            {
                "date": prices.index[formation + 1],
                "config_id": config.config_id,
                "gross_return": gross_return,
                "net_return": net_return,
                "turnover": turnover,
                "invested_fraction": float(investable.abs().sum()),
                "cash_fraction": 1.0 - float(investable.abs().sum()),
                "positions": len(held_symbols),
                "held_symbols": held_symbols,
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
