from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import hashlib
import json
import math
from statistics import NormalDist

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PitConfig:
    strategy_type: str = "cross_sectional"
    lookback_months: int = 12
    skip_months: int = 1
    number_of_assets: int = 10
    universe_size: int = 200
    require_positive_momentum: bool = True
    use_fixed_slots: bool = False
    rebalance_interval_months: int = 1
    signal_scaling: str = "raw"
    weighting: str = "equal"
    volatility_window_months: int = 6
    market_filter_months: int = 0
    transaction_cost_rate: float = 0.0005

    def validate(self) -> None:
        if self.strategy_type not in {"cross_sectional", "time_series"}:
            raise ValueError("unsupported strategy_type")
        if self.lookback_months <= self.skip_months:
            raise ValueError("lookback_months must exceed skip_months")
        if self.number_of_assets <= 0 or self.universe_size <= 0:
            raise ValueError("asset counts must be positive")
        if self.number_of_assets > self.universe_size:
            raise ValueError("number_of_assets cannot exceed universe_size")
        if self.rebalance_interval_months <= 0:
            raise ValueError("rebalance interval must be positive")
        if self.signal_scaling not in {"raw", "volatility_adjusted"}:
            raise ValueError("unsupported signal scaling")
        if self.weighting not in {"equal", "inverse_volatility"}:
            raise ValueError("unsupported weighting")
        if self.volatility_window_months < 2:
            raise ValueError("volatility window must be at least two months")
        if self.market_filter_months < 0:
            raise ValueError("market filter cannot be negative")
        if not 0 <= self.transaction_cost_rate < 1:
            raise ValueError("transaction cost rate must be in [0, 1)")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def config_id(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


def _validate_inputs(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
    liquidity: pd.DataFrame,
) -> None:
    if prices.empty or membership.empty or liquidity.empty:
        raise ValueError("point-in-time inputs cannot be empty")
    if not prices.index.equals(membership.index) or not prices.index.equals(
        liquidity.index
    ):
        raise ValueError("price, membership, and liquidity dates must match")
    if list(prices.columns) != list(membership.columns) or list(
        prices.columns
    ) != list(liquidity.columns):
        raise ValueError("point-in-time input symbols must match and be ordered")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("point-in-time dates must be unique and sorted")


def _target_weights(
    raw_signal: pd.Series,
    volatility: pd.Series,
    eligible: pd.Series,
    liquidity: pd.Series,
    config: PitConfig,
) -> pd.Series:
    candidates = eligible & raw_signal.notna() & liquidity.notna()
    liquid_names = (
        liquidity.where(candidates)
        .dropna()
        .nlargest(config.universe_size)
        .index
    )
    in_universe = pd.Series(False, index=raw_signal.index)
    in_universe.loc[liquid_names] = True
    if config.signal_scaling == "volatility_adjusted":
        score = raw_signal.divide(volatility.where(volatility > 0.0))
    else:
        score = raw_signal.copy()
    score = score.where(in_universe)

    if config.strategy_type == "time_series":
        selected = in_universe & raw_signal.gt(0.0)
        target_slots = max(1, len(liquid_names))
    else:
        if config.require_positive_momentum:
            score = score.where(raw_signal > 0.0)
        chosen = score.dropna().nlargest(config.number_of_assets).index
        selected = pd.Series(False, index=raw_signal.index)
        selected.loc[chosen] = True
        target_slots = config.number_of_assets

    weights = pd.Series(0.0, index=raw_signal.index)
    if config.weighting == "inverse_volatility":
        base = volatility.where(selected & volatility.gt(0.0)).rdiv(1.0)
    else:
        base = selected.astype(float)
    base = base.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if base.sum() > 0.0:
        weights = base / base.sum()
        if config.use_fixed_slots:
            weights *= float(selected.sum()) / float(target_slots)
    return weights


def backtest_point_in_time(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
    liquidity: pd.DataFrame,
    config: PitConfig,
    market_prices: pd.Series | None = None,
) -> pd.DataFrame:
    """Run one monthly close-to-close backtest with formation-time eligibility."""
    config.validate()
    _validate_inputs(prices, membership, liquidity)
    returns = prices.pct_change(fill_method=None)
    start = max(
        config.lookback_months,
        config.skip_months + config.volatility_window_months,
        config.market_filter_months + config.skip_months,
    )
    if len(prices) - start < 3:
        raise ValueError("configuration leaves too few evaluation months")

    current_weights = pd.Series(0.0, index=prices.columns)
    records: list[dict[str, object]] = []
    for formation in range(start, len(prices) - 1):
        member_now = membership.iloc[formation].fillna(False).astype(bool)
        price_now = prices.iloc[formation]
        forced_eligible = member_now & price_now.notna()
        new_weights = current_weights.where(forced_eligible, 0.0)
        forced_turnover = 0.5 * float((new_weights - current_weights).abs().sum())
        scheduled = (formation - start) % config.rebalance_interval_months == 0

        market_ok = True
        if config.market_filter_months:
            if market_prices is None:
                raise ValueError("market prices are required by the market filter")
            market_end = formation - config.skip_months
            market_start = market_end - config.market_filter_months
            first = float(market_prices.iloc[market_start])
            last = float(market_prices.iloc[market_end])
            market_ok = math.isfinite(first) and math.isfinite(last) and last > first

        rebalance_turnover = 0.0
        if scheduled:
            signal_end = formation - config.skip_months
            signal_start = formation - config.lookback_months
            raw_signal = prices.iloc[signal_end].divide(
                prices.iloc[signal_start]
            ) - 1.0
            vol_end = signal_end
            vol_start = vol_end - config.volatility_window_months + 1
            volatility = returns.iloc[vol_start : vol_end + 1].std(ddof=1)
            if market_ok:
                target = _target_weights(
                    raw_signal,
                    volatility,
                    forced_eligible,
                    liquidity.iloc[formation],
                    config,
                )
            else:
                target = pd.Series(0.0, index=prices.columns)
            rebalance_turnover = 0.5 * float((target - new_weights).abs().sum())
            new_weights = target

        turnover = forced_turnover + rebalance_turnover
        next_returns = returns.iloc[formation + 1]
        held = new_weights.abs() > 1e-14
        missing_held = list(next_returns.index[held & next_returns.isna()])
        if missing_held:
            month = prices.index[formation + 1].date()
            raise ValueError(
                f"missing holding-period return for {missing_held[:8]} at {month}"
            )
        gross_return = float((new_weights * next_returns.fillna(0.0)).sum())
        net_return = gross_return - turnover * config.transaction_cost_rate
        records.append(
            {
                "date": prices.index[formation + 1],
                "net_return": net_return,
                "gross_return": gross_return,
                "turnover": turnover,
                "invested_fraction": float(new_weights.sum()),
                "positions": int(held.sum()),
                "market_filter_passed": market_ok,
            }
        )

        end_values = new_weights * (1.0 + next_returns.fillna(0.0))
        gross_growth = 1.0 + gross_return
        current_weights = (
            end_values / gross_growth
            if gross_growth > 0.0
            else pd.Series(0.0, index=prices.columns)
        )

    result = pd.DataFrame(records).set_index("date")
    result.index = pd.DatetimeIndex(result.index, name="date")
    return result


def performance_metrics(
    returns: pd.Series,
    turnover: pd.Series | None = None,
) -> dict[str, float]:
    values = returns.dropna().astype(float)
    if len(values) < 2:
        raise ValueError("at least two returns are required")
    growth = float((1.0 + values).prod())
    cagr = growth ** (12.0 / len(values)) - 1.0 if growth > 0.0 else math.nan
    volatility = float(values.std(ddof=1) * math.sqrt(12.0))
    sharpe = (
        float(values.mean() / values.std(ddof=1) * math.sqrt(12.0))
        if values.std(ddof=1) > 0.0
        else math.nan
    )
    equity = (1.0 + values).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "months": float(len(values)),
        "total_return": growth - 1.0,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "maximum_drawdown": float(drawdown.min()),
        "positive_month_rate": float((values > 0.0).mean()),
        "annualized_turnover": (
            float(turnover.reindex(values.index).mean() * 12.0)
            if turnover is not None
            else math.nan
        ),
    }


def walk_forward_summary(
    backtest: pd.DataFrame,
    test_years: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025),
) -> dict[str, object]:
    fold_metrics: list[dict[str, object]] = []
    oos_parts: list[pd.DataFrame] = []
    for year in test_years:
        train = backtest.loc[backtest.index.year < year]
        test = backtest.loc[backtest.index.year == year]
        if len(train) < 24 or len(test) < 12:
            raise ValueError(f"walk-forward fold {year} lacks complete train/test data")
        train_metrics = performance_metrics(train["net_return"], train["turnover"])
        test_metrics = performance_metrics(test["net_return"], test["turnover"])
        fold_metrics.append(
            {"test_year": year, "train": train_metrics, "test": test_metrics}
        )
        oos_parts.append(test)
    oos = pd.concat(oos_parts).sort_index()
    return {
        "folds": fold_metrics,
        "oos": performance_metrics(oos["net_return"], oos["turnover"]),
        "positive_oos_years": sum(
            float(fold["test"]["total_return"]) > 0.0 for fold in fold_metrics
        ),
        "oos_returns": oos["net_return"],
        "oos_turnover": oos["turnover"],
    }


def _monthly_sharpe(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return -math.inf
    deviation = float(np.std(finite, ddof=1))
    return float(np.mean(finite) / deviation) if deviation > 0.0 else -math.inf


def deflated_sharpe_probability(
    selected_returns: pd.Series,
    trial_monthly_sharpes: np.ndarray,
    number_of_trials: int,
) -> dict[str, float]:
    values = selected_returns.dropna().to_numpy(dtype=float)
    if len(values) < 3 or number_of_trials < 1:
        raise ValueError("DSR requires returns and at least one recorded trial")
    sr = _monthly_sharpe(values)
    finite_trials = trial_monthly_sharpes[np.isfinite(trial_monthly_sharpes)]
    trial_mean = float(np.mean(finite_trials))
    trial_std = float(np.std(finite_trials, ddof=1)) if len(finite_trials) > 1 else 0.0
    if number_of_trials == 1 or trial_std == 0.0:
        expected_max = trial_mean
    else:
        normal = NormalDist()
        gamma = 0.5772156649015329
        first = normal.inv_cdf(1.0 - 1.0 / number_of_trials)
        second = normal.inv_cdf(1.0 - 1.0 / (number_of_trials * math.e))
        expected_max = trial_mean + trial_std * (
            (1.0 - gamma) * first + gamma * second
        )
    skew = float(pd.Series(values).skew())
    kurtosis = float(pd.Series(values).kurt()) + 3.0
    variance_term = 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr * sr
    statistic = (
        (sr - expected_max) * math.sqrt(len(values) - 1.0)
        / math.sqrt(variance_term)
        if variance_term > 0.0
        else -math.inf
    )
    return {
        "dsr_probability": float(NormalDist().cdf(statistic)),
        "observed_monthly_sharpe": sr,
        "expected_max_monthly_sharpe": expected_max,
        "trials_penalty_count": float(number_of_trials),
    }


def probability_of_backtest_overfitting(
    return_matrix: pd.DataFrame,
    number_of_blocks: int = 10,
) -> dict[str, object]:
    clean = return_matrix.dropna(axis=0, how="any").dropna(axis=1, how="any")
    if number_of_blocks < 4 or number_of_blocks % 2:
        raise ValueError("PBO block count must be even and at least four")
    if len(clean) < number_of_blocks * 2 or clean.shape[1] < 2:
        raise ValueError("PBO needs more observations and at least two strategies")
    blocks = np.array_split(np.arange(len(clean)), number_of_blocks)
    logits: list[float] = []
    degradations: list[float] = []
    winner_counts: dict[str, int] = {}
    for in_block_ids in combinations(range(number_of_blocks), number_of_blocks // 2):
        in_locations = np.concatenate([blocks[index] for index in in_block_ids])
        out_locations = np.concatenate(
            [blocks[index] for index in range(number_of_blocks) if index not in in_block_ids]
        )
        in_values = clean.iloc[in_locations].to_numpy(dtype=float)
        out_values = clean.iloc[out_locations].to_numpy(dtype=float)
        in_deviation = np.std(in_values, axis=0, ddof=1)
        out_deviation = np.std(out_values, axis=0, ddof=1)
        in_vector = np.divide(
            np.mean(in_values, axis=0),
            in_deviation,
            out=np.full(clean.shape[1], -np.inf),
            where=in_deviation > 0.0,
        )
        out_vector = np.divide(
            np.mean(out_values, axis=0),
            out_deviation,
            out=np.full(clean.shape[1], -np.inf),
            where=out_deviation > 0.0,
        )
        in_scores = pd.Series(in_vector, index=clean.columns)
        winner = str(in_scores.idxmax())
        out_scores = pd.Series(out_vector, index=clean.columns)
        ranks = out_scores.rank(method="average")
        relative_rank = float(ranks[winner] / (len(ranks) + 1.0))
        relative_rank = min(max(relative_rank, 1e-12), 1.0 - 1e-12)
        logits.append(math.log(relative_rank / (1.0 - relative_rank)))
        degradations.append(float(out_scores[winner] - in_scores[winner]))
        winner_counts[winner] = winner_counts.get(winner, 0) + 1
    return {
        "pbo": float(np.mean(np.asarray(logits) <= 0.0)),
        "split_count": len(logits),
        "number_of_blocks": number_of_blocks,
        "median_logit": float(np.median(logits)),
        "median_sharpe_degradation": float(np.median(degradations)),
        "most_selected_in_sample": sorted(
            winner_counts.items(), key=lambda item: (-item[1], item[0])
        )[:10],
    }


def stationary_bootstrap_metrics(
    returns: pd.Series,
    turnover: pd.Series,
    simulations: int,
    seed: int,
    average_block_length: float = 6.0,
) -> dict[str, float]:
    values = returns.to_numpy(dtype=float)
    turns = turnover.reindex(returns.index).to_numpy(dtype=float)
    if simulations <= 0 or len(values) < 2 or average_block_length < 1.0:
        raise ValueError("invalid stationary-bootstrap dimensions")
    rng = np.random.default_rng(seed)
    random_locations = rng.integers(len(values), size=(simulations, len(values)))
    restart = rng.random((simulations, len(values) - 1)) < 1.0 / average_block_length
    locations = np.empty_like(random_locations)
    locations[:, 0] = random_locations[:, 0]
    for position in range(1, len(values)):
        locations[:, position] = np.where(
            restart[:, position - 1],
            random_locations[:, position],
            (locations[:, position - 1] + 1) % len(values),
        )
    paths = values[locations]
    growth = np.prod(1.0 + paths, axis=1)
    cagr = np.where(growth > 0.0, growth ** (12.0 / len(values)) - 1.0, np.nan)
    volatility = np.std(paths, axis=1, ddof=1) * math.sqrt(12.0)
    equity = np.cumprod(1.0 + paths, axis=1)
    drawdown = equity / np.maximum.accumulate(equity, axis=1) - 1.0
    maximum_drawdown = np.min(drawdown, axis=1)
    annualized_turnover = np.mean(turns[locations], axis=1) * 12.0
    result: dict[str, float] = {"simulations": float(simulations)}
    for name, vector in (
        ("cagr", cagr),
        ("annualized_volatility", volatility),
        ("maximum_drawdown", maximum_drawdown),
        ("annualized_turnover", annualized_turnover),
    ):
        result[f"{name}_q05"] = float(np.nanquantile(vector, 0.05))
        result[f"{name}_median"] = float(np.nanquantile(vector, 0.50))
        result[f"{name}_q95"] = float(np.nanquantile(vector, 0.95))
    return result
