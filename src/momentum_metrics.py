from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MomentumMetrics:
    net_cagr: float
    sharpe_ratio: float
    maximum_drawdown: float
    annualized_volatility: float
    annualized_turnover: float
    cost_stress_cagr: float
    observations: int

    def to_dict(self) -> dict[str, object]:
        def finite_or_none(value: float) -> float | None:
            return value if math.isfinite(value) else None

        return {
            "net_cagr": finite_or_none(self.net_cagr),
            "sharpe_ratio": finite_or_none(self.sharpe_ratio),
            "maximum_drawdown": finite_or_none(self.maximum_drawdown),
            "annualized_volatility": finite_or_none(self.annualized_volatility),
            "annualized_turnover": finite_or_none(self.annualized_turnover),
            "cost_stress_cagr": finite_or_none(self.cost_stress_cagr),
            "observations": self.observations,
        }


def _cagr(returns: pd.Series) -> float:
    growth = float((1.0 + returns).prod())
    if growth <= 0.0:
        return -1.0
    return growth ** (12.0 / len(returns)) - 1.0


def calculate_momentum_metrics(
    result: pd.DataFrame,
    cost_stress_rate: float,
) -> MomentumMetrics:
    required = {"gross_return", "net_return", "turnover"}
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"momentum result is missing fields: {sorted(missing)}")
    if len(result) < 2:
        raise ValueError("momentum metrics require at least two monthly observations")
    numeric = result[list(sorted(required))].astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("momentum result contains non-finite values")
    if (numeric["turnover"] < 0.0).any():
        raise ValueError("momentum turnover cannot be negative")
    if (numeric[["gross_return", "net_return"]] <= -1.0).to_numpy().any():
        raise ValueError("momentum monthly returns must exceed -100%")
    if not 0.0 <= cost_stress_rate < 1.0:
        raise ValueError("momentum cost stress rate must be in [0, 1)")

    net = numeric["net_return"]
    standard_deviation = float(net.std(ddof=1))
    sharpe = (
        float(net.mean()) / standard_deviation * math.sqrt(12.0)
        if standard_deviation > 0.0
        else math.nan
    )
    wealth = np.concatenate(([1.0], (1.0 + net).cumprod().to_numpy(dtype=float)))
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    stressed = numeric["gross_return"] - numeric["turnover"] * cost_stress_rate
    if (stressed <= -1.0).any():
        raise ValueError("cost-stressed monthly returns must exceed -100%")
    return MomentumMetrics(
        net_cagr=_cagr(net),
        sharpe_ratio=sharpe,
        maximum_drawdown=float(drawdown.min()),
        annualized_volatility=standard_deviation * math.sqrt(12.0),
        annualized_turnover=float(numeric["turnover"].mean()) * 12.0,
        cost_stress_cagr=_cagr(stressed),
        observations=len(numeric),
    )
