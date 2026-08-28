from __future__ import annotations

import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from momentum_engine import MomentumConfig, backtest_momentum, momentum_signal


class MomentumEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dates = pd.date_range("2016-01-31", periods=72, freq="ME")
        rng = np.random.default_rng(505)
        values = rng.normal(0.008, 0.035, size=(72, 40))
        cls.prices = pd.DataFrame(
            100.0 * np.cumprod(1.0 + values, axis=0),
            index=cls.dates,
            columns=[f"S{index:02d}" for index in range(40)],
        )
        cls.membership = pd.DataFrame(True, index=cls.dates, columns=cls.prices.columns)
        cls.liquidity = pd.DataFrame(
            np.arange(1, 41)[None, :].repeat(72, axis=0),
            index=cls.dates,
            columns=cls.prices.columns,
        )
        cls.benchmark = pd.Series(
            100.0 * np.cumprod(np.full(72, 1.01)), index=cls.dates, name="SPY"
        )

    @staticmethod
    def config(
        lookback: int = 12,
        holdings: int = 20,
        rebalance: int = 1,
        weighting: str = "equal",
        positive: bool = True,
        cost: float = 0.001,
    ) -> MomentumConfig:
        return MomentumConfig(
            config_id="equivalence",
            family_id="EXPLORATORY_UI_CROSS_SECTIONAL",
            strategy_type="cross_sectional",
            lookback_months=lookback,
            skip_months=1,
            universe_size=300,
            number_of_assets=holdings,
            require_positive_momentum=positive,
            use_fixed_slots=True,
            rebalance_interval_months=rebalance,
            weighting=weighting,
            volatility_window_months=6,
            market_filter_months=0,
            target_volatility=None,
            maximum_leverage=1.0,
            transaction_cost_rate=cost,
        )

    def test_all_288_allowed_ui_configs_match_the_equivalence_fingerprint(self) -> None:
        digest = hashlib.sha256()
        count = 0
        for values in product(
            (6, 9, 12),
            (10, 20, 30),
            (1, 3),
            ("equal", "inverse_volatility"),
            (False, True),
            (0.0, 0.001, 0.0025, 0.005),
        ):
            config = self.config(*values)
            result = backtest_momentum(
                self.prices,
                self.membership,
                self.liquidity,
                self.benchmark,
                config,
            )
            digest.update(json.dumps(values, separators=(",", ":")).encode())
            digest.update(
                result.to_csv(
                    float_format="%.17g",
                    date_format="%Y-%m-%d",
                ).encode()
            )
            count += 1
        self.assertEqual(count, 288)
        self.assertEqual(
            digest.hexdigest(),
            "5cf6ec17f06f3e8ea83ce784535c2a96ed474580d10ba23eeeca30835a5975e6",
        )

    def test_signal_does_not_read_formation_or_future_prices(self) -> None:
        before = momentum_signal(self.prices, 30, 12, 1)
        changed = self.prices.copy()
        changed.iloc[30:] *= 1000.0
        after = momentum_signal(changed, 30, 12, 1)
        pd.testing.assert_series_equal(before, after)

    def test_missing_held_return_fails_loudly(self) -> None:
        prices = self.prices.copy()
        prices.iloc[-1, :] = np.nan
        with self.assertRaisesRegex(ValueError, "holding-period return gap"):
            backtest_momentum(
                prices,
                self.membership,
                self.liquidity,
                self.benchmark,
                self.config(),
            )


if __name__ == "__main__":
    unittest.main()
