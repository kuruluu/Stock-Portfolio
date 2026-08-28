from __future__ import annotations

from itertools import product
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from momentum_engine import (
    MomentumConfig,
    backtest_momentum,
    momentum_signal,
    validate_etf_inputs,
)


class MomentumEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dates = pd.date_range("2016-01-31", periods=72, freq="ME")
        rng = np.random.default_rng(505)
        values = rng.normal(0.008, 0.035, size=(72, 36))
        cls.prices = pd.DataFrame(
            100.0 * np.cumprod(1.0 + values, axis=0),
            index=cls.dates,
            columns=[f"E{index:02d}" for index in range(36)],
        )
        cls.eligibility = pd.DataFrame(
            True, index=cls.dates, columns=cls.prices.columns
        )
        cls.benchmark = cls.prices.iloc[:, 0].rename("SPY")

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
            config_id="test-etf-config",
            family_id="EXPLORATORY_ETF_CROSS_SECTIONAL",
            strategy_type="cross_sectional",
            lookback_months=lookback,
            skip_months=1,
            number_of_assets=holdings,
            require_positive_momentum=positive,
            use_fixed_slots=True,
            rebalance_interval_months=rebalance,
            weighting=weighting,
            volatility_window_months=6,
            market_filter_months=0,
            maximum_leverage=1.0,
            transaction_cost_rate=cost,
        )

    def run_config(self, config: MomentumConfig) -> pd.DataFrame:
        return backtest_momentum(
            self.prices,
            self.eligibility,
            self.benchmark,
            config,
        )

    def test_representative_288_frontend_configurations_execute(self) -> None:
        count = 0
        for values in product(
            (6, 9, 12),
            (1, 20, 36),
            (1, 3),
            ("equal", "inverse_volatility"),
            (False, True),
            (0.0, 0.001, 0.0025, 0.005),
        ):
            with self.subTest(values=values):
                result = self.run_config(self.config(*values))
                self.assertFalse(result.empty)
                self.assertTrue((result["positions"] <= values[1]).all())
                self.assertTrue((result["turnover"] >= 0.0).all())
                self.assertTrue(
                    all(
                        len(symbols) == positions
                        for symbols, positions in zip(
                            result["held_symbols"], result["positions"], strict=True
                        )
                    )
                )
                count += 1
        self.assertEqual(count, 288)

    def test_backtest_is_deterministic(self) -> None:
        first = self.run_config(self.config())
        second = self.run_config(self.config())
        pd.testing.assert_frame_equal(first, second)

    def test_signal_does_not_read_formation_or_future_prices(self) -> None:
        before = momentum_signal(self.prices, 30, 12, 1)
        changed = self.prices.copy()
        changed.iloc[30:] *= 1000.0
        after = momentum_signal(changed, 30, 12, 1)
        pd.testing.assert_series_equal(before, after)

    def test_held_symbols_match_positions_and_breadth(self) -> None:
        result = self.run_config(self.config(holdings=10, positive=False))
        for symbols, positions in zip(
            result["held_symbols"], result["positions"], strict=True
        ):
            self.assertEqual(len(symbols), positions)
            self.assertLessEqual(positions, 10)
            self.assertEqual(len(symbols), len(set(symbols)))

    def test_quarterly_rebalance_carries_holdings_between_rebalances(self) -> None:
        result = self.run_config(
            self.config(holdings=10, rebalance=3, positive=False)
        )
        self.assertEqual(result.iloc[0]["held_symbols"], result.iloc[1]["held_symbols"])
        self.assertEqual(result.iloc[1]["held_symbols"], result.iloc[2]["held_symbols"])

    def test_missing_held_return_fails_loudly(self) -> None:
        prices = self.prices.copy()
        prices.iloc[-1, :] = np.nan
        eligibility = self.eligibility.copy()
        eligibility.iloc[-1, :] = False
        with self.assertRaisesRegex(ValueError, "holding-period return gap"):
            backtest_momentum(
                prices,
                eligibility,
                self.benchmark,
                self.config(positive=False),
            )

    def test_eligibility_must_be_complete_and_aligned(self) -> None:
        eligibility = self.eligibility.astype("boolean")
        eligibility.iloc[0, 0] = pd.NA
        with self.assertRaisesRegex(ValueError, "eligibility"):
            validate_etf_inputs(self.prices, eligibility, self.benchmark)


if __name__ == "__main__":
    unittest.main()
