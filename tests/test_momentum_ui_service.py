from __future__ import annotations

from itertools import product
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from momentum_ui_service import (
    ALLOWED_LOOKBACKS,
    ALLOWED_REBALANCE_INTERVALS,
    ALLOWED_TRANSACTION_COSTS,
    ALLOWED_WEIGHTINGS,
    ChartData,
    MAX_HOLDINGS,
    MIN_HOLDINGS,
    DataIdentityError,
    DevelopmentFrames,
    StrategyInputs,
    WorkbenchResult,
    build_chart_data,
    file_sha256,
    load_development_frames,
    normalize_prices,
    run_exploratory_backtest,
    select_chart_view,
)


def synthetic_prices(periods: int = 72, symbols: int = 36) -> pd.DataFrame:
    dates = pd.date_range("2018-01-31", periods=periods, freq="ME")
    rng = np.random.default_rng(42)
    changes = rng.normal(0.008, 0.035, size=(periods, symbols))
    return pd.DataFrame(
        100.0 * np.cumprod(1.0 + changes, axis=0),
        index=dates,
        columns=["SPY", *(f"E{index:03d}" for index in range(1, symbols))],
    )


class StrategyInputsTests(unittest.TestCase):
    def test_defaults_map_to_the_etf_engine_contract(self) -> None:
        config = StrategyInputs().to_engine_config()
        self.assertEqual(config.strategy_type, "cross_sectional")
        self.assertEqual(config.lookback_months, 12)
        self.assertEqual(config.skip_months, 1)
        self.assertEqual(config.number_of_assets, 20)
        self.assertTrue(config.require_positive_momentum)
        self.assertEqual(config.transaction_cost_rate, 0.001)
        self.assertTrue(config.config_id.startswith("exploratory-ui-"))

    def test_all_3456_frontend_combinations_are_allowed(self) -> None:
        combinations = list(
            product(
                ALLOWED_LOOKBACKS,
                range(MIN_HOLDINGS, MAX_HOLDINGS + 1),
                ALLOWED_REBALANCE_INTERVALS,
                ALLOWED_WEIGHTINGS,
                (False, True),
                ALLOWED_TRANSACTION_COSTS,
            )
        )
        self.assertEqual(len(combinations), 3456)
        for values in combinations:
            StrategyInputs(*values)

    def test_holdings_must_be_a_bounded_integer(self) -> None:
        StrategyInputs(number_of_assets=MIN_HOLDINGS)
        StrategyInputs(number_of_assets=MAX_HOLDINGS)
        for value in (0, 37, 1.5, True):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "integer from 1 to 36"
            ):
                StrategyInputs(number_of_assets=value)


class ReadOnlyLoaderTests(unittest.TestCase):
    def _workspace(self, root: Path) -> Path:
        data = root / "data"
        data.mkdir(parents=True)
        prices = synthetic_prices()
        prices_path = data / "etf_prices.parquet"
        manifest_path = data / "etf_manifest.json"
        prices.to_parquet(prices_path)
        manifest_path.write_text(
            json.dumps(
                {
                    "accepted": list(prices.columns),
                    "rows": len(prices),
                    "start": str(prices.index.min().date()),
                    "end": str(prices.index.max().date()),
                }
            ),
            encoding="utf-8",
        )
        workbench = {
            "schema_revision": 2,
            "market_id": "US_ETF_EXPLORATORY",
            "benchmark": "SPY",
            "eligibility_policy": "all_complete_symbols",
            "research_only": True,
            "read_only": True,
            "cost_stress_rate": 0.005,
            "inputs": {
                "prices": "data/etf_prices.parquet",
                "manifest": "data/etf_manifest.json",
            },
            "sha256": {
                "prices": file_sha256(prices_path),
                "manifest": file_sha256(manifest_path),
            },
        }
        workbench_path = root / "momentum_workbench.json"
        workbench_path.write_text(json.dumps(workbench), encoding="utf-8")
        return workbench_path

    def test_loader_accepts_hash_bound_complete_etf_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbench_path = self._workspace(root)
            bundle = load_development_frames(root, workbench_path)
            self.assertEqual(bundle.prices.shape, (72, 36))
            self.assertTrue(bundle.eligibility.to_numpy().all())
            self.assertEqual(bundle.benchmark.name, "SPY")
            self.assertEqual(bundle.cost_stress_rate, 0.005)

    def test_loader_rejects_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbench_path = self._workspace(root)
            manifest = json.loads(workbench_path.read_text(encoding="utf-8"))
            manifest["sha256"]["prices"] = "0" * 64
            workbench_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(DataIdentityError, "identity"):
                load_development_frames(root, workbench_path)

    def test_loader_rejects_manifest_symbol_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbench_path = self._workspace(root)
            data_manifest_path = root / "data" / "etf_manifest.json"
            data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
            data_manifest["accepted"] = data_manifest["accepted"][:-1]
            data_manifest_path.write_text(json.dumps(data_manifest), encoding="utf-8")
            workbench = json.loads(workbench_path.read_text(encoding="utf-8"))
            workbench["sha256"]["manifest"] = file_sha256(data_manifest_path)
            workbench_path.write_text(json.dumps(workbench), encoding="utf-8")
            with self.assertRaisesRegex(DataIdentityError, "symbols"):
                load_development_frames(root, workbench_path)


class ChartDataTests(unittest.TestCase):
    def test_normalization_rebases_without_mutating_prices(self) -> None:
        prices = pd.DataFrame(
            {"AAA": [10.0, 12.0, 9.0], "BBB": [200.0, 100.0, 300.0]},
            index=pd.date_range("2024-01-31", periods=3, freq="ME"),
        )
        original = prices.copy()
        normalized = normalize_prices(prices)
        pd.testing.assert_frame_equal(prices, original)
        self.assertEqual(normalized.iloc[0].to_dict(), {"AAA": 100.0, "BBB": 100.0})
        self.assertEqual(normalized.iloc[1].to_dict(), {"AAA": 120.0, "BBB": 50.0})

    def test_held_segments_cover_exact_return_intervals(self) -> None:
        dates = pd.date_range("2024-01-31", periods=4, freq="ME")
        prices = pd.DataFrame(
            {"AAA": [10.0, 11.0, 12.0, 13.0], "BBB": [20.0, 19.0, 18.0, 17.0]},
            index=dates,
        )
        result = pd.DataFrame(
            {"held_symbols": [("AAA",), ("BBB",)]},
            index=pd.DatetimeIndex(dates[2:], name="date"),
        )
        chart = build_chart_data(prices, result)
        self.assertEqual(set(chart.background["symbol"]), {"AAA", "BBB"})
        self.assertEqual(len(chart.background), 6)
        self.assertEqual(chart.segments["segment_id"].nunique(), 2)
        self.assertTrue((chart.segments.groupby("segment_id").size() == 2).all())
        first = chart.segments[chart.segments["segment_id"] == "AAA|2024-03-31"]
        self.assertEqual(list(first["date"]), [dates[1], dates[2]])
        self.assertEqual(list(chart.quarter_guides["date"]), [dates[2]])

    def test_chart_views_filter_rebase_and_keep_complete_segments(self) -> None:
        dates = pd.date_range("2020-01-31", periods=49, freq="ME")
        prices = pd.DataFrame(
            {
                "AAA": np.linspace(10.0, 58.0, len(dates)),
                "BBB": np.linspace(200.0, 152.0, len(dates)),
            },
            index=dates,
        )
        result = pd.DataFrame(
            {"held_symbols": [("AAA",)] * (len(dates) - 1)},
            index=pd.DatetimeIndex(dates[1:], name="date"),
        )
        chart = build_chart_data(prices, result)

        one_year = select_chart_view(chart, "1Y")
        three_year = select_chart_view(chart, "3Y")
        maximum = select_chart_view(chart, "Max")

        self.assertTrue((one_year.background.groupby("symbol").size() == 13).all())
        self.assertTrue((three_year.background.groupby("symbol").size() == 37).all())
        self.assertTrue((maximum.background.groupby("symbol").size() == 49).all())
        self.assertTrue(
            (one_year.background.groupby("symbol")["normalized_value"].first() == 100.0).all()
        )
        self.assertEqual(
            one_year.background.groupby("symbol")["adjusted_close"].first().to_dict(),
            {"AAA": 46.0, "BBB": 164.0},
        )
        self.assertTrue((one_year.segments.groupby("segment_id").size() == 2).all())
        self.assertTrue(
            set(one_year.quarter_guides["date"].dt.month).issubset({3, 6, 9, 12})
        )

    def test_chart_view_rejects_unknown_values(self) -> None:
        dates = pd.date_range("2024-01-31", periods=3, freq="ME")
        chart = ChartData(
            background=pd.DataFrame(
                {
                    "date": dates,
                    "symbol": ["AAA"] * 3,
                    "adjusted_close": [10.0, 11.0, 12.0],
                    "normalized_value": [100.0, 110.0, 120.0],
                    "held": ["Not held"] * 3,
                }
            ),
            segments=pd.DataFrame(
                columns=(
                    "segment_id",
                    "date",
                    "symbol",
                    "adjusted_close",
                    "normalized_value",
                    "held",
                )
            ),
            quarter_guides=pd.DataFrame({"date": [dates[-1]]}),
        )
        with self.assertRaisesRegex(ValueError, "Unsupported chart view"):
            select_chart_view(chart, "5Y")


class BacktestServiceTests(unittest.TestCase):
    def test_run_returns_chart_first_workbench_result(self) -> None:
        inputs = StrategyInputs()
        prices = synthetic_prices()
        eligibility = pd.DataFrame(True, index=prices.index, columns=prices.columns)
        bundle = DevelopmentFrames(
            prices=prices,
            eligibility=eligibility,
            benchmark=prices["SPY"],
            cost_stress_rate=0.005,
        )
        with patch("momentum_ui_service.load_development_frames", return_value=bundle):
            output = run_exploratory_backtest(inputs)
        self.assertIsInstance(output, WorkbenchResult)
        self.assertEqual(len(output.chart.background["symbol"].unique()), 36)
        self.assertFalse(output.chart.segments.empty)
        self.assertEqual(
            tuple(label for label, _, _ in output.summary.metric_items()),
            (
                "Net CAGR",
                "Sharpe ratio",
                "Maximum drawdown",
                "Annualized volatility",
                "Annualized turnover",
                "Cost-stressed CAGR",
            ),
        )
        self.assertIn("lifecycle", output.disclosure.lower())


if __name__ == "__main__":
    unittest.main()
