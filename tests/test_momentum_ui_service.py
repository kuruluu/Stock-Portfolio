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
    ALLOWED_HOLDINGS,
    ALLOWED_LOOKBACKS,
    ALLOWED_REBALANCE_INTERVALS,
    ALLOWED_TRANSACTION_COSTS,
    ALLOWED_WEIGHTINGS,
    BacktestSummary,
    DataIdentityError,
    DevelopmentFrames,
    StrategyInputs,
    file_sha256,
    load_development_frames,
    run_exploratory_backtest,
)


def synthetic_frames(
    periods: int = 72,
    symbols: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    dates = pd.date_range("2018-01-31", periods=periods, freq="ME")
    rng = np.random.default_rng(42)
    changes = rng.normal(0.008, 0.035, size=(periods, symbols))
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + changes, axis=0),
        index=dates,
        columns=[f"S{index:03d}" for index in range(symbols)],
    )
    membership = pd.DataFrame(True, index=dates, columns=prices.columns)
    liquidity = pd.DataFrame(
        np.arange(1, symbols + 1)[None, :].repeat(periods, axis=0),
        index=dates,
        columns=prices.columns,
    )
    benchmark = pd.Series(
        100.0 * np.cumprod(np.full(periods, 1.007)),
        index=dates,
        name="SPY",
    )
    return prices, membership, liquidity, benchmark


class StrategyInputsTests(unittest.TestCase):
    def test_defaults_map_to_the_engine_contract(self) -> None:
        config = StrategyInputs().to_engine_config()
        self.assertEqual(config.strategy_type, "cross_sectional")
        self.assertEqual(config.lookback_months, 12)
        self.assertEqual(config.skip_months, 1)
        self.assertEqual(config.universe_size, 300)
        self.assertEqual(config.number_of_assets, 20)
        self.assertTrue(config.require_positive_momentum)
        self.assertEqual(config.transaction_cost_rate, 0.001)
        self.assertTrue(config.config_id.startswith("exploratory-ui-"))

    def test_exactly_288_frontend_combinations_are_allowed(self) -> None:
        combinations = list(
            product(
                ALLOWED_LOOKBACKS,
                ALLOWED_HOLDINGS,
                ALLOWED_REBALANCE_INTERVALS,
                ALLOWED_WEIGHTINGS,
                (False, True),
                ALLOWED_TRANSACTION_COSTS,
            )
        )
        self.assertEqual(len(combinations), 288)
        for values in combinations:
            StrategyInputs(*values)

    def test_every_input_dimension_is_constrained(self) -> None:
        invalid = (
            {"lookback_months": 3},
            {"number_of_assets": 300},
            {"rebalance_interval_months": 2},
            {"weighting": "market_cap"},
            {"require_positive_momentum": 1},
            {"transaction_cost_rate": 0.02},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                StrategyInputs(**values)


class ReadOnlyLoaderTests(unittest.TestCase):
    def _workspace(self, root: Path) -> Path:
        data = root / "data"
        data.mkdir(parents=True)
        prices, membership, liquidity, benchmark = synthetic_frames()
        paths = {
            "prices": data / "prices.parquet",
            "membership": data / "membership.parquet",
            "liquidity": data / "liquidity.parquet",
            "benchmark": data / "benchmark.parquet",
            "manifest": data / "manifest.json",
        }
        prices.to_parquet(paths["prices"])
        membership.to_parquet(paths["membership"])
        liquidity.to_parquet(paths["liquidity"])
        benchmark.to_frame().to_parquet(paths["benchmark"])
        paths["manifest"].write_text('{"research_only": true}', encoding="utf-8")
        workbench = {
            "schema_revision": 1,
            "market_id": "US_SP500_PIT",
            "benchmark": "SPY",
            "research_only": True,
            "read_only": True,
            "cost_stress_rate": 0.005,
            "inputs": {
                name: str(path.relative_to(root)).replace("\\", "/")
                for name, path in paths.items()
            },
            "sha256": {name: file_sha256(path) for name, path in paths.items()},
        }
        workbench_path = root / "momentum_workbench.json"
        workbench_path.write_text(json.dumps(workbench), encoding="utf-8")
        return workbench_path

    def test_loader_accepts_only_hash_bound_aligned_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbench_path = self._workspace(root)
            bundle = load_development_frames(root, workbench_path)
            self.assertEqual(bundle.prices.shape, (72, 40))
            self.assertEqual(bundle.benchmark.name, "SPY")
            self.assertEqual(bundle.cost_stress_rate, 0.005)

    def test_loader_rejects_identity_drift_before_loading_performance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbench_path = self._workspace(root)
            manifest = json.loads(workbench_path.read_text(encoding="utf-8"))
            manifest["sha256"]["prices"] = "0" * 64
            workbench_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(DataIdentityError, "identity"):
                load_development_frames(root, workbench_path)

    def test_loader_refuses_a_non_us_market(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbench_path = self._workspace(root)
            manifest = json.loads(workbench_path.read_text(encoding="utf-8"))
            manifest["market_id"] = "AU_ASX_PIT"
            workbench_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(DataIdentityError, "approved US stock market"):
                load_development_frames(root, workbench_path)

    def test_loader_refuses_a_path_outside_the_data_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbench_path = self._workspace(root)
            manifest = json.loads(workbench_path.read_text(encoding="utf-8"))
            manifest["inputs"]["prices"] = "outside.parquet"
            workbench_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(DataIdentityError, "outside the approved data area"):
                load_development_frames(root, workbench_path)


class BacktestServiceTests(unittest.TestCase):
    def test_run_returns_the_fixed_ordered_summary_without_writing(self) -> None:
        inputs = StrategyInputs()
        prices, membership, liquidity, benchmark = synthetic_frames()
        bundle = DevelopmentFrames(prices, membership, liquidity, benchmark, 0.005)
        with patch("momentum_ui_service.load_development_frames", return_value=bundle) as loader:
            summary = run_exploratory_backtest(inputs)
        self.assertIsInstance(summary, BacktestSummary)
        self.assertEqual(
            tuple(label for label, _, _ in summary.metric_items()),
            (
                "Net CAGR",
                "Sharpe ratio",
                "Maximum drawdown",
                "Annualized volatility",
                "Annualized turnover",
                "Cost-stressed CAGR",
            ),
        )
        self.assertGreater(summary.observations, 1)
        self.assertGreaterEqual(summary.elapsed_seconds, 0.0)
        loader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
