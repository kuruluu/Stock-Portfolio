from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from momentum_metrics import calculate_momentum_metrics


def synthetic_result() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gross_return": [0.012, 0.007, -0.003, 0.014, 0.006, 0.009],
            "net_return": [0.011, 0.006, -0.004, 0.013, 0.005, 0.008],
            "turnover": [0.10, 0.08, 0.10, 0.10, 0.08, 0.10],
        },
        index=pd.date_range("2020-01-31", periods=6, freq="ME"),
    )


class MomentumMetricsTests(unittest.TestCase):
    def test_metrics_match_the_equivalence_fingerprint(self) -> None:
        self.assertEqual(
            calculate_momentum_metrics(synthetic_result(), 0.005).to_dict(),
            {
                "net_cagr": 0.08065987885296932,
                "sharpe_ratio": 3.779112348269772,
                "maximum_drawdown": -0.0040000000000000036,
                "annualized_volatility": 0.020639767440550294,
                "annualized_turnover": 1.12,
                "cost_stress_cagr": 0.08755210398671176,
                "observations": 6,
            },
        )

    def test_non_finite_result_is_rejected(self) -> None:
        result = synthetic_result()
        result.iloc[0, result.columns.get_loc("net_return")] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            calculate_momentum_metrics(result, 0.005)

    def test_invalid_cost_stress_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cost stress rate"):
            calculate_momentum_metrics(synthetic_result(), 1.0)


if __name__ == "__main__":
    unittest.main()
