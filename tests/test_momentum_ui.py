from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from momentum_ui import _build_chart
from momentum_ui_service import ChartData, DataUnavailableError
from streamlit.testing.v1 import AppTest


APP_PATH = ROOT / "src" / "momentum_ui.py"


def chart_fixture() -> ChartData:
    dates = pd.date_range("2024-01-31", periods=3, freq="ME")
    background = pd.DataFrame(
        {
            "date": [*dates, *dates],
            "symbol": ["AAA"] * 3 + ["BBB"] * 3,
            "adjusted_close": [10.0, 11.0, 12.0, 20.0, 19.0, 18.0],
            "normalized_value": [100.0, 110.0, 120.0, 100.0, 95.0, 90.0],
            "held": ["Not held", "Held", "Not held", "Not held", "Not held", "Held"],
        }
    )
    segments = pd.DataFrame(
        {
            "segment_id": ["AAA|2024-02-29"] * 2,
            "date": dates[:2],
            "symbol": ["AAA", "AAA"],
            "adjusted_close": [10.0, 11.0],
            "normalized_value": [100.0, 110.0],
            "held": ["Held", "Held"],
        }
    )
    quarter_guides = pd.DataFrame({"date": [dates[2]]})
    return ChartData(
        background=background,
        segments=segments,
        quarter_guides=quarter_guides,
    )


class MomentumChartTests(unittest.TestCase):
    def test_chart_layers_all_histories_beneath_held_segments(self) -> None:
        specification = _build_chart(chart_fixture()).to_dict()
        self.assertEqual(len(specification["layer"]), 3)
        self.assertEqual(specification["layer"][0]["mark"]["type"], "rule")
        self.assertEqual(specification["layer"][1]["mark"]["opacity"], 0.16)
        self.assertEqual(specification["layer"][2]["mark"]["strokeWidth"], 3)
        self.assertEqual(
            specification["layer"][2]["encoding"]["detail"]["field"],
            "segment_id",
        )
        self.assertEqual(
            specification["layer"][1]["encoding"]["x"]["axis"]["tickCount"],
            "year",
        )
        self.assertEqual(
            specification["layer"][1]["encoding"]["x"]["axis"]["format"],
            "%Y",
        )
        tooltip_fields = {
            item["field"]
            for layer in specification["layer"]
            for item in layer["encoding"].get("tooltip", [])
        }
        self.assertEqual(
            tooltip_fields,
            {"date", "symbol", "adjusted_close", "normalized_value", "held"},
        )


class MomentumUIAppTests(unittest.TestCase):
    def app(self) -> AppTest:
        return AppTest.from_file(APP_PATH, default_timeout=15).run()

    def test_initial_page_has_horizontal_settings_contract_and_empty_state(self) -> None:
        app = self.app()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "ETF momentum workbench")
        self.assertEqual(len(app.selectbox), 4)
        self.assertEqual(len(app.number_input), 1)
        self.assertEqual(app.number_input[0].value, 20)
        self.assertEqual(app.number_input[0].min, 1)
        self.assertEqual(app.number_input[0].max, 36)
        self.assertEqual(len(app.segmented_control), 1)
        self.assertEqual(app.segmented_control[0].value, "Max")
        self.assertEqual(len(app.toggle), 1)
        self.assertEqual(len(app.button), 1)
        self.assertIn("Run a backtest", app.info[0].value)

    def test_metrics_have_no_individual_box_styling(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertNotIn('[data-testid="stMetric"]', source)

    def test_run_renders_one_chart_before_six_metrics(self) -> None:
        app = self.app()
        app.button[0].click().run(timeout=15)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("vega_lite_chart")), 1)
        self.assertEqual(len(app.metric), 6)
        self.assertEqual(
            tuple(metric.label for metric in app.metric),
            (
                "Net CAGR",
                "Sharpe ratio",
                "Maximum drawdown",
                "Annualized volatility",
                "Annualized turnover",
                "Cost-stressed CAGR",
            ),
        )
        self.assertTrue(
            any("lifecycle history is unavailable" in warning.value for warning in app.warning)
        )

    def test_changing_a_setting_marks_the_chart_and_metrics_stale(self) -> None:
        app = self.app()
        app.button[0].click().run(timeout=15)
        app.number_input[0].set_value(1).run()
        self.assertFalse(app.exception)
        self.assertIn("Settings changed", app.warning[0].value)
        self.assertEqual(len(app.get("vega_lite_chart")), 1)

    def test_changing_graph_view_does_not_mark_results_stale(self) -> None:
        app = self.app()
        app.button[0].click().run(timeout=15)
        app.segmented_control[0].set_value("1Y").run()
        self.assertFalse(app.exception)
        self.assertFalse(
            any("Settings changed" in warning.value for warning in app.warning)
        )
        self.assertEqual(len(app.get("vega_lite_chart")), 1)

    def test_safe_data_error_is_rendered_without_an_exception(self) -> None:
        with patch(
            "momentum_ui_service.run_exploratory_backtest",
            side_effect=DataUnavailableError("Required ETF development data is missing: prices."),
        ):
            app = self.app()
            app.button[0].click().run(timeout=15)
        self.assertFalse(app.exception)
        self.assertIn("Required ETF development data is missing", app.error[0].value)


if __name__ == "__main__":
    unittest.main()
