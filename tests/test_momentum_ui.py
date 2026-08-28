from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from momentum_ui_service import DataUnavailableError
from streamlit.testing.v1 import AppTest


APP_PATH = ROOT / "src" / "momentum_ui.py"


class MomentumUIAppTests(unittest.TestCase):
    def app(self) -> AppTest:
        return AppTest.from_file(APP_PATH, default_timeout=15).run()

    def test_initial_page_has_exactly_six_controls_and_no_result(self) -> None:
        app = self.app()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Momentum workbench")
        self.assertEqual(len(app.selectbox), 5)
        self.assertEqual(len(app.toggle), 1)
        self.assertEqual(len(app.button), 1)
        self.assertIn("Choose settings", app.info[0].value)

    def test_run_renders_the_fixed_metric_order(self) -> None:
        app = self.app()
        app.button[0].click().run(timeout=15)
        self.assertFalse(app.exception)
        rendered = "\n".join(element.value for element in app.markdown)
        labels = (
            "Net CAGR",
            "Sharpe ratio",
            "Maximum drawdown",
            "Annualized volatility",
            "Annualized turnover",
            "Cost-stressed CAGR",
        )
        positions = [rendered.index(label) for label in labels]
        self.assertEqual(positions, sorted(positions))

    def test_changing_a_setting_marks_the_result_stale_without_running(self) -> None:
        app = self.app()
        app.button[0].click().run(timeout=15)
        app.selectbox[0].set_value("6 months").run()
        self.assertFalse(app.exception)
        self.assertIn("Settings changed", app.warning[0].value)

    def test_safe_data_error_is_rendered_without_an_exception(self) -> None:
        with patch(
            "momentum_ui_service.run_exploratory_backtest",
            side_effect=DataUnavailableError("Required US development data is missing: prices."),
        ):
            app = self.app()
            app.button[0].click().run(timeout=15)
        self.assertFalse(app.exception)
        self.assertIn("Required US development data is missing", app.error[0].value)


if __name__ == "__main__":
    unittest.main()
