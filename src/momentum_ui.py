from __future__ import annotations

from html import escape
import logging

import streamlit as st

import momentum_ui_service as service


LOGGER = logging.getLogger("momentum_workbench")

LOOKBACK_VALUES = {"6 months": 6, "9 months": 9, "12 months": 12}
HOLDING_VALUES = {"10": 10, "20": 20, "30": 30}
REBALANCE_VALUES = {"Monthly": 1, "Quarterly": 3}
WEIGHTING_VALUES = {"Equal": "equal", "Inverse volatility": "inverse_volatility"}
COST_VALUES = {"0 bps": 0.0, "10 bps": 0.001, "25 bps": 0.0025, "50 bps": 0.005}


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #1f2933;
            --muted: #667085;
            --line: #e4e7ec;
            --paper: #ffffff;
            --wash: #f8fafc;
            --accent: #315c55;
        }
        .stApp { background: var(--wash); color: var(--ink); }
        .block-container { max-width: 980px; padding-top: 3.5rem; padding-bottom: 4rem; }
        h1 { letter-spacing: -0.035em; font-weight: 650 !important; }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--paper);
            border-color: var(--line);
            border-radius: 12px;
        }
        .stButton > button {
            width: 100%;
            border-radius: 8px;
            background: var(--accent);
            border-color: var(--accent);
            color: #ffffff;
            font-weight: 600;
            min-height: 2.75rem;
        }
        .stButton > button:hover { background: #274a45; border-color: #274a45; }
        .stButton > button:focus-visible,
        [data-baseweb="select"]:focus-within,
        [data-testid="stCheckbox"] input:focus-visible + div {
            outline: 3px solid rgba(49, 92, 85, 0.28);
            outline-offset: 2px;
        }
        .result-list { margin: 0; }
        .result-row {
            display: flex;
            justify-content: space-between;
            gap: 1.5rem;
            padding: 0.88rem 0;
            border-bottom: 1px solid var(--line);
        }
        .result-row:last-child { border-bottom: 0; }
        .result-row dt { color: var(--muted); font-size: 0.92rem; }
        .result-row dd { margin: 0; color: var(--ink); font-weight: 650; }
        .result-meta {
            margin-top: 1.25rem;
            padding-top: 1rem;
            border-top: 1px solid var(--line);
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.6;
        }
        @media (max-width: 640px) {
            .block-container { padding-top: 2rem; padding-left: 1rem; padding-right: 1rem; }
            .result-row { gap: 0.75rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _selected_inputs() -> service.StrategyInputs:
    lookback = st.selectbox(
        "Momentum lookback",
        tuple(LOOKBACK_VALUES),
        index=2,
        help="Return measured through the most recent completed signal month.",
    )
    holdings = st.selectbox(
        "Holdings",
        tuple(HOLDING_VALUES),
        index=1,
        help="Highest-ranked eligible stocks held in fixed portfolio slots.",
    )
    rebalance = st.selectbox(
        "Rebalance frequency",
        tuple(REBALANCE_VALUES),
        index=0,
    )
    weighting = st.selectbox(
        "Weighting",
        tuple(WEIGHTING_VALUES),
        index=0,
        help="Equal allocation or allocation tilted toward lower trailing volatility.",
    )
    require_positive = st.toggle(
        "Require positive momentum",
        value=True,
        help="Leave unfilled portfolio slots in cash when qualifying momentum is absent.",
    )
    transaction_cost = st.selectbox(
        "Transaction cost",
        tuple(COST_VALUES),
        index=1,
        help="Applied to each unit of portfolio turnover.",
    )
    return service.StrategyInputs(
        lookback_months=LOOKBACK_VALUES[lookback],
        number_of_assets=HOLDING_VALUES[holdings],
        rebalance_interval_months=REBALANCE_VALUES[rebalance],
        weighting=WEIGHTING_VALUES[weighting],
        require_positive_momentum=require_positive,
        transaction_cost_rate=COST_VALUES[transaction_cost],
    )


def _render_summary(summary: service.BacktestSummary) -> None:
    rows = "".join(
        (
            '<div class="result-row">'
            f"<dt>{escape(label)}</dt>"
            f"<dd>{escape(service.format_metric(value, kind))}</dd>"
            "</div>"
        )
        for label, value, kind in summary.metric_items()
    )
    configuration = " · ".join(
        f"{escape(label)}: {escape(value)}"
        for label, value in summary.configuration.display_items()
    )
    st.markdown(
        f"""
        <section aria-label="Backtest statistics">
          <dl class="result-list">{rows}</dl>
          <div class="result-meta">
            {escape(summary.period_start)} to {escape(summary.period_end)} ·
            {summary.observations} monthly observations ·
            {summary.elapsed_seconds:.2f}s<br>
            {configuration}<br>
            Universe: audited US point-in-time stocks · Benchmark: SPY ·
            Cost stress: 50 bps per unit of turnover
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Momentum workbench",
        page_icon=None,
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    _apply_theme()

    st.title("Momentum workbench")
    st.write("Configure a focused cross-sectional momentum strategy and inspect its core statistics.")
    st.caption(
        "Exploratory local backtest only. Runs are not saved, ranked, or registered as research trials."
    )

    settings_column, results_column = st.columns((0.9, 1.1), gap="large")
    with settings_column:
        with st.container(border=True):
            st.subheader("Strategy")
            current_inputs = _selected_inputs()
            run_clicked = st.button(
                "Run backtest",
                type="primary",
                disabled=bool(st.session_state.get("running", False)),
            )

    if run_clicked:
        st.session_state["running"] = True
        st.session_state["error"] = None
        try:
            with st.spinner("Running backtest…"):
                summary = service.run_exploratory_backtest(current_inputs)
            st.session_state["summary"] = summary
            st.session_state["last_inputs"] = current_inputs
        except service.MomentumUIError as exc:
            st.session_state["summary"] = None
            st.session_state["error"] = str(exc)
        except Exception:
            LOGGER.exception("Unexpected momentum workbench failure")
            st.session_state["summary"] = None
            st.session_state["error"] = (
                "The backtest could not be completed. Check the local diagnostic output."
            )
        finally:
            st.session_state["running"] = False

    with results_column:
        with st.container(border=True):
            st.subheader("Key statistics")
            error = st.session_state.get("error")
            summary = st.session_state.get("summary")
            last_inputs = st.session_state.get("last_inputs")
            if error:
                st.error(error)
            elif summary is None:
                st.info("Choose settings and run a backtest to see results.")
            else:
                if last_inputs != current_inputs:
                    st.warning("Settings changed — run again to refresh these results.")
                _render_summary(summary)


if __name__ == "__main__":
    main()
