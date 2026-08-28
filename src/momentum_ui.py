from __future__ import annotations

import colorsys
from html import escape
import logging

import altair as alt
import streamlit as st

import momentum_ui_service as service


LOGGER = logging.getLogger("momentum_workbench")

LOOKBACK_VALUES = {"6 months": 6, "9 months": 9, "12 months": 12}
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
        .block-container {
            max-width: 1400px;
            padding-top: 2.5rem;
            padding-bottom: 4rem;
        }
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
        [data-testid="stNumberInput"] input:focus-visible,
        [data-testid="stCheckbox"] input:focus-visible + div {
            outline: 3px solid rgba(49, 92, 85, 0.28);
            outline-offset: 2px;
        }
        .result-meta {
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.6;
        }
        .chart-note {
            color: var(--muted);
            font-size: 0.86rem;
            margin: -0.2rem 0 0.65rem;
        }
        @media (max-width: 640px) {
            .block-container {
                padding-top: 1.5rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _selected_inputs(columns: list[object]) -> tuple[service.StrategyInputs, bool]:
    with columns[0]:
        lookback = st.selectbox(
            "Lookback",
            tuple(LOOKBACK_VALUES),
            index=2,
            help="Momentum return through the most recent completed signal month.",
        )
    with columns[1]:
        holdings = st.number_input(
            "Holdings",
            min_value=service.MIN_HOLDINGS,
            max_value=service.MAX_HOLDINGS,
            value=20,
            step=1,
            help="Highest-ranked eligible ETFs held in fixed portfolio slots.",
        )
    with columns[2]:
        rebalance = st.selectbox(
            "Rebalance",
            tuple(REBALANCE_VALUES),
            index=0,
        )
    with columns[3]:
        weighting = st.selectbox(
            "Weighting",
            tuple(WEIGHTING_VALUES),
            index=0,
            help="Equal allocation or allocation tilted toward lower trailing volatility.",
        )
    with columns[4]:
        require_positive = st.toggle(
            "Positive momentum only",
            value=True,
            help="Leave unfilled portfolio slots in cash when qualifying momentum is absent.",
        )
    with columns[5]:
        transaction_cost = st.selectbox(
            "Transaction cost",
            tuple(COST_VALUES),
            index=1,
            help="Applied to each unit of portfolio turnover.",
        )
    with columns[6]:
        run_clicked = st.button(
            "Run backtest",
            type="primary",
            disabled=bool(st.session_state.get("running", False)),
        )
    return (
        service.StrategyInputs(
            lookback_months=LOOKBACK_VALUES[lookback],
            number_of_assets=int(holdings),
            rebalance_interval_months=REBALANCE_VALUES[rebalance],
            weighting=WEIGHTING_VALUES[weighting],
            require_positive_momentum=require_positive,
            transaction_cost_rate=COST_VALUES[transaction_cost],
        ),
        run_clicked,
    )


def _symbol_palette(symbols: list[str]) -> list[str]:
    colors: list[str] = []
    for index in range(len(symbols)):
        hue = (index * 0.618033988749895) % 1.0
        red, green, blue = colorsys.hls_to_rgb(hue, 0.46, 0.60)
        colors.append(f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}")
    return colors


def _chart_tooltips() -> list[alt.Tooltip]:
    return [
        alt.Tooltip("symbol:N", title="ETF"),
        alt.Tooltip("date:T", title="Month", format="%b %Y"),
        alt.Tooltip("adjusted_close:Q", title="Adjusted close", format=",.2f"),
        alt.Tooltip("normalized_value:Q", title="Indexed value", format=",.1f"),
        alt.Tooltip("held:N", title="Strategy status"),
    ]


def _build_chart(chart_data: service.ChartData) -> alt.LayerChart:
    symbols = sorted(str(value) for value in chart_data.background["symbol"].unique())
    x_encoding = alt.X(
        "date:T",
        title=None,
        axis=alt.Axis(
            format="%Y",
            tickCount="year",
            grid=False,
            labelAngle=0,
            labelColor="#667085",
        ),
    )
    y_encoding = alt.Y(
        "normalized_value:Q",
        title="Indexed value (start = 100)",
        scale=alt.Scale(zero=False),
        axis=alt.Axis(grid=True, gridColor="#edf0f2", labelColor="#667085"),
    )
    quarter_guides = (
        alt.Chart(chart_data.quarter_guides)
        .mark_rule(color="#d7dce2", opacity=0.55, strokeWidth=1)
        .encode(x=x_encoding)
    )
    background = (
        alt.Chart(chart_data.background)
        .mark_line(color="#98a2b3", opacity=0.16, strokeWidth=1)
        .encode(
            x=x_encoding,
            y=y_encoding,
            detail=alt.Detail("symbol:N"),
            tooltip=_chart_tooltips(),
        )
    )
    selected = (
        alt.Chart(chart_data.segments)
        .mark_line(opacity=0.96, strokeWidth=3)
        .encode(
            x=x_encoding,
            y=y_encoding,
            detail=alt.Detail("segment_id:N"),
            color=alt.Color(
                "symbol:N",
                legend=None,
                scale=alt.Scale(domain=symbols, range=_symbol_palette(symbols)),
            ),
            tooltip=_chart_tooltips(),
        )
    )
    return (
        alt.layer(quarter_guides, background, selected)
        .properties(height=520, title=alt.TitleParams("ETF performance", anchor="start"))
        .configure_view(stroke=None)
        .configure_title(fontSize=18, fontWeight=600, color="#1f2933", offset=18)
    )


def _render_metrics(summary: service.BacktestSummary) -> None:
    with st.container(border=True):
        st.subheader("Key statistics")
        metric_columns = st.columns(6, gap="small")
        for column, (label, value, kind) in zip(
            metric_columns, summary.metric_items(), strict=True
        ):
            with column:
                st.metric(label, service.format_metric(value, kind))
        configuration = " · ".join(
            f"{escape(label)}: {escape(value)}"
            for label, value in summary.configuration.display_items()
        )
        st.markdown(
            f"""
            <div class="result-meta">
              {escape(summary.period_start)} to {escape(summary.period_end)} ·
              {summary.observations} monthly observations ·
              {summary.elapsed_seconds:.2f}s<br>
              {configuration}<br>
              Universe: 36 exploratory US ETFs · Benchmark: SPY ·
              Cost stress: 50 bps per unit of turnover
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="ETF momentum workbench",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _apply_theme()

    st.title("ETF momentum workbench")
    st.write("Configure a cross-sectional ETF momentum strategy and see what it held each month.")
    st.caption("Exploratory local backtest only. Runs are not saved, ranked, or registered.")

    with st.container(border=True):
        st.subheader("Strategy settings")
        settings_columns = st.columns(
            (1.0, 0.72, 0.92, 1.15, 1.12, 1.0, 0.92),
            gap="small",
            vertical_alignment="bottom",
        )
        current_inputs, run_clicked = _selected_inputs(settings_columns)

    if run_clicked:
        st.session_state["running"] = True
        st.session_state["error"] = None
        try:
            with st.spinner("Running ETF backtest…"):
                output = service.run_exploratory_backtest(current_inputs)
            st.session_state["output"] = output
            st.session_state["last_inputs"] = current_inputs
        except service.MomentumUIError as exc:
            st.session_state["output"] = None
            st.session_state["error"] = str(exc)
        except Exception:
            LOGGER.exception("Unexpected ETF momentum workbench failure")
            st.session_state["output"] = None
            st.session_state["error"] = (
                "The backtest could not be completed. Check the local diagnostic output."
            )
        finally:
            st.session_state["running"] = False

    error = st.session_state.get("error")
    output = st.session_state.get("output")
    last_inputs = st.session_state.get("last_inputs")
    with st.container(border=True):
        graph_view = st.segmented_control(
            "Graph view",
            service.CHART_VIEWS,
            default="Max",
            required=True,
            key="graph-view",
            help="Rebase and display the final 1 year, 3 years, or full evaluation period.",
        )
        if error:
            st.error(error)
        elif output is None:
            st.info("Run a backtest to see normalized ETF histories and monthly holdings.")
        else:
            if last_inputs != current_inputs:
                st.warning("Settings changed — run again to refresh the chart and statistics.")
            st.markdown(
                "<div class='chart-note'>All ETFs start at 100. Colored segments were held "
                "for that monthly return interval; gray lines were not held.</div>",
                unsafe_allow_html=True,
            )
            st.warning(output.disclosure)
            st.altair_chart(
                _build_chart(service.select_chart_view(output.chart, str(graph_view))),
                width="stretch",
                key="etf-holdings-chart",
            )

    if output is not None:
        _render_metrics(output.summary)


if __name__ == "__main__":
    main()
