# ETF momentum workbench

A minimalist local frontend for configuring one cross-sectional ETF momentum strategy, seeing which ETFs it held each month, and reviewing six core backtest statistics. Runs stay in memory and the market-data files remain read-only.

## Install

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\momentum_ui.py
```

Open the local URL printed by Streamlit. The application verifies the configured SHA-256 identities of the ETF prices and universe manifest before reading performance. Missing, changed, malformed, or out-of-scope data fails with a clear message; the application never substitutes another dataset.

## Strategy controls

The settings appear in one horizontal desktop row and stack on narrow screens:

- Momentum lookback: 6, 9, or 12 months
- Holdings: any whole number from 1 through 36 ETFs
- Rebalance frequency: monthly or quarterly
- Weighting: equal or inverse volatility
- Positive momentum: required or not required
- Transaction cost: 0, 10, 25, or 50 basis points per unit of turnover

These controls create 3,456 allowed combinations. The strategy skips one month between signal and formation, uses a six-month volatility window, holds qualifying ETFs in fixed portfolio slots, applies no market filter, and caps leverage at 1.0.

## ETF chart

The main chart shows every ETF in the 36-symbol exploratory universe. Each adjusted-close history is normalized with:

```text
indexed value = adjusted close / adjusted close at the first plotted month × 100
```

This preserves proportional performance while allowing ETFs with different nominal prices to share one scale. Hover details retain the actual adjusted close and indexed value.

Choose **1Y**, **3Y**, or **Max** above the graph. Each view is rebased to 100 at its first visible month without rerunning the backtest or changing the statistics. The x-axis labels years only, and subtle vertical rules mark each quarter end.

- Thin gray lines show every ETF throughout the evaluation period.
- Thick colored segments show ETFs held by the strategy for that monthly return interval.
- Segment colors remain tied to ticker identity within the chart.
- The chart has no permanent 36-item legend; hover identifies each ETF.

## Statistics

The chart is followed by six unboxed statistics:

1. Net CAGR
2. Sharpe ratio
3. Maximum drawdown
4. Annualized volatility
5. Annualized turnover
6. Cost-stressed CAGR at 50 basis points per unit of turnover

## Exploratory-data boundary

The current ETF file contains 36 complete adjusted month-end histories from January 2016 through December 2025. Point-in-time ETF lifecycle history is unavailable, so every complete-history symbol is treated as eligible in every month. The application displays this limitation with every result.

The workbench is exploratory only. It does not qualify a strategy, place orders, connect to a brokerage, download live data, save runs, rank configurations, or provide production-trading output.

## Verify

Run the focused suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_momentum_*.py"
```

The suite validates all 3,456 settings combinations and executes a representative 288-case engine matrix spanning the holdings boundaries. It also covers no-lookahead behavior, ETF selection and holding identity, hash-bound input validation, view-local normalization, quarter guides, held-period segment construction, chart layers, metric ordering, stale results, and safe errors.

## Troubleshooting

- **Required ETF data is missing:** restore the named input and rerun; no substitute dataset is used.
- **Data identity changed:** restore the expected input or deliberately update `momentum_workbench.json` only after validating the replacement.
- **ETF manifest symbols do not match:** regenerate or restore a manifest whose accepted symbols exactly match the price columns.
- **Backtest could not be completed:** inspect the terminal running Streamlit for diagnostics; selected settings remain visible in the page.
