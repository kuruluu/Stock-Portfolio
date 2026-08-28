# Momentum workbench

A minimalist local frontend for configuring one cross-sectional momentum strategy and viewing its most important backtest statistics. Runs stay in memory and the market-data files remain read-only.

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

Open the local URL printed by Streamlit. The application verifies the configured SHA-256 identity of every required point-in-time input before reading it. Missing, changed, misaligned, or out-of-scope data fails with a clear message; the application never substitutes synthetic data.

## Strategy controls

- Momentum lookback: 6, 9, or 12 months
- Holdings: 10, 20, or 30 stocks
- Rebalance frequency: monthly or quarterly
- Weighting: equal or inverse volatility
- Positive momentum: required or not required
- Transaction cost: 0, 10, 25, or 50 basis points per unit of turnover

These controls create 288 allowed combinations. The calculation fixes the investable universe at the 300 most liquid eligible stocks, skips one month between signal and formation, uses a six-month volatility window, applies no market filter or volatility target, and caps leverage at 1.0.

## Statistics

The output is intentionally limited to:

1. Net CAGR
2. Sharpe ratio
3. Maximum drawdown
4. Annualized volatility
5. Annualized turnover
6. Cost-stressed CAGR at 50 basis points per unit of turnover

The data period, observation count, runtime, and complete chosen configuration appear as secondary context.

## Research boundary

- Every run is exploratory and is not a registered research trial.
- The application does not rank, compare, save, export, or register runs.
- The application reads only the US point-in-time stock paths and hashes declared in `momentum_workbench.json`.
- The application has no brokerage, account, credential, or production-trading integration.

## Verify

Run all focused tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_momentum_*.py"
```

The suite covers all 288 frontend configurations, calculation equivalence, input validation and identity checks, metric ordering, safe error display, and stale-result behavior.

## Troubleshooting

- **Required data is missing:** restore the named input and rerun; no substitute dataset is used.
- **Data identity changed:** restore the expected input or deliberately update `momentum_workbench.json` only after validating the replacement.
- **Backtest could not be completed:** inspect the terminal running Streamlit for diagnostics; selected settings remain visible in the page.
