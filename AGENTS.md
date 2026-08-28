# Project execution defaults

- Treat this repository as a standalone, local momentum frontend and its tested calculation core.
- Keep the market-data inputs read-only; never rewrite files under `data/` from the application or tests.
- Verify every input against `momentum_workbench.json` before reading performance data.
- Keep strategy runs in memory; do not save, rank, export, or register them.
- Preserve the six-control interface and six-statistic output unless a requested change includes matching tests.
- Run the focused unit and Streamlit interaction tests after frontend, engine, metric, manifest, or dependency changes.
- Do not add account creation, brokerage connectivity, secrets, production trading, or paid services without explicit authorization.
