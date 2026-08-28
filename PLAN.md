# Plan: Flexible Holdings and Focused ETF Chart Views

One-line goal: Users can enter a valid ETF holding count, view unboxed key statistics, and switch the ETF chart between 1Y, 3Y, and Max ranges with yearly labels and quarterly guides (user).

## Classification

Track: UI build — the work changes frontend controls, chart presentation, and matching interaction tests while preserving the existing strategy calculation (verified: `src/momentum_ui.py`, `src/momentum_ui_service.py`, `src/momentum_engine.py`). Parked secondary asks: none (user).

## Interview Ledger

No questions were required; reversible presentation defaults are recorded below (verified: repository inspection and user request).

## Goal & Success Criteria

- Holdings accepts whole numbers from 1 through 36 and defaults to 20 [assumed: the manifest-bound 36-symbol universe defines the maximum — if wrong: derive the limit from a different approved universe contract].
- The six statistics have no individual background, border, rounded box, or box padding (user).
- The graph offers 1Y, 3Y, and Max views without changing strategy statistics [assumed: graph view is presentation-only — if wrong: make it a backtest setting].
- Each visible graph window rebases every ETF to 100 at its first displayed observation [assumed: per-view rebasing is intended — if wrong: retain the Max-period baseline].
- The x-axis labels years only and quarter-end observations have subtle vertical rules (user; [assumed: quarters mean March, June, September, and December month ends — if wrong: use quarter starts]).
- Focused engine, service, chart, and Streamlit interaction tests pass (verified: repository test conventions).

## Current State

- The approved market-data input contains 36 aligned ETF columns (verified: `momentum_workbench.json` and `load_development_frames`).
- The engine accepts positive breadth and rejects values larger than the loaded universe (verified: `src/momentum_engine.py`).
- Strategy runs remain in memory and market-data inputs remain read-only (verified: `AGENTS.md`).

## Scope (v1)

- Replace the fixed holdings selector with a bounded integer input (user).
- Add a display-only 1Y/3Y/Max segmented control above the chart [A1].
- Filter and rebase the existing chart data without rerunning the strategy [A2].
- Add quarter-end rule marks behind the ETF lines and yearly x-axis ticks (user).
- Remove individual metric decorations while retaining the outer Key statistics section [A3].
- Update focused tests and user documentation (verified: existing repository structure).

## Out of Scope & Parked Items

- Custom date ranges, zooming, panning, and additional presets are excluded because the requested choices are 1Y, 3Y, and Max (user).
- Backtest formulas, metric definitions, export, saved runs, live data, and brokerage connectivity remain unchanged (verified: `AGENTS.md`).
- No new dependency is required (verified: `requirements.txt` and installed API signature inspection).

## Approach

`momentum_ui_service.py` owns a pure chart-view transformation that filters the requested range, rebases adjusted closes, preserves complete visible holding segments, and derives quarter-end guide dates [A1, A2]. `momentum_ui.py` renders the range control and a three-layer Altair chart containing guides, faded histories, and highlighted holding segments [A1].

## Requirements

1. WHEN settings render, THE SYSTEM SHALL expose Holdings as an integer from 1 through 36 with default 20 (user; [A4]).
2. WHEN holdings is invalid, THE SYSTEM SHALL fail validation before engine execution [assumed: fail-loudly behavior — if wrong: clamp the input].
3. WHEN statistics render, THE SYSTEM SHALL preserve six values without per-stat boxes or backgrounds (user).
4. WHEN graph view changes, THE SYSTEM SHALL render 1Y, 3Y, or Max without marking the strategy result stale [A1].
5. WHEN 1Y or 3Y is selected, THE SYSTEM SHALL display 12 or 36 monthly intervals including both endpoints [A2].
6. WHEN any view renders, THE SYSTEM SHALL rebase visible ETFs to 100 while retaining actual adjusted closes in tooltips [A2].
7. WHEN the chart renders, THE SYSTEM SHALL show year-only x-axis labels and quarter-end vertical guides behind ETF lines (user; [A5]).

## Key Decisions

- Holdings control: `st.number_input` bounded to 1–36 [A4].
- Graph range: single-select `st.segmented_control` defaulting to Max [A1].
- Quarter guides: March, June, September, and December month ends [A5].
- Range changes remain outside `StrategyInputs` and therefore do not stale statistics [A1].
- Tests validate all 3,456 legal settings cheaply and execute a representative 288-case engine matrix spanning holdings 1, 20, and 36 [assumed: this balances breadth and runtime — if wrong: execute all 3,456 engine cases].

## Data & State Changes

No persistent data, schema, or manifest changes are required because all view transformations remain in memory (verified: implementation architecture and `AGENTS.md`).

## Interfaces, Integrations & Credentials

No external integration, credential, API, or account is required (verified: local application architecture).

## Edge Cases & Failure Handling

- Holdings below 1, above 36, boolean, or non-integral → reject clearly [A4].
- Requested window exceeds available history → display and rebase all available observations [assumed: graceful truncation — if wrong: disable the preset].
- Segment crosses the visible boundary → omit incomplete off-screen portion and retain complete visible intervals [A2].
- Empty or invalid prices → preserve existing data-identity error behavior (verified: `normalize_prices`).

## Risks, Landmines & Adaptations

- Cropping already-normalized values would not start each view at 100 → rebase from unchanged adjusted closes [A2].
- Streamlit can hot-reload the UI while caching an imported service module → restart the local server after service changes (verified: live execution check).
- Existing modified files overlap this work → patch only requested sections and preserve unrelated changes (verified: `git status --short`).

## Assumptions Ledger

| ID | Assumption | Basis | Blast radius if wrong | Check |
|---|---|---|---|---|
| A1 | Range selection is display-only and defaults to Max. | The user called it a graph view setting (user). | Strategy state and metrics would change. | Streamlit stale-state test |
| A2 | 1Y and 3Y use 12 and 36 intervals and rebase locally. | Existing chart models monthly return intervals (verified: `build_chart_data`). | Window counts and indexed values change. | Service chart-view tests |
| A3 | The outer statistics container remains. | The request targets boxes around each stat (user). | Results-section styling only. | Live computed-style check |
| A4 | Valid holdings are integers 1–36. | Approved universe has 36 columns (verified: loader tests). | Input contract changes. | Boundary and combination tests |
| A5 | Quarter guides use quarter-end months. | Chart data is month-end (verified: service fixtures). | Guide dates shift. | Guide-date and chart-layer tests |

## Open Items (none blocking)

- Exact guide opacity remains a reversible visual choice within the existing muted palette [assumed: minor styling latitude — if wrong: adjust after visual review].

## Verification

- Run `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_momentum_*.py"` (verified: `README.md`).
- Inspect the live Streamlit DOM, chart labels, computed metric styles, and 390px overflow behavior (verified: browser test capability).
- Confirm the local endpoint returns HTTP 200 and leave the app running (user; [assumed: port 8501 remains the desired local endpoint — if wrong: choose another port]).

## Build Phases

- [x] Phase 1: Lock input and chart-view contracts

      Done when: boundary, range, rebasing, guide, and chart-specification tests cover the requested behavior (verified: focused tests).

      Steps: add strict holdings, chart-window, quarterly-guide, yearly-axis, metric-style, and stale-state regression tests (verified: test files).

      Covers: R1–R7; checks: A1–A5.

- [x] Phase 2: Implement service-level input and view behavior

      Done when: all legal holdings validate and view data filters, rebases, and preserves complete segments (verified: service tests).

      Steps: add bounded holdings validation, `select_chart_view`, complete-segment filtering, and quarter-guide derivation (verified: `src/momentum_ui_service.py`).

      Covers: R1–R2 and R4–R7; checks: A1, A2, A4, A5.

- [x] Phase 3: Implement the minimalist frontend revision

      Done when: numeric holdings, range control, three chart layers, yearly labels, and unboxed metrics render (verified: UI tests and live DOM).

      Steps: update widgets, Altair layers and axis, CSS, and view selection without changing strategy state (verified: `src/momentum_ui.py`).

      Covers: R1 and R3–R7; checks: A1–A5.

- [x] Phase 4: Verify, document, and relaunch

      Done when: all focused tests pass, README matches the interface, HTTP returns 200, and live desktop/narrow checks pass (verified: test output, `README.md`, HTTP request, and browser inspection).

      Steps: run focused tests, update documentation, restart Streamlit, inspect the default and boundary flows, and leave the app open at the local endpoint (verified: execution results).

      Covers: R1–R7; checks: A1–A5.
