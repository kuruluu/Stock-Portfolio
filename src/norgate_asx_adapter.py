from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import importlib
from typing import Any, Iterable

import pandas as pd


ASX300_INDEX_NAME = "S&P/ASX 300"
ASX300_INDEX_SYMBOL = "$XKO.au"
AUTHENTIC_START = pd.Timestamp("2000-04-01")


class NorgateAccessError(RuntimeError):
    """A stable, reportable Norgate access or entitlement failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AccessProbe:
    ready: bool
    code: str
    message: str
    package_version: str | None
    updater_running: bool
    required_api_present: bool
    entitlement_verified: bool
    smoke_test: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


REQUIRED_API = (
    "status",
    "price_timeseries",
    "index_constituent_timeseries",
    "watchlists",
    "watchlist_symbols",
    "assetid",
    "last_quoted_date",
    "StockPriceAdjustmentType",
    "PaddingType",
)


def _provider() -> Any:
    try:
        return importlib.import_module("norgatedata")
    except Exception as exc:  # pragma: no cover - depends on local installation
        raise NorgateAccessError(
            "norgatedata_package_missing",
            "The pinned norgatedata package is not importable in this environment.",
        ) from exc


def _string_values(values: Iterable[object]) -> list[str]:
    return sorted({str(value) for value in values if str(value).strip()})


def discover_asx300_watchlist(provider: Any) -> tuple[str, list[str]]:
    names = _string_values(provider.watchlists())
    matches = [
        name
        for name in names
        if "asx 300" in name.lower()
        and "current" in name.lower()
        and "past" in name.lower()
    ]
    if not matches:
        matches = [name for name in names if "asx 300" in name.lower()]
    if not matches:
        raise NorgateAccessError(
            "asx300_current_past_watchlist_missing",
            "No S&P/ASX 300 current-and-past watchlist is available from Norgate.",
        )
    name = matches[0]
    symbols = _string_values(provider.watchlist_symbols(name))
    if not symbols:
        raise NorgateAccessError(
            "asx300_current_past_watchlist_empty",
            "The S&P/ASX 300 current-and-past watchlist contains no securities.",
        )
    return name, symbols


def _timestamp(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    converted = pd.to_datetime(value, errors="coerce")
    if pd.isna(converted):
        return None
    return pd.Timestamp(converted).tz_localize(None)


def _choose_smoke_symbols(provider: Any, symbols: list[str]) -> tuple[str, str]:
    dated: list[tuple[pd.Timestamp, str]] = []
    for symbol in symbols:
        quoted = _timestamp(provider.last_quoted_date(symbol))
        if quoted is not None:
            dated.append((quoted, symbol))
    if len(dated) < 2:
        raise NorgateAccessError(
            "listed_and_delisted_examples_missing",
            "The ASX 300 universe did not expose both dated listed and delisted examples.",
        )
    dated.sort()
    newest_date, listed = dated[-1]
    old = [row for row in dated if row[0] < newest_date - pd.Timedelta(days=365)]
    if not old:
        raise NorgateAccessError(
            "delisted_example_missing",
            "No security ending at least one year before the latest ASX quote was found.",
        )
    return listed, old[0][1]


def _frame_schema(value: object) -> dict[str, object]:
    if not isinstance(value, pd.DataFrame):
        raise NorgateAccessError(
            "unexpected_norgate_timeseries_type",
            "Norgate did not return the requested pandas DataFrame.",
        )
    return {
        "columns": [str(column) for column in value.columns],
        "index_type": type(value.index).__name__,
        "row_count": int(len(value)),
        "nonempty": not value.empty,
    }


def run_read_only_smoke_test(provider: Any) -> dict[str, object]:
    watchlist, symbols = discover_asx300_watchlist(provider)
    listed, delisted = _choose_smoke_symbols(provider, symbols)
    price_settings = provider.StockPriceAdjustmentType
    padding = provider.PaddingType.NONE
    listed_prices = provider.price_timeseries(
        listed,
        stock_price_adjustment_setting=price_settings.TOTALRETURN,
        padding_setting=padding,
        start_date="2020-01-01",
        timeseriesformat="pandas-dataframe",
    )
    delisted_prices = provider.price_timeseries(
        delisted,
        stock_price_adjustment_setting=price_settings.TOTALRETURN,
        padding_setting=padding,
        start_date="2000-04-01",
        timeseriesformat="pandas-dataframe",
    )
    membership = provider.index_constituent_timeseries(
        listed,
        ASX300_INDEX_NAME,
        padding_setting=padding,
        start_date="2000-04-01",
        timeseriesformat="pandas-dataframe",
    )
    schemas = {
        "listed_price": _frame_schema(listed_prices),
        "delisted_price": _frame_schema(delisted_prices),
        "membership": _frame_schema(membership),
    }
    if not all(schema["nonempty"] for schema in schemas.values()):
        raise NorgateAccessError(
            "norgate_smoke_timeseries_empty",
            "At least one required Norgate smoke-test time series is empty.",
        )
    member_values = normalize_membership_frame(membership)
    if not bool(member_values.any()):
        raise NorgateAccessError(
            "historical_membership_entitlement_missing",
            "Historical S&P/ASX 300 membership returned no true observations.",
        )
    return {
        "watchlist": watchlist,
        "universe_symbol_count": len(symbols),
        "listed_example_asset_id": str(provider.assetid(listed)),
        "delisted_example_asset_id": str(provider.assetid(delisted)),
        "schemas": schemas,
        "performance_values_redacted": True,
    }


def probe_access(run_live_smoke: bool = True) -> AccessProbe:
    try:
        provider = _provider()
    except NorgateAccessError as exc:
        return AccessProbe(False, exc.code, str(exc), None, False, False, False, {})

    version = str(getattr(provider, "__version__", "unknown"))
    missing = [name for name in REQUIRED_API if not hasattr(provider, name)]
    if missing:
        return AccessProbe(
            False,
            "norgatedata_api_incomplete",
            f"The installed package lacks required API members: {', '.join(missing)}.",
            version,
            False,
            False,
            False,
            {},
        )
    try:
        updater_running = bool(provider.status())
    except Exception:
        updater_running = False
    if not updater_running:
        return AccessProbe(
            False,
            "norgate_updater_unavailable",
            "Norgate Data Updater is not running or its local service is unavailable.",
            version,
            False,
            True,
            False,
            {},
        )
    if not run_live_smoke:
        return AccessProbe(
            True,
            "norgate_updater_ready_unverified_entitlement",
            "The Norgate service is reachable; the entitlement smoke test was skipped.",
            version,
            True,
            True,
            False,
            {},
        )
    try:
        smoke = run_read_only_smoke_test(provider)
    except NorgateAccessError as exc:
        return AccessProbe(
            False,
            exc.code,
            str(exc),
            version,
            True,
            True,
            False,
            {},
        )
    except Exception as exc:  # pragma: no cover - depends on vendor service
        return AccessProbe(
            False,
            "norgate_live_smoke_failed",
            f"The read-only Norgate smoke test failed: {type(exc).__name__}: {exc}",
            version,
            True,
            True,
            False,
            {},
        )
    return AccessProbe(
        True,
        "ready",
        "Norgate Data Updater and Australian Platinum historical membership are ready.",
        version,
        True,
        True,
        True,
        smoke,
    )


def _dated_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if not isinstance(result.index, pd.DatetimeIndex):
        date_columns = [column for column in result.columns if str(column).lower() == "date"]
        if not date_columns:
            raise ValueError("Norgate frame has no datetime index or Date column")
        result = result.set_index(date_columns[0])
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index)).tz_localize(None)
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result.loc[result.index >= AUTHENTIC_START]


def _column(frame: pd.DataFrame, names: tuple[str, ...]) -> object:
    lookup = {str(column).strip().lower(): column for column in frame.columns}
    for name in names:
        if name in lookup:
            return lookup[name]
    raise ValueError(f"required column missing; expected one of {names}")


def normalize_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    dated = _dated_frame(frame)
    close_column = _column(dated, ("close", "closing price"))
    volume_column = _column(dated, ("volume",))
    result = pd.DataFrame(
        {
            "close": pd.to_numeric(dated[close_column], errors="coerce"),
            "volume": pd.to_numeric(dated[volume_column], errors="coerce"),
        },
        index=dated.index,
    )
    return result.replace([float("inf"), float("-inf")], pd.NA)


def normalize_membership_frame(frame: pd.DataFrame) -> pd.Series:
    dated = _dated_frame(frame)
    candidates = [
        column
        for column in dated.columns
        if "constituent" in str(column).lower()
        or "membership" in str(column).lower()
        or str(column).strip().lower() in {"value", "index constituent"}
    ]
    if not candidates and len(dated.columns) == 1:
        candidates = [dated.columns[0]]
    if not candidates:
        raise ValueError("historical membership value column is missing")
    values = pd.to_numeric(dated[candidates[0]], errors="coerce").fillna(0.0)
    return values.ne(0.0).rename("member")


def iso_now() -> str:
    return datetime.now().astimezone().isoformat()

