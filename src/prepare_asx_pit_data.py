from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from norgate_asx_adapter import (
    ASX300_INDEX_NAME,
    ASX300_INDEX_SYMBOL,
    AUTHENTIC_START,
    NorgateAccessError,
    _provider,
    discover_asx300_watchlist,
    normalize_membership_frame,
    normalize_price_frame,
    probe_access,
)
from v3_validation import file_sha256
from v4_protocol import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "v4_markets" / "AU_ASX300_PIT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build audited ASX 300 PIT inputs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _monthly_last(values: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    clean = values.loc[values.index >= AUTHENTIC_START].sort_index()
    return clean.resample("ME").last().reindex(index)


def _fetch_security(provider: Any, symbol: str) -> dict[str, object]:
    padding = provider.PaddingType.NONE
    settings = provider.StockPriceAdjustmentType
    total_return = provider.price_timeseries(
        symbol,
        stock_price_adjustment_setting=settings.TOTALRETURN,
        padding_setting=padding,
        start_date="2000-04-01",
        timeseriesformat="pandas-dataframe",
    )
    unadjusted = provider.price_timeseries(
        symbol,
        stock_price_adjustment_setting=settings.NONE,
        padding_setting=padding,
        start_date="2000-04-01",
        timeseriesformat="pandas-dataframe",
    )
    membership = provider.index_constituent_timeseries(
        symbol,
        ASX300_INDEX_NAME,
        padding_setting=padding,
        start_date="2000-04-01",
        timeseriesformat="pandas-dataframe",
    )
    adjusted = normalize_price_frame(total_return)
    raw = normalize_price_frame(unadjusted)
    member = normalize_membership_frame(membership)
    if adjusted.empty or raw.empty or member.empty:
        raise NorgateAccessError(
            "empty_security_history", f"Required history is empty for {symbol}."
        )
    asset_id = str(provider.assetid(symbol))
    last_quoted = pd.to_datetime(provider.last_quoted_date(symbol), errors="coerce")
    return {
        "symbol": symbol,
        "asset_id": asset_id,
        "adjusted": adjusted,
        "raw": raw,
        "membership": member,
        "last_quoted_date": (
            None if pd.isna(last_quoted) else pd.Timestamp(last_quoted).date().isoformat()
        ),
    }


def _benchmark(provider: Any, month_index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = provider.price_timeseries(
        ASX300_INDEX_SYMBOL,
        stock_price_adjustment_setting=provider.StockPriceAdjustmentType.TOTALRETURN,
        padding_setting=provider.PaddingType.NONE,
        start_date="2000-04-01",
        timeseriesformat="pandas-dataframe",
    )
    normalized = normalize_price_frame(frame)
    values = _monthly_last(normalized["close"], month_index)
    if values.isna().any():
        raise NorgateAccessError(
            "benchmark_coverage_gap",
            "The S&P/ASX 300 benchmark does not cover every output month.",
        )
    return values.rename(ASX300_INDEX_SYMBOL).to_frame()


def build_dataset(output: Path) -> dict[str, object]:
    access = probe_access(run_live_smoke=True)
    if not access.ready:
        raise NorgateAccessError(access.code, access.message)
    provider = _provider()
    watchlist, symbols = discover_asx300_watchlist(provider)
    securities = [_fetch_security(provider, symbol) for symbol in symbols]
    latest_daily = max(
        row["adjusted"].index.max() for row in securities if not row["adjusted"].empty
    )
    final_month = pd.Timestamp(latest_daily).to_period("M").to_timestamp("M")
    if pd.Timestamp(latest_daily).normalize() < final_month.normalize():
        final_month -= pd.offsets.MonthEnd(1)
    month_index = pd.date_range(
        AUTHENTIC_START.to_period("M").to_timestamp("M"), final_month, freq="ME"
    )

    prices: dict[str, pd.Series] = {}
    membership: dict[str, pd.Series] = {}
    liquidity: dict[str, pd.Series] = {}
    identities: list[dict[str, object]] = []
    for row in securities:
        key = f"A{row['asset_id']}"
        adjusted = row["adjusted"]
        raw = row["raw"]
        member = row["membership"]
        prices[key] = _monthly_last(adjusted["close"], month_index)
        membership[key] = (
            member.resample("ME").last().reindex(month_index).fillna(False).astype(bool)
        )
        daily_liquidity = (raw["close"] * raw["volume"]).rolling(
            63, min_periods=20
        ).mean()
        liquidity[key] = _monthly_last(daily_liquidity, month_index)
        identities.append(
            {
                "asset_id": str(row["asset_id"]),
                "dataset_key": key,
                "symbol_at_export": str(row["symbol"]),
                "last_quoted_date": row["last_quoted_date"],
            }
        )

    price_frame = pd.DataFrame(prices, index=month_index).sort_index(axis=1)
    member_frame = pd.DataFrame(membership, index=month_index).reindex(
        columns=price_frame.columns
    )
    liquidity_frame = pd.DataFrame(liquidity, index=month_index).reindex(
        columns=price_frame.columns
    )

    terminal_events: list[dict[str, object]] = []
    unexplained: list[dict[str, object]] = []
    for column in price_frame.columns:
        for location in range(len(month_index) - 1):
            if not bool(member_frame.iloc[location][column]):
                continue
            if pd.isna(price_frame.iloc[location][column]):
                unexplained.append(
                    {"asset": column, "date": month_index[location].date().isoformat(), "kind": "member_without_formation_price"}
                )
                continue
            if pd.isna(price_frame.iloc[location + 1][column]):
                if bool(member_frame.iloc[location + 1][column]):
                    unexplained.append(
                        {"asset": column, "date": month_index[location + 1].date().isoformat(), "kind": "member_without_holding_return"}
                    )
                else:
                    price_frame.iloc[location + 1, price_frame.columns.get_loc(column)] = price_frame.iloc[location][column]
                    liquidity_frame.iloc[location + 1, liquidity_frame.columns.get_loc(column)] = 0.0
                    terminal_events.append(
                        {
                            "asset": column,
                            "formation_date": month_index[location].date().isoformat(),
                            "proxy_date": month_index[location + 1].date().isoformat(),
                            "treatment": "terminal_cash_proxy_zero_return",
                        }
                    )
    if unexplained:
        raise NorgateAccessError(
            "unexplained_holding_gaps",
            f"Australian audit found {len(unexplained)} unexplained holding gaps.",
        )

    benchmark = _benchmark(provider, month_index)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "prices": output / "monthly_close.parquet",
        "membership": output / "membership.parquet",
        "liquidity": output / "liquidity.parquet",
        "benchmark": output / "benchmark.parquet",
        "terminal_events": output / "terminal_events.parquet",
        "identities": output / "identities.parquet",
    }
    price_frame.to_parquet(paths["prices"])
    member_frame.to_parquet(paths["membership"])
    liquidity_frame.to_parquet(paths["liquidity"])
    benchmark.to_parquet(paths["benchmark"])
    pd.DataFrame(terminal_events).to_parquet(paths["terminal_events"], index=False)
    pd.DataFrame(identities).to_parquet(paths["identities"], index=False)
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    manifest = {
        "market_id": "AU_ASX300_PIT",
        "country": "Australia",
        "provider": "Norgate Data Australian Platinum",
        "provider_package_version": access.package_version,
        "watchlist": watchlist,
        "index_name": ASX300_INDEX_NAME,
        "index_symbol": ASX300_INDEX_SYMBOL,
        "created_at": datetime.now().astimezone().isoformat(),
        "first_month": month_index.min().date().isoformat(),
        "last_month": month_index.max().date().isoformat(),
        "usable_months": len(month_index),
        "security_count": len(price_frame.columns),
        "point_in_time_membership": True,
        "includes_delisted_returns": True,
        "adjusted_total_return_prices": True,
        "audited_terminal_exit_treatment": True,
        "terminal_cash_proxy_count": len(terminal_events),
        "unexplained_holding_gaps": 0,
        "coverage_gaps": 0,
        "synthetic_pre_march_2000_excluded": True,
        "performance_inspected_before_seal": False,
        "file_sha256": hashes,
    }
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    try:
        manifest = build_dataset(args.output.resolve())
    except NorgateAccessError as exc:
        print(json.dumps({"status": "blocked_missing_norgate_access", "code": exc.code, "message": str(exc)}))
        raise SystemExit(2) from exc
    print(json.dumps({"status": "certified", "manifest": manifest}, sort_keys=True))


if __name__ == "__main__":
    main()

