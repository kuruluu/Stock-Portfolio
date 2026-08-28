from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "c31ac3cc56f28cf9a02b4e694eff7ceab596a0ff"
SOURCE_FILE = "S%26P%20500%20Historical%20Components%20%26%20Changes%20%28Updated%29.csv"
SOURCE_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/"
    f"{SOURCE_COMMIT}/{SOURCE_FILE}"
)
SOURCE_SHA256 = "39a9202c9ef69a74c0ff07e2113ad41fb6da7c8c5b6cd9541f0185fb4391e717"
START_DATE = "2016-01-01"
END_DATE = "2026-07-01"
MONTHS = pd.date_range("2016-01-31", "2026-06-30", freq="ME", name="month_end")

RAW_MEMBERSHIP_PATH = ROOT / "data" / "pit_sp500_membership_source.csv"
PRICE_PATH = ROOT / "data" / "pit_sp500_monthly_close.parquet"
MEMBERSHIP_PATH = ROOT / "data" / "pit_sp500_membership.parquet"
LIQUIDITY_PATH = ROOT / "data" / "pit_sp500_monthly_liquidity.parquet"
MARKET_PATH = ROOT / "data" / "pit_spy_monthly_close.parquet"
AUDIT_PATH = ROOT / "data" / "pit_sp500_coverage_audit.csv"
MANIFEST_PATH = ROOT / "data" / "pit_sp500_manifest.json"
CHUNK_DIR = ROOT / "data" / "pit_stock_chunks"
EXACT_CHUNK_DIR = ROOT / "data" / "pit_stock_chunks_exact_symbol_20260801"
ALIAS_CHUNK_PATH = ROOT / "data" / "pit_stock_alias_chunks" / "aliases.parquet"
TERMINAL_PROXY_PATH = ROOT / "data" / "pit_sp500_terminal_proxies.csv"

ALIAS_RULES: dict[str, tuple[str, str]] = {
    "AABA": ("YHOO", "2017-06-19"),
    "ANDV": ("TSO", "2017-08-01"),
    "APTV": ("DLPH", "2017-11-17"),
    "ARNC": ("AA", "2016-11-01"),
    "BHGE": ("BHI", "2017-07-05"),
    "UAA": ("UA", "2016-04-08"),
    "WYND": ("WYN", "2018-06-01"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a point-in-time S&P 500 membership and price dataset."
    )
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch_membership_source(refresh: bool) -> pd.DataFrame:
    RAW_MEMBERSHIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not RAW_MEMBERSHIP_PATH.exists():
        request = Request(SOURCE_URL, headers={"User-Agent": "stockPortfolio-v2/1.0"})
        with urlopen(request, timeout=60) as response:
            payload = response.read()
        observed = sha256_bytes(payload)
        if observed != SOURCE_SHA256:
            raise ValueError(
                f"historical-membership checksum changed: {observed}"
            )
        RAW_MEMBERSHIP_PATH.write_bytes(payload)
    observed_file = hashlib.sha256(RAW_MEMBERSHIP_PATH.read_bytes()).hexdigest()
    if observed_file != SOURCE_SHA256:
        raise ValueError("cached historical-membership checksum is invalid")
    source = pd.read_csv(RAW_MEMBERSHIP_PATH, parse_dates=["date"])
    source = source.sort_values("date").drop_duplicates("date", keep="last")
    if source.empty or set(source.columns) != {"date", "tickers"}:
        raise ValueError("historical-membership source has an unexpected schema")
    return source


def monthly_membership(source: pd.DataFrame) -> pd.DataFrame:
    indexed = source.set_index("date")["tickers"]
    snapshots = indexed.reindex(MONTHS, method="ffill")
    if snapshots.isna().any():
        raise ValueError("membership source does not cover every requested month")
    symbol_sets = [set(str(value).split(",")) for value in snapshots]
    symbols = sorted(set().union(*symbol_sets))
    matrix = pd.DataFrame(False, index=MONTHS, columns=symbols, dtype=bool)
    for date, names in zip(MONTHS, symbol_sets):
        matrix.loc[date, list(names)] = True
    counts = matrix.sum(axis=1)
    if counts.min() < 490 or counts.max() > 510:
        raise ValueError(
            f"implausible constituent count range {counts.min()}-{counts.max()}"
        )
    return matrix


def download_chunk(
    client: StockHistoricalDataClient,
    symbols: list[str],
    chunk_path: Path,
) -> pd.DataFrame:
    if chunk_path.exists():
        return pd.read_parquet(chunk_path)
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                start=START_DATE,
                end=END_DATE,
                timeframe=TimeFrame.Day,
                adjustment=Adjustment.ALL,
                feed=DataFeed.SIP,
                asof="2026-06-30",
            )
            bars = client.get_stock_bars(request).df
            frame = bars.reset_index() if not bars.empty else pd.DataFrame()
            chunk_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(chunk_path, index=False)
            return frame
        except Exception as error:
            last_error = error
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to download chunk beginning {symbols[0]}") from last_error


def normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "timestamp", "close", "volume"}
    if not required.issubset(bars.columns):
        raise ValueError(f"bar data is missing {sorted(required.difference(bars.columns))}")
    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["date"] = (
        frame["timestamp"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    frame["dollar_volume"] = frame["close"] * frame["volume"]
    return frame


def load_chunk_directory(directory: Path) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in sorted(directory.glob("*.parquet"))]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return normalize_bars(pd.concat(frames, ignore_index=True))


def download_all_bars(symbols: list[str], chunk_size: int) -> pd.DataFrame:
    load_dotenv(ROOT / ".env")
    key = os.getenv("API_KEY")
    secret = os.getenv("API_SECRET_KEY")
    if not key or not secret:
        raise ValueError("API_KEY and API_SECRET_KEY must exist in the environment")
    client = StockHistoricalDataClient(key, secret)
    frames: list[pd.DataFrame] = []
    for offset in range(0, len(symbols), chunk_size):
        chunk = symbols[offset : offset + chunk_size]
        digest = hashlib.sha256(",".join(chunk).encode("utf-8")).hexdigest()[:12]
        frame = download_chunk(client, chunk, CHUNK_DIR / f"{offset:04d}_{digest}.parquet")
        if not frame.empty:
            frames.append(frame)
        print(
            f"PIT data {min(offset + chunk_size, len(symbols))}/{len(symbols)} symbols",
            flush=True,
        )
    if not frames:
        raise ValueError("market-data provider returned no bars")
    bars = pd.concat(frames, ignore_index=True)
    return normalize_bars(bars)


def download_alias_bars() -> pd.DataFrame:
    if ALIAS_CHUNK_PATH.exists():
        return normalize_bars(pd.read_parquet(ALIAS_CHUNK_PATH))
    load_dotenv(ROOT / ".env")
    client = StockHistoricalDataClient(
        os.getenv("API_KEY"), os.getenv("API_SECRET_KEY")
    )
    aliases = sorted({source for source, _ in ALIAS_RULES.values()})
    request = StockBarsRequest(
        symbol_or_symbols=aliases,
        start=START_DATE,
        end=END_DATE,
        timeframe=TimeFrame.Day,
        adjustment=Adjustment.ALL,
        feed=DataFeed.SIP,
        asof="-",
    )
    bars = client.get_stock_bars(request).df
    if bars.empty:
        raise ValueError("market-data provider returned no alias bars")
    frame = bars.reset_index()
    ALIAS_CHUNK_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(ALIAS_CHUNK_PATH, index=False)
    return normalize_bars(frame)


def build_monthly_bars(
    bars: pd.DataFrame,
    symbols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = (
        bars.set_index("date")
        .groupby("symbol")["close"]
        .resample("ME")
        .last()
        .unstack(0)
        .reindex(index=MONTHS, columns=symbols)
    )
    liquidity = (
        bars.set_index("date")
        .groupby("symbol")["dollar_volume"]
        .resample("ME")
        .median()
        .unstack(0)
        .reindex(index=MONTHS, columns=symbols)
    )
    return close, liquidity


def coverage_audit(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for location in range(len(membership) - 1):
        members = membership.iloc[location]
        formation_missing = members & prices.iloc[location].isna()
        next_missing = members & prices.iloc[location + 1].isna()
        for kind, mask in (
            ("formation_price", formation_missing),
            ("next_month_price", next_missing),
        ):
            for symbol in mask.index[mask]:
                rows.append(
                    {
                        "formation_month": membership.index[location],
                        "holding_month": membership.index[location + 1],
                        "symbol": symbol,
                        "missing": kind,
                        "member_next_month": bool(membership.iloc[location + 1][symbol]),
                    }
                )
    return pd.DataFrame(rows)


def choose_provider_series(
    mapped_prices: pd.DataFrame,
    mapped_liquidity: pd.DataFrame,
    exact_prices: pd.DataFrame,
    exact_liquidity: pd.DataFrame,
    membership: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    prices = mapped_prices.copy()
    liquidity = mapped_liquidity.copy()
    choices: dict[str, str] = {}
    for symbol in membership.columns:
        member = membership[symbol]
        mapped_score = int((member & mapped_prices[symbol].notna()).sum())
        exact_score = int((member & exact_prices[symbol].notna()).sum())
        if exact_score > mapped_score:
            prices[symbol] = exact_prices[symbol]
            liquidity[symbol] = exact_liquidity[symbol]
            choices[symbol] = "exact_symbol"
        else:
            choices[symbol] = "entity_mapped"
    return prices, liquidity, choices


def apply_alias_rules(
    prices: pd.DataFrame,
    liquidity: pd.DataFrame,
    alias_bars: pd.DataFrame,
) -> list[dict[str, object]]:
    alias_symbols = sorted({source for source, _ in ALIAS_RULES.values()})
    alias_prices, alias_liquidity = build_monthly_bars(alias_bars, alias_symbols)
    applications: list[dict[str, object]] = []
    for target, (source, cutover) in ALIAS_RULES.items():
        locations = prices.index < pd.Timestamp(cutover)
        missing = locations & prices[target].isna() & alias_prices[source].notna()
        prices.loc[missing, target] = alias_prices.loc[missing, source]
        liquidity.loc[missing, target] = alias_liquidity.loc[missing, source]
        applications.append(
            {
                "target": target,
                "source": source,
                "before": cutover,
                "filled_months": int(missing.sum()),
            }
        )
    return applications


def apply_terminal_cash_proxies(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
) -> list[dict[str, object]]:
    proxies: list[dict[str, object]] = []
    for location in range(len(membership) - 1):
        exiting = membership.iloc[location] & ~membership.iloc[location + 1]
        candidates = (
            exiting
            & prices.iloc[location].notna()
            & prices.iloc[location + 1].isna()
        )
        for symbol in candidates.index[candidates]:
            prices.iloc[location + 1, prices.columns.get_loc(symbol)] = prices.iloc[
                location, prices.columns.get_loc(symbol)
            ]
            proxies.append(
                {
                    "symbol": symbol,
                    "formation_month": membership.index[location],
                    "exit_month": membership.index[location + 1],
                    "proxy_return": 0.0,
                    "reason": "no post-removal monthly bar; carry terminal close into cash exit",
                }
            )
    return proxies


def main() -> None:
    args = parse_args()
    if args.chunk_size < 1:
        raise ValueError("chunk size must be positive")
    source = fetch_membership_source(args.refresh)
    membership = monthly_membership(source)
    requested_symbols = sorted(set(membership.columns).union({"SPY"}))
    mapped_bars = download_all_bars(requested_symbols, args.chunk_size)
    mapped_prices, mapped_liquidity = build_monthly_bars(
        mapped_bars, requested_symbols
    )
    if not EXACT_CHUNK_DIR.exists():
        raise ValueError("preserved exact-symbol cache is required for identity audit")
    exact_bars = load_chunk_directory(EXACT_CHUNK_DIR)
    exact_prices, exact_liquidity = build_monthly_bars(
        exact_bars, requested_symbols
    )
    prices, liquidity, provider_choices = choose_provider_series(
        mapped_prices.loc[:, membership.columns],
        mapped_liquidity.loc[:, membership.columns],
        exact_prices.loc[:, membership.columns],
        exact_liquidity.loc[:, membership.columns],
        membership,
    )
    alias_applications = apply_alias_rules(
        prices, liquidity, download_alias_bars()
    )
    terminal_proxies = apply_terminal_cash_proxies(prices, membership)
    pd.DataFrame(terminal_proxies).to_csv(TERMINAL_PROXY_PATH, index=False)
    market = mapped_prices["SPY"].rename("SPY")

    audit = coverage_audit(prices, membership)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(AUDIT_PATH, index=False)
    if not audit.empty:
        sample = audit.head(12).to_dict("records")
        raise ValueError(
            f"point-in-time price coverage has {len(audit)} gaps; sample={sample}"
        )
    if market.isna().any():
        raise ValueError("SPY market-filter series has missing months")

    prices.to_parquet(PRICE_PATH)
    membership.to_parquet(MEMBERSHIP_PATH)
    liquidity.to_parquet(LIQUIDITY_PATH)
    market.to_frame().to_parquet(MARKET_PATH)
    manifest = {
        "created": datetime.now().astimezone().isoformat(),
        "membership_source": SOURCE_URL,
        "membership_source_commit": SOURCE_COMMIT,
        "membership_source_sha256": SOURCE_SHA256,
        "membership_source_license": "MIT",
        "membership_source_caveat": (
            "Community-maintained history derived from an original historical file, "
            "Wikipedia changes, and manual research; not an official S&P licensed feed."
        ),
        "market_data_provider": "Alpaca SIP historical stock bars",
        "adjustment": "all",
        "symbol_mapping": (
            "Alpaca entity mapping as of 2026-06-30; this joins historical ticker "
            "names for renamed entities while unmatched inactive symbols remain exact"
        ),
        "start": str(MONTHS.min().date()),
        "end": str(MONTHS.max().date()),
        "months": len(MONTHS),
        "unique_historical_members": len(membership.columns),
        "minimum_monthly_members": int(membership.sum(axis=1).min()),
        "maximum_monthly_members": int(membership.sum(axis=1).max()),
        "coverage_gaps": len(audit),
        "complete_coverage_required": True,
        "provider_series_choices": {
            key: list(provider_choices.values()).count(key)
            for key in sorted(set(provider_choices.values()))
        },
        "alias_applications": alias_applications,
        "terminal_cash_proxies": len(terminal_proxies),
        "terminal_proxy_policy": (
            "When a member exits before another monthly bar exists, carry its last "
            "observable close forward for a zero-return cash exit; every use is audited."
        ),
        "research_only": True,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
