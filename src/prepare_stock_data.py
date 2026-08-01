from __future__ import annotations

import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
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
SOURCE_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OUTPUT_PATH = ROOT / "data" / "stock_monthly_close_2016_2025.parquet"
CANDIDATE_PATH = ROOT / "data" / "stock_universe_candidates.csv"
SELECTED_PATH = ROOT / "data" / "stock_universe_200.csv"
MANIFEST_PATH = ROOT / "data" / "stock_universe_manifest.json"
EXPECTED_INDEX = pd.date_range(
    "2016-01-31", "2025-12-31", freq="ME", name="month_end"
)
TARGET_SIZE = 200
MINIMUM_PER_SECTOR = 8
MAXIMUM_PER_SECTOR = 35


class ConstituentsParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.current_cell: list[str] = []
        self.current_row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        if tag == "table" and attributes.get("id") == "constituents":
            self.in_table = True
        elif self.in_table and tag == "tr":
            self.in_row = True
            self.current_row = []
        elif self.in_row and tag in {"th", "td"}:
            self.in_cell = True
            self.current_cell = []

    def handle_endtag(self, tag: str) -> None:
        if self.in_cell and tag in {"th", "td"}:
            value = re.sub(r"\s+", " ", "".join(self.current_cell)).strip()
            self.current_row.append(value)
            self.in_cell = False
        elif self.in_row and tag == "tr":
            if self.current_row:
                self.rows.append(self.current_row)
            self.in_row = False
        elif self.in_table and tag == "table":
            self.in_table = False

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.current_cell.append(data)


def fetch_candidates() -> pd.DataFrame:
    request = Request(
        SOURCE_URL,
        headers={"User-Agent": "stockPortfolio research/1.0"},
    )
    with urlopen(request, timeout=30) as response:
        content = response.read().decode("utf-8")
    parser = ConstituentsParser()
    parser.feed(html.unescape(content))
    if len(parser.rows) < 2:
        raise ValueError("Could not parse the S&P 500 constituent table")
    header, *rows = parser.rows
    frame = pd.DataFrame(rows, columns=header)
    required = {"Symbol", "Security", "GICS Sector", "GICS Sub-Industry"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Constituent source is missing columns: {sorted(missing)}")
    frame = frame.loc[:, sorted(required)].rename(
        columns={
            "Symbol": "symbol",
            "Security": "company",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
        }
    )
    frame["symbol"] = frame["symbol"].str.strip()
    return frame.drop_duplicates("symbol").sort_values("symbol")


def download_bars(symbols: list[str], chunk_size: int = 50) -> pd.DataFrame:
    load_dotenv(ROOT / ".env")
    client = StockHistoricalDataClient(
        os.getenv("API_KEY"), os.getenv("API_SECRET_KEY")
    )
    frames: list[pd.DataFrame] = []
    for offset in range(0, len(symbols), chunk_size):
        chunk = symbols[offset : offset + chunk_size]
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=chunk,
                    start="2016-01-01",
                    end="2026-01-01",
                    timeframe=TimeFrame.Day,
                    adjustment=Adjustment.ALL,
                    feed=DataFeed.SIP,
                    asof="2026-08-01",
                )
                bars = client.get_stock_bars(request).df
                if not bars.empty:
                    frames.append(bars.reset_index())
                print(
                    f"Downloaded candidates {offset + 1}-"
                    f"{min(offset + chunk_size, len(symbols))} / {len(symbols)}",
                    flush=True,
                )
                last_error = None
                break
            except Exception as error:
                last_error = error
                time.sleep(2**attempt)
        if last_error is not None:
            raise RuntimeError(f"Failed candidate chunk beginning {chunk[0]}") from last_error
    if not frames:
        raise ValueError("The market-data provider returned no stock bars")
    return pd.concat(frames, ignore_index=True)


def allocate_sector_quotas(eligible: pd.DataFrame) -> dict[str, int]:
    counts = eligible["sector"].value_counts().sort_index()
    quotas = {
        sector: min(MINIMUM_PER_SECTOR, int(count))
        for sector, count in counts.items()
    }
    while sum(quotas.values()) < TARGET_SIZE:
        candidates = [
            sector
            for sector, count in counts.items()
            if quotas[sector] < min(int(count), MAXIMUM_PER_SECTOR)
        ]
        if not candidates:
            raise ValueError("Sector constraints cannot supply 200 eligible stocks")
        sector = max(
            candidates,
            key=lambda value: (
                TARGET_SIZE * counts[value] / counts.sum() - quotas[value],
                counts[value] - quotas[value],
                value,
            ),
        )
        quotas[sector] += 1
    return quotas


def main() -> None:
    candidates = fetch_candidates()
    bars = download_bars(candidates["symbol"].tolist())
    bars["date"] = (
        bars["timestamp"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    bars["dollar_volume"] = bars["close"] * bars["volume"]

    daily_close = bars.pivot(
        index="date", columns="symbol", values="close"
    ).sort_index()
    monthly = daily_close.resample("ME").last().reindex(EXPECTED_INDEX)
    liquidity = (
        bars.loc[bars["date"] >= "2025-01-01"]
        .groupby("symbol")["dollar_volume"]
        .median()
        .rename("median_daily_dollar_volume_2025")
    )

    coverage = monthly.notna().sum().rename("complete_months")
    candidates = candidates.join(coverage, on="symbol").join(
        liquidity, on="symbol"
    )
    candidates["complete_months"] = candidates["complete_months"].fillna(0).astype(int)
    candidates["eligible"] = (
        candidates["complete_months"].eq(len(EXPECTED_INDEX))
        & candidates["median_daily_dollar_volume_2025"].notna()
        & np.isfinite(candidates["median_daily_dollar_volume_2025"])
    )
    eligible = candidates.loc[candidates["eligible"]].copy()
    if len(eligible) < TARGET_SIZE:
        raise ValueError(
            f"Only {len(eligible)} candidates have complete coverage; need {TARGET_SIZE}"
        )

    quotas = allocate_sector_quotas(eligible)
    selected_parts: list[pd.DataFrame] = []
    for sector, quota in quotas.items():
        sector_rows = eligible.loc[eligible["sector"] == sector].nlargest(
            quota, "median_daily_dollar_volume_2025"
        )
        selected_parts.append(sector_rows)
    selected = pd.concat(selected_parts).sort_values(
        ["sector", "median_daily_dollar_volume_2025"],
        ascending=[True, False],
    )
    if len(selected) != TARGET_SIZE or selected["symbol"].duplicated().any():
        raise ValueError("The selected stock universe is not exactly 200 unique symbols")

    selected_symbols = selected["symbol"].tolist()
    selected_prices = monthly.loc[:, selected_symbols]
    if selected_prices.shape != (120, 200) or selected_prices.isna().any().any():
        raise ValueError("The selected price matrix failed complete-coverage validation")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    selected_prices.to_parquet(OUTPUT_PATH)
    candidates.to_csv(CANDIDATE_PATH, index=False)
    selected.to_csv(SELECTED_PATH, index=False)
    manifest = {
        "created": pd.Timestamp.now(tz="America/New_York").isoformat(),
        "candidate_source": SOURCE_URL,
        "candidate_source_type": "current constituent snapshot",
        "market_data_provider": "Alpaca SIP historical stock bars",
        "adjustment": "all",
        "asof": "2026-08-01",
        "price_start": str(EXPECTED_INDEX.min().date()),
        "price_end": str(EXPECTED_INDEX.max().date()),
        "rows": len(selected_prices),
        "selected_count": len(selected),
        "eligible_candidate_count": len(eligible),
        "sector_quotas": quotas,
        "selected": selected[
            [
                "symbol",
                "company",
                "sector",
                "sub_industry",
                "median_daily_dollar_volume_2025",
                "complete_months",
            ]
        ].to_dict("records"),
        "limitations": [
            "The fixed universe starts from a current constituent snapshot and is subject to survivorship and membership-selection bias.",
            "Requiring complete 2016-2025 data excludes later IPOs and securities that ceased trading during the period.",
            "Results are research simulations and not forecasts.",
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(selected)} stocks to {OUTPUT_PATH}")
    print(selected["sector"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
