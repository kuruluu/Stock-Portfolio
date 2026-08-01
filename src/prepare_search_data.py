from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

ADDITIONAL_SYMBOLS = [
    "XLB", "XLU", "XLRE",
    "IWD", "IWF", "USMV", "QUAL", "MTUM",
    "AGG", "HYG", "EMB", "BIL",
    "EWJ", "EWC", "EWZ",
]

BASE_DATA_PATH = Path("data/monthly_close_2016_2025.parquet")
OUTPUT_PATH = Path("data/search_monthly_close_2016_2025.parquet")
MANIFEST_PATH = Path("data/search_universe_manifest.json")
START_DATE = datetime(2016, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def download_monthly_prices(symbols: list[str]) -> pd.DataFrame:
    load_dotenv()
    client = StockHistoricalDataClient(
        os.getenv("API_KEY"), os.getenv("API_SECRET_KEY")
    )
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        start=START_DATE,
        end=END_DATE,
        timeframe=TimeFrame.Day,
        adjustment=Adjustment.ALL,
        feed=DataFeed.SIP,
    )
    bars = client.get_stock_bars(request).df.reset_index()
    bars["date"] = (
        bars["timestamp"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    daily = bars.pivot(
        index="date", columns="symbol", values="close"
    ).sort_index()
    monthly = daily.resample("ME").last().loc[
        "2016-01-01":"2025-12-31"
    ]
    monthly.index.name = "month_end"
    return monthly


def main() -> None:
    if not BASE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {BASE_DATA_PATH}. Run data.py first."
        )
    base_symbols = list(pd.read_parquet(BASE_DATA_PATH).columns)
    requested = list(dict.fromkeys([*base_symbols, *ADDITIONAL_SYMBOLS]))
    monthly = download_monthly_prices(requested)
    expected_index = pd.date_range(
        "2016-01-31", "2025-12-31", freq="ME", name="month_end"
    )
    monthly = monthly.reindex(expected_index)

    accepted: list[str] = []
    rejected: dict[str, str] = {}
    for symbol in requested:
        if symbol not in monthly.columns:
            rejected[symbol] = "not returned by provider"
        elif monthly[symbol].notna().sum() != len(expected_index):
            rejected[symbol] = (
                f"{monthly[symbol].notna().sum()} of "
                f"{len(expected_index)} complete months"
            )
        else:
            accepted.append(symbol)

    complete = monthly.loc[:, accepted]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    complete.to_parquet(OUTPUT_PATH)
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "requested": requested,
                "accepted": accepted,
                "rejected": rejected,
                "rows": len(complete),
                "start": str(complete.index.min().date()),
                "end": str(complete.index.max().date()),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Saved {len(accepted)} complete ETFs to {OUTPUT_PATH}")
    if rejected:
        print("Rejected:")
        for symbol, reason in rejected.items():
            print(f"  {symbol}: {reason}")


if __name__ == "__main__":
    main()
