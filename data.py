from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv


SYMBOLS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "TLT", "GLD"]
START_DATE = datetime(2016, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2026, 1, 1, tzinfo=timezone.utc)

OUTPUT_DIRECTORY = Path("data")
CSV_PATH = OUTPUT_DIRECTORY / "monthly_close_2016_2025.csv"
PARQUET_PATH = OUTPUT_DIRECTORY / "monthly_close_2016_2025.parquet"

def create_client() -> StockHistoricalDataClient:
    load_dotenv()
    api_key = os.getenv("API_KEY")
    api_secret_key = os.getenv("API_SECRET_KEY")
    return StockHistoricalDataClient(api_key, api_secret_key)

def download_daily_bars() -> pd.DataFrame:
    client = create_client()
    request = StockBarsRequest(
        symbol_or_symbols=SYMBOLS,
        start=START_DATE,
        end=END_DATE,
        timeframe=TimeFrame.Day,
        adjustment=Adjustment.ALL,
        feed=DataFeed.SIP
    )
    bars = client.get_stock_bars(request).df.reset_index()
    bars["date"] = (bars["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize())
    bars = bars.sort_values(["date", "symbol"])
    return bars

def calculate_monthly_end_closes(bars: pd.DataFrame) -> pd.DataFrame:
    daily_close = bars.pivot(index="date", columns="symbol", values="close").sort_index()
    monthly_close = daily_close.resample("ME").last()
    monthly_close = monthly_close.loc["2016-01-01":"2025-12-31"]
    monthly_close = monthly_close.reindex(columns=SYMBOLS)
    monthly_close.index.name = "month_end"
    return monthly_close

def validate_data(monthly_close: pd.DataFrame) -> None:
    expected_months = 10 * 12
    if len(monthly_close) != expected_months:
        raise ValueError(f"Expected {expected_months} months of data, but got {len(monthly_close)}.")

def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    bars = download_daily_bars()
    monthly_close = calculate_monthly_end_closes(bars)
    validate_data(monthly_close)
    monthly_close.to_csv(CSV_PATH)
    monthly_close.to_parquet(PARQUET_PATH)

if __name__ == "__main__":
    main()