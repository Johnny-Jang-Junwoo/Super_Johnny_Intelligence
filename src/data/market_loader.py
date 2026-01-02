from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yfinance as yf

DRIVE_ROOT = Path(r"G:\My Drive\SuperJohnnyTrader")
TRADE_PATTERN = "*trade*.csv"

DATE_COLUMNS = ["date", "datetime", "timestamp", "time"]
TICKER_COLUMNS = ["ticker", "symbol", "instrument", "asset"]


def _latest_file(files: Iterable[Path]) -> Path:
    return max(files, key=lambda p: p.stat().st_mtime)


def load_trade_logs(drive_root: Path = DRIVE_ROOT, pattern: str = TRADE_PATTERN) -> pd.DataFrame:
    drive_root = Path(drive_root)
    files = sorted(drive_root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No trade log CSVs found at {drive_root} with pattern {pattern}")

    data_frames = [pd.read_csv(path) for path in files]
    return pd.concat(data_frames, ignore_index=True)


def load_latest_trade_log(
    drive_root: Path = DRIVE_ROOT, pattern: str = TRADE_PATTERN
) -> tuple[pd.DataFrame, Path]:
    drive_root = Path(drive_root)
    files = sorted(drive_root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No trade log CSVs found at {drive_root} with pattern {pattern}")

    latest_path = _latest_file(files)
    return pd.read_csv(latest_path), latest_path


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def normalize_trade_log(trade_df: pd.DataFrame) -> pd.DataFrame:
    date_col = _find_column(trade_df.columns.tolist(), DATE_COLUMNS)
    if date_col is None:
        raise ValueError("Trade log must include a date column")

    df = trade_df.copy()
    if date_col != "Date":
        df = df.rename(columns={date_col: "Date"})

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"])


def infer_ticker(trade_df: pd.DataFrame) -> Optional[str]:
    ticker_col = _find_column(trade_df.columns.tolist(), TICKER_COLUMNS)
    if ticker_col is None:
        return None

    series = trade_df[ticker_col].dropna().astype(str).str.upper()
    if series.empty:
        return None
    return series.mode().iloc[0]


def fetch_market_data(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    if interval != "1d":
        raise ValueError("Only interval='1d' is supported for macro analysis")

    kwargs = {"interval": interval, "progress": False}
    if start or end:
        kwargs.update({"start": start, "end": end})
    else:
        kwargs["period"] = "2y"

    data = yf.download(ticker, **kwargs)
    if data.empty:
        raise ValueError(f"No market data returned for {ticker} with interval {interval}")

    data = data.reset_index()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    data.rename(columns={"Datetime": "Date", "Adj Close": "Adj_Close"}, inplace=True)
    return data

