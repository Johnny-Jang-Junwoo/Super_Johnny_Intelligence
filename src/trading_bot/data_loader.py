from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yfinance as yf

DRIVE_PATH = Path(r"G:\My Drive\SuperJohnnyTrader")


def _latest_file(files: Iterable[Path]) -> Path:
    return max(files, key=lambda p: p.stat().st_mtime)


def load_trade_logs(drive_path: Path = DRIVE_PATH, pattern: str = "*trade*.csv") -> pd.DataFrame:
    drive_path = Path(drive_path)
    files = sorted(drive_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No trade log CSVs found at {drive_path} with pattern {pattern}")

    data_frames = [pd.read_csv(path) for path in files]
    return pd.concat(data_frames, ignore_index=True)


def load_latest_trade_log(
    drive_path: Path = DRIVE_PATH, pattern: str = "*trade*.csv"
) -> tuple[pd.DataFrame, Path]:
    drive_path = Path(drive_path)
    files = sorted(drive_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No trade log CSVs found at {drive_path} with pattern {pattern}")

    latest_path = _latest_file(files)
    return pd.read_csv(latest_path), latest_path


def load_sentiment_scores(
    drive_path: Path = DRIVE_PATH, filename: str = "processed_sentiment.csv"
) -> Optional[pd.DataFrame]:
    path = Path(drive_path) / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


def fetch_market_data(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    if interval not in {"1m", "1d"}:
        raise ValueError("interval must be '1m' or '1d'")

    kwargs = {"interval": interval, "progress": False}
    if start or end:
        kwargs.update({"start": start, "end": end})
    else:
        kwargs["period"] = "7d" if interval == "1m" else "2y"

    data = yf.download(ticker, **kwargs)
    if data.empty:
        raise ValueError(f"No market data returned for {ticker} with interval {interval}")

    data = data.reset_index()
    data.rename(columns={"Datetime": "Date", "Adj Close": "Adj_Close"}, inplace=True)
    return data

