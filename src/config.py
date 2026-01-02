"""Centralized configuration for Super Johnny Intelligence autonomous system."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

DRIVE_ROOT = Path(r"G:\My Drive\SuperJohnnyTrader")
LOCAL_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TICKERS = ["SPY", "AAPL", "TSLA", "NVDA", "BTC-USD"]


@dataclass
class AutonomousConfig:
    """Configuration for autonomous multi-ticker operation."""

    tickers: List[str] = field(default_factory=lambda: DEFAULT_TICKERS.copy())
    run_every_hours: float = 24.0
    drive_root: Path = field(default_factory=lambda: DRIVE_ROOT)
    local_root: Path = field(default_factory=lambda: LOCAL_ROOT)
    model_subdir: str = "models"
    prediction_log_name: str = "prediction_log.csv"
    raw_news_name: str = "raw_news.csv"
    processed_sentiment_name: str = "processed_sentiment.csv"
    sentiment_default: float = 0.0

    @property
    def model_dir(self) -> Path:
        """Root directory for all models."""
        return self.drive_root / self.model_subdir

    @property
    def raw_news_path(self) -> Path:
        """Path to raw news CSV."""
        return self.drive_root / self.raw_news_name

    @property
    def sentiment_path(self) -> Path:
        """Path to processed sentiment CSV."""
        return self.drive_root / self.processed_sentiment_name

    @property
    def drive_prediction_log(self) -> Path:
        """Path to prediction log on Google Drive."""
        return self.drive_root / self.prediction_log_name

    @property
    def local_prediction_log(self) -> Path:
        """Path to local backup of prediction log."""
        return self.local_root / self.prediction_log_name

    def ticker_model_dir(self, ticker: str) -> Path:
        """Per-ticker model directory (sanitizes ticker name for filesystem)."""
        safe_ticker = ticker.replace("-", "_").replace(".", "_")
        return self.model_dir / safe_ticker

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        for ticker in self.tickers:
            self.ticker_model_dir(ticker).mkdir(parents=True, exist_ok=True)
