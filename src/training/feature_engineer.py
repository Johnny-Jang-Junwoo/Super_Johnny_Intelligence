from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd

try:
    import pandas_ta as ta
except ImportError:  # pragma: no cover - optional dependency
    ta = None


@dataclass
class FeatureEngineerConfig:
    rsi_length: int = 14
    ema_length: int = 20
    sentiment_default: float = 0.0


class FeatureEngineer:
    def __init__(self, config: Optional[FeatureEngineerConfig] = None) -> None:
        self.config = config or FeatureEngineerConfig()

    def _ensure_close(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if "close" in df.columns:
            return df
        if "Close" in df.columns:
            df["close"] = df["Close"]
            return df
        raise ValueError("Expected a 'close' or 'Close' column for indicator calculations.")

    def _ensure_timestamp(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        elif "Date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
        elif "date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        else:
            raise ValueError("Expected a 'timestamp', 'Date', or 'date' column for sentiment merge.")

        return df.dropna(subset=["timestamp"])

    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_close(data)
        # 1. EMA
        df["ema"] = df["close"].ewm(span=self.config.ema_length, adjust=False).mean()

        # 2. RSI
        if ta is not None:
            df["rsi"] = ta.rsi(df["close"], length=self.config.rsi_length)
        else:
            delta = df["close"].diff()
            gains = delta.clip(lower=0.0)
            losses = -delta.clip(upper=0.0)
            avg_gain = gains.rolling(self.config.rsi_length, min_periods=self.config.rsi_length).mean()
            avg_loss = losses.rolling(self.config.rsi_length, min_periods=self.config.rsi_length).mean()
            rs = avg_gain / avg_loss.replace(0.0, pd.NA)
            df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        return df

    def add_sentiment_score(
        self,
        data: pd.DataFrame,
        sentiment_file: Union[str, Path, None] = None,
    ) -> pd.DataFrame:
        """
        Merges AI sentiment scores into the market data.
        Logic: fill forward - the last known news score applies until new news arrives.
        """
        df = self._ensure_timestamp(data)

        # 1. If no file provided, use default (Neutral)
        if sentiment_file is None or not Path(sentiment_file).exists():
            print("WARNING: No sentiment file found. Using default neutral sentiment.")
            df["sentiment_score"] = float(self.config.sentiment_default)
            return df

        # 2. Load the AI Scores
        try:
            sent_df = pd.read_csv(sentiment_file)
            date_col = None
            if "date" in sent_df.columns:
                date_col = "date"
            elif "timestamp" in sent_df.columns:
                date_col = "timestamp"
            elif "Date" in sent_df.columns:
                date_col = "Date"

            sentiment_col = None
            if "ai_sentiment" in sent_df.columns:
                sentiment_col = "ai_sentiment"
            elif "sentiment" in sent_df.columns:
                sentiment_col = "sentiment"

            if not date_col or not sentiment_col:
                print("WARNING: Sentiment file missing date or sentiment columns.")
                df["sentiment_score"] = float(self.config.sentiment_default)
                return df

            # Convert to datetime and sort
            sent_df["timestamp"] = pd.to_datetime(sent_df[date_col], utc=True, errors="coerce")
            sent_df[sentiment_col] = pd.to_numeric(sent_df[sentiment_col], errors="coerce")
            sent_df = sent_df.dropna(subset=["timestamp"])
            sent_df = sent_df.sort_values("timestamp")

            # Keep only relevant columns
            sent_df = sent_df[["timestamp", sentiment_col]]

            # 3. Merge with Market Data (As-Of Merge)
            df = df.sort_values("timestamp")

            merged = pd.merge_asof(
                df,
                sent_df,
                on="timestamp",
                direction="backward",
            )

            # Rename and fill missing (before first news) with default
            merged = merged.rename(columns={sentiment_col: "sentiment_score"})
            merged["sentiment_score"] = merged["sentiment_score"].fillna(self.config.sentiment_default)

            return merged

        except Exception as exc:
            print(f"ERROR: Error merging sentiment: {exc}")
            df["sentiment_score"] = float(self.config.sentiment_default)
            return df

    def transform(self, data: pd.DataFrame, sentiment_file: Union[str, Path, None] = None) -> pd.DataFrame:
        df = self.add_indicators(data)
        df = self.add_sentiment_score(df, sentiment_file)
        return df


def engineer_features(
    price_df: pd.DataFrame,
    sentiment_file: Union[str, Path, None] = None,
    default_sentiment: float = 0.0,
) -> pd.DataFrame:
    config = FeatureEngineerConfig(sentiment_default=default_sentiment)
    return FeatureEngineer(config).transform(price_df, sentiment_file)
