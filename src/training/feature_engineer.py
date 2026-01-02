from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd

try:
    import pandas_ta as ta
except ImportError:  # pragma: no cover - optional dependency
    ta = None

SENTIMENT_WINDOWS = {
    "1h": "1h",
    "1d": "1d",
    "1w": "7d",
    "1m": "30d",
}
REGIONS = ["GLOBAL", "US", "TICKER"]


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
        df["ema"] = df["close"].ewm(span=self.config.ema_length, adjust=False).mean()

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

    def _load_sentiment(self, sentiment_file: Union[str, Path, None]) -> pd.DataFrame | None:
        if sentiment_file is None or not Path(sentiment_file).exists():
            return None

        sent_df = pd.read_csv(sentiment_file)
        return _normalize_sentiment_frame(sent_df)

    def add_sentiment_features(
        self,
        data: pd.DataFrame,
        sentiment_file: Union[str, Path, None] = None,
    ) -> pd.DataFrame:
        df = self._ensure_timestamp(data)
        sent_df = self._load_sentiment(sentiment_file)

        sentiment_columns = [
            f"sent_{region.lower()}_{window}"
            for region in REGIONS
            for window in SENTIMENT_WINDOWS
        ]

        if sent_df is None or sent_df.empty:
            for column in sentiment_columns:
                df[column] = float(self.config.sentiment_default)
            return df

        sentiment_features = _build_sentiment_feature_frame(sent_df)

        df = df.sort_values("timestamp")
        merged = pd.merge_asof(
            df,
            sentiment_features,
            left_on="timestamp",
            right_index=True,
            direction="backward",
        )

        merged[sentiment_columns] = merged[sentiment_columns].ffill().fillna(self.config.sentiment_default)
        return merged

    def transform(self, data: pd.DataFrame, sentiment_file: Union[str, Path, None] = None) -> pd.DataFrame:
        df = self.add_indicators(data)
        df = self.add_sentiment_features(df, sentiment_file)
        return df


def engineer_features(
    price_df: pd.DataFrame,
    sentiment_file: Union[str, Path, None] = None,
    default_sentiment: float = 0.0,
) -> pd.DataFrame:
    config = FeatureEngineerConfig(sentiment_default=default_sentiment)
    return FeatureEngineer(config).transform(price_df, sentiment_file)


def get_sentiment_feature_columns() -> list[str]:
    return [f"sent_{region.lower()}_{window}" for region in REGIONS for window in SENTIMENT_WINDOWS]


def _normalize_sentiment_frame(sent_df: pd.DataFrame) -> pd.DataFrame | None:
    if sent_df.empty:
        return None

    date_col = None
    if "timestamp" in sent_df.columns:
        date_col = "timestamp"
    elif "date" in sent_df.columns:
        date_col = "date"
    elif "published" in sent_df.columns:
        date_col = "published"

    if date_col is None or "sentiment" not in sent_df.columns:
        return None

    region_col = "region" if "region" in sent_df.columns else None

    sent_df = sent_df.copy()
    sent_df["timestamp"] = pd.to_datetime(sent_df[date_col], utc=True, errors="coerce")
    sent_df["sentiment"] = pd.to_numeric(sent_df["sentiment"], errors="coerce")
    sent_df = sent_df.dropna(subset=["timestamp", "sentiment"])
    if region_col is None:
        sent_df["region"] = "US"
    else:
        sent_df["region"] = sent_df[region_col].fillna("US").astype(str).str.upper()

    return sent_df[["timestamp", "region", "sentiment"]].sort_values("timestamp")


def _build_sentiment_feature_frame(sent_df: pd.DataFrame) -> pd.DataFrame:
    region_frames = []
    for region in REGIONS:
        region_df = sent_df[sent_df["region"] == region].copy()
        if region_df.empty:
            region_frames.append(
                pd.DataFrame(
                    columns=[f"sent_{region.lower()}_{window}" for window in SENTIMENT_WINDOWS],
                    index=pd.DatetimeIndex([], name="timestamp"),
                )
            )
            continue

        region_df = region_df.set_index("timestamp").sort_index()
        rolling_data: dict[str, pd.Series] = {}
        for window_key, window in SENTIMENT_WINDOWS.items():
            rolling_data[f"sent_{region.lower()}_{window_key}"] = region_df["sentiment"].rolling(
                window,
                min_periods=1,
            ).mean()
        region_frames.append(pd.DataFrame(rolling_data))

    return pd.concat(region_frames, axis=1).sort_index()


def generate_rolling_features(
    sentiment_df: pd.DataFrame,
    asof_timestamp: pd.Timestamp,
    default_sentiment: float = 0.0,
) -> dict[str, float]:
    sentiment_columns = get_sentiment_feature_columns()
    if sentiment_df is None or sentiment_df.empty:
        return {column: float(default_sentiment) for column in sentiment_columns}

    normalized = _normalize_sentiment_frame(sentiment_df)
    if normalized is None or normalized.empty:
        return {column: float(default_sentiment) for column in sentiment_columns}

    sentiment_features = _build_sentiment_feature_frame(normalized)
    target_df = pd.DataFrame(
        {"timestamp": [pd.to_datetime(asof_timestamp, utc=True, errors="coerce")]},
    ).dropna(subset=["timestamp"])

    if target_df.empty:
        return {column: float(default_sentiment) for column in sentiment_columns}

    merged = pd.merge_asof(
        target_df.sort_values("timestamp"),
        sentiment_features,
        left_on="timestamp",
        right_index=True,
        direction="backward",
    )

    for column in sentiment_columns:
        if column not in merged.columns:
            merged[column] = float(default_sentiment)
    merged[sentiment_columns] = merged[sentiment_columns].ffill().fillna(default_sentiment)

    return {column: float(merged.iloc[0][column]) for column in sentiment_columns}
