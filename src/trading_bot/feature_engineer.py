from __future__ import annotations

import pandas as pd
import ta


def _ensure_date_column(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    if "Date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"])


def add_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()
    if "Close" not in df.columns:
        raise ValueError("Price data must include a Close column")

    close = df["Close"]
    df["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    df["ema_12"] = ta.trend.EMAIndicator(close=close, window=12).ema_indicator()
    df["ema_26"] = ta.trend.EMAIndicator(close=close, window=26).ema_indicator()

    if "Volume" in df.columns:
        df["volume_change"] = df["Volume"].pct_change().fillna(0.0)
    else:
        df["volume_change"] = 0.0

    return df


def _normalize_sentiment(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    df = sentiment_df.copy()
    date_col = None
    for column in df.columns:
        if column.lower() in {"date", "datetime", "timestamp", "published"}:
            date_col = column
            break

    if date_col is None:
        raise ValueError("Sentiment data must include a date column")

    if date_col != "Date":
        df = df.rename(columns={date_col: "Date"})

    sentiment_col = None
    for column in df.columns:
        if column.lower() == "sentiment":
            sentiment_col = column
            break

    if sentiment_col is None:
        raise ValueError("Sentiment data must include a sentiment column")

    if sentiment_col != "sentiment":
        df = df.rename(columns={sentiment_col: "sentiment"})

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    return df.dropna(subset=["Date"])


def merge_sentiment(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame | None,
    default_sentiment: float = 0.5,
) -> pd.DataFrame:
    df = _ensure_date_column(price_df)
    df["date_only"] = df["Date"].dt.date

    if sentiment_df is None or sentiment_df.empty:
        df["sentiment"] = default_sentiment
        return df.drop(columns=["date_only"])

    sentiment = _normalize_sentiment(sentiment_df)
    sentiment["date_only"] = sentiment["Date"].dt.date
    sentiment = sentiment[["date_only", "sentiment"]].dropna()

    df = df.merge(sentiment, on="date_only", how="left")
    df["sentiment"] = df["sentiment"].fillna(default_sentiment)
    return df.drop(columns=["date_only"])


def engineer_features(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame | None = None,
    default_sentiment: float = 0.5,
) -> pd.DataFrame:
    df = add_indicators(price_df)
    df = merge_sentiment(df, sentiment_df, default_sentiment=default_sentiment)
    return df
