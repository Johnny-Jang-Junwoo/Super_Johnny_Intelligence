from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis import news_processor
from data import market_loader, news_fetcher
from training import ai_model


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def _latest_sentiment_context(processed_path: Path) -> tuple[float, str]:
    if not processed_path.exists():
        return 0.0, "unknown"

    df = pd.read_csv(processed_path)
    if "sentiment" not in df.columns or df.empty:
        return 0.0, "unknown"

    date_col = _find_column(df, ["timestamp", "published", "date", "datetime"])
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        if df.empty:
            return 0.0, "unknown"
        df = df.sort_values(date_col)
        row = df.iloc[-1]
    else:
        row = df.iloc[-1]

    sentiment = pd.to_numeric(row.get("sentiment", 0.0), errors="coerce")
    if pd.isna(sentiment):
        sentiment = 0.0

    sector = row.get("sector", "unknown")
    sector = "unknown" if pd.isna(sector) else str(sector).strip()

    return float(sentiment), sector


def _mood_label(score: float) -> str:
    if score <= -0.2:
        return "Bearish"
    if score >= 0.2:
        return "Bullish"
    return "Neutral"


def _signal_label(signal: int) -> str:
    return {1: "BUY", -1: "SELL", 0: "HOLD"}.get(signal, "HOLD")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the daily trading cycle.")
    parser.add_argument("--ticker", help="Ticker symbol for market data.")
    parser.add_argument("--drive-path", default=str(market_loader.DRIVE_ROOT))
    parser.add_argument("--trade-pattern", default=market_loader.TRADE_PATTERN)
    parser.add_argument("--interval", default="1d", choices=["1d"])
    parser.add_argument("--model-dir", default=str(market_loader.DRIVE_ROOT / "models"))
    args = parser.parse_args()

    drive_root = Path(args.drive_path)
    model_dir = Path(args.model_dir)

    news_ok = False
    try:
        news_fetcher.fetch_firehose()
        print("News fetch completed.")
        news_ok = True
    except Exception as exc:
        print(f"News fetch failed ({exc}); continuing with technicals only.")

    sentiment_ok = False
    if news_ok:
        try:
            processed_count = news_processor.process_news(drive_root)
            print(f"News processing completed: {processed_count} headlines scored.")
            sentiment_ok = True
        except Exception as exc:
            print(f"News processing failed ({exc}); defaulting sentiment to 0.")

    processed_path = drive_root / "processed_sentiment.csv"
    if sentiment_ok:
        sentiment_score, sector = _latest_sentiment_context(processed_path)
    else:
        sentiment_score, sector = 0.0, "unknown"
    sentiment_file = processed_path if sentiment_ok and processed_path.exists() else None

    trade_df, trade_path = market_loader.load_latest_trade_log(drive_root, pattern=args.trade_pattern)
    trade_df = market_loader.normalize_trade_log(trade_df)

    ticker = args.ticker or market_loader.infer_ticker(trade_df)
    if not ticker:
        raise ValueError("Ticker not provided and not found in trade logs")

    trade_dates = trade_df["Date"].dropna()
    start = trade_dates.min() if not trade_dates.empty else None
    end = trade_dates.max() + pd.Timedelta(days=1) if not trade_dates.empty else None

    price_df = market_loader.fetch_market_data(ticker, start=start, end=end, interval=args.interval)

    try:
        model, model_path, signal = ai_model.train_and_predict(
            trade_df=trade_df,
            price_df=price_df,
            sentiment_file=sentiment_file,
            default_sentiment=0.0,
            model_dir=model_dir,
        )
        print(f"Model trained: {model_path}")
        print(f"Using trade log: {trade_path}")
    except Exception as exc:
        print(f"Training failed ({exc}); attempting to load latest model.")
        try:
            model_path = ai_model.find_latest_model(model_dir)
            model = ai_model.load_model(model_path)
            features_df = ai_model.build_features(
                price_df=price_df,
                sentiment_file=sentiment_file,
                default_sentiment=0.0,
            )
            signal = ai_model.predict_latest(model, features_df)
            print(f"Loaded model: {model_path}")
        except Exception as inner_exc:
            print(f"Fallback model load failed ({inner_exc}); defaulting to HOLD.")
            signal = 0

    mood = _mood_label(sentiment_score)
    recommendation = _signal_label(signal)

    summary = f"Market Mood: {mood} ({sentiment_score:.2f}). AI Recommendation: {recommendation}."
    if sector:
        summary += f" Sector: {sector}."
    print(summary)


if __name__ == "__main__":
    main()
