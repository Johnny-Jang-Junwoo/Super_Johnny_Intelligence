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
from training import ai_model, feature_engineer


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

def _recommendation_from_pred(prediction: float | None) -> str:
    if prediction is None:
        return "HOLD"
    if prediction > 0.002:
        return "BUY"
    if prediction < -0.002:
        return "SELL"
    return "HOLD"


def _format_move(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _conviction_label(value: float | None) -> str:
    if value is None:
        return "No Signal"
    magnitude = abs(value)
    if magnitude >= 0.01:
        return "High Conviction"
    if magnitude >= 0.005:
        return "Medium Conviction"
    return "Low Conviction"


def _direction_label(value: float | None, positive: str, negative: str, neutral: str = "Neutral") -> str:
    if value is None:
        return "No Signal"
    if value > 0:
        return positive
    if value < 0:
        return negative
    return neutral


def _top_sentiment_feature(report: ai_model.TrainingReport) -> tuple[str, float] | None:
    if report.feature_importance.empty:
        return None
    sent_mask = report.feature_importance["feature"].astype(str).str.startswith("sent_")
    sent_df = report.feature_importance[sent_mask]
    if sent_df.empty:
        return None
    top_row = sent_df.iloc[0]
    return str(top_row["feature"]), float(top_row["importance"])


def _sentiment_alignment(features_df: pd.DataFrame) -> tuple[int, str, float | None, float | None]:
    if features_df.empty:
        return 50, "Unknown", None, None

    short_windows = ["1h", "1d"]
    long_windows = ["1w", "1m"]
    short_cols = [
        f"sent_{region.lower()}_{window}"
        for region in feature_engineer.REGIONS
        for window in short_windows
    ]
    long_cols = [
        f"sent_{region.lower()}_{window}"
        for region in feature_engineer.REGIONS
        for window in long_windows
    ]

    latest = features_df.iloc[-1]
    short_vals = [
        float(latest[col])
        for col in short_cols
        if col in features_df.columns and pd.notna(latest[col])
    ]
    long_vals = [
        float(latest[col])
        for col in long_cols
        if col in features_df.columns and pd.notna(latest[col])
    ]

    if not short_vals or not long_vals:
        return 50, "Unknown", None, None

    short_mean = float(pd.Series(short_vals).mean())
    long_mean = float(pd.Series(long_vals).mean())

    if short_mean * long_mean < 0:
        return 25, "Low Conviction", short_mean, long_mean

    diff = abs(short_mean - long_mean)
    score = int(round(70 + 30 * (1 - min(1.0, diff))))
    return score, "Aligned", short_mean, long_mean


def _load_predictions_from_models(
    model_dir: Path,
    features_df: pd.DataFrame,
) -> dict[str, float]:
    predictions: dict[str, float] = {}
    for horizon in ai_model.TARGET_COLUMNS:
        model_path = model_dir / f"model_next_{horizon}.pkl"
        if not model_path.exists():
            continue
        model = ai_model.load_model(model_path)
        predictions[horizon] = ai_model.predict_latest(model, features_df)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the daily trading cycle.")
    parser.add_argument("--ticker", default="SPY", help="Ticker symbol for market data.")
    parser.add_argument("--drive-path", default=str(market_loader.DRIVE_ROOT))
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

    ticker = args.ticker or "SPY"
    price_df = market_loader.fetch_market_data(ticker, interval=args.interval)

    features_df = ai_model.build_features(
        price_df=price_df,
        sentiment_file=sentiment_file,
        default_sentiment=0.0,
    )

    predictions: dict[str, float] = {}
    reports: dict[str, ai_model.TrainingReport] = {}
    try:
        predictions, reports, _training_df = ai_model.train_and_predict(
            price_df=price_df,
            sentiment_file=sentiment_file,
            default_sentiment=0.0,
            model_dir=model_dir,
        )
        for report in reports.values():
            print(f"Model trained ({report.horizon}): {report.model_path} (MAE {report.mae:.4f})")
            top_sent = _top_sentiment_feature(report)
            if top_sent:
                feature, importance = top_sent
                print(f"Top sentiment feature ({report.horizon}): {feature} ({importance:.3f})")
    except Exception as exc:
        print(f"Training failed ({exc}); attempting to load latest models.")
        try:
            predictions = _load_predictions_from_models(model_dir, features_df)
            if predictions:
                print(f"Loaded models from: {model_dir}")
        except Exception as inner_exc:
            print(f"Fallback model load failed ({inner_exc}); defaulting to HOLD.")
            predictions = {}

    mood = _mood_label(sentiment_score)
    pred_1d = predictions.get("1d")
    pred_1w = predictions.get("1w")
    pred_1m = predictions.get("1m")
    recommendation = _recommendation_from_pred(pred_1d)

    alignment_score, alignment_label, short_mean, long_mean = _sentiment_alignment(features_df)

    summary = (
        f"Market Mood: {mood} ({sentiment_score:.2f}). "
        f"AI Recommendation: {recommendation}. "
        f"Strategy Alignment: {alignment_label} ({alignment_score}/100)."
    )
    if sector:
        summary += f" Sector: {sector}."
    if short_mean is not None and long_mean is not None:
        summary += f" Short Sent: {short_mean:.2f}, Long Sent: {long_mean:.2f}."
    print(summary)

    print(
        f"Short-term (1D) Prediction: {_format_move(pred_1d)} ({_conviction_label(pred_1d)})."
    )
    print(
        f"Mid-term (1W) Prediction: {_format_move(pred_1w)} "
        f"({_direction_label(pred_1w, 'Macro Push', 'Macro Drag')})."
    )
    print(
        f"Long-term (1M) Prediction: {_format_move(pred_1m)} "
        f"({_direction_label(pred_1m, 'Macro Tailwind', 'Macro Headwind')})."
    )


if __name__ == "__main__":
    main()
