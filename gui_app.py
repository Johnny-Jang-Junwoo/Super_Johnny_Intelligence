from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis import news_processor
from data import market_loader, news_fetcher
from training import ai_model, feature_engineer

DRIVE_ROOT = market_loader.DRIVE_ROOT
RAW_NEWS_PATH = DRIVE_ROOT / "raw_news.csv"
PROCESSED_PATH = DRIVE_ROOT / "processed_sentiment.csv"
MODEL_DIR = DRIVE_ROOT / "models"


@dataclass
class CycleResult:
    market_mood_score: float
    ai_signal: str
    latest_sentiment_score: float
    model_path: str
    model_time: str
    predictions: dict[str, float]
    alignment_score: int
    alignment_label: str
    price_df: pd.DataFrame | None
    features_df: pd.DataFrame | None
    news_df: pd.DataFrame | None
    warnings: list[str]


st.set_page_config(
    page_title="Super Johnny Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    :root {
        --bg: #0f172a;
        --panel: #1e293b;
        --panel-strong: #111827;
        --accent: #10b981;
        --danger: #ef4444;
        --muted: #94a3b8;
        --text: #e2e8f0;
    }
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text);
    }
    .stApp {
        background: radial-gradient(1200px 600px at 10% -20%, #1e293b 0%, #0f172a 55%, #0b1220 100%);
    }
    .metric-card {
        background: var(--panel);
        border: 1px solid rgba(148, 163, 184, 0.2);
        padding: 18px 20px;
        border-radius: 16px;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.35);
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--muted);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-top: 6px;
    }
    .metric-sub {
        margin-top: 4px;
        font-size: 0.9rem;
        color: var(--muted);
    }
    .sidebar-section {
        background: var(--panel-strong);
        padding: 14px 16px;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_news_table() -> pd.DataFrame:
    if PROCESSED_PATH.exists():
        df = pd.read_csv(PROCESSED_PATH)
    elif RAW_NEWS_PATH.exists():
        df = pd.read_csv(RAW_NEWS_PATH)
    else:
        return pd.DataFrame()

    timestamp_col = next(
        (col for col in ["timestamp", "date", "published", "datetime"] if col in df.columns),
        None,
    )
    if timestamp_col:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.sort_values(timestamp_col, ascending=False)
    return df


def load_price_data(ticker: str, interval: str) -> pd.DataFrame:
    return market_loader.fetch_market_data(ticker, interval=interval)


def get_latest_model_info(model_dir: Path) -> tuple[str, str]:
    try:
        model_path = ai_model.find_latest_model(model_dir)
        mtime = datetime.fromtimestamp(model_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        return str(model_path), mtime
    except Exception:
        return "No model found", ""


def latest_sentiment_score() -> float:
    if not PROCESSED_PATH.exists():
        return 0.0
    df = pd.read_csv(PROCESSED_PATH)
    if "sentiment" not in df.columns:
        return 0.0
    date_col = next(
        (col for col in ["timestamp", "date", "published", "datetime"] if col in df.columns),
        None,
    )
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        if df.empty:
            return 0.0
        df = df.sort_values(date_col)
        latest_date = df[date_col].dt.date.iloc[-1]
        df = df[df[date_col].dt.date == latest_date]
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    df = df.dropna(subset=["sentiment"])
    if df.empty:
        return 0.0
    return float(df["sentiment"].mean())


def mood_label(score: float) -> str:
    if score <= -0.2:
        return "Bearish"
    if score >= 0.2:
        return "Bullish"
    return "Neutral"


def signal_label_from_return(prediction: float | None) -> str:
    if prediction is None:
        return "HOLD"
    if prediction > 0.002:
        return "BUY"
    if prediction < -0.002:
        return "SELL"
    return "HOLD"


def format_move(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def alignment_summary(features_df: pd.DataFrame) -> tuple[int, str]:
    if features_df is None or features_df.empty:
        return 50, "Unknown"

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
        return 50, "Unknown"

    short_mean = float(pd.Series(short_vals).mean())
    long_mean = float(pd.Series(long_vals).mean())
    if short_mean * long_mean < 0:
        return 25, "Low Conviction"

    diff = abs(short_mean - long_mean)
    score = int(round(70 + 30 * (1 - min(1.0, diff))))
    return score, "Aligned"


def load_predictions_from_models(model_dir: Path, features_df: pd.DataFrame) -> dict[str, float]:
    predictions: dict[str, float] = {}
    for horizon in ai_model.TARGET_COLUMNS:
        model_path = model_dir / f"model_next_{horizon}.pkl"
        if not model_path.exists():
            continue
        model = ai_model.load_model(model_path)
        predictions[horizon] = ai_model.predict_latest(model, features_df)
    return predictions


def run_cycle(ticker: str, interval: str) -> CycleResult:
    warnings: list[str] = []

    try:
        news_fetcher.fetch_firehose()
    except Exception as exc:
        warnings.append(f"News fetch failed: {exc}")

    try:
        news_processor.process_news(DRIVE_ROOT)
    except Exception as exc:
        warnings.append(f"News processing failed: {exc}")

    model_path, model_time = get_latest_model_info(MODEL_DIR)
    ai_signal = "HOLD"
    predictions: dict[str, float] = {}
    alignment_score = 50
    alignment_label = "Unknown"
    price_df = None
    features_df = None

    try:
        ticker = ticker or "SPY"
        price_df = market_loader.fetch_market_data(ticker, interval=interval)
        sentiment_file = PROCESSED_PATH if PROCESSED_PATH.exists() else None

        features_df = ai_model.build_features(
            price_df=price_df,
            sentiment_file=sentiment_file,
            default_sentiment=0.0,
        )
        alignment_score, alignment_label = alignment_summary(features_df)

        predictions, reports, _training_df = ai_model.train_and_predict(
            price_df=price_df,
            sentiment_file=sentiment_file,
            default_sentiment=0.0,
            model_dir=MODEL_DIR,
        )
        if reports:
            latest_report = max(reports.values(), key=lambda report: report.model_path.stat().st_mtime)
            model_path = str(latest_report.model_path)
            model_time = datetime.fromtimestamp(latest_report.model_path.stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M"
            )
    except Exception as exc:
        warnings.append(f"Training failed: {exc}")
        try:
            if features_df is None and price_df is not None:
                sentiment_file = PROCESSED_PATH if PROCESSED_PATH.exists() else None
                features_df = ai_model.build_features(
                    price_df=price_df,
                    sentiment_file=sentiment_file,
                    default_sentiment=0.0,
                )
            if features_df is not None:
                alignment_score, alignment_label = alignment_summary(features_df)
            if features_df is not None:
                predictions = load_predictions_from_models(MODEL_DIR, features_df)
        except Exception as inner_exc:
            warnings.append(f"Model fallback failed: {inner_exc}")

    pred_1d = predictions.get("1d")
    ai_signal = signal_label_from_return(pred_1d)

    sentiment_score = latest_sentiment_score()

    return CycleResult(
        market_mood_score=sentiment_score,
        ai_signal=ai_signal,
        latest_sentiment_score=sentiment_score,
        model_path=model_path,
        model_time=model_time,
        predictions=predictions,
        alignment_score=alignment_score,
        alignment_label=alignment_label,
        price_df=price_df,
        features_df=features_df,
        news_df=None,
        warnings=warnings,
    )


def get_executor() -> ThreadPoolExecutor:
    if "executor" not in st.session_state:
        st.session_state.executor = ThreadPoolExecutor(max_workers=1)
    return st.session_state.executor


def start_cycle(ticker: str, interval: str) -> None:
    executor = get_executor()
    st.session_state.run_future = executor.submit(run_cycle, ticker, interval)
    st.session_state.run_started = datetime.now().strftime("%H:%M:%S")


def consume_future() -> CycleResult | None:
    future = st.session_state.get("run_future")
    if not future:
        return None
    if future.done():
        result = future.result()
        st.session_state.last_result = result
        st.session_state.run_future = None
        return result
    return None


with st.sidebar:
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.subheader("Controls")
    ticker_input = st.text_input("Ticker", value=st.session_state.get("ticker", "SPY"))
    interval_input = st.selectbox("Interval", ["1d"], index=0)
    run_clicked = st.button("Run Daily Cycle")
    refresh_clicked = st.button("Refresh Status")
    st.markdown("</div>", unsafe_allow_html=True)

    model_path, model_time = get_latest_model_info(MODEL_DIR)
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.subheader("Model")
    st.caption(model_path)
    if model_time:
        st.caption(f"Last trained: {model_time}")
    st.markdown("</div>", unsafe_allow_html=True)

if run_clicked:
    st.session_state.ticker = ticker_input
    start_cycle(ticker_input, interval_input)

if refresh_clicked:
    st.experimental_rerun()

consume_future()

result: CycleResult | None = st.session_state.get("last_result")
if result is None:
    sentiment_score = latest_sentiment_score()
    model_path, model_time = get_latest_model_info(MODEL_DIR)
    result = CycleResult(
        market_mood_score=sentiment_score,
        ai_signal="HOLD",
        latest_sentiment_score=sentiment_score,
        model_path=model_path,
        model_time=model_time,
        predictions={},
        alignment_score=50,
        alignment_label="Unknown",
        price_df=None,
        features_df=None,
        news_df=load_news_table(),
        warnings=[],
    )

if st.session_state.get("run_future"):
    st.info("Daily cycle running in the background. Click 'Refresh Status' to update.")

if result.warnings:
    st.warning(" | ".join(result.warnings))

st.title("Super Johnny Intelligence")

col1, col2, col3 = st.columns(3)

mood_text = mood_label(result.market_mood_score)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Market Mood</div>
            <div class="metric-value">{result.market_mood_score:.2f}</div>
            <div class="metric-sub">{mood_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    signal_color = "#10B981" if result.ai_signal == "BUY" else "#EF4444" if result.ai_signal == "SELL" else "#94A3B8"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">AI Signal</div>
            <div class="metric-value" style="color: {signal_color}">{result.ai_signal}</div>
            <div class="metric-sub">Model output</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Latest Sentiment</div>
            <div class="metric-value">{result.latest_sentiment_score:.2f}</div>
            <div class="metric-sub">From news scores</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("## Predicted Movement")
pred_1d = result.predictions.get("1d") if result.predictions else None
pred_1w = result.predictions.get("1w") if result.predictions else None
pred_1m = result.predictions.get("1m") if result.predictions else None

pm1, pm2, pm3 = st.columns(3)
pm1.metric("Next 1D", format_move(pred_1d))
pm2.metric("Next 1W", format_move(pred_1w))
pm3.metric("Next 1M", format_move(pred_1m))

st.caption(f"Strategy Alignment: {result.alignment_label} ({result.alignment_score}/100)")

st.markdown("## Market Chart")

chart_placeholder = st.empty()

ticker_for_chart = ticker_input or "SPY"
try:
    price_df = result.price_df or load_price_data(ticker_for_chart, interval_input)
    sentiment_file = PROCESSED_PATH if PROCESSED_PATH.exists() else None
    features_df = result.features_df or feature_engineer.engineer_features(
        price_df, sentiment_file, default_sentiment=0.0
    )
    if "timestamp" not in features_df.columns:
        if "Date" in features_df.columns:
            features_df["timestamp"] = pd.to_datetime(features_df["Date"], errors="coerce")
        elif "date" in features_df.columns:
            features_df["timestamp"] = pd.to_datetime(features_df["date"], errors="coerce")
        else:
            raise ValueError("Missing timestamp/Date column for charting.")
    features_df = features_df.dropna(subset=["timestamp"]).sort_values("timestamp")

    close_series = features_df["close"] if "close" in features_df.columns else features_df["Close"]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
    )
    fig.add_trace(
        go.Scatter(
            x=features_df["timestamp"],
            y=close_series,
            name="Close",
            line=dict(color="#E2E8F0", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=features_df["timestamp"],
            y=features_df["ema"],
            name="EMA",
            line=dict(color="#10B981", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=features_df["timestamp"],
            y=features_df["rsi"],
            name="RSI",
            line=dict(color="#F59E0B", width=2),
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=520,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])

    chart_placeholder.plotly_chart(fig, width="stretch")
except Exception as exc:
    chart_placeholder.error(f"Chart unavailable: {exc}")

st.markdown("## News Firehose")
news_df = result.news_df if result.news_df is not None else load_news_table()
if news_df.empty:
    st.info("No news data available yet.")
else:
    display_cols = [
        col
        for col in ["timestamp", "headline", "sentiment", "impact", "sector", "reasoning"]
        if col in news_df.columns
    ]
    st.dataframe(
        news_df[display_cols].head(50),
        width="stretch",
        height=320,
    )
