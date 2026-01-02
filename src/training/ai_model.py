from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from . import feature_engineer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"

FEATURE_COLUMNS = ["ema", "rsi", "sentiment_score"]
DATE_COLUMNS = ["date", "datetime", "timestamp", "time"]
LABEL_COLUMNS = ["signal", "action", "side", "trade_action", "position"]

SIGNAL_MAP = {
    "buy": 1,
    "long": 1,
    "sell": -1,
    "short": -1,
    "hold": 0,
    "flat": 0,
}


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


def _map_signal(value) -> int | None:
    if value is None or pd.isna(value):
        return None

    if isinstance(value, str):
        text = value.strip().lower()
        if text in SIGNAL_MAP:
            return SIGNAL_MAP[text]
        try:
            numeric = float(text)
        except ValueError:
            return None
        return int(numeric) if numeric in (-1, 0, 1) else None

    if isinstance(value, (int, float)) and value in (-1, 0, 1):
        return int(value)

    return None


def _extract_labels(trade_df: pd.DataFrame) -> pd.DataFrame:
    label_col = _find_column(trade_df.columns.tolist(), LABEL_COLUMNS)
    if label_col is None:
        raise ValueError("Trade log must include a signal/action column for behavioral cloning")

    labels = trade_df[["Date", label_col]].copy()
    labels["signal"] = labels[label_col].apply(_map_signal)
    labels = labels.dropna(subset=["signal"])

    if labels.empty:
        raise ValueError("No valid trade signals found in trade log")

    labels["signal"] = labels["signal"].astype(int)
    return labels[["Date", "signal"]]


def build_features(
    price_df: pd.DataFrame,
    sentiment_file: Path | str | None = None,
    default_sentiment: float = 0.0,
) -> pd.DataFrame:
    return feature_engineer.engineer_features(
        price_df=price_df,
        sentiment_file=sentiment_file,
        default_sentiment=default_sentiment,
    )


def prepare_training_data(
    price_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    sentiment_file: Path | str | None = None,
    default_sentiment: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features_df = build_features(price_df, sentiment_file, default_sentiment=default_sentiment)
    features_df["Date"] = pd.to_datetime(features_df["Date"], errors="coerce")
    features_df = features_df.dropna(subset=["Date"])

    normalized_trades = normalize_trade_log(trade_df)
    labels_df = _extract_labels(normalized_trades)
    labels_df["Date"] = pd.to_datetime(labels_df["Date"], errors="coerce")
    labels_df = labels_df.dropna(subset=["Date"])

    features_df["date_only"] = features_df["Date"].dt.date
    labels_df["date_only"] = labels_df["Date"].dt.date
    labels_daily = labels_df.groupby("date_only", as_index=False).last()

    training_df = features_df.merge(labels_daily[["date_only", "signal"]], on="date_only", how="left")
    training_df = training_df.drop(columns=["date_only"]).dropna(subset=["signal"])

    training_df = training_df.dropna(subset=FEATURE_COLUMNS + ["signal"])
    if training_df.empty:
        raise ValueError("No training rows after merging trade signals with features")

    return training_df, features_df


def train_model(
    training_df: pd.DataFrame,
    model_dir: Path | None = None,
) -> tuple[RandomForestClassifier, Path]:
    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pkl"

    X = training_df[FEATURE_COLUMNS]
    y = training_df["signal"].astype(int)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X, y)

    joblib.dump(model, model_path)
    return model, model_path


def predict_latest(model: RandomForestClassifier, features_df: pd.DataFrame) -> int:
    valid = features_df.dropna(subset=FEATURE_COLUMNS)
    if valid.empty:
        raise ValueError("No valid feature rows available for prediction")

    latest = valid.iloc[[-1]][FEATURE_COLUMNS]
    return int(model.predict(latest)[0])


def train_and_predict(
    trade_df: pd.DataFrame,
    price_df: pd.DataFrame,
    sentiment_file: Path | str | None = None,
    default_sentiment: float = 0.0,
    model_dir: Path | None = None,
) -> tuple[RandomForestClassifier, Path, int]:
    training_df, features_df = prepare_training_data(
        price_df=price_df,
        trade_df=trade_df,
        sentiment_file=sentiment_file,
        default_sentiment=default_sentiment,
    )
    model, model_path = train_model(training_df, model_dir=model_dir)
    signal = predict_latest(model, features_df)
    return model, model_path, signal


def find_latest_model(model_dir: Path | None = None) -> Path:
    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    models = list(model_dir.glob("*.pkl"))
    if not models:
        raise FileNotFoundError(f"No .pkl models found in {model_dir}")
    return max(models, key=lambda p: p.stat().st_mtime)


def load_model(model_path: Path) -> RandomForestClassifier:
    return joblib.load(Path(model_path))
