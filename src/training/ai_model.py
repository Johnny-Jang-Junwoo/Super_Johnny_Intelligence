from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import feature_engineer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"

HORIZON_PERIODS = {
    "1d": 1,
    "1w": 5,
    "1m": 21,
}
TARGET_COLUMNS = {
    "1d": "target_1d_change",
    "1w": "target_1w_change",
    "1m": "target_1m_change",
}


@dataclass
class TrainingReport:
    horizon: str
    mae: float
    feature_importance: pd.DataFrame
    model_path: Path


def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "timestamp" not in data.columns:
        if "Date" in data.columns:
            data["timestamp"] = pd.to_datetime(data["Date"], errors="coerce", utc=True)
        elif "date" in data.columns:
            data["timestamp"] = pd.to_datetime(data["date"], errors="coerce", utc=True)
        else:
            raise ValueError("Price data must include a timestamp or Date column.")
    else:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)

    return data.dropna(subset=["timestamp"])


def _get_feature_columns() -> list[str]:
    sentiment_cols = feature_engineer.get_sentiment_feature_columns()
    return ["ema", "rsi"] + sentiment_cols


def add_targets(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    close = df["close"] if "close" in df.columns else df["Close"]

    for horizon, periods in HORIZON_PERIODS.items():
        target_col = TARGET_COLUMNS[horizon]
        df[target_col] = (close.shift(-periods) / close) - 1

    return df


def build_features(
    price_df: pd.DataFrame,
    sentiment_file: Path | str | None = None,
    default_sentiment: float = 0.0,
) -> pd.DataFrame:
    features_df = feature_engineer.engineer_features(
        price_df=price_df,
        sentiment_file=sentiment_file,
        default_sentiment=default_sentiment,
    )
    return _ensure_timestamp(features_df)


def prepare_training_frame(
    price_df: pd.DataFrame,
    sentiment_file: Path | str | None = None,
    default_sentiment: float = 0.0,
) -> pd.DataFrame:
    features_df = build_features(price_df, sentiment_file, default_sentiment=default_sentiment)
    features_df = add_targets(features_df)

    feature_cols = _get_feature_columns()
    targets = list(TARGET_COLUMNS.values())
    features_df = features_df.dropna(subset=feature_cols + targets)

    if features_df.empty:
        raise ValueError("No training rows available after feature and target preparation")

    return features_df


def walk_forward_train(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_months: int = 6,
) -> tuple[float, pd.Series]:
    data = df.sort_values("timestamp").copy()
    data["month"] = data["timestamp"].dt.to_period("M")
    months = sorted(data["month"].unique())

    if len(months) <= train_months:
        raise ValueError("Not enough history for walk-forward training")

    maes: list[float] = []
    importances: list[pd.Series] = []

    for idx in range(train_months, len(months)):
        train_month_set = months[idx - train_months : idx]
        test_month = months[idx]

        train_df = data[data["month"].isin(train_month_set)]
        test_df = data[data["month"] == test_month]

        if train_df.empty or test_df.empty:
            continue

        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(train_df[feature_cols], train_df[target_col])

        preds = model.predict(test_df[feature_cols])
        maes.append(mean_absolute_error(test_df[target_col], preds))
        importances.append(pd.Series(model.feature_importances_, index=feature_cols))

    if not maes:
        raise ValueError("Walk-forward training produced no evaluation windows")

    avg_mae = float(sum(maes) / len(maes))
    avg_importance = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    return avg_mae, avg_importance


def train_regressor(
    df: pd.DataFrame,
    horizon: str,
    model_dir: Path | None = None,
) -> TrainingReport:
    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = _get_feature_columns()
    target_col = TARGET_COLUMNS[horizon]

    mae, importance = walk_forward_train(df, feature_cols, target_col)

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(df[feature_cols], df[target_col])

    model_path = model_dir / f"model_next_{horizon}.pkl"
    joblib.dump(model, model_path)

    importance_df = importance.reset_index()
    importance_df.columns = ["feature", "importance"]

    return TrainingReport(
        horizon=horizon,
        mae=mae,
        feature_importance=importance_df,
        model_path=model_path,
    )


def predict_latest(model: RandomForestRegressor, df: pd.DataFrame) -> float:
    feature_cols = _get_feature_columns()
    latest = df.dropna(subset=feature_cols).iloc[[-1]][feature_cols]
    return float(model.predict(latest)[0])


def train_and_predict(
    price_df: pd.DataFrame,
    sentiment_file: Path | str | None = None,
    default_sentiment: float = 0.0,
    model_dir: Path | None = None,
) -> tuple[dict[str, float], dict[str, TrainingReport], pd.DataFrame]:
    training_df = prepare_training_frame(
        price_df=price_df,
        sentiment_file=sentiment_file,
        default_sentiment=default_sentiment,
    )

    predictions: dict[str, float] = {}
    reports: dict[str, TrainingReport] = {}

    for horizon in TARGET_COLUMNS:
        report = train_regressor(training_df, horizon, model_dir=model_dir)
        model = joblib.load(report.model_path)
        predictions[horizon] = predict_latest(model, training_df)
        reports[horizon] = report

    save_feature_importance_report(reports.values(), model_dir)

    return predictions, reports, training_df


def execute_rolling_study(
    price_df: pd.DataFrame,
    sentiment_file: Path | str | None = None,
    default_sentiment: float = 0.0,
    model_dir: Path | None = None,
    min_train_days: int = 60,
    n_estimators: int = 200,
) -> pd.DataFrame:
    return daily_walk_forward_train(
        price_df=price_df,
        sentiment_file=sentiment_file,
        default_sentiment=default_sentiment,
        model_dir=model_dir,
        min_train_days=min_train_days,
        n_estimators=n_estimators,
    )


def daily_walk_forward_train(
    price_df: pd.DataFrame,
    sentiment_file: Path | str | None = None,
    default_sentiment: float = 0.0,
    model_dir: Path | None = None,
    min_train_days: int = 60,
    n_estimators: int = 200,
) -> pd.DataFrame:
    features_df = build_features(
        price_df=price_df,
        sentiment_file=sentiment_file,
        default_sentiment=default_sentiment,
    )
    features_df = add_targets(features_df)
    features_df = _ensure_timestamp(features_df)

    feature_cols = _get_feature_columns()
    features_df = features_df.dropna(subset=feature_cols)
    features_df = features_df.sort_values("timestamp").reset_index(drop=True)

    report_rows: list[dict[str, float | str]] = []

    for idx in range(min_train_days, len(features_df)):
        train_df = features_df.iloc[:idx]
        test_row = features_df.iloc[idx]
        date_str = test_row["timestamp"].date().isoformat()

        row_result: dict[str, float | str] = {"date": date_str}

        for horizon, target_col in TARGET_COLUMNS.items():
            train_subset = train_df.dropna(subset=[target_col])
            actual = test_row.get(target_col)

            if train_subset.empty or pd.isna(actual):
                row_result[f"predicted_{horizon}_change"] = float("nan")
                row_result[f"actual_{horizon}_change"] = float("nan")
                row_result[f"mse_{horizon}"] = float("nan")
                continue

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(train_subset[feature_cols], train_subset[target_col])

            test_features = test_row[feature_cols].to_frame().T
            prediction = float(model.predict(test_features)[0])
            actual_value = float(actual)
            mse_value = mean_squared_error([actual_value], [prediction])

            row_result[f"predicted_{horizon}_change"] = prediction
            row_result[f"actual_{horizon}_change"] = actual_value
            row_result[f"mse_{horizon}"] = mse_value

        report_rows.append(row_result)

    report_df = pd.DataFrame(report_rows)
    if model_dir is not None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(model_dir / "daily_backtest_report.csv", index=False)

    return report_df


def save_feature_importance_report(reports: Iterable[TrainingReport], model_dir: Path | None) -> None:
    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    rows = []
    for report in reports:
        for _, row in report.feature_importance.iterrows():
            rows.append(
                {
                    "horizon": report.horizon,
                    "feature": row["feature"],
                    "importance": row["importance"],
                    "mae": report.mae,
                }
            )
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(model_dir / "feature_importance.csv", index=False)


def find_latest_model(model_dir: Path | None = None) -> Path:
    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    models = list(model_dir.glob("model_next_*.pkl"))
    if not models:
        raise FileNotFoundError(f"No .pkl models found in {model_dir}")
    return max(models, key=lambda p: p.stat().st_mtime)


def load_model(model_path: Path) -> RandomForestRegressor:
    return joblib.load(Path(model_path))
