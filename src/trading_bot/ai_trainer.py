from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FEATURE_COLUMNS = ["rsi_14", "ema_12", "ema_26", "volume_change", "sentiment"]


def train_model(
    data: pd.DataFrame,
    target_col: str = "signal",
    model_path: Path = Path("models/model.pkl"),
) -> RandomForestClassifier:
    missing = [col for col in FEATURE_COLUMNS + [target_col] if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = data[FEATURE_COLUMNS]
    y = data[target_col]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X, y)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model


def load_model(model_path: Path) -> RandomForestClassifier:
    return joblib.load(Path(model_path))
