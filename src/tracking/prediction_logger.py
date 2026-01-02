"""Prediction logging and truth tracking for autonomous trading system."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.config import AutonomousConfig

logger = logging.getLogger(__name__)

PREDICTION_LOG_COLUMNS = [
    "run_id",
    "prediction_timestamp",
    "ticker",
    "prediction_1d",
    "prediction_1w",
    "prediction_1m",
    "price_at_prediction",
    "model_mae_1d",
    "model_mae_1w",
    "model_mae_1m",
    "validated",
    "actual_1d_return",
    "actual_1w_return",
    "actual_1m_return",
    "error_1d",
    "error_1w",
    "error_1m",
    "validation_timestamp",
]


@dataclass
class PredictionRecord:
    """Single prediction record for the truth log."""

    ticker: str
    predictions: dict  # {"1d": float, "1w": float, "1m": float}
    price_at_prediction: float
    model_mae: dict  # {"1d": float, "1w": float, "1m": float}
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    prediction_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame row."""
        return {
            "run_id": self.run_id,
            "prediction_timestamp": self.prediction_timestamp.isoformat(),
            "ticker": self.ticker,
            "prediction_1d": self.predictions.get("1d", 0.0),
            "prediction_1w": self.predictions.get("1w", 0.0),
            "prediction_1m": self.predictions.get("1m", 0.0),
            "price_at_prediction": self.price_at_prediction,
            "model_mae_1d": self.model_mae.get("1d", 0.0),
            "model_mae_1w": self.model_mae.get("1w", 0.0),
            "model_mae_1m": self.model_mae.get("1m", 0.0),
            "validated": False,
            "actual_1d_return": None,
            "actual_1w_return": None,
            "actual_1m_return": None,
            "error_1d": None,
            "error_1w": None,
            "error_1m": None,
            "validation_timestamp": None,
        }


def load_prediction_log(path: Path) -> pd.DataFrame:
    """Load existing prediction log or create empty DataFrame with schema."""
    if path.exists():
        try:
            df = pd.read_csv(path)
            for col in PREDICTION_LOG_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            return df
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e}. Starting fresh.")

    return pd.DataFrame(columns=PREDICTION_LOG_COLUMNS)


def _write_log(df: pd.DataFrame, path: Path) -> bool:
    """Write DataFrame to CSV, creating parent directories if needed."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return True
    except Exception as e:
        logger.error(f"Failed to write to {path}: {e}")
        return False


def log_predictions(
    records: List[PredictionRecord],
    config: AutonomousConfig,
) -> bool:
    """
    Append prediction records to CSV in BOTH Google Drive AND local directory.

    Returns True if at least one write succeeded.
    """
    if not records:
        logger.info("No predictions to log.")
        return True

    new_rows = [r.to_dict() for r in records]
    new_df = pd.DataFrame(new_rows)

    success = False

    # Write to local first (more reliable)
    local_df = load_prediction_log(config.local_prediction_log)
    combined_local = pd.concat([local_df, new_df], ignore_index=True)
    if _write_log(combined_local, config.local_prediction_log):
        logger.info(f"Logged {len(records)} predictions to {config.local_prediction_log}")
        success = True

    # Write to Google Drive
    drive_df = load_prediction_log(config.drive_prediction_log)
    combined_drive = pd.concat([drive_df, new_df], ignore_index=True)
    if _write_log(combined_drive, config.drive_prediction_log):
        logger.info(f"Logged {len(records)} predictions to {config.drive_prediction_log}")
        success = True
    else:
        logger.warning("Google Drive write failed, but local backup succeeded.")

    return success


def get_unvalidated_predictions(config: AutonomousConfig) -> pd.DataFrame:
    """Load predictions that haven't been validated yet."""
    df = load_prediction_log(config.drive_prediction_log)
    if df.empty:
        return df
    return df[df["validated"] == False].copy()


def update_validation(
    config: AutonomousConfig,
    updates: pd.DataFrame,
) -> bool:
    """
    Update the prediction log with validation results.

    Args:
        config: Configuration with file paths
        updates: DataFrame with validated rows (must have same index as original)

    Returns True if both writes succeeded.
    """
    success = True

    for log_path in [config.local_prediction_log, config.drive_prediction_log]:
        try:
            df = load_prediction_log(log_path)
            if df.empty:
                continue

            # Update rows by run_id and ticker
            for _, update_row in updates.iterrows():
                mask = (df["run_id"] == update_row["run_id"]) & (
                    df["ticker"] == update_row["ticker"]
                )
                for col in [
                    "validated",
                    "actual_1d_return",
                    "actual_1w_return",
                    "actual_1m_return",
                    "error_1d",
                    "error_1w",
                    "error_1m",
                    "validation_timestamp",
                ]:
                    if col in update_row:
                        df.loc[mask, col] = update_row[col]

            if not _write_log(df, log_path):
                success = False
        except Exception as e:
            logger.error(f"Failed to update {log_path}: {e}")
            success = False

    return success
