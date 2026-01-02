#!/usr/bin/env python
"""
Prediction Validation Script

Validates matured predictions by comparing them to actual price movements.

Maturity windows:
- 1d predictions: 1 trading day (use 2 calendar days to be safe)
- 1w predictions: 5 trading days (use 7 calendar days)
- 1m predictions: 21 trading days (use 30 calendar days)

Usage:
    python validate_predictions.py
    python validate_predictions.py --dry-run  # Preview without updating
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AutonomousConfig
from src.data import market_loader
from src.tracking.prediction_logger import (
    load_prediction_log,
    update_validation,
    PREDICTION_LOG_COLUMNS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Maturity periods in calendar days (slightly longer than trading days for safety)
MATURITY_CALENDAR_DAYS = {
    "1d": 2,   # 1 trading day = ~2 calendar days (accounts for weekends)
    "1w": 8,   # 5 trading days = ~7-8 calendar days
    "1m": 32,  # 21 trading days = ~30-32 calendar days
}


def fetch_price_at_date(ticker: str, target_date: datetime) -> float | None:
    """
    Fetch the closing price for a ticker at or near a target date.

    Returns the close price, or None if data unavailable.
    """
    try:
        # Fetch a small window around the target date
        start = (target_date - timedelta(days=5)).strftime("%Y-%m-%d")
        end = (target_date + timedelta(days=5)).strftime("%Y-%m-%d")

        df = market_loader.fetch_market_data(ticker, start=start, end=end)
        if df.empty:
            return None

        # Find closest date to target
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        # Get prices on or before target date
        before_target = df[df["Date"] <= target_date]
        if before_target.empty:
            # If no data before, use first available
            close_col = "Close" if "Close" in df.columns else "close"
            return float(df[close_col].iloc[0])

        close_col = "Close" if "Close" in before_target.columns else "close"
        return float(before_target[close_col].iloc[-1])

    except Exception as e:
        logger.warning(f"Failed to fetch price for {ticker} at {target_date}: {e}")
        return None


def calculate_actual_return(
    ticker: str,
    prediction_time: datetime,
    horizon_days: int,
) -> float | None:
    """
    Calculate the actual return for a ticker from prediction time.

    Args:
        ticker: Stock symbol
        prediction_time: When the prediction was made
        horizon_days: Number of calendar days to look ahead

    Returns:
        Actual return as decimal (e.g., 0.05 for 5%), or None if unavailable
    """
    # Get price at prediction time
    price_at_pred = fetch_price_at_date(ticker, prediction_time)
    if price_at_pred is None or price_at_pred == 0:
        return None

    # Get price at horizon
    horizon_date = prediction_time + timedelta(days=horizon_days)
    price_at_horizon = fetch_price_at_date(ticker, horizon_date)
    if price_at_horizon is None:
        return None

    # Calculate return
    return (price_at_horizon / price_at_pred) - 1


def validate_matured_predictions(
    config: AutonomousConfig,
    dry_run: bool = False,
) -> int:
    """
    Validate predictions that have matured.

    Checks if enough time has passed for each horizon and calculates actual returns.

    Args:
        config: Configuration with file paths
        dry_run: If True, preview changes without saving

    Returns:
        Number of predictions validated
    """
    # Load prediction log
    df = load_prediction_log(config.drive_prediction_log)
    if df.empty:
        logger.info("No predictions in log to validate.")
        return 0

    # Filter to unvalidated predictions
    unvalidated = df[df["validated"] == False].copy()
    if unvalidated.empty:
        logger.info("All predictions already validated.")
        return 0

    logger.info(f"Found {len(unvalidated)} unvalidated predictions")

    now = datetime.now(timezone.utc)
    validated_count = 0
    updates = []

    for idx, row in unvalidated.iterrows():
        try:
            pred_time = pd.to_datetime(row["prediction_timestamp"])
            if pred_time.tzinfo is None:
                pred_time = pred_time.replace(tzinfo=timezone.utc)

            ticker = row["ticker"]
            age_days = (now - pred_time).days

            # Check each horizon
            horizons_validated = 0
            update_row = row.copy()

            for horizon, maturity_days in MATURITY_CALENDAR_DAYS.items():
                actual_col = f"actual_{horizon}_return"
                error_col = f"error_{horizon}"
                pred_col = f"prediction_{horizon}"

                # Skip if already has actual value
                if pd.notna(row.get(actual_col)):
                    horizons_validated += 1
                    continue

                # Check if matured
                if age_days < maturity_days:
                    continue

                # Calculate actual return
                actual = calculate_actual_return(ticker, pred_time, maturity_days)
                if actual is not None:
                    update_row[actual_col] = actual
                    predicted = row.get(pred_col, 0)
                    update_row[error_col] = abs(actual - predicted) if pd.notna(predicted) else None
                    horizons_validated += 1

                    logger.info(
                        f"[{ticker}] {horizon}: predicted={predicted*100:.2f}%, "
                        f"actual={actual*100:.2f}%, error={abs(actual-predicted)*100:.2f}%"
                    )

            # Mark as validated if all horizons are done
            # (either they have values or haven't matured yet)
            all_matured = age_days >= max(MATURITY_CALENDAR_DAYS.values())
            if all_matured or horizons_validated == len(MATURITY_CALENDAR_DAYS):
                update_row["validated"] = True
                update_row["validation_timestamp"] = now.isoformat()
                validated_count += 1

            updates.append(update_row)

        except Exception as e:
            logger.error(f"Error validating row {idx}: {e}")
            continue

    if not updates:
        logger.info("No predictions ready for validation.")
        return 0

    # Apply updates
    updates_df = pd.DataFrame(updates)

    if dry_run:
        logger.info(f"DRY RUN: Would validate {validated_count} predictions")
        print("\nPending updates:")
        print(updates_df[["ticker", "prediction_timestamp", "validated",
                         "actual_1d_return", "actual_1w_return", "actual_1m_return"]].to_string())
    else:
        update_validation(config, updates_df)
        logger.info(f"Validated {validated_count} predictions")

    return validated_count


def print_validation_stats(config: AutonomousConfig) -> None:
    """Print statistics about validated predictions."""
    df = load_prediction_log(config.drive_prediction_log)
    if df.empty:
        print("No predictions in log.")
        return

    total = len(df)
    validated = df["validated"].sum()

    print("\n" + "=" * 60)
    print("VALIDATION STATISTICS")
    print("=" * 60)
    print(f"Total predictions: {total}")
    print(f"Validated: {validated} ({validated/total*100:.1f}%)")
    print(f"Pending: {total - validated}")

    if validated > 0:
        validated_df = df[df["validated"] == True]

        for horizon in ["1d", "1w", "1m"]:
            error_col = f"error_{horizon}"
            actual_col = f"actual_{horizon}_return"
            pred_col = f"prediction_{horizon}"

            if error_col in validated_df.columns:
                errors = validated_df[error_col].dropna()
                if not errors.empty:
                    mae = errors.mean()
                    print(f"\n{horizon.upper()} Horizon:")
                    print(f"  MAE: {mae*100:.3f}%")
                    print(f"  Samples: {len(errors)}")

                    # Direction accuracy
                    if actual_col in validated_df.columns and pred_col in validated_df.columns:
                        actuals = validated_df[actual_col].dropna()
                        preds = validated_df.loc[actuals.index, pred_col]
                        correct_dir = ((actuals > 0) == (preds > 0)).sum()
                        print(f"  Direction accuracy: {correct_dir/len(actuals)*100:.1f}%")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Validate matured predictions")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview validation without updating files",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show validation statistics only",
    )
    parser.add_argument(
        "--drive-path",
        type=str,
        default=None,
        help="Override Google Drive root path",
    )
    args = parser.parse_args()

    config = AutonomousConfig()
    if args.drive_path:
        config.drive_root = Path(args.drive_path)

    if args.stats:
        print_validation_stats(config)
        return

    validated = validate_matured_predictions(config, dry_run=args.dry_run)
    print(f"\nValidated {validated} predictions")

    if not args.dry_run:
        print_validation_stats(config)


if __name__ == "__main__":
    main()
