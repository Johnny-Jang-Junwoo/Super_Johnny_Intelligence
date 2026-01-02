#!/usr/bin/env python
"""
Autonomous Multi-Ticker Trading Intelligence Engine

Usage:
    python run_autonomous.py --tickers SPY AAPL TSLA --run-every-hours 24
    python run_autonomous.py --run-once  # Single run, no loop
    python run_autonomous.py  # Uses default tickers, runs every 24 hours
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AutonomousConfig, DEFAULT_TICKERS
from src.engine.multi_ticker import process_all_tickers, TickerCycleResult
from src.tracking.prediction_logger import PredictionRecord, log_predictions

# Graceful shutdown handling
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logging.info("Shutdown signal received. Completing current cycle...")
    shutdown_requested = True


def setup_logging(config: AutonomousConfig) -> None:
    """Configure logging to both console and file."""
    log_file = config.local_root / "autonomous.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )


def fetch_and_process_news(config: AutonomousConfig) -> bool:
    """Fetch news and process sentiment. Returns True if successful."""
    try:
        from src.data import news_fetcher
        from src.analysis import news_processor

        logging.info("Fetching news firehose...")
        news_fetcher.fetch_firehose()  # Uses module-level DRIVE_FOLDER constant

        logging.info("Processing news sentiment...")
        news_processor.process_news(drive_root=config.drive_root)

        return True
    except Exception as e:
        logging.error(f"News processing failed: {e}")
        return False


def run_validation(config: AutonomousConfig) -> int:
    """Run validation on matured predictions. Returns count validated."""
    try:
        # Import here to avoid circular imports
        from validate_predictions import validate_matured_predictions
        return validate_matured_predictions(config)
    except ImportError:
        logging.warning("Validation script not found. Skipping validation.")
        return 0
    except Exception as e:
        logging.error(f"Validation failed: {e}")
        return 0


def results_to_records(results: list[TickerCycleResult], run_id: str) -> list[PredictionRecord]:
    """Convert TickerCycleResult objects to PredictionRecord objects."""
    records = []
    timestamp = datetime.now(timezone.utc)

    for result in results:
        if not result.success:
            continue

        record = PredictionRecord(
            ticker=result.ticker,
            predictions=result.predictions,
            price_at_prediction=result.price_at_prediction,
            model_mae=result.model_mae,
            run_id=run_id,
            prediction_timestamp=timestamp,
        )
        records.append(record)

    return records


def run_cycle(config: AutonomousConfig) -> bool:
    """
    Execute one complete autonomous cycle.

    Steps:
    1. Fetch and process news (shared sentiment)
    2. Process all tickers (per-ticker models)
    3. Log predictions to truth log
    4. Validate matured predictions

    Returns True on success.
    """
    import uuid

    run_id = str(uuid.uuid4())[:8]
    cycle_start = datetime.now(timezone.utc)

    logging.info(f"{'=' * 60}")
    logging.info(f"Cycle {run_id} started at {cycle_start.isoformat()}")
    logging.info(f"Tickers: {config.tickers}")
    logging.info(f"{'=' * 60}")

    try:
        # Step 1: News and sentiment (shared across all tickers)
        news_ok = fetch_and_process_news(config)
        if not news_ok:
            logging.warning("News processing failed. Continuing with existing sentiment data.")

        # Step 2: Process all tickers
        results = process_all_tickers(config.tickers, config)

        # Step 3: Log predictions
        records = results_to_records(results, run_id)
        if records:
            log_predictions(records, config)
            logging.info(f"Logged {len(records)} predictions to truth log")
        else:
            logging.warning("No successful predictions to log")

        # Step 4: Print summary
        print_cycle_summary(results)

        # Step 5: Validate past predictions
        validated_count = run_validation(config)
        if validated_count > 0:
            logging.info(f"Validated {validated_count} matured predictions")

        cycle_duration = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        logging.info(f"Cycle {run_id} complete in {cycle_duration:.1f}s")

        return True

    except Exception as e:
        logging.error(f"Cycle {run_id} failed: {e}", exc_info=True)
        return False


def print_cycle_summary(results: list[TickerCycleResult]) -> None:
    """Print a summary of the cycle results."""
    print("\n" + "=" * 60)
    print("CYCLE SUMMARY")
    print("=" * 60)

    for result in results:
        status = "OK" if result.success else f"FAILED: {result.error}"
        print(f"\n{result.ticker}: {status}")

        if result.success:
            print(f"  Price: ${result.price_at_prediction:.2f}")
            print(f"  Predictions:")
            for horizon, pred in result.predictions.items():
                mae = result.model_mae.get(horizon, 0)
                direction = "UP" if pred > 0 else "DOWN" if pred < 0 else "FLAT"
                print(f"    {horizon}: {pred*100:+.2f}% ({direction}) [MAE: {mae:.4f}]")

    successful = sum(1 for r in results if r.success)
    print(f"\n{'=' * 60}")
    print(f"Total: {successful}/{len(results)} tickers processed successfully")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Multi-Ticker Trading Intelligence Engine"
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help=f"Ticker symbols to process (default: {DEFAULT_TICKERS})",
    )
    parser.add_argument(
        "--run-every-hours",
        type=float,
        default=24.0,
        help="Hours between cycles (default: 24.0)",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run a single cycle and exit",
    )
    parser.add_argument(
        "--drive-path",
        type=str,
        default=None,
        help="Override Google Drive root path",
    )
    args = parser.parse_args()

    # Build configuration
    config = AutonomousConfig(
        tickers=args.tickers,
        run_every_hours=args.run_every_hours,
    )
    if args.drive_path:
        config.drive_root = Path(args.drive_path)

    # Setup logging
    setup_logging(config)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logging.info("Super Johnny Intelligence - Autonomous Engine Starting")
    logging.info(f"Tickers: {config.tickers}")
    logging.info(f"Run interval: {config.run_every_hours} hours")
    logging.info(f"Drive root: {config.drive_root}")

    # Ensure directories exist
    config.ensure_directories()

    if args.run_once:
        logging.info("Single run mode")
        success = run_cycle(config)
        sys.exit(0 if success else 1)

    # Continuous loop
    sleep_seconds = config.run_every_hours * 3600

    logging.info("Continuous mode - Press Ctrl+C to stop gracefully")

    while not shutdown_requested:
        run_cycle(config)

        if shutdown_requested:
            break

        next_run = datetime.now(timezone.utc).timestamp() + sleep_seconds
        next_run_dt = datetime.fromtimestamp(next_run, tz=timezone.utc)
        logging.info(f"Next cycle at {next_run_dt.isoformat()} (sleeping {config.run_every_hours} hours)")

        # Interruptible sleep (check every 60 seconds)
        while time.time() < next_run and not shutdown_requested:
            time.sleep(min(60, next_run - time.time()))

    logging.info("Autonomous engine shut down gracefully.")


if __name__ == "__main__":
    main()
