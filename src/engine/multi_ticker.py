"""Multi-ticker processing engine for autonomous trading intelligence."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.config import AutonomousConfig
from src.data import market_loader
from src.training import ai_model

logger = logging.getLogger(__name__)


@dataclass
class TickerCycleResult:
    """Result of processing a single ticker."""

    ticker: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    predictions: dict = field(default_factory=dict)  # {"1d": float, "1w": float, "1m": float}
    model_mae: dict = field(default_factory=dict)  # {"1d": float, "1w": float, "1m": float}
    price_at_prediction: float = 0.0
    success: bool = False
    error: Optional[str] = None


def process_single_ticker(
    ticker: str,
    sentiment_file: Path | None,
    model_dir: Path,
    default_sentiment: float = 0.0,
) -> TickerCycleResult:
    """
    Process one ticker: fetch data, engineer features, train models, predict.

    Args:
        ticker: Stock symbol (e.g., "AAPL", "BTC-USD")
        sentiment_file: Path to processed sentiment CSV (shared across tickers)
        model_dir: Directory to save/load per-ticker models
        default_sentiment: Default sentiment value if no data available

    Returns:
        TickerCycleResult with predictions or error information
    """
    result = TickerCycleResult(ticker=ticker)

    try:
        # Step 1: Fetch market data
        logger.info(f"[{ticker}] Fetching market data...")
        price_df = market_loader.fetch_market_data(ticker, interval="1d")

        if price_df.empty:
            result.error = "No market data returned"
            return result

        # Get latest price for logging
        close_col = "Close" if "Close" in price_df.columns else "close"
        result.price_at_prediction = float(price_df[close_col].iloc[-1])

        # Step 2: Train models and get predictions
        logger.info(f"[{ticker}] Training models and generating predictions...")
        predictions, reports, _ = ai_model.train_and_predict(
            price_df=price_df,
            sentiment_file=sentiment_file,
            default_sentiment=default_sentiment,
            model_dir=model_dir,
        )

        result.predictions = predictions
        result.model_mae = {horizon: report.mae for horizon, report in reports.items()}
        result.success = True

        logger.info(
            f"[{ticker}] Predictions: 1d={predictions.get('1d', 0):.4f}, "
            f"1w={predictions.get('1w', 0):.4f}, 1m={predictions.get('1m', 0):.4f}"
        )

    except Exception as e:
        result.error = str(e)
        logger.error(f"[{ticker}] Processing failed: {e}")

    return result


def process_all_tickers(
    tickers: List[str],
    config: AutonomousConfig,
    sentiment_file: Path | None = None,
) -> List[TickerCycleResult]:
    """
    Orchestrate processing for all tickers with shared sentiment.

    Sentiment features are computed from the same news data (global market mood),
    but each ticker gets its own trained model based on its price history.

    Args:
        tickers: List of ticker symbols to process
        config: Autonomous configuration
        sentiment_file: Path to processed sentiment CSV (optional override)

    Returns:
        List of TickerCycleResult for each ticker
    """
    results: List[TickerCycleResult] = []

    # Use sentiment file from config if not explicitly provided
    if sentiment_file is None:
        sentiment_file = config.sentiment_path
        if not sentiment_file.exists():
            logger.warning(f"Sentiment file not found: {sentiment_file}. Using defaults.")
            sentiment_file = None

    # Ensure model directories exist
    config.ensure_directories()

    logger.info(f"Processing {len(tickers)} tickers: {tickers}")

    for ticker in tickers:
        ticker_model_dir = config.ticker_model_dir(ticker)
        result = process_single_ticker(
            ticker=ticker,
            sentiment_file=sentiment_file,
            model_dir=ticker_model_dir,
            default_sentiment=config.sentiment_default,
        )
        results.append(result)

    # Summary
    successful = sum(1 for r in results if r.success)
    logger.info(f"Cycle complete: {successful}/{len(tickers)} tickers processed successfully")

    return results


def fetch_market_data_batch(
    tickers: List[str],
) -> dict[str, pd.DataFrame | None]:
    """
    Fetch market data for multiple tickers.

    Returns dict mapping ticker to DataFrame (or None if fetch failed).
    """
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = market_loader.fetch_market_data(ticker, interval="1d")
        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {e}")
            results[ticker] = None
    return results
