from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import market_loader
from training import ai_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily walk-forward backtest report.")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--drive-path", default=str(market_loader.DRIVE_ROOT))
    parser.add_argument("--interval", default="1d", choices=["1d"])
    parser.add_argument("--model-dir", default=str(market_loader.DRIVE_ROOT / "models"))
    parser.add_argument("--min-train-days", type=int, default=60)
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args()

    drive_root = Path(args.drive_path)
    model_dir = Path(args.model_dir)
    ticker = args.ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker symbol is required.")

    sentiment_file = drive_root / "processed_sentiment.csv"
    sentiment_path = sentiment_file if sentiment_file.exists() else None

    price_df = market_loader.fetch_market_data(ticker, interval=args.interval)
    report_df = ai_model.daily_walk_forward_train(
        price_df=price_df,
        sentiment_file=sentiment_path,
        default_sentiment=0.0,
        model_dir=model_dir,
        min_train_days=args.min_train_days,
        n_estimators=args.n_estimators,
    )

    if report_df.empty:
        print("No backtest rows generated.")
        return

    report_path = model_dir / "daily_backtest_report.csv"
    display_cols = [
        "date",
        "predicted_1d_change",
        "actual_1d_change",
        "mse_1d",
    ]
    display_df = report_df[display_cols].copy()
    pd.set_option("display.max_rows", 20)
    pd.set_option("display.width", 120)
    print(display_df.tail(20).to_string(index=False))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
