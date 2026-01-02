"""
Firehose News Fetcher (RSS)
Aggregates hundreds of financial headlines from 30+ global sources.
"""
from __future__ import annotations

import os
from datetime import datetime

import feedparser
import pandas as pd
from dateutil import parser as date_parser

# --- CONFIGURATION ---
DRIVE_FOLDER = r"G:\My Drive\SuperJohnnyTrader"
OUTPUT_FILE = os.path.join(DRIVE_FOLDER, "raw_news.csv")

# --- THE FIREHOSE LIST ---
# Categorized feeds to give the AI context
RSS_FEEDS = {
    "MACRO_ECONOMY": [
        "https://feeds.content.dowjones.io/public/rss/mw_topstories",  # MarketWatch Top Stories
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # CNBC Top News
        "https://finance.yahoo.com/news/rssindex",  # Yahoo Finance
        "https://www.investing.com/rss/news.rss",  # Investing.com All News
        "https://www.federalreserve.gov/feeds/press_all.xml",  # Federal Reserve (Rates/Policy)
    ],
    "STOCKS_TECH": [
        "http://feeds.marketwatch.com/marketwatch/technology",  # MW Tech
        "https://www.cnbc.com/id/19854910/device/rss/rss.html",  # CNBC Tech
        "https://techcrunch.com/feed/",  # TechCrunch
        "https://www.theverge.com/rss/index.xml",  # The Verge (Product news)
    ],
    "CRYPTO": [
        "https://cointelegraph.com/rss",  # CoinTelegraph
        "https://www.coindesk.com/arc/outboundfeeds/rss/",  # CoinDesk
        "https://decrypt.co/feed",  # Decrypt
        "https://bitcoinmagazine.com/.rss/full/",  # Bitcoin Magazine
        "https://theblock.co/rss.xml",  # The Block
    ],
    "FOREX_COMMODITIES": [
        "https://www.dailyfx.com/feeds/market-news",  # DailyFX
        "https://www.investing.com/rss/commodities.rss",  # Commodities (Gold/Oil)
        "https://www.investing.com/rss/forex_news.rss",  # Forex News
    ],
}


def clean_date(date_str: str) -> str:
    try:
        dt = date_parser.parse(date_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fetch_firehose() -> None:
    print("=" * 60)
    print("🔥 IGNITING NEWS FIREHOSE...")
    print("=" * 60)

    all_headlines: list[dict[str, str]] = []

    for category, urls in RSS_FEEDS.items():
        print(f"\n📡 Scanning Category: {category}")
        for url in urls:
            try:
                feed = feedparser.parse(url)
                count = len(feed.entries)
                print(f"   • {feed.feed.get('title', url)[:30]}... ({count} articles)")

                for entry in feed.entries:
                    all_headlines.append(
                        {
                            "date": clean_date(entry.get("published", str(datetime.now()))),
                            "category": category,  # Keep category for the AI!
                            "headline": entry.get("title", "").strip(),
                            "link": entry.get("link", ""),
                            "source": feed.feed.get("title", "Unknown"),
                        }
                    )
            except Exception as exc:
                print(f"   ❌ Failed: {url} ({str(exc)[:20]}...)")

    # --- PROCESSING ---
    new_df = pd.DataFrame(all_headlines)

    if new_df.empty:
        print("\n⚠️  No news found. Check internet connection.")
        return

    # --- SAVE TO DRIVE (Deduplication) ---
    if os.path.exists(OUTPUT_FILE):
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            # Combine
            combined_df = pd.concat([existing_df, new_df])
            # Remove duplicates based on Headline (keep newest)
            combined_df = combined_df.drop_duplicates(subset=["headline"], keep="last")
            # Sort by date (newest first)
            combined_df = combined_df.sort_values("date", ascending=False)
        except Exception as exc:
            print(f"⚠️  Could not read existing file: {exc}. Creating new one.")
            combined_df = new_df
    else:
        combined_df = new_df

    combined_df.to_csv(OUTPUT_FILE, index=False)

    print("=" * 60)
    print("📝 STATS:")
    print(f"   • Fetched Today: {len(new_df)}")
    print(f"   • Total Database: {len(combined_df)}")
    print(f"   • Location: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    fetch_firehose()
