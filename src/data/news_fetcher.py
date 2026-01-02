"""
Firehose News Fetcher (RSS)
Aggregates financial headlines from multiple sources.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import feedparser
import pandas as pd
from dateutil import parser as date_parser

DRIVE_FOLDER = Path(r"G:\My Drive\SuperJohnnyTrader")
OUTPUT_FILE = DRIVE_FOLDER / "raw_news.csv"

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

CATEGORY_REGION = {
    "MACRO_ECONOMY": "US",
    "STOCKS_TECH": "US",
    "CRYPTO": "GLOBAL",
    "FOREX_COMMODITIES": "GLOBAL",
}

GLOBAL_DOMAINS = {
    "investing.com",
    "coindesk.com",
    "cointelegraph.com",
    "decrypt.co",
    "bitcoinmagazine.com",
    "theblock.co",
    "dailyfx.com",
}

TICKER_PATTERN = re.compile(r"\$(?P<ticker>[A-Z]{1,5})\b")
EXCHANGE_PATTERN = re.compile(r"\b(?:NYSE|NASDAQ):\s*[A-Z]{1,5}\b")


def clean_timestamp(entry) -> str:
    raw = entry.get("published") or entry.get("updated")
    if raw:
        try:
            dt = date_parser.parse(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except Exception:
            pass

    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        try:
            dt = datetime(*parsed[:6], tzinfo=timezone.utc)
            return dt.isoformat()
        except Exception:
            pass

    return datetime.now(timezone.utc).isoformat()


def infer_region(headline: str, source_url: str, category: str) -> str:
    if TICKER_PATTERN.search(headline) or EXCHANGE_PATTERN.search(headline):
        return "TICKER"

    domain = urlparse(source_url).netloc.replace("www.", "")
    if domain in GLOBAL_DOMAINS:
        return "GLOBAL"

    return CATEGORY_REGION.get(category, "US")


def fetch_firehose() -> None:
    print("=" * 60)
    print("Starting news firehose...")
    print("=" * 60)

    all_headlines: list[dict[str, str]] = []

    for category, urls in RSS_FEEDS.items():
        print(f"Scanning Category: {category}")
        for url in urls:
            try:
                feed = feedparser.parse(url)
                count = len(feed.entries)
                title = feed.feed.get("title", url)
                print(f"  {title[:40]}... ({count} articles)")

                for entry in feed.entries:
                    headline = str(entry.get("title", "")).strip()
                    if not headline:
                        continue

                    all_headlines.append(
                        {
                            "timestamp": clean_timestamp(entry),
                            "category": category,
                            "region": infer_region(headline, url, category),
                            "headline": headline,
                            "link": str(entry.get("link", "")).strip(),
                            "source": str(feed.feed.get("title", "Unknown")),
                        }
                    )
            except Exception as exc:
                print(f"  Failed: {url} ({str(exc)[:50]}...)")

    new_df = pd.DataFrame(all_headlines)

    if new_df.empty:
        print("No news found. Check internet connection.")
        return

    DRIVE_FOLDER.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["headline"], keep="last")
            combined_df = combined_df.sort_values("timestamp", ascending=False)
        except Exception as exc:
            print(f"Could not read existing file: {exc}. Creating new one.")
            combined_df = new_df
    else:
        combined_df = new_df

    combined_df.to_csv(OUTPUT_FILE, index=False)

    print("=" * 60)
    print("STATS:")
    print(f"  Fetched Today: {len(new_df)}")
    print(f"  Total Database: {len(combined_df)}")
    print(f"  Location: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    fetch_firehose()
