from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from urllib.parse import quote

import feedparser
import ollama
import pandas as pd

DRIVE_PATH = Path(r"G:\My Drive\SuperJohnnyTrader")
RAW_NEWS_FILE = DRIVE_PATH / "raw_news.csv"
OUTPUT_FILE = DRIVE_PATH / "processed_sentiment.csv"

PROMPT_TEMPLATE = """
[FINANCIAL ANALYST MISSION]
Score the following headline on a continuous scale for the US STOCK MARKET.

HEADLINE: "{headline}"

[SCORING CRITERIA]
- SENTIMENT: Use any decimal between -1.00 and 1.00.
  Be precise. (e.g., -0.12 for slightly bad, -0.95 for a total crash).
- IMPACT: 0 to 5.
  CRITICAL: If the news is about personal relationships, family, or non-business advice, the IMPACT MUST BE 0.

[DIVERSITY EXAMPLES - DO NOT ONLY USE THESE NUMBERS]
- "Inflation drops 0.1%": Sentiment 0.15, Impact 4
- "Minor software bug in iPhone": Sentiment -0.05, Impact 1
- "Nvidia earnings beat by 20%": Sentiment 0.88, Impact 5
- "Family feud over inheritance": Sentiment -0.40, Impact 0 (Irrelevant to market)

[OUTPUT]
Return ONLY JSON:
{{
    "sentiment": float,
    "impact": int,
    "sector": "string",
    "reasoning": "why this specific score?"
}}
"""


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    return df.columns[0]


def _dedupe_key(headline: str, link: str) -> str:
    key = link or headline
    return key.strip().lower()


def _parse_timestamp(entry) -> str:
    raw = entry.get("published") or entry.get("updated")
    if raw:
        parsed = pd.to_datetime(raw, utc=True, errors="coerce")
        if not pd.isna(parsed):
            return parsed.isoformat()
    parsed_struct = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed_struct:
        try:
            dt = pd.Timestamp(*parsed_struct[:6], tz="UTC")
            return dt.isoformat()
        except Exception:
            pass
    return pd.Timestamp.utcnow().isoformat()


def analyze_headline(headline: str, model: str = "llama3.1") -> dict:
    prompt = PROMPT_TEMPLATE.format(headline=headline)
    content = ""

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.1},
        )

        content = response.get("message", {}).get("content", "")
        data = json.loads(content)

        return {
            "sentiment": float(data.get("sentiment", 0.0)),
            "impact": int(data.get("impact", 0)),
            "sector": str(data.get("sector", "Unknown")),
            "reasoning": str(data.get("reasoning", "No reason provided")),
        }
    except Exception as exc:
        preview = content.replace("\n", " ")[:200]
        print(f"PARSE ERROR on headline: {headline[:30]}...")
        print(f"Raw AI Response: {preview or 'NO RESPONSE'}")
        print(f"Error: {exc}")
        return {
            "sentiment": 0.0,
            "impact": 0,
            "sector": "Error",
            "reasoning": f"Parse Fail: {str(exc)[:50]}",
        }


def backfill_ticker_news(ticker: str, days: int = 365) -> int:
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker symbol is required for backfill mode.")

    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
    existing_keys: set[str] = set()
    existing_columns = ["timestamp", "category", "region", "headline", "link", "source"]

    if RAW_NEWS_FILE.exists():
        existing_df = pd.read_csv(RAW_NEWS_FILE)
        existing_columns = list(existing_df.columns)
        headline_col = "headline" if "headline" in existing_df.columns else _pick_column(existing_df, ["headline"])
        link_col = "link" if "link" in existing_df.columns else _pick_column(existing_df, ["link"])
        for _, row in existing_df.iterrows():
            headline = row.get(headline_col, "")
            link = row.get(link_col, "")
            headline = "" if pd.isna(headline) else str(headline).strip()
            link = "" if pd.isna(link) else str(link).strip()
            key = _dedupe_key(headline, link)
            if key:
                existing_keys.add(key)

    query = f"{ticker} stock when:{days}d"
    encoded_query = quote(query)
    feed_urls = [
        f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en",
    ]

    new_rows = []
    for url in feed_urls:
        feed = feedparser.parse(url)
        source = feed.feed.get("title", "Google News")
        for entry in feed.entries:
            headline = str(entry.get("title", "")).strip()
            if not headline:
                continue
            link = str(entry.get("link", "")).strip()
            key = _dedupe_key(headline, link)
            if not key or key in existing_keys:
                continue

            timestamp = _parse_timestamp(entry)
            parsed_ts = pd.to_datetime(timestamp, utc=True, errors="coerce")
            if pd.isna(parsed_ts) or parsed_ts < cutoff:
                continue

            new_rows.append(
                {
                    "timestamp": timestamp,
                    "category": "TICKER_BACKFILL",
                    "region": "TICKER",
                    "headline": headline,
                    "link": link,
                    "source": str(source),
                }
            )
            existing_keys.add(key)

    if not new_rows:
        print("No new backfill headlines found.")
        return 0

    new_df = pd.DataFrame(new_rows)
    RAW_NEWS_FILE.parent.mkdir(parents=True, exist_ok=True)

    if RAW_NEWS_FILE.exists():
        combined_df = pd.concat([pd.read_csv(RAW_NEWS_FILE), new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["headline"], keep="last")
    else:
        combined_df = new_df

    for column in existing_columns:
        if column not in combined_df.columns:
            combined_df[column] = ""
    combined_df = combined_df.sort_values("timestamp", ascending=False)
    combined_df.to_csv(RAW_NEWS_FILE, index=False)

    print(f"Backfill complete. Added {len(new_df)} headlines to {RAW_NEWS_FILE}")
    return len(new_df)


def run(model: str = "llama3.1") -> None:
    if not RAW_NEWS_FILE.exists():
        raise FileNotFoundError(f"Missing raw news file at {RAW_NEWS_FILE}")

    news_df = pd.read_csv(RAW_NEWS_FILE)
    headline_col = _pick_column(news_df, ["headline", "title", "news", "text"])
    date_col = _pick_column(news_df, ["date", "datetime", "timestamp", "published"]) if not news_df.empty else "date"

    results = []
    for _, row in news_df.iterrows():
        headline = str(row.get(headline_col, "")).strip()
        if not headline:
            results.append(
                {
                    "Date": row.get(date_col),
                    "headline": headline,
                    "ai_sentiment": 0.0,
                    "ai_impact": 0,
                    "ai_reasoning": "No reasoning provided",
                    "sector": "Unknown",
                }
            )
            continue

        try:
            score_data = analyze_headline(headline, model=model)
        except Exception:
            score_data = {
                "reasoning": "Parse Fail",
                "sentiment": 0.0,
                "impact": 0,
                "sector": "Error",
            }

        results.append(
            {
                "Date": row.get(date_col),
                "headline": headline,
                "ai_sentiment": score_data.get("sentiment", 0.0),
                "ai_impact": score_data.get("impact", 0),
                "ai_reasoning": score_data.get("reasoning", "No reasoning provided"),
                "sector": score_data.get("sector", "Unknown"),
            }
        )

    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="News librarian utilities.")
    parser.add_argument("--mode", choices=["process", "backfill"], default="process")
    parser.add_argument("--model", default="llama3.1")
    parser.add_argument("--ticker", help="Ticker symbol for backfill mode.")
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()

    if args.mode == "backfill":
        backfill_ticker_news(args.ticker or "", days=args.days)
    else:
        run(model=args.model)

