from __future__ import annotations

import json
from pathlib import Path

import ollama
import pandas as pd

DRIVE_ROOT = Path(r"G:\My Drive\SuperJohnnyTrader")
RAW_NEWS_PATH = DRIVE_ROOT / "raw_news.csv"
PROCESSED_PATH = DRIVE_ROOT / "processed_sentiment.csv"

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


def _dedupe_key(headline: str, link: str) -> str:
    key = link or headline
    return key.strip().lower()


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def analyze_headline(headline: str, model: str = "llama3.1") -> dict:
    """
    Sends the headline to the local LLM using Ollama's JSON mode.
    """
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


def process_news(drive_root: Path = DRIVE_ROOT, model: str = "llama3.1") -> int:
    drive_root = Path(drive_root)
    raw_path = drive_root / "raw_news.csv"
    processed_path = drive_root / "processed_sentiment.csv"

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw news file at {raw_path}")

    raw_df = pd.read_csv(raw_path)
    headline_col = _find_column(raw_df, ["headline", "title", "news", "text"])
    timestamp_col = _find_column(raw_df, ["timestamp", "published", "date", "datetime"])

    if headline_col is None:
        raise ValueError("raw_news.csv must include a headline column")

    existing_keys: set[str] = set()
    if processed_path.exists():
        processed_df = pd.read_csv(processed_path)
        for _, row in processed_df.iterrows():
            headline = row.get("headline", "")
            link = row.get("link", "")
            headline = "" if pd.isna(headline) else str(headline).strip()
            link = "" if pd.isna(link) else str(link).strip()
            key = _dedupe_key(headline, link)
            if key:
                existing_keys.add(key)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    new_keys: set[str] = set()
    processed_count = 0
    start_index = 0

    print("Starting real-time processing...")

    for i in range(start_index, len(raw_df)):
        row = raw_df.iloc[i]
        headline = row.get(headline_col, "")
        headline = "" if pd.isna(headline) else str(headline).strip()
        if not headline:
            continue

        link = row.get("link", "") if "link" in raw_df.columns else ""
        source = row.get("source", "") if "source" in raw_df.columns else ""
        link = "" if pd.isna(link) else str(link).strip()
        source = "" if pd.isna(source) else str(source).strip()
        timestamp = row.get(timestamp_col) if timestamp_col else None

        key = _dedupe_key(headline, link)
        if not key or key in existing_keys or key in new_keys:
            continue

        print(f"   [{i + 1}/{len(raw_df)}] Analyzing: {headline[:50]}...", end="", flush=True)
        score_data = analyze_headline(headline, model=model)
        print(f" -> Done! (Sent: {score_data.get('sentiment', 0.0)})", flush=True)

        new_row = pd.DataFrame(
            [
                {
                    "timestamp": timestamp,
                    "headline": headline,
                    "link": link,
                    "source": source,
                    "sentiment": score_data.get("sentiment", 0.0),
                    "impact": score_data.get("impact", 0),
                    "sector": score_data.get("sector", "Unknown"),
                    "reasoning": score_data.get("reasoning", "N/A"),
                }
            ]
        )

        file_exists = processed_path.exists()
        new_row.to_csv(processed_path, mode="a", index=False, header=not file_exists)
        new_keys.add(key)
        processed_count += 1

    return processed_count


if __name__ == "__main__":
    count = process_news()
    print(f"Processed {count} headlines into {PROCESSED_PATH}")

