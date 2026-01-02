from __future__ import annotations

import json
from pathlib import Path

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
    run()

