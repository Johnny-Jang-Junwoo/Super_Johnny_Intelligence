from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Sequence

import ollama

PROMPT_TEMPLATE = """
[SYSTEM ROLE]
You are a Senior Macro Quant Strategist. Your goal is to predict price direction over a 1-day to 1-week horizon.
You ignore short-term noise and focus on "Information Density" and "Market Alignment."

[INPUT DATA PACKET]
- TARGET TICKER: {ticker_symbol}
- TICKER NEWS: {ticker_headlines_list}
- MACRO ENVIRONMENT: {global_market_sentiment_score} (Scale -1 to 1)
- TECHNICAL CONTEXT: Volume is {relative_volume}x average; 5-day trend is {price_trend}.
- FUNDAMENTAL HEALTH: {earnings_summary_or_risk_factors}
- CALENDAR: Month: {current_month}, Quarter-End: {is_quarter_end}

[ANALYSIS ALGORITHM]
1. ALIGNMENT: Does the ticker news match the Macro Environment? (e.g., Good news in a Bear market is a "Trap").
2. MOMENTUM: If Relative Volume > 1.5x and Sentiment is < -0.6, flag as "High Conviction Capitulation."
3. FUNDAMENTAL ANCHOR: Does the news contradict the long-term financial health?
4. SEASONALITY: Adjust for historical volatility (e.g., September/October weakness).

[OUTPUT REQUIREMENT]
Return ONLY a JSON object:
{{
    "trade_direction": "LONG | SHORT | NEUTRAL",
    "conviction_score": int (0-100),
    "expected_horizon": "1-day | 3-day | 1-week",
    "macro_weight": float (How much the overall market is dragging/pushing this ticker),
    "thesis": "A 2-sentence explanation of the interaction between news, volume, and macro."
}}
"""


@dataclass
class MacroPromptInputs:
    ticker_symbol: str
    ticker_headlines_list: Sequence[str] | str
    global_market_sentiment_score: float
    relative_volume: float
    price_trend: str
    earnings_summary_or_risk_factors: str
    current_month: str
    is_quarter_end: str

    def normalized_headlines(self) -> str:
        if isinstance(self.ticker_headlines_list, str):
            return self.ticker_headlines_list
        return " | ".join(str(item).strip() for item in self.ticker_headlines_list if str(item).strip())


def analyze_macro_signal(inputs: MacroPromptInputs, model: str = "llama3.1") -> dict:
    prompt = PROMPT_TEMPLATE.format(
        ticker_symbol=inputs.ticker_symbol,
        ticker_headlines_list=inputs.normalized_headlines(),
        global_market_sentiment_score=inputs.global_market_sentiment_score,
        relative_volume=inputs.relative_volume,
        price_trend=inputs.price_trend,
        earnings_summary_or_risk_factors=inputs.earnings_summary_or_risk_factors,
        current_month=inputs.current_month,
        is_quarter_end=inputs.is_quarter_end,
    )

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
    except Exception as exc:
        preview = content.replace("\n", " ")[:200]
        print(f"PARSE ERROR for {inputs.ticker_symbol}: {exc}. Raw preview: {preview}")
        return {
            "trade_direction": "NEUTRAL",
            "conviction_score": 0,
            "expected_horizon": "1-day",
            "macro_weight": 0.0,
            "thesis": "Parse Fail",
        }

    trade_direction = str(data.get("trade_direction", "NEUTRAL")).upper()
    conviction = int(data.get("conviction_score", 0))
    expected_horizon = str(data.get("expected_horizon", "1-day"))
    macro_weight = float(data.get("macro_weight", 0.0))
    thesis = str(data.get("thesis", ""))

    if trade_direction not in {"LONG", "SHORT", "NEUTRAL"}:
        trade_direction = "NEUTRAL"
    conviction = max(0, min(100, conviction))

    return {
        "trade_direction": trade_direction,
        "conviction_score": conviction,
        "expected_horizon": expected_horizon,
        "macro_weight": macro_weight,
        "thesis": thesis,
    }


def summarize_macro_signal(signals: Iterable[dict]) -> dict:
    signals = list(signals)
    if not signals:
        return {
            "trade_direction": "NEUTRAL",
            "conviction_score": 0,
            "expected_horizon": "1-day",
            "macro_weight": 0.0,
            "thesis": "No signals available.",
        }

    direction_votes = {"LONG": 0, "SHORT": 0, "NEUTRAL": 0}
    total_conviction = 0
    macro_weights = []

    for signal in signals:
        direction = str(signal.get("trade_direction", "NEUTRAL")).upper()
        if direction in direction_votes:
            direction_votes[direction] += 1
        total_conviction += int(signal.get("conviction_score", 0))
        macro_weights.append(float(signal.get("macro_weight", 0.0)))

    final_direction = max(direction_votes, key=direction_votes.get)
    avg_conviction = int(total_conviction / len(signals))
    avg_macro_weight = sum(macro_weights) / len(macro_weights) if macro_weights else 0.0

    return {
        "trade_direction": final_direction,
        "conviction_score": avg_conviction,
        "expected_horizon": "1-week",
        "macro_weight": avg_macro_weight,
        "thesis": "Aggregate macro signal across inputs.",
    }
