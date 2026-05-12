def score_news_sentiment(news_items, ticker, groq_client):
    """
    Step 7: Score news sentiment using Groq (Llama 3.3 70B).

    We send up to 18 headlines to the LLM and ask for a single +1/0/-1 signal.
    This uses very few tokens (headlines only, no full articles) so it barely
    touches the Groq free tier quota.

    Returns:
    {
        "score": -1 | 0 | 1,
        "sentiment": "POSITIVE" | "NEUTRAL" | "NEGATIVE",
        "reason": str,
        "items_scored": int
    }
    """
    import json

    if not news_items:
        return {"score": 0, "sentiment": "NEUTRAL", "reason": "No news available", "items_scored": 0}

    # build a compact headline list — titles only, no URLs or metadata
    headlines = "\n".join([
        f"- {n.get('title', '').strip()}"
        for n in news_items[:18]
        if n.get("title")
    ])

    if not headlines:
        return {"score": 0, "sentiment": "NEUTRAL", "reason": "No readable headlines", "items_scored": 0}

    prompt = f"""You are scoring recent news for {ticker} to assess revenue outlook.
Rate the overall revenue sentiment from these headlines.
Return ONLY valid JSON — nothing else, no explanation outside the JSON.

Format: {{"score": -1, "reason": "one sentence"}}
score must be: -1 (negative for revenue), 0 (neutral/mixed), or 1 (positive for revenue)

Headlines:
{headlines}"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        raw = response.choices[0].message.content.strip()

        # handle cases where the model wraps JSON in markdown code blocks
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()

        result = json.loads(raw)
        score = int(result.get("score", 0))
        score = max(-1, min(1, score))  # clamp to -1/0/1

        sentiment_map = {1: "POSITIVE", 0: "NEUTRAL", -1: "NEGATIVE"}
        return {
            "score": score,
            "sentiment": sentiment_map[score],
            "reason": result.get("reason", ""),
            "items_scored": len(news_items[:18])
        }

    except Exception as e:
        print(f"News sentiment scoring failed: {e}")
        return {"score": 0, "sentiment": "NEUTRAL", "reason": "Scoring failed", "items_scored": 0}


def build_signal_summary(price_signal, news_sentiment, beat_rate):
    """
    Aggregate all Step 7-9 signals into a single summary dict.
    Computes an overall directional bias from the three signals.

    Scoring:
    - Price near 52W low  → +1 (easy bar)
    - Price near 52W high → -1 (elevated bar)
    - News positive       → +1
    - News negative       → -1
    - Beat rate > 60%     → +1 (company tends to beat)
    - Beat rate < 40%     → -1 (company tends to miss)

    Overall bias: POSITIVE (score > 0), NEUTRAL (0), NEGATIVE (score < 0)
    """
    score = 0

    # price signal contribution
    ps = price_signal.get("signal", "NEUTRAL")
    if ps == "LOW_BAR":
        score += 1
    elif ps == "HIGH_BAR":
        score -= 1

    # news sentiment contribution
    score += news_sentiment.get("score", 0)

    # beat rate contribution
    br = beat_rate.get("beat_rate")
    if br is not None:
        if br > 60:
            score += 1
        elif br < 40:
            score -= 1

    if score > 0:
        overall_bias = "POSITIVE"
    elif score < 0:
        overall_bias = "NEGATIVE"
    else:
        overall_bias = "NEUTRAL"

    return {
        "price_signal": price_signal,
        "news_sentiment": news_sentiment,
        "beat_rate": beat_rate,
        "overall_bias": overall_bias,
        "signal_score": score
    }
