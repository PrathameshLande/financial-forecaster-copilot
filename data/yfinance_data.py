import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def get_stock_price_history(ticker, period="2y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df[["Close"]].reset_index()
    df.columns = ["date", "close_price"]
    df["date"] = pd.to_datetime(df["date"])
    df["date"] = df["date"].dt.tz_localize(None)
    print(f"Fetched {len(df)} days of price data for {ticker}")
    return df


def get_analyst_consensus(ticker):
    try:
        stock = yf.Ticker(ticker)
        estimates = stock.revenue_estimate
        if estimates is None or estimates.empty:
            print(f"No analyst estimates found for {ticker}")
            return None
        estimates = estimates.reset_index()
        print(f"Analyst estimates found for {ticker}")
        return estimates
    except Exception as e:
        print(f"Could not fetch analyst estimates: {e}")
        return None


def get_company_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
        "market_cap": info.get("marketCap", None),
        "description": info.get("longBusinessSummary", "")
    }


def get_news(ticker, months=3):
    """
    Step 7: Fetch recent news articles for the ticker.
    yfinance API for news has changed across versions — we try multiple approaches
    so it works regardless of which version is installed.
    """
    cutoff_ts = (datetime.now() - timedelta(days=months * 30)).timestamp()
    articles = []

    # Approach 1: yf.Search (works in yfinance >= 0.2.50)
    try:
        search = yf.Search(ticker, news_count=25, enable_fuzzy_query=False)
        raw = search.news or []
        if raw:
            articles = raw
            print(f"News via yf.Search: {len(articles)} items for {ticker}")
    except Exception:
        pass

    # Approach 2: Ticker.news (older yfinance versions)
    if not articles:
        try:
            stock = yf.Ticker(ticker)
            raw = stock.news or []
            if raw:
                articles = raw
                print(f"News via Ticker.news: {len(articles)} items for {ticker}")
        except Exception:
            pass

    if not articles:
        print(f"No news found for {ticker}")
        return []

    # Normalise: yfinance changed the key names across versions
    # Old: {"title": ..., "providerPublishTime": <unix ts>}
    # New: {"content": {"title": ..., "pubDate": "2026-05-07T..."}}
    normalised = []
    for item in articles:
        title = (
            item.get("title")
            or item.get("content", {}).get("title")
            or ""
        )
        # get timestamp — try old key first, then parse new ISO string
        ts = item.get("providerPublishTime", 0)
        if not ts:
            pub = item.get("content", {}).get("pubDate", "")
            if pub:
                try:
                    from datetime import timezone
                    ts = datetime.fromisoformat(pub.replace("Z", "+00:00")).timestamp()
                except Exception:
                    ts = 0

        if title:
            normalised.append({"title": title, "providerPublishTime": ts})

    # filter to last `months` months (skip items with no timestamp)
    recent = [n for n in normalised if n["providerPublishTime"] == 0 or n["providerPublishTime"] > cutoff_ts]
    print(f"Fetched {len(recent)} news items for {ticker} (last {months} months)")
    return recent[:18]


def get_52w_signal(ticker):
    """
    Step 8: Compare current price to 52-week high/low.
    Near 52W high → bar is elevated (harder to beat expectations).
    Near 52W low  → bar is easy (lower expectations).
    Returns a signal dict with: signal, note, current, high_52w, low_52w.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        current = info.get("currentPrice") or info.get("regularMarketPrice")
        high_52w = info.get("fiftyTwoWeekHigh")
        low_52w = info.get("fiftyTwoWeekLow")

        if not all([current, high_52w, low_52w]):
            return {"signal": "NEUTRAL", "note": "Price data unavailable", "current": None,
                    "high_52w": None, "low_52w": None}

        pct_from_high = (high_52w - current) / high_52w * 100
        pct_from_low = (current - low_52w) / low_52w * 100

        if pct_from_high < 10:
            signal = "HIGH_BAR"
            note = (f"Stock is within {pct_from_high:.1f}% of 52W high (${high_52w:.2f}). "
                    "Market expectations are elevated.")
        elif pct_from_low < 10:
            signal = "LOW_BAR"
            note = (f"Stock is within {pct_from_low:.1f}% of 52W low (${low_52w:.2f}). "
                    "Bar is lower — easier to beat consensus.")
        else:
            signal = "NEUTRAL"
            note = f"Stock at ${current:.2f}, mid-range between 52W low (${low_52w:.2f}) and high (${high_52w:.2f})."

        return {
            "signal": signal,
            "note": note,
            "current": current,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "pct_from_high": round(pct_from_high, 1),
            "pct_from_low": round(pct_from_low, 1)
        }
    except Exception as e:
        print(f"Could not fetch price signal for {ticker}: {e}")
        return {"signal": "NEUTRAL", "note": f"Price signal unavailable: {e}",
                "current": None, "high_52w": None, "low_52w": None}


def get_beat_rate(ticker):
    """
    Step 9: Historical EPS beat rate over the last 8 quarters.
    We use EPS (earnings per share) beats as a proxy for the company's
    tendency to beat or miss analyst expectations in general.
    A company that beats EPS >60% of the time likely beats revenue too.
    Returns: {beat_rate (%), quarters_checked, beats, misses}
    """
    try:
        stock = yf.Ticker(ticker)
        history = stock.earnings_history
        if history is None or history.empty:
            print(f"No earnings history found for {ticker}")
            return {"beat_rate": None, "quarters_checked": 0, "beats": 0, "misses": 0}

        recent = history.head(8).copy()
        # beat = actual EPS > estimated EPS
        valid = recent.dropna(subset=["epsActual", "epsEstimate"])
        if valid.empty:
            return {"beat_rate": None, "quarters_checked": 0, "beats": 0, "misses": 0}

        beats = int((valid["epsActual"] > valid["epsEstimate"]).sum())
        misses = len(valid) - beats
        beat_rate = round(beats / len(valid) * 100, 1)

        print(f"Beat rate for {ticker}: {beat_rate}% ({beats}/{len(valid)} quarters)")
        return {
            "beat_rate": beat_rate,
            "quarters_checked": len(valid),
            "beats": beats,
            "misses": misses
        }
    except Exception as e:
        print(f"Could not fetch beat rate for {ticker}: {e}")
        return {"beat_rate": None, "quarters_checked": 0, "beats": 0, "misses": 0}
