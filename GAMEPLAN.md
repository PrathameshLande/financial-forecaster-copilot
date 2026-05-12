# Financial Forecaster v2.0 — Full Game Plan

> **Purpose**: This document is the coding spec. Hand it to Claude Code / any coding agent and it has everything needed to build the v2.0 upgrade without ambiguity.

---

## What Exists (Don't Break It)

- `app.py` — Streamlit UI, 2 tabs (Earnings Forecast + Stock Trend)
- `models/individual/` — 6 models: Seasonal Naive, Linear Regression, ETS, SARIMA, Prophet, Theta
- `models/ensemble.py` — inverse-RMSE weighted ensemble (will be replaced by pipeline)
- `models/runner.py` — calls all models, collects results
- `data/edgar.py` — SEC EDGAR quarterly revenue (3-layer fallback to yfinance)
- `data/yfinance_data.py` — analyst consensus, company info, stock price history
- `data/fred.py` — CPI, Fed rate, GDP, jobless claims
- `chatbot/analyst.py` — Groq Llama 3.3 70B for AI brief + chat
- `.env` — FRED_API_KEY and GROQ_API_KEY already set

---

## What to Build (13 steps from the screenshot, strictly)

### New Libraries Needed
```
pmdarima   # AutoARIMA
tbats      # TBATS model
```
Add both to `requirements.txt`.

---

## Phase 1: Four New Models

All models follow the same return interface:
```python
{
    "model": "ModelName",
    "forecast": float_or_none,
    "status": "success" | "skipped",
    "reason": None_or_string,
    "rmse": float_or_none
}
```

### `models/individual/naive_drift.py`
- Forecast = last observed value + average quarter-over-quarter delta (mean of all consecutive differences)
- Minimum: 4 quarters
- RMSE: train on all except last 4, predict 4 steps, compute RMSE
- Sanity check: no negatives, not > 1.5x historical max

### `models/individual/holts_linear.py`
- Use `statsmodels.tsa.holtwinters.Holt` (double exponential smoothing, no seasonality)
- `optimized=True` to let statsmodels find best alpha/beta
- Minimum: 8 quarters
- RMSE: same train/test split as above

### `models/individual/tbats_model.py`
- Use `tbats.TBATS(seasonal_periods=[4])`
- Fit on full series, forecast 1 step
- Minimum: 16 quarters (TBATS needs enough data for seasonal estimation)
- RMSE: retrain on n-4, forecast 4, compute RMSE
- Wrap in try/except — TBATS can be slow, set a 60-second timeout using `signal` or `threading`

### `models/individual/autoarima.py`
- Use `pmdarima.auto_arima`
- Two variants in same file:
  - `run_autoarima(df)` — regular, on raw revenue
  - `run_autoarima_log(df)` — log-transform revenue before fitting, exp() the forecast back
  - Use log variant automatically when company YoY growth > 10% (pass `high_growth=True` param)
- `auto_arima(y, seasonal=True, m=4, suppress_warnings=True, error_action='ignore', stepwise=True)`
- Minimum: 16 quarters
- Return as model name "AutoARIMA" or "AutoARIMA(log)" depending on variant used

### Update `models/runner.py`
- Import and call all 10 models
- Pass `high_growth` flag to autoarima (compute as: mean YoY growth of last 4Q > 10%)
- Model list order:
  1. Seasonal Naive
  2. Naive Drift (NEW)
  3. Holts Linear (NEW)
  4. ETS
  5. SARIMA
  6. TBATS (NEW)
  7. Prophet
  8. AutoARIMA / AutoARIMA(log) (NEW)
  9. Theta
  10. Linear Regression

---

## Phase 2: Pipeline Folder (Steps 1, 3, 4)

Create folder `pipeline/` with `__init__.py`.

### `pipeline/stage1_data.py`

```python
def validate_data(df):
    """
    Step 1 from screenshot: verify 20 quarters of data.
    Returns a validation result dict, not an exception.
    App can show warning but still proceed.
    """
    result = {
        "quarters": len(df),
        "passes": len(df) >= 20,
        "warnings": [],
        "errors": []
    }
    if len(df) < 20:
        result["errors"].append(f"Only {len(df)} quarters — need 20 for full pipeline")
    if len(df) < 8:
        result["errors"].append("Critically insufficient data — models will mostly skip")
    # check for nulls
    null_count = df["revenue"].isna().sum()
    if null_count > 0:
        result["warnings"].append(f"{null_count} null revenue values found")
    # check for negative revenue
    neg_count = (df["revenue"] < 0).sum()
    if neg_count > 0:
        result["warnings"].append(f"{neg_count} negative revenue values — may distort models")
    # check for large gaps (>2 missing quarters in a row)
    df_sorted = df.sort_values("date").reset_index(drop=True)
    for i in range(1, len(df_sorted)):
        gap = (df_sorted["date"].iloc[i] - df_sorted["date"].iloc[i-1]).days
        if gap > 200:
            result["warnings"].append(f"Large gap detected around {df_sorted['date'].iloc[i].date()}")
    return result
```

### `pipeline/stage3_backtest.py`

```python
def run_walk_forward_backtest(df, model_fn, model_name, n_test=12, min_valid=4):
    """
    Step 3: 12-quarter walk-forward backtest.
    For each of the last n_test quarters, train on data up to that point,
    predict 1 quarter ahead, compare to actual.
    
    Returns:
    {
        "model": model_name,
        "mape": float (mean absolute percentage error),
        "valid_quarters": int (how many test quarters had valid predictions),
        "eligible": bool (mape <= 10% AND valid_quarters >= min_valid),
        "predictions": list of {actual, predicted, ape}
    }
    """
    if len(df) < n_test + 8:  # need at least 8 quarters to train on
        return {"model": model_name, "mape": None, "valid_quarters": 0, "eligible": False}
    
    df = df.sort_values("date").reset_index(drop=True)
    predictions = []
    
    for i in range(n_test, 0, -1):
        train = df.iloc[:-i].copy()
        actual = df.iloc[-i]["revenue"]
        
        try:
            result = model_fn(train, n_quarters=1)
            if result["status"] == "success" and result["forecast"]:
                predicted = result["forecast"]
                ape = abs(actual - predicted) / actual * 100
                predictions.append({"actual": actual, "predicted": predicted, "ape": ape})
        except:
            pass
    
    if len(predictions) < min_valid:
        return {"model": model_name, "mape": None, "valid_quarters": len(predictions), "eligible": False}
    
    mape = sum(p["ape"] for p in predictions) / len(predictions)
    eligible = mape <= 10.0 and len(predictions) >= min_valid
    
    return {
        "model": model_name,
        "mape": round(mape, 2),
        "valid_quarters": len(predictions),
        "eligible": eligible,
        "predictions": predictions
    }
```

### `pipeline/stage4_ensemble.py`

```python
def run_backtest_weighted_ensemble(model_results, backtest_results, analyst_consensus=None):
    """
    Step 4: weight models by 1/MAPE from backtest.
    Only eligible models (mape<=10%, valid_quarters>=4) get weight.
    Analyst consensus gets 30% if available.
    """
    # build lookup: model_name -> backtest result
    backtest_map = {b["model"]: b for b in backtest_results}
    
    eligible = []
    for r in model_results:
        if r["status"] != "success":
            continue
        bt = backtest_map.get(r["model"])
        if bt and bt["eligible"] and bt["mape"] and bt["mape"] > 0:
            eligible.append({"result": r, "mape": bt["mape"]})
    
    if not eligible:
        # fallback: use all successful models with equal weight
        eligible = [{"result": r, "mape": 5.0} for r in model_results if r["status"] == "success"]
    
    # 1/MAPE weights
    inv_mape = [1.0 / e["mape"] for e in eligible]
    total = sum(inv_mape)
    weights = [w / total for w in inv_mape]
    
    # shrink model weights if consensus available
    if analyst_consensus and analyst_consensus > 0:
        weights = [w * 0.70 for w in weights]
        consensus_weight = 0.30
    else:
        consensus_weight = 0.0
    
    # compute ensemble
    ensemble = sum(e["result"]["forecast"] * w for e, w in zip(eligible, weights))
    if analyst_consensus and analyst_consensus > 0:
        ensemble += analyst_consensus * consensus_weight
    
    model_weights = {e["result"]["model"]: round(w * 100, 1) for e, w in zip(eligible, weights)}
    model_forecasts = {e["result"]["model"]: e["result"]["forecast"] for e in eligible}
    if analyst_consensus and analyst_consensus > 0:
        model_weights["Analyst Consensus"] = round(consensus_weight * 100, 1)
        model_forecasts["Analyst Consensus"] = analyst_consensus
    
    return {
        "ensemble_forecast": round(ensemble, 0),
        "status": "success",
        "model_weights": model_weights,
        "model_forecasts": model_forecasts
    }
```

---

## Phase 3: Scenarios + Blend (Steps 5–6)

### `pipeline/stage5_scenarios.py`

```python
def build_scenarios(df):
    """
    Step 5: Bear/Base/Bull from same-quarter-last-year growth history.
    
    Logic:
    1. Identify which quarter we're forecasting (Q1/Q2/Q3/Q4)
    2. Collect all historical revenue for that same quarter
    3. Compute YoY growth rates
    4. Bear = 25th percentile growth applied to most recent same-quarter value
    5. Base = 50th percentile (median)
    6. Bull = 75th percentile
    
    Returns: {"bear": float, "base": float, "bull": float, "growth_rates": list}
    """
    import numpy as np
    
    df = df.sort_values("date").reset_index(drop=True)
    next_quarter = (df["date"].iloc[-1].month - 1) // 3 + 1  # 1-4
    
    # get all rows for this quarter
    same_q = df[df["date"].dt.quarter == next_quarter].sort_values("date")
    
    if len(same_q) < 3:
        # fallback: use overall average growth
        revenues = df["revenue"].values
        growth_rates = [(revenues[i] - revenues[i-4]) / revenues[i-4] * 100 
                        for i in range(4, len(revenues)) if revenues[i-4] > 0]
    else:
        revenues = same_q["revenue"].values
        growth_rates = [(revenues[i] - revenues[i-1]) / revenues[i-1] * 100 
                        for i in range(1, len(revenues))]
    
    if not growth_rates:
        last_rev = df["revenue"].iloc[-4] if len(df) >= 4 else df["revenue"].iloc[-1]
        return {"bear": last_rev, "base": last_rev, "bull": last_rev, "growth_rates": []}
    
    p25 = np.percentile(growth_rates, 25) / 100
    p50 = np.percentile(growth_rates, 50) / 100
    p75 = np.percentile(growth_rates, 75) / 100
    
    base_value = same_q["revenue"].iloc[-1] if not same_q.empty else df["revenue"].iloc[-4]
    
    return {
        "bear": round(base_value * (1 + p25), 0),
        "base": round(base_value * (1 + p50), 0),
        "bull": round(base_value * (1 + p75), 0),
        "growth_rates": growth_rates,
        "percentiles": {"p25": round(p25*100,1), "p50": round(p50*100,1), "p75": round(p75*100,1)}
    }
```

### `pipeline/stage6_blend.py`

```python
def blend_forecast(ensemble_forecast, scenarios):
    """
    Step 6: Blend ensemble (60%) with base scenario (40%).
    Check convergence between the two.
    """
    base = scenarios["base"]
    blended = round(0.6 * ensemble_forecast + 0.4 * base, 0)
    
    # convergence check
    diff_pct = abs(ensemble_forecast - base) / base * 100
    
    if diff_pct <= 2:
        convergence = "high"
        note = f"Ensemble and scenario agree within {diff_pct:.1f}%"
    elif diff_pct <= 5:
        convergence = "moderate"
        note = f"Ensemble and scenario differ by {diff_pct:.1f}% — blending smooths this"
    else:
        convergence = "low"
        note = f"Warning: Ensemble and scenario diverge by {diff_pct:.1f}% — investigate anchor"
    
    return {
        "blended_forecast": blended,
        "ensemble_contribution": round(0.6 * ensemble_forecast, 0),
        "scenario_contribution": round(0.4 * base, 0),
        "convergence": convergence,
        "convergence_note": note,
        "diff_pct": round(diff_pct, 1)
    }
```

---

## Phase 4: Signals (Steps 7–10)

### Add to `data/yfinance_data.py`

```python
def get_news(ticker, months=3):
    """Step 7: Get recent news for the ticker."""
    import yfinance as yf
    from datetime import datetime, timedelta
    stock = yf.Ticker(ticker)
    news = stock.news
    cutoff = datetime.now() - timedelta(days=months * 30)
    recent = [n for n in news if datetime.fromtimestamp(n.get("providerPublishTime", 0)) > cutoff]
    return recent[:18]  # cap at 18 items (3 months x 6 items)


def get_52w_signal(ticker):
    """Step 8: Compare current price to 52-week range."""
    import yfinance as yf
    stock = yf.Ticker(ticker)
    info = stock.info
    current = info.get("currentPrice") or info.get("regularMarketPrice")
    high_52w = info.get("fiftyTwoWeekHigh")
    low_52w = info.get("fiftyTwoWeekLow")
    
    if not all([current, high_52w, low_52w]):
        return {"signal": "NEUTRAL", "note": "Price data unavailable"}
    
    pct_from_high = (high_52w - current) / high_52w * 100
    pct_from_low = (current - low_52w) / low_52w * 100
    
    if pct_from_high < 10:
        return {"signal": "HIGH_BAR", "note": f"Stock within {pct_from_high:.1f}% of 52W high — bar is elevated", 
                "current": current, "high_52w": high_52w, "low_52w": low_52w}
    elif pct_from_low < 10:
        return {"signal": "LOW_BAR", "note": f"Stock within {pct_from_low:.1f}% of 52W low — bar is easier",
                "current": current, "high_52w": high_52w, "low_52w": low_52w}
    else:
        return {"signal": "NEUTRAL", "note": "Stock in mid-range — neutral bar",
                "current": current, "high_52w": high_52w, "low_52w": low_52w}


def get_beat_rate(ticker):
    """Step 9: Historical revenue beat rate over last 8 quarters."""
    import yfinance as yf
    stock = yf.Ticker(ticker)
    try:
        history = stock.earnings_history
        if history is None or history.empty:
            return {"beat_rate": None, "quarters_checked": 0}
        # Check how often actual > estimated
        recent = history.head(8)
        beats = (recent["epsActual"] > recent["epsEstimate"]).sum()  # proxy using EPS
        rate = beats / len(recent) * 100
        return {"beat_rate": round(rate, 1), "quarters_checked": len(recent)}
    except:
        return {"beat_rate": None, "quarters_checked": 0}
```

### `pipeline/stage7_signals.py`

```python
def score_news_sentiment(news_items, ticker, groq_client):
    """
    Score news items using Groq — send headline+snippet, get +1/0/-1 signal.
    Very short prompt = very few tokens = negligible Groq usage.
    """
    if not news_items:
        return {"score": 0, "items_scored": 0, "sentiment": "NEUTRAL"}
    
    headlines = "\n".join([
        f"- {n.get('title', '')}" for n in news_items[:18]
    ])
    
    prompt = f"""Rate overall revenue sentiment for {ticker} from these headlines.
Return ONLY a JSON: {{"score": -1|0|1, "reason": "one sentence"}}
-1 = clearly negative for revenue, 0 = neutral/mixed, 1 = clearly positive

Headlines:
{headlines}"""
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80
        )
        import json
        result = json.loads(response.choices[0].message.content.strip())
        return {
            "score": result.get("score", 0),
            "reason": result.get("reason", ""),
            "items_scored": len(news_items[:18]),
            "sentiment": {1: "POSITIVE", 0: "NEUTRAL", -1: "NEGATIVE"}.get(result.get("score", 0), "NEUTRAL")
        }
    except:
        return {"score": 0, "items_scored": 0, "sentiment": "NEUTRAL", "reason": "scoring failed"}


def build_signal_summary(price_signal, news_sentiment, beat_rate):
    """Aggregate all signals into a single summary dict."""
    signals = {
        "price_signal": price_signal,
        "news_sentiment": news_sentiment,
        "beat_rate": beat_rate,
        "overall_bias": "NEUTRAL"
    }
    # compute overall directional bias
    score = 0
    if price_signal.get("signal") == "LOW_BAR":
        score += 1
    elif price_signal.get("signal") == "HIGH_BAR":
        score -= 1
    score += news_sentiment.get("score", 0)
    if beat_rate.get("beat_rate") and beat_rate["beat_rate"] > 60:
        score += 1
    elif beat_rate.get("beat_rate") and beat_rate["beat_rate"] < 40:
        score -= 1
    
    if score > 0:
        signals["overall_bias"] = "POSITIVE"
    elif score < 0:
        signals["overall_bias"] = "NEGATIVE"
    
    return signals
```

---

## Phase 5: Validation (Steps 11–13)

### `pipeline/stage8_validate.py`

```python
def run_validation(blended_forecast, scenarios, analyst_consensus, ensemble_forecast):
    """
    Steps 11-13: final three-gate validation before output.
    
    Step 11: Same-day consensus anchor (already fetched from yfinance)
    Step 12: 3% error band check
    Step 13: Convergence check (3 methods within 3%)
    """
    results = {
        "consensus_check": {},
        "error_band_check": {},
        "convergence_check": {},
        "passes_all": False,
        "flags": []
    }
    
    # Step 11: Consensus anchor — must not be >5% from analyst consensus
    if analyst_consensus and analyst_consensus > 0:
        consensus_diff = (blended_forecast - analyst_consensus) / analyst_consensus * 100
        passes_consensus = abs(consensus_diff) <= 5
        results["consensus_check"] = {
            "passes": passes_consensus,
            "diff_pct": round(consensus_diff, 1),
            "note": f"Forecast is {consensus_diff:+.1f}% vs analyst consensus"
        }
        if not passes_consensus:
            results["flags"].append(f"Consensus anchor violated: {consensus_diff:+.1f}% (max ±5%)")
    else:
        results["consensus_check"] = {"passes": True, "note": "No consensus available — skipped"}
    
    # Step 12: 3% error band — forecast must sit within bear–bull range and within 3% of base
    base = scenarios["base"]
    bear = scenarios["bear"]
    bull = scenarios["bull"]
    
    within_range = bear <= blended_forecast <= bull
    base_diff = abs(blended_forecast - base) / base * 100
    passes_band = within_range and base_diff <= 3
    
    results["error_band_check"] = {
        "passes": passes_band,
        "within_bear_bull": within_range,
        "base_diff_pct": round(base_diff, 1),
        "note": f"Forecast ${blended_forecast/1e9:.2f}B vs bear ${bear/1e9:.2f}B / base ${base/1e9:.2f}B / bull ${bull/1e9:.2f}B"
    }
    if not passes_band:
        results["flags"].append(f"Error band violated: {base_diff:.1f}% from base (max 3%)")
    
    # Step 13: Convergence — ensemble, blended, and scenario base all within 3% of each other
    methods = [ensemble_forecast, blended_forecast, base]
    max_val = max(methods)
    min_val = min(methods)
    spread_pct = (max_val - min_val) / min_val * 100
    
    if spread_pct <= 3:
        convergence_level = "HIGH"
        convergence_note = f"All 3 methods converge within {spread_pct:.1f}% ✅"
    elif spread_pct <= 6:
        convergence_level = "MODERATE"
        convergence_note = f"Methods within {spread_pct:.1f}% — acceptable"
    else:
        convergence_level = "LOW"
        convergence_note = f"Methods diverge by {spread_pct:.1f}% — investigate"
        results["flags"].append(f"Convergence failed: {spread_pct:.1f}% spread (max 3%)")
    
    results["convergence_check"] = {
        "level": convergence_level,
        "spread_pct": round(spread_pct, 1),
        "note": convergence_note
    }
    
    results["passes_all"] = len(results["flags"]) == 0
    return results
```

---

## Phase 6: UI Refresh (app.py changes)

### New pipeline orchestration in app.py (replacing current run_button block):

```python
# 1. validate data
validation = validate_data(df)
if not validation["passes"]:
    st.warning(f"⚠️ {validation['errors'][0]} — results may be less reliable")

# 2. run 10 models
results = run_all_models(df, analyst_consensus=consensus)

# 3. backtest
backtest_results = []
for model_fn, model_name in MODEL_REGISTRY:
    bt = run_walk_forward_backtest(df, model_fn, model_name)
    backtest_results.append(bt)

# 4. backtest-weighted ensemble
ensemble = run_backtest_weighted_ensemble(results["individual_results"], backtest_results, consensus)

# 5. scenarios
scenarios = build_scenarios(df)

# 6. blend
blend = blend_forecast(ensemble["ensemble_forecast"], scenarios)

# 7. signals
news = get_news(ticker)
price_signal = get_52w_signal(ticker)
beat_rate = get_beat_rate(ticker)
news_sentiment = score_news_sentiment(news, ticker, groq_client)
signals = build_signal_summary(price_signal, news_sentiment, beat_rate)

# 8. validate
validation_result = run_validation(blend["blended_forecast"], scenarios, consensus, ensemble["ensemble_forecast"])
```

### New UI sections to add:

1. **Bear / Base / Bull cards** — 3 metric columns showing the three scenarios with % growth labels
2. **Signal row** — 3 small badges: News (🟢/🟡🔴), Price (📈📉➡️), Beat Rate (X% of last 8Q)
3. **Validation gates row** — 3 checkmarks for Steps 11/12/13 — green or red
4. **Confidence badge** — derived from validation + convergence, calibrated to 50–80% range per rules
5. **Pipeline progress** — `st.progress()` bar showing which stage completed

### Update chatbot/analyst.py:
Pass `scenarios`, `signals`, `validation_result`, and `blend` to `build_context()` so the AI brief references all pipeline stages.

---

## Universal Rules to Enforce in Code (from Screenshot)

| Rule | Where to implement |
|------|-------------------|
| Convergence rule: ≥3 independent methods | stage8_validate.py Step 13 |
| Same-day consensus: always re-fetch | fetch consensus just before stage8, not at startup |
| Consensus anchor: never >5% from consensus | stage8_validate.py Step 11 |
| Historical beat rate always applied | stage7_signals.py + shown in UI |
| Error band: within bear-bull range | stage8_validate.py Step 12 |
| Model exclusion: MAPE >10% or <4 valid Q | stage3_backtest.py eligible flag |
| Confidence calibration: 50–80% max | app.py confidence display |

---

## What NOT to Do (keep it recruiter-explainable)

- No neural networks, no transformers, no paid ML APIs
- No complex microservices — it's still a single Streamlit app
- Every weight/blend constant has a comment explaining why that number
- Each pipeline file does exactly ONE thing
- Max file length ~100 lines per file (split if longer)
- All models use the same input/output interface — makes it easy to add/remove

---

## Final Recruiter Pitch

> "I built a multi-stage revenue forecasting pipeline for public equities. It pulls quarterly earnings from SEC EDGAR, runs 10 statistical models — SARIMA, Prophet, Theta, AutoARIMA, TBATS and more — then backtests each one on the last 12 quarters to score accuracy. Models are weighted by their backtest error, not arbitrarily. The forecast gets broken into bear, base, and bull scenarios based on how the company has historically grown in the same quarter. Before finalizing, the app checks today's analyst consensus, scans recent news for sentiment, looks at whether the stock is near a 52-week high or low, and applies the company's own historical beat rate. If at least 3 independent signals converge within 3%, the forecast locks. A Groq LLM then reads all of this and writes a plain-English analyst brief. Everything runs on free APIs."

---

*Generated: 2026-05-07 | Project: forecaster*
