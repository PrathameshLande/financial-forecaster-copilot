# Financial Forecaster

A Streamlit app that forecasts next-quarter revenue for any publicly traded US company using a 13-step ensemble pipeline, 10 statistical models, and an AI analyst powered by Llama 3.3 (via Groq's free tier).

**Live demo:** *(add your Streamlit Community Cloud URL here)*

---

## What it does

Enter any ticker (e.g. `AAPL`, `WMT`, `DAL`) and the app:

1. Pulls 5+ years of quarterly revenue from SEC EDGAR (free, no API key)
2. Runs 10 forecast models in parallel
3. Backtests each model over 12 quarters using walk-forward validation — no lookahead bias
4. Builds a MAPE-weighted ensemble (lower error = higher weight)
5. Computes bear / base / bull scenarios from historical same-quarter YoY growth
6. Blends the ensemble with the base scenario (60/40)
7. Pulls live signals — news sentiment (Groq), 52-week price position, and EPS beat rate
8. Runs 3 validation gates: consensus anchor, error band, and model convergence
9. Generates a structured AI analyst brief with headline, scenarios, signals, macro, and key risk

---

## Pipeline overview

```
SEC EDGAR → Quarterly Revenue (20+ quarters)
         ↓
[Stage 1]  Data validation — gaps, nulls, min quarters
         ↓
[Stage 2]  10 models run in parallel:
           Seasonal Naive · Naive Drift · Holt's Linear · ETS
           SARIMA · TBATS · Prophet · AutoARIMA · Theta · Linear Regression
         ↓
[Stage 3]  12-quarter walk-forward backtest → MAPE per model
         ↓
[Stage 4]  Backtest-weighted ensemble (1/MAPE weights)
           + 30% analyst consensus if available
         ↓
[Stage 5]  Bear / Base / Bull scenarios (p25 / p50 / p75 same-quarter YoY)
         ↓
[Stage 6]  Blended forecast = 60% ensemble + 40% base scenario
         ↓
[Stages 7–9]  Market signals: news sentiment · 52W price · EPS beat rate
         ↓
[Stages 11–13]  Validation gates: consensus ±5% · error band · convergence
         ↓
         AI Analyst Brief (Groq Llama 3.3 70B, free tier)
```

---

## Tech stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Revenue data | SEC EDGAR (XBRL API, free) |
| Price / signals | yfinance |
| Macro data | FRED API (free) |
| Forecast models | statsmodels, Prophet, pmdarima, tbats, scikit-learn |
| Charts | Plotly |
| AI analyst | Groq API — Llama 3.3 70B (free tier) |

All APIs used are free. No Bloomberg, no paid data providers.

---

## Project structure

```
forecaster/
├── app.py                     # Streamlit entry point
├── data/
│   ├── edgar.py               # SEC EDGAR quarterly revenue
│   ├── yfinance_data.py       # Price, news, beat rate, consensus
│   └── fred.py                # Macro: CPI, Fed rate, GDP, jobless claims
├── models/
│   ├── runner.py              # Runs all 10 models, returns results list
│   └── individual/
│       ├── seasonal_naive.py
│       ├── naive_drift.py
│       ├── holts_linear.py
│       ├── ets.py
│       ├── sarima.py
│       ├── tbats_model.py
│       ├── prophet_model.py
│       ├── autoarima.py
│       ├── theta.py
│       └── linear_regression.py
├── pipeline/
│   ├── stage1_data.py         # Data validation
│   ├── stage3_backtest.py     # 12-quarter walk-forward backtest
│   ├── stage4_ensemble.py     # MAPE-weighted ensemble
│   ├── stage5_scenarios.py    # Bear / Base / Bull scenarios
│   ├── stage6_blend.py        # 60/40 ensemble-scenario blend
│   ├── stage7_signals.py      # News sentiment + price + beat rate
│   └── stage8_validate.py     # Validation gates (Steps 11–13)
├── chatbot/
│   └── analyst.py             # Groq AI brief + chat
├── requirements.txt
└── .streamlit/
    └── config.toml            # Dark theme + server settings
```

---

## Setup

```bash
# 1. Clone
git clone https://github.com/<your-username>/forecaster.git
cd forecaster

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API keys
cp .env.example .env
# Edit .env and fill in GROQ_API_KEY and FRED_API_KEY

# 5. Run
streamlit run app.py
```

---

## Environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
FRED_API_KEY=your_fred_api_key_here
```

- **Groq** — free at [console.groq.com](https://console.groq.com)
- **FRED** — free at [fred.stlouisfed.org/docs/api](https://fred.stlouisfed.org/docs/api/api_key.html)

---

## Deploy to Streamlit Community Cloud (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set `app.py` as the entry point
4. Add `GROQ_API_KEY` and `FRED_API_KEY` under **Settings → Secrets**

---

## Key design decisions

**Why walk-forward backtest?** Training on all data and testing on a holdout leaks future information. Walk-forward trains on data available at each point in time — the same way a model would have been used in production.

**Why MAPE weighting?** MAPE (Mean Absolute Percentage Error) is scale-independent, so a model's weight is comparable across companies of any size. Lower MAPE = higher weight in the ensemble.

**Why 60/40 blend?** The ensemble captures long-run statistical patterns; the base scenario captures recent same-quarter momentum. The blend anchors the final number in both.

**Why Groq (free tier)?** The app is intentionally zero-cost. Groq's free tier provides fast Llama 3.3 70B inference with enough quota for interactive use.
