import numpy as np


def build_scenarios(df):
    """
    Step 5 from the 13-step process: build Bear / Base / Bull scenarios.

    Logic:
    1. Figure out which quarter we're forecasting (Q1, Q2, Q3, or Q4)
    2. Gather all historical revenue values for that SAME quarter across years
       (e.g. if we're forecasting Q1 2026, collect Q1 2019, Q1 2020, ... Q1 2025)
    3. Compute year-over-year growth rates between consecutive same-quarter values
    4. Bear = 25th percentile growth rate applied to the most recent same-quarter value
    5. Base = 50th percentile (median) — our central estimate
    6. Bull = 75th percentile

    Why same-quarter YoY growth? Because comparing Q1 to Q4 mixes seasonality with
    actual growth. Same-quarter comparison isolates the true growth trend.

    Returns:
    {
        "bear": float,
        "base": float,
        "bull": float,
        "growth_rates": list of floats (pct),
        "percentiles": {"p25": float, "p50": float, "p75": float},
        "quarters_used": int
    }
    """
    df = df.sort_values("date").reset_index(drop=True)

    # which quarter is next? same as the most recent quarter in the data
    next_quarter_num = df["date"].iloc[-1].quarter  # 1, 2, 3, or 4

    # get all historical rows for this same quarter number
    same_q = df[df["date"].dt.quarter == next_quarter_num].sort_values("date").reset_index(drop=True)

    if len(same_q) >= 3:
        # compute YoY growth between consecutive same-quarter values
        revenues = same_q["revenue"].values
        growth_rates = [
            (revenues[i] - revenues[i - 1]) / revenues[i - 1] * 100
            for i in range(1, len(revenues))
            if revenues[i - 1] > 0
        ]
        base_value = float(same_q["revenue"].iloc[-1])
    else:
        # fallback: not enough same-quarter history — use all consecutive quarter growth
        revenues = df["revenue"].values
        growth_rates = [
            (revenues[i] - revenues[i - 4]) / revenues[i - 4] * 100
            for i in range(4, len(revenues))
            if revenues[i - 4] > 0
        ]
        base_value = float(df["revenue"].iloc[-4]) if len(df) >= 4 else float(df["revenue"].iloc[-1])

    if not growth_rates:
        # last resort fallback: flat forecast
        flat = float(df["revenue"].iloc[-4]) if len(df) >= 4 else float(df["revenue"].iloc[-1])
        return {
            "bear": flat,
            "base": flat,
            "bull": flat,
            "growth_rates": [],
            "percentiles": {"p25": 0.0, "p50": 0.0, "p75": 0.0},
            "quarters_used": len(same_q)
        }

    p25 = float(np.percentile(growth_rates, 25))
    p50 = float(np.percentile(growth_rates, 50))
    p75 = float(np.percentile(growth_rates, 75))

    return {
        "bear": round(base_value * (1 + p25 / 100), 0),
        "base": round(base_value * (1 + p50 / 100), 0),
        "bull": round(base_value * (1 + p75 / 100), 0),
        "growth_rates": [round(g, 2) for g in growth_rates],
        "percentiles": {
            "p25": round(p25, 1),
            "p50": round(p50, 1),
            "p75": round(p75, 1)
        },
        "quarters_used": len(same_q)
    }
