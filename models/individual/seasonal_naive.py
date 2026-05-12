import pandas as pd
import numpy as np


def run_seasonal_naive(df, n_quarters=4):
    # seasonal naive = predict next quarter using same quarter from last year
    # example: Q1 2026 forecast = Q1 2025 actual
    # this is our baseline - every other model must beat this to be worth using

    # minimum data requirement: at least 4 quarters (one full year)
    if len(df) < 4:
        return {
            "model": "Seasonal Naive",
            "forecast": None,
            "status": "skipped",
            "reason": "insufficient data - need at least 4 quarters"
        }

    # get the last 4 quarters so we know same quarter last year
    df = df.sort_values("date").reset_index(drop=True)

    # the forecast is simply the value from exactly 4 quarters ago
    last_value = df["revenue"].iloc[-4]
    forecast = float(last_value)

    # sanity check - forecast should be positive
    if forecast <= 0:
        return {
            "model": "Seasonal Naive",
            "forecast": None,
            "status": "skipped",
            "reason": "sanity check failed - negative forecast"
        }

    # calculate RMSE on historical data so we can weight this model later
    # we test it by seeing how well it predicted the last 4 quarters
    actuals = df["revenue"].iloc[-4:].values
    naive_preds = df["revenue"].iloc[-8:-4].values

    if len(actuals) == len(naive_preds):
        rmse = float(np.sqrt(np.mean((actuals - naive_preds) ** 2)))
    else:
        rmse = None

    return {
        "model": "Seasonal Naive",
        "forecast": forecast,
        "status": "success",
        "reason": None,
        "rmse": rmse
    }
