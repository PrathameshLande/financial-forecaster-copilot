import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def run_ets(df, n_quarters=1):
    # ETS = Error Trend Seasonality (also called Holt-Winters)
    # unlike linear regression it explicitly models seasonality
    # it gives more weight to recent data than older data
    # great for companies with consistent seasonal patterns like airlines

    # minimum requirement: at least 2 full years (8 quarters)
    # needs to see seasonality repeat at least twice to learn it
    if len(df) < 8:
        return {
            "model": "ETS",
            "forecast": None,
            "status": "skipped",
            "reason": "insufficient data - need at least 8 quarters"
        }

    df = df.sort_values("date").reset_index(drop=True)
    revenue = df["revenue"].values

    try:
        # additive seasonality - seasonal swings stay roughly constant in size
        # multiplicative would be used if swings grow proportionally with level
        # for airlines additive works well
        model = ExponentialSmoothing(
            revenue,
            trend="add",
            seasonal="add",
            seasonal_periods=4
        )
        fit = model.fit(optimized=True)
        forecast = float(fit.forecast(n_quarters)[0])

        # sanity checks
        historical_max = df["revenue"].max()
        historical_min = df["revenue"].min()

        if forecast <= 0:
            return {
                "model": "ETS",
                "forecast": None,
                "status": "skipped",
                "reason": "sanity check failed - negative forecast"
            }
        if forecast > historical_max * 1.5:
            return {
                "model": "ETS",
                "forecast": None,
                "status": "skipped",
                "reason": "sanity check failed - forecast unrealistically high"
            }

        # calculate RMSE on last 4 quarters
        train = revenue[:-4]
        test = revenue[-4:]

        model_test = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=4
        )
        fit_test = model_test.fit(optimized=True)
        preds = fit_test.forecast(4)
        rmse = float(np.sqrt(np.mean((test - preds) ** 2)))

        return {
            "model": "ETS",
            "forecast": forecast,
            "status": "success",
            "reason": None,
            "rmse": rmse
        }

    except Exception as e:
        return {
            "model": "ETS",
            "forecast": None,
            "status": "skipped",
            "reason": f"model fitting failed: {e}"
        }
        