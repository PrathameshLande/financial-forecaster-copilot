import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")


def run_sarima(df, n_quarters=1):
    # SARIMA = Seasonal AutoRegressive Integrated Moving Average
    # the most sophisticated pure time series model we use
    # it models trend, seasonality, and autocorrelation all together
    # autocorrelation means: this quarter's revenue is influenced
    # by what happened in previous quarters, not just the overall trend

    # minimum requirement: at least 16 quarters (4 full years)
    # needs enough data to estimate all its parameters reliably
    if len(df) < 16:
        return {
            "model": "SARIMA",
            "forecast": None,
            "status": "skipped",
            "reason": "insufficient data - need at least 16 quarters"
        }

    df = df.sort_values("date").reset_index(drop=True)
    revenue = df["revenue"].values

    try:
        # SARIMA(1,1,1)(1,1,0,4) is the order we use
        # (1,1,1) = non-seasonal: 1 autoregressive, 1 difference, 1 moving average
        # (1,1,0,4) = seasonal: 1 seasonal AR, 1 seasonal difference, period=4 quarters
        # this is the same order used in YFL and proven to work well for quarterly revenue
        model = SARIMAX(
            revenue,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 0, 4),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fit = model.fit(disp=False)
        forecast = float(fit.forecast(n_quarters)[0])

        # sanity checks
        historical_max = df["revenue"].max()

        if forecast <= 0:
            return {
                "model": "SARIMA",
                "forecast": None,
                "status": "skipped",
                "reason": "sanity check failed - negative forecast"
            }
        if forecast > historical_max * 1.5:
            return {
                "model": "SARIMA",
                "forecast": None,
                "status": "skipped",
                "reason": "sanity check failed - forecast unrealistically high"
            }

        # calculate RMSE on last 4 quarters
        train = revenue[:-4]
        test = revenue[-4:]

        model_test = SARIMAX(
            train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 0, 4),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fit_test = model_test.fit(disp=False)
        preds = fit_test.forecast(4)
        rmse = float(np.sqrt(np.mean((test - preds) ** 2)))

        return {
            "model": "SARIMA",
            "forecast": forecast,
            "status": "success",
            "reason": None,
            "rmse": rmse
        }

    except Exception as e:
        return {
            "model": "SARIMA",
            "forecast": None,
            "status": "skipped",
            "reason": f"model fitting failed: {e}"
        }
    