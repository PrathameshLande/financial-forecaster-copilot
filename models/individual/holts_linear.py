import numpy as np
from statsmodels.tsa.holtwinters import Holt


def run_holts_linear(df, n_quarters=1):
    # Holt's Linear (Double Exponential Smoothing) - captures trend but not seasonality
    # unlike ETS which handles seasonality, Holt's focuses purely on the trend component
    # it uses two smoothing equations: one for the level, one for the trend
    # useful for companies where trend matters more than seasonal pattern
    # statsmodels handles the alpha/beta optimization automatically

    # minimum requirement: 8 quarters for a reliable trend estimate
    if len(df) < 8:
        return {
            "model": "Holts Linear",
            "forecast": None,
            "status": "skipped",
            "reason": "insufficient data - need at least 8 quarters"
        }

    df = df.sort_values("date").reset_index(drop=True)
    revenue = df["revenue"].values

    try:
        # optimized=True lets statsmodels find the best smoothing parameters
        # exponential=False means additive (linear) trend, not multiplicative
        model = Holt(revenue, exponential=False)
        fit = model.fit(optimized=True)
        forecast = float(fit.forecast(n_quarters)[0])

        # sanity checks
        historical_max = df["revenue"].max()

        if forecast <= 0:
            return {
                "model": "Holts Linear",
                "forecast": None,
                "status": "skipped",
                "reason": "sanity check failed - negative forecast"
            }
        if forecast > historical_max * 1.5:
            return {
                "model": "Holts Linear",
                "forecast": None,
                "status": "skipped",
                "reason": "sanity check failed - forecast unrealistically high"
            }

        # RMSE: retrain on all except last 4, forecast those 4
        train = revenue[:-4]
        test = revenue[-4:]

        model_test = Holt(train, exponential=False)
        fit_test = model_test.fit(optimized=True)
        preds = fit_test.forecast(4)
        rmse = float(np.sqrt(np.mean((test - preds) ** 2)))

        return {
            "model": "Holts Linear",
            "forecast": forecast,
            "status": "success",
            "reason": None,
            "rmse": rmse
        }

    except Exception as e:
        return {
            "model": "Holts Linear",
            "forecast": None,
            "status": "skipped",
            "reason": f"model fitting failed: {e}"
        }
