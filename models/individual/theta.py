import pandas as pd
import numpy as np


def run_theta(df, n_quarters=1):
    # Theta method decomposes the series into two lines called theta lines
    # theta=0 line captures the long term trend
    # theta=2 line captures the short term fluctuations
    # the forecast blends both lines together
    # it consistently outperforms more complex models on quarterly financial data
    # and is extremely easy to explain in interviews

    # minimum requirement: at least 8 quarters
    if len(df) < 8:
        return {
            "model": "Theta",
            "forecast": None,
            "status": "skipped",
            "reason": "insufficient data - need at least 8 quarters"
        }

    df = df.sort_values("date").reset_index(drop=True)
    revenue = df["revenue"].values
    n = len(revenue)

    try:
        # step 1: remove seasonality by dividing each value by
        # the average of the same quarter across all years
        # seasonal period is 4 for quarterly data
        period = 4
        seasonal_factors = []

        for i in range(period):
            # get all values for this quarter position
            quarter_values = revenue[i::period]
            seasonal_factors.append(np.mean(quarter_values))

        # normalize factors so they average to 1
        overall_mean = np.mean(seasonal_factors)
        seasonal_factors = [f / overall_mean for f in seasonal_factors]

        # deseasonalize the series
        deseasonalized = np.array([
            revenue[i] / seasonal_factors[i % period]
            for i in range(n)
        ])

        # step 2: fit a simple linear trend to deseasonalized data
        x = np.arange(n)
        slope, intercept = np.polyfit(x, deseasonalized, 1)

        # step 3: fit simple exponential smoothing to deseasonalized data
        # find optimal alpha using grid search
        best_alpha = 0.1
        best_sse = float("inf")

        for alpha in np.arange(0.1, 1.0, 0.1):
            smoothed = [deseasonalized[0]]
            for t in range(1, n):
                smoothed.append(
                    alpha * deseasonalized[t] + (1 - alpha) * smoothed[-1]
                )
            sse = sum((deseasonalized[t] - smoothed[t-1])**2 for t in range(1, n))
            if sse < best_sse:
                best_sse = sse
                best_alpha = alpha

        # generate smoothed series with best alpha
        smoothed = [deseasonalized[0]]
        for t in range(1, n):
            smoothed.append(
                best_alpha * deseasonalized[t] + (1 - best_alpha) * smoothed[-1]
            )

        # step 4: forecast = blend of trend line and smoothed value
        trend_forecast = slope * n + intercept
        ses_forecast = smoothed[-1]
        deseasonalized_forecast = (trend_forecast + ses_forecast) / 2

        # step 5: reapply seasonality for the next quarter
        next_quarter_idx = n % period
        forecast = deseasonalized_forecast * seasonal_factors[next_quarter_idx]

        # sanity checks
        historical_max = df["revenue"].max()

        if forecast <= 0:
            return {
                "model": "Theta",
                "forecast": None,
                "status": "skipped",
                "reason": "sanity check failed - negative forecast"
            }
        if forecast > historical_max * 1.5:
            return {
                "model": "Theta",
                "forecast": None,
                "status": "skipped",
                "reason": "sanity check failed - forecast unrealistically high"
            }

        # calculate RMSE on last 4 quarters
        train = revenue[:-4]
        test = revenue[-4:]
        n_train = len(train)

        seasonal_factors_train = []
        for i in range(period):
            quarter_values = train[i::period]
            seasonal_factors_train.append(np.mean(quarter_values))
        overall_mean_train = np.mean(seasonal_factors_train)
        seasonal_factors_train = [f / overall_mean_train for f in seasonal_factors_train]

        deseasonalized_train = np.array([
            train[i] / seasonal_factors_train[i % period]
            for i in range(n_train)
        ])

        slope_t, intercept_t = np.polyfit(np.arange(n_train), deseasonalized_train, 1)
        smoothed_t = [deseasonalized_train[0]]
        for t in range(1, n_train):
            smoothed_t.append(
                best_alpha * deseasonalized_train[t] + (1 - best_alpha) * smoothed_t[-1]
            )

        preds = []
        for step in range(4):
            tf = slope_t * (n_train + step) + intercept_t
            sf = smoothed_t[-1]
            df_cast = (tf + sf) / 2
            qi = (n_train + step) % period
            preds.append(df_cast * seasonal_factors_train[qi])

        rmse = float(np.sqrt(np.mean((test - np.array(preds)) ** 2)))

        return {
            "model": "Theta",
            "forecast": float(forecast),
            "status": "success",
            "reason": None,
            "rmse": rmse
        }

    except Exception as e:
        return {
            "model": "Theta",
            "forecast": None,
            "status": "skipped",
            "reason": f"model fitting failed: {e}"
        }
    