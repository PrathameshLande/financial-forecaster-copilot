import pandas as pd
import numpy as np
from prophet import Prophet
import logging
logging.getLogger("prophet").setLevel(logging.ERROR)


def run_prophet(df, n_quarters=1):
    # Prophet is Facebook's forecasting library
    # it decomposes time series into trend + seasonality + holidays
    # very good at handling missing data and outliers
    # key risk: COVID period can distort the trend badly
    # we have a COVID exclusion built in to handle this

    # minimum requirement: at least 8 quarters
    if len(df) < 8:
        return {
            "model": "Prophet",
            "forecast": None,
            "status": "skipped",
            "reason": "insufficient data - need at least 8 quarters"
        }

    df = df.sort_values("date").reset_index(drop=True)

    # prophet requires columns named ds (date) and y (value)
    prophet_df = df[["date", "revenue"]].copy()
    prophet_df.columns = ["ds", "y"]

    # exclude COVID period - March 2020 to June 2021
    # this prevents the massive revenue crash from distorting the trend
    # exactly what we did manually in YFL
    covid_mask = (
        (prophet_df["ds"] >= "2020-03-01") &
        (prophet_df["ds"] <= "2021-06-30")
    )
    prophet_df_clean = prophet_df[~covid_mask].copy()

    # need at least 8 quarters after COVID exclusion
    if len(prophet_df_clean) < 8:
        return {
            "model": "Prophet",
            "forecast": None,
            "status": "skipped",
            "reason": "insufficient data after COVID exclusion"
        }

    try:
        model = Prophet(
            seasonality_mode="additive",
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        model.fit(prophet_df_clean)

        # create a future dataframe for the next quarter
        future = model.make_future_dataframe(periods=n_quarters, freq="QE")
        forecast_df = model.predict(future)
        forecast = float(forecast_df["yhat"].iloc[-1])

        # sanity checks
        historical_max = df["revenue"].max()

        if forecast <= 0:
            return {
                "model": "Prophet",
                "forecast": None,
                "status": "skipped",
                "reason": "sanity check failed - negative forecast"
            }
        if forecast > historical_max * 1.5:
            return {
                "model": "Prophet",
                "forecast": None,
                "status": "skipped",
                "reason": "sanity check failed - forecast unrealistically high"
            }

        # calculate RMSE on last 4 quarters
        train_df = prophet_df_clean.iloc[:-4]
        test_y = prophet_df["y"].values[-4:]

        model_test = Prophet(
            seasonality_mode="additive",
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        model_test.fit(train_df)
        future_test = model_test.make_future_dataframe(periods=4, freq="QE")
        forecast_test = model_test.predict(future_test)
        preds = forecast_test["yhat"].values[-4:]
        rmse = float(np.sqrt(np.mean((test_y - preds) ** 2)))

        return {
            "model": "Prophet",
            "forecast": forecast,
            "status": "success",
            "reason": None,
            "rmse": rmse
        }

    except Exception as e:
        return {
            "model": "Prophet",
            "forecast": None,
            "status": "skipped",
            "reason": f"model fitting failed: {e}"
        }
        