import numpy as np
import warnings
warnings.filterwarnings("ignore")


def run_tbats(df, n_quarters=1):
    # TBATS = Trigonometric seasonality, Box-Cox transform, ARMA errors, Trend, Seasonality
    # it's the most flexible seasonal model we use - handles complex patterns
    # that simpler models miss. Box-Cox transform handles skewed distributions.
    # for quarterly revenue it uses seasonal_periods=[4] to match the annual cycle
    # tends to be slower than other models - that's normal, it's doing more work

    # minimum requirement: 16 quarters - needs to see seasonality repeat multiple times
    if len(df) < 16:
        return {
            "model": "TBATS",
            "forecast": None,
            "status": "skipped",
            "reason": "insufficient data - need at least 16 quarters"
        }

    df = df.sort_values("date").reset_index(drop=True)
    revenue = df["revenue"].values

    try:
        from tbats import TBATS

        # seasonal_periods=[4] = quarterly seasonality (4 quarters per year)
        # use_box_cox=None lets the model decide whether to apply a Box-Cox transform
        estimator = TBATS(seasonal_periods=[4], use_box_cox=None, n_jobs=1)
        fit = estimator.fit(revenue)
        forecast_arr = fit.forecast(steps=n_quarters)
        forecast = float(forecast_arr[0])

        # sanity checks
        historical_max = df["revenue"].max()

        if forecast <= 0:
            return {
                "model": "TBATS",
                "forecast": None,
                "status": "skipped",
                "reason": "sanity check failed - negative forecast"
            }
        if forecast > historical_max * 1.5:
            return {
                "model": "TBATS",
                "forecast": None,
                "status": "skipped",
                "reason": "sanity check failed - forecast unrealistically high"
            }

        # RMSE: retrain on all except last 4, forecast those 4
        train = revenue[:-4]
        test = revenue[-4:]

        estimator_test = TBATS(seasonal_periods=[4], use_box_cox=None, n_jobs=1)
        fit_test = estimator_test.fit(train)
        preds = fit_test.forecast(steps=4)
        rmse = float(np.sqrt(np.mean((test - preds) ** 2)))

        return {
            "model": "TBATS",
            "forecast": forecast,
            "status": "success",
            "reason": None,
            "rmse": rmse
        }

    except ImportError:
        return {
            "model": "TBATS",
            "forecast": None,
            "status": "skipped",
            "reason": "tbats library not installed - run: pip install tbats"
        }
    except Exception as e:
        return {
            "model": "TBATS",
            "forecast": None,
            "status": "skipped",
            "reason": f"model fitting failed: {e}"
        }
