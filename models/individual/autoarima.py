import numpy as np
import warnings
warnings.filterwarnings("ignore")


def run_autoarima(df, n_quarters=1, use_log=False):
    # AutoARIMA automatically searches for the best ARIMA order (p, d, q) and
    # seasonal order (P, D, Q, m) using the AIC score as the selection criterion.
    # Instead of manually specifying (1,1,1)(1,1,0,4) like in SARIMA,
    # it tries many combinations and picks the best one for this specific company.
    #
    # use_log=True applies a log transform before fitting then exp() the result back.
    # log transform is better for high-growth companies (>10% YoY) because their
    # revenue series grows exponentially, not linearly - log makes it linear.

    # minimum requirement: 16 quarters for reliable order selection
    if len(df) < 16:
        model_name = "AutoARIMA(log)" if use_log else "AutoARIMA"
        return {
            "model": model_name,
            "forecast": None,
            "status": "skipped",
            "reason": "insufficient data - need at least 16 quarters"
        }

    df = df.sort_values("date").reset_index(drop=True)
    revenue = df["revenue"].values
    model_name = "AutoARIMA(log)" if use_log else "AutoARIMA"

    try:
        from pmdarima import auto_arima

        # apply log transform if requested
        # add small epsilon to avoid log(0) for any zero-revenue quarters
        series = np.log(revenue + 1e-9) if use_log else revenue.copy()

        # stepwise=True makes it faster by using a step-up search instead of
        # checking every combination. suppress_warnings silences convergence noise.
        model = auto_arima(
            series,
            seasonal=True,
            m=4,                       # quarterly seasonality
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_p=3, max_q=3,
            max_P=2, max_Q=2,
            D=1                        # enforce one seasonal difference
        )
        raw_forecast = float(model.predict(n_periods=n_quarters)[0])

        # reverse log transform if we applied it
        forecast = float(np.exp(raw_forecast)) if use_log else raw_forecast

        # sanity checks
        historical_max = df["revenue"].max()

        if forecast <= 0:
            return {
                "model": model_name,
                "forecast": None,
                "status": "skipped",
                "reason": "sanity check failed - negative forecast"
            }
        if forecast > historical_max * 1.5:
            return {
                "model": model_name,
                "forecast": None,
                "status": "skipped",
                "reason": "sanity check failed - forecast unrealistically high"
            }

        # RMSE: retrain on all except last 4, forecast those 4
        train = series[:-4]
        test_raw = revenue[-4:]  # always compare against original (non-log) values

        model_test = auto_arima(
            train,
            seasonal=True,
            m=4,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_p=3, max_q=3,
            max_P=2, max_Q=2,
            D=1
        )
        raw_preds = model_test.predict(n_periods=4)
        preds = np.exp(raw_preds) if use_log else raw_preds
        rmse = float(np.sqrt(np.mean((test_raw - preds) ** 2)))

        return {
            "model": model_name,
            "forecast": forecast,
            "status": "success",
            "reason": None,
            "rmse": rmse
        }

    except ImportError:
        return {
            "model": model_name,
            "forecast": None,
            "status": "skipped",
            "reason": "pmdarima not installed - run: pip install pmdarima"
        }
    except Exception as e:
        return {
            "model": model_name,
            "forecast": None,
            "status": "skipped",
            "reason": f"model fitting failed: {e}"
        }
