import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def run_linear_regression(df, n_quarters=1):
    # linear regression fits a straight trend line through all historical data
    # then projects that line forward to forecast the next quarter
    # good for companies with steady consistent growth
    # bad for highly seasonal companies - it ignores seasonality completely

    # minimum requirement: at least 8 quarters for a meaningful trend
    if len(df) < 8:
        return {
            "model": "Linear Regression",
            "forecast": None,
            "status": "skipped",
            "reason": "insufficient data - need at least 8 quarters"
        }

    df = df.sort_values("date").reset_index(drop=True)

    # convert dates to numbers so regression can work with them
    # we use quarter index (0, 1, 2, 3...) as our x variable
    df["quarter_idx"] = range(len(df))

    X = df[["quarter_idx"]].values
    y = df["revenue"].values

    model = LinearRegression()
    model.fit(X, y)

    # predict the next quarter by extending the index by 1
    next_idx = np.array([[len(df)]])
    forecast = float(model.predict(next_idx)[0])

    # sanity check - forecast must be positive
    # also check it isn't more than 50% above the historical max
    # that would be unrealistic
    historical_max = df["revenue"].max()
    if forecast <= 0:
        return {
            "model": "Linear Regression",
            "forecast": None,
            "status": "skipped",
            "reason": "sanity check failed - negative forecast"
        }
    if forecast > historical_max * 1.5:
        return {
            "model": "Linear Regression",
            "forecast": None,
            "status": "skipped",
            "reason": "sanity check failed - forecast unrealistically high"
        }

    # calculate RMSE using last 4 quarters as test set
    test_X = df[["quarter_idx"]].values[-4:]
    test_y = df["revenue"].values[-4:]

    # refit on everything except last 4 quarters
    train_X = df[["quarter_idx"]].values[:-4]
    train_y = df["revenue"].values[:-4]
    model_test = LinearRegression()
    model_test.fit(train_X, train_y)
    preds = model_test.predict(test_X)
    rmse = float(np.sqrt(np.mean((test_y - preds) ** 2)))

    return {
        "model": "Linear Regression",
        "forecast": forecast,
        "status": "success",
        "reason": None,
        "rmse": rmse
    }
    