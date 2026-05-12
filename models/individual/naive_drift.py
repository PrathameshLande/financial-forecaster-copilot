import numpy as np


def run_naive_drift(df, n_quarters=1):
    # Naive Drift = last observed value + average quarter-over-quarter change
    # it asks: "if the company keeps growing at its average historical pace,
    # what would next quarter look like?"
    # simpler than regression - it doesn't fit a line, it just extends the drift
    # good as a sanity check against fancier models

    # minimum requirement: at least 4 quarters to compute a meaningful average drift
    if len(df) < 4:
        return {
            "model": "Naive Drift",
            "forecast": None,
            "status": "skipped",
            "reason": "insufficient data - need at least 4 quarters"
        }

    df = df.sort_values("date").reset_index(drop=True)
    revenue = df["revenue"].values
    n = len(revenue)

    # drift = average of all quarter-over-quarter changes
    # e.g. if revenue went [100, 105, 103, 108] the drifts are [5, -2, 5]
    # average drift = 2.67, so forecast = 108 + 2.67 = 110.67
    diffs = [revenue[i] - revenue[i - 1] for i in range(1, n)]
    avg_drift = float(np.mean(diffs))

    forecast = float(revenue[-1] + avg_drift * n_quarters)

    # sanity checks
    historical_max = df["revenue"].max()

    if forecast <= 0:
        return {
            "model": "Naive Drift",
            "forecast": None,
            "status": "skipped",
            "reason": "sanity check failed - negative forecast"
        }
    if forecast > historical_max * 1.5:
        return {
            "model": "Naive Drift",
            "forecast": None,
            "status": "skipped",
            "reason": "sanity check failed - forecast unrealistically high"
        }

    # RMSE: retrain on all except last 4, predict those 4
    train = revenue[:-4]
    test = revenue[-4:]

    train_diffs = [train[i] - train[i - 1] for i in range(1, len(train))]
    train_drift = float(np.mean(train_diffs))

    preds = [train[-1] + train_drift * (i + 1) for i in range(4)]
    rmse = float(np.sqrt(np.mean((test - np.array(preds)) ** 2)))

    return {
        "model": "Naive Drift",
        "forecast": forecast,
        "status": "success",
        "reason": None,
        "rmse": rmse
    }
