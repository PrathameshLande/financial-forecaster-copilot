import numpy as np


def run_ensemble(model_results, analyst_consensus=None):
    # filter to only successful models
    successful = [r for r in model_results if r["status"] == "success"]

    if not successful:
        return {
            "ensemble_forecast": None,
            "status": "failed",
            "reason": "no models succeeded",
            "model_weights": {},
            "model_forecasts": {}
        }

    # step 1: assign weights by inverse RMSE
    # lower RMSE = better model = higher weight
    # inverse RMSE means a model with RMSE of 500M gets 2x the weight
    # of a model with RMSE of 1B
    rmse_values = []
    for r in successful:
        if r.get("rmse") and r["rmse"] > 0:
            rmse_values.append(r["rmse"])
        else:
            # if no RMSE available give it average weight
            rmse_values.append(None)

    # fill missing RMSEs with the mean of available ones
    valid_rmses = [r for r in rmse_values if r is not None]
    mean_rmse = np.mean(valid_rmses) if valid_rmses else 1.0
    rmse_values = [r if r is not None else mean_rmse for r in rmse_values]

    # inverse RMSE weights - normalized to sum to 1
    inverse_rmse = [1.0 / r for r in rmse_values]
    total = sum(inverse_rmse)
    weights = [w / total for w in inverse_rmse]

    # step 2: if analyst consensus is available add it as an input
    # analyst consensus gets 30% weight - it represents real world expectations
    # we down-weight all model outputs proportionally to make room for it
    if analyst_consensus and analyst_consensus > 0:
        model_weight_total = 0.70
        weights = [w * model_weight_total for w in weights]
        analyst_weight = 0.30
    else:
        analyst_weight = 0.0

    # step 3: compute weighted ensemble forecast
    ensemble = sum(
        r["forecast"] * w
        for r, w in zip(successful, weights)
    )

    if analyst_consensus and analyst_consensus > 0:
        ensemble = ensemble + analyst_consensus * analyst_weight

    # step 4: build output dictionary
    model_weights = {}
    model_forecasts = {}

    for r, w in zip(successful, weights):
        model_weights[r["model"]] = round(w * 100, 1)
        model_forecasts[r["model"]] = round(r["forecast"], 0)

    if analyst_consensus and analyst_consensus > 0:
        model_weights["Analyst Consensus"] = round(analyst_weight * 100, 1)
        model_forecasts["Analyst Consensus"] = round(analyst_consensus, 0)

    return {
        "ensemble_forecast": round(ensemble, 0),
        "status": "success",
        "reason": None,
        "model_weights": model_weights,
        "model_forecasts": model_forecasts
    }
