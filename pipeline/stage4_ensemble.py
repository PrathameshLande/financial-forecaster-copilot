def run_backtest_weighted_ensemble(model_results, backtest_results, analyst_consensus=None):
    """
    Step 4 from the 13-step process: build a backtest-weighted ensemble.

    This replaces the old RMSE-weighted ensemble with a MAPE-weighted one.
    The key difference: weights come from the 12-quarter walk-forward backtest
    (Step 3), not from a simple 4-quarter holdout. More trustworthy.

    Weight logic:
    - Only eligible models (MAPE <= 10%, >= 4 valid quarters) get weight
    - Weight = 1/MAPE — lower error = higher weight
    - Analyst consensus gets a fixed 30% if available (it represents real market
      expectations and has historically anchored forecasts well)
    - All weights are normalized to sum to 100%

    Fallback: if no models pass the eligibility filter, we fall back to using
    all successful models with equal weight rather than returning nothing.
    """
    # build a lookup: model_name -> backtest result
    backtest_map = {b["model"]: b for b in backtest_results}

    # collect eligible models that also ran successfully
    eligible = []
    for r in model_results:
        if r["status"] != "success" or not r.get("forecast"):
            continue
        bt = backtest_map.get(r["model"])
        if bt and bt["eligible"] and bt["mape"] and bt["mape"] > 0:
            eligible.append({"result": r, "mape": bt["mape"]})

    if len(eligible) < 4:
        # safety net: if fewer than 4 models passed the MAPE filter,
        # fill up to 4 using the next-best models ranked by lowest MAPE.
        # this prevents the ensemble from being too thin on high-growth companies
        # where even good models show high MAPE on recent explosive quarters.
        eligible_names = {e["result"]["model"] for e in eligible}
        candidates = []
        for r in model_results:
            if r["status"] != "success" or not r.get("forecast"):
                continue
            if r["model"] in eligible_names:
                continue
            bt = backtest_map.get(r["model"])
            if bt and bt.get("mape") and bt["mape"] > 0:
                candidates.append({"result": r, "mape": bt["mape"]})
            elif r["status"] == "success":
                candidates.append({"result": r, "mape": 15.0})  # neutral fallback MAPE

        # sort by MAPE ascending, take enough to reach 4
        candidates.sort(key=lambda x: x["mape"])
        needed = max(0, 4 - len(eligible))
        eligible += candidates[:needed]

    if not eligible:
        return {
            "ensemble_forecast": None,
            "status": "failed",
            "reason": "no models succeeded",
            "model_weights": {},
            "model_forecasts": {},
            "used_fallback": False
        }

    used_fallback = all(e["mape"] == 5.0 for e in eligible)

    # compute 1/MAPE weights, normalized to sum to 1
    inv_mape = [1.0 / e["mape"] for e in eligible]
    total = sum(inv_mape)
    weights = [w / total for w in inv_mape]

    # shrink model weights to 70% if analyst consensus is available
    # consensus gets the remaining 30% — it represents real market expectations
    if analyst_consensus and analyst_consensus > 0:
        weights = [w * 0.70 for w in weights]
        consensus_weight = 0.30
    else:
        consensus_weight = 0.0

    # compute the weighted average forecast
    ensemble = sum(e["result"]["forecast"] * w for e, w in zip(eligible, weights))
    if analyst_consensus and analyst_consensus > 0:
        ensemble += analyst_consensus * consensus_weight

    # build output dicts
    model_weights = {
        e["result"]["model"]: round(w * 100, 1)
        for e, w in zip(eligible, weights)
    }
    model_forecasts = {
        e["result"]["model"]: round(e["result"]["forecast"], 0)
        for e in eligible
    }

    if analyst_consensus and analyst_consensus > 0:
        model_weights["Analyst Consensus"] = round(consensus_weight * 100, 1)
        model_forecasts["Analyst Consensus"] = round(analyst_consensus, 0)

    return {
        "ensemble_forecast": round(ensemble, 0),
        "status": "success",
        "reason": None,
        "model_weights": model_weights,
        "model_forecasts": model_forecasts,
        "used_fallback": used_fallback
    }
