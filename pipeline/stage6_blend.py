def blend_forecast(ensemble_forecast, scenarios):
    """
    Step 6 from the 13-step process: blend the ensemble forecast with the
    base scenario to produce a single final blended number.

    Why blend instead of just using the ensemble?
    The ensemble is purely model-driven (statistical extrapolation).
    The scenario is purely data-driven (historical same-quarter growth patterns).
    Blending 60/40 combines both views: trend + history.

    The convergence check is important: if the two approaches agree closely
    (within 2%), that's a strong signal we're in the right ballpark.
    If they diverge sharply, something unusual is happening and we flag it.

    Returns:
    {
        "blended_forecast": float,
        "ensemble_contribution": float,
        "scenario_contribution": float,
        "convergence": "high" | "moderate" | "low",
        "convergence_note": str,
        "diff_pct": float
    }
    """
    ensemble_val = float(ensemble_forecast)
    base_val = float(scenarios["base"])

    # 60% ensemble + 40% base scenario
    # 60/40 is the split from the screenshot — ensemble gets slightly more
    # weight because it's informed by 10 models + backtesting
    blended = round(0.6 * ensemble_val + 0.4 * base_val, 0)

    # measure how far apart the two inputs are
    diff_pct = abs(ensemble_val - base_val) / base_val * 100 if base_val > 0 else 0.0

    if diff_pct <= 2:
        convergence = "high"
        convergence_note = (
            f"Ensemble and base scenario agree within {diff_pct:.1f}% — strong signal."
        )
    elif diff_pct <= 5:
        convergence = "moderate"
        convergence_note = (
            f"Ensemble and base scenario differ by {diff_pct:.1f}% — blending smooths this out."
        )
    else:
        convergence = "low"
        convergence_note = (
            f"Warning: ensemble and scenario diverge by {diff_pct:.1f}%. "
            "One may be anchored on an unusual period. Check model weights and scenario history."
        )

    return {
        "blended_forecast": blended,
        "ensemble_contribution": round(0.6 * ensemble_val, 0),
        "scenario_contribution": round(0.4 * base_val, 0),
        "convergence": convergence,
        "convergence_note": convergence_note,
        "diff_pct": round(diff_pct, 1)
    }
