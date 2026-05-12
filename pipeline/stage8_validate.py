def run_validation(blended_forecast, scenarios, analyst_consensus, ensemble_forecast):
    """
    Steps 11–13 from the 13-step process: three final validation gates.

    These are the last checks before the forecast is published. If any gate
    fails, the app flags it clearly but does not block the output — it just
    lowers the confidence level shown to the user.

    Step 11 — Same-Day Consensus Check (MANDATORY):
        The blended forecast must not stray more than 5% from analyst consensus.
        If it does, we need a documented specific reason — not vague assumptions.
        This is the single most important rule in the screenshot.

    Step 12 — 3% Error Band Check:
        The blended forecast must sit within the bear–bull range AND
        be within 3% of the base scenario. This ensures we're not producing
        an outlier that doesn't match any reasonable scenario.

    Step 13 — Final Convergence Check:
        Three independent estimates must converge within 3%:
        (1) ensemble forecast, (2) blended forecast, (3) base scenario.
        If all three agree → high confidence. Two agree → moderate. None → red flag.

    Returns:
    {
        "consensus_check": {...},
        "error_band_check": {...},
        "convergence_check": {...},
        "passes_all": bool,
        "confidence_level": "high" | "moderate" | "low",
        "confidence_pct": int (50-80, per screenshot calibration rules),
        "flags": list of warning strings
    }
    """
    results = {
        "consensus_check": {},
        "error_band_check": {},
        "convergence_check": {},
        "passes_all": False,
        "confidence_level": "moderate",
        "confidence_pct": 65,
        "flags": []
    }

    # ── Step 11: Same-Day Consensus Anchor ─────────────────────────────────
    # Rule: never stray more than 5% from analyst consensus without documented reason.
    # Consensus is fetched fresh from yfinance right before this check.
    if analyst_consensus and analyst_consensus > 0:
        consensus_diff_pct = (blended_forecast - analyst_consensus) / analyst_consensus * 100
        passes_consensus = abs(consensus_diff_pct) <= 5.0

        direction = "above" if consensus_diff_pct > 0 else "below"
        results["consensus_check"] = {
            "passes": passes_consensus,
            "diff_pct": round(consensus_diff_pct, 1),
            "analyst_consensus": analyst_consensus,
            "note": (
                f"Forecast is {abs(consensus_diff_pct):.1f}% {direction} analyst consensus "
                f"(${analyst_consensus/1e9:.2f}B)."
            )
        }
        if not passes_consensus:
            results["flags"].append(
                f"Step 11: Consensus anchor violated — forecast is {consensus_diff_pct:+.1f}% "
                f"vs consensus (max ±5%). Document specific reason or adjust forecast."
            )
    else:
        results["consensus_check"] = {
            "passes": True,
            "diff_pct": None,
            "analyst_consensus": None,
            "note": "No analyst consensus available — Step 11 skipped."
        }

    # ── Step 12: 3% Error Band Check ──────────────────────────────────────
    bear = float(scenarios["bear"])
    base = float(scenarios["base"])
    bull = float(scenarios["bull"])

    within_range = bear <= blended_forecast <= bull
    base_diff_pct = abs(blended_forecast - base) / base * 100 if base > 0 else 0.0
    passes_band = within_range and base_diff_pct <= 3.0

    results["error_band_check"] = {
        "passes": passes_band,
        "within_bear_bull": within_range,
        "base_diff_pct": round(base_diff_pct, 1),
        "bear": bear,
        "base": base,
        "bull": bull,
        "note": (
            f"Forecast ${blended_forecast/1e9:.2f}B vs "
            f"bear ${bear/1e9:.2f}B / base ${base/1e9:.2f}B / bull ${bull/1e9:.2f}B. "
            f"Within range: {within_range}. Distance from base: {base_diff_pct:.1f}%."
        )
    }
    if not passes_band:
        if not within_range:
            results["flags"].append(
                f"Step 12: Forecast falls outside bear–bull range "
                f"(${bear/1e9:.2f}B – ${bull/1e9:.2f}B). Revise safer scenario."
            )
        else:
            results["flags"].append(
                f"Step 12: Forecast is {base_diff_pct:.1f}% from base scenario (max 3%). "
                "Blend may be over-indexing on ensemble. Check outlier models."
            )

    # ── Step 13: Final Convergence Check ──────────────────────────────────
    # Three independent estimates: ensemble, blended, and base scenario.
    # All three must converge within 3% for high confidence.
    methods = {
        "Ensemble": float(ensemble_forecast),
        "Blended":  float(blended_forecast),
        "Base scenario": base
    }
    values = list(methods.values())
    max_val = max(values)
    min_val = min(values)
    spread_pct = (max_val - min_val) / min_val * 100 if min_val > 0 else 0.0

    # find the outlier method (furthest from the mean)
    mean_val = sum(values) / 3
    outlier = max(methods, key=lambda k: abs(methods[k] - mean_val))

    if spread_pct <= 3:
        convergence_level = "high"
        convergence_note = f"All 3 methods converge within {spread_pct:.1f}% ✅"
    elif spread_pct <= 6:
        convergence_level = "moderate"
        convergence_note = f"Methods within {spread_pct:.1f}% — acceptable divergence."
    else:
        convergence_level = "low"
        convergence_note = (
            f"Methods diverge by {spread_pct:.1f}%. "
            f"Outlier: {outlier} (${methods[outlier]/1e9:.2f}B). Investigate before publishing."
        )
        results["flags"].append(
            f"Step 13: Convergence failed — {spread_pct:.1f}% spread across 3 methods "
            f"(max 3%). Outlier: {outlier}."
        )

    results["convergence_check"] = {
        "level": convergence_level,
        "spread_pct": round(spread_pct, 1),
        "methods": {k: round(v, 0) for k, v in methods.items()},
        "outlier": outlier if convergence_level == "low" else None,
        "note": convergence_note
    }

    # ── Compute overall confidence ─────────────────────────────────────────
    # Per screenshot rules: start at 50%, max out at 80% for earnings forecasts
    # Each gate that passes adds confidence; each flag reduces it
    n_flags = len(results["flags"])
    passes_consensus = results["consensus_check"]["passes"]
    passes_band = results["error_band_check"]["passes"]

    if n_flags == 0 and convergence_level == "high":
        confidence_level = "high"
        confidence_pct = 80
    elif n_flags <= 1 and convergence_level in ("high", "moderate"):
        confidence_level = "moderate"
        confidence_pct = 65
    else:
        confidence_level = "low"
        confidence_pct = 50

    results["passes_all"] = n_flags == 0
    results["confidence_level"] = confidence_level
    results["confidence_pct"] = confidence_pct

    return results
