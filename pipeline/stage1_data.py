def validate_data(df):
    """
    Step 1 from the 13-step process: verify data quality before running any models.

    We need at least 20 quarters for the full pipeline to work reliably.
    This function doesn't raise exceptions - it returns a result dict so the
    app can show warnings to the user and still proceed (with reduced confidence).

    Returns:
    {
        "quarters": int,
        "passes": bool,       # True if >= 20 quarters
        "warnings": list,     # non-fatal issues
        "errors": list        # serious issues that may affect results
    }
    """
    import pandas as pd

    result = {
        "quarters": len(df),
        "passes": len(df) >= 20,
        "warnings": [],
        "errors": []
    }

    # check minimum data requirements
    if len(df) < 20:
        result["errors"].append(
            f"Only {len(df)} quarters available — need 20 for full pipeline. "
            "Walk-forward backtest will use a shorter window."
        )
    if len(df) < 8:
        result["errors"].append(
            "Critically low data — most models will skip. Forecast unreliable."
        )

    # check for null revenues
    null_count = df["revenue"].isna().sum()
    if null_count > 0:
        result["warnings"].append(f"{null_count} missing revenue values found and will be skipped.")

    # check for negative revenue (unusual but possible for loss-making periods)
    neg_count = (df["revenue"] < 0).sum()
    if neg_count > 0:
        result["warnings"].append(
            f"{neg_count} quarters with negative revenue detected. "
            "This may distort trend models."
        )

    # check for large time gaps (>2 consecutive missing quarters)
    df_sorted = df.sort_values("date").reset_index(drop=True)
    for i in range(1, len(df_sorted)):
        gap_days = (df_sorted["date"].iloc[i] - df_sorted["date"].iloc[i - 1]).days
        if gap_days > 200:  # more than ~2 quarters gap
            gap_date = df_sorted["date"].iloc[i].strftime("%Y-%m")
            result["warnings"].append(f"Data gap detected around {gap_date} ({gap_days} days).")

    return result
