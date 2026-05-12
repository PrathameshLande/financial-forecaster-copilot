from models.individual.seasonal_naive import run_seasonal_naive
from models.individual.naive_drift import run_naive_drift
from models.individual.holts_linear import run_holts_linear
from models.individual.ets import run_ets
from models.individual.sarima import run_sarima
from models.individual.tbats_model import run_tbats
from models.individual.prophet_model import run_prophet
from models.individual.autoarima import run_autoarima
from models.individual.theta import run_theta
from models.individual.linear_regression import run_linear_regression
from models.ensemble import run_ensemble


def _is_high_growth(df):
    # check if the company is growing >10% YoY on average over the last 4 quarters
    # if yes, we use AutoARIMA with a log transform instead of the regular version
    # log transform handles exponential growth series better
    if len(df) < 8:
        return False
    revenue = df.sort_values("date")["revenue"].values
    recent = revenue[-4:]
    year_ago = revenue[-8:-4]
    yoy_growths = [(recent[i] - year_ago[i]) / year_ago[i] * 100
                   for i in range(4) if year_ago[i] > 0]
    if not yoy_growths:
        return False
    return sum(yoy_growths) / len(yoy_growths) > 10.0


def run_all_models(df, analyst_consensus=None):
    if df is None or len(df) == 0:
        print("No data available - cannot run models")
        return None

    print("\nRunning all 10 models...")

    # decide whether to use log-transform AutoARIMA based on YoY growth
    high_growth = _is_high_growth(df)
    if high_growth:
        print("  High-growth company detected (>10% YoY) - using AutoARIMA(log)")

    results = []

    print("  [1/10] Seasonal Naive...")
    results.append(run_seasonal_naive(df))

    print("  [2/10] Naive Drift...")
    results.append(run_naive_drift(df))

    print("  [3/10] Holts Linear...")
    results.append(run_holts_linear(df))

    print("  [4/10] ETS...")
    results.append(run_ets(df))

    print("  [5/10] SARIMA...")
    results.append(run_sarima(df))

    print("  [6/10] TBATS...")
    results.append(run_tbats(df))

    print("  [7/10] Prophet...")
    results.append(run_prophet(df))

    print("  [8/10] AutoARIMA...")
    # use log variant automatically for high-growth companies
    results.append(run_autoarima(df, use_log=high_growth))

    print("  [9/10] Theta...")
    results.append(run_theta(df))

    print("  [10/10] Linear Regression...")
    results.append(run_linear_regression(df))

    # print summary table
    print("\nModel Results:")
    print(f"{'Model':<22} {'Forecast':>15} {'RMSE':>15} {'Status':<10}")
    print("-" * 67)

    for r in results:
        if r["status"] == "success":
            forecast_str = f"${r['forecast']:,.0f}"
            rmse_str = f"${r['rmse']:,.0f}" if r.get("rmse") else "N/A"
        else:
            forecast_str = "SKIPPED"
            rmse_str = "N/A"
        print(f"{r['model']:<22} {forecast_str:>15} {rmse_str:>15} {r['status']:<10}")
        if r["status"] == "skipped":
            print(f"  → {r['reason']}")

    # run ensemble
    print("\nRunning ensemble...")
    ensemble_result = run_ensemble(results, analyst_consensus=analyst_consensus)

    print(f"\nEnsemble Forecast: ${ensemble_result['ensemble_forecast']:,.0f}")
    print("\nModel Weights:")
    for model, weight in ensemble_result["model_weights"].items():
        print(f"  {model:<28} {weight}%")

    return {
        "individual_results": results,
        "ensemble": ensemble_result
    }

