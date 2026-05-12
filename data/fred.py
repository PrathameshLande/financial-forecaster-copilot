from fredapi import Fred
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()


def get_macro_context():
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))

    try:
        # CPI - inflation rate
        cpi = fred.get_series("CPIAUCSL").dropna()
        latest_cpi = round(float(cpi.iloc[-1]), 2)
        prev_cpi = round(float(cpi.iloc[-13]), 2)
        cpi_yoy = round(((latest_cpi - prev_cpi) / prev_cpi) * 100, 2)

        # Federal funds rate
        fed_rate = fred.get_series("FEDFUNDS").dropna()
        latest_fed_rate = round(float(fed_rate.iloc[-1]), 2)

        # Weekly jobless claims
        jobless = fred.get_series("ICSA").dropna()
        latest_jobless = int(jobless.iloc[-1])

        # GDP growth
        gdp = fred.get_series("A191RL1Q225SBEA").dropna()
        latest_gdp = round(float(gdp.iloc[-1]), 2)

        context = {
            "cpi_latest": latest_cpi,
            "cpi_yoy_pct": cpi_yoy,
            "fed_rate": latest_fed_rate,
            "jobless_claims": latest_jobless,
            "gdp_growth": latest_gdp,
        }

        print("Macro context fetched successfully")
        return context

    except Exception as e:
        print(f"FRED fetch failed: {e}")
        return {}