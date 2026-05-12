from fredapi import Fred
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()


_MACRO_FALLBACK = {
    "cpi_yoy_pct": "N/A",
    "fed_rate": "N/A",
    "jobless_claims": 0,
    "gdp_growth": "N/A",
}


def get_macro_context():
    """
    Fetch macro indicators from FRED.
    Returns sensible defaults (N/A) instead of crashing when:
    - FRED_API_KEY is not set (e.g. Streamlit Cloud without secrets configured)
    - Any network / API error occurs
    This way the rest of the pipeline always gets a valid dict.
    """
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        print("FRED_API_KEY not set — returning N/A macro context")
        return _MACRO_FALLBACK.copy()

    try:
        fred = Fred(api_key=api_key)

        # CPI — year-over-year inflation
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

        # GDP growth (real, annualised)
        gdp = fred.get_series("A191RL1Q225SBEA").dropna()
        latest_gdp = round(float(gdp.iloc[-1]), 2)

        print("Macro context fetched successfully")
        return {
            "cpi_latest": latest_cpi,
            "cpi_yoy_pct": cpi_yoy,
            "fed_rate": latest_fed_rate,
            "jobless_claims": latest_jobless,
            "gdp_growth": latest_gdp,
        }

    except Exception as e:
        print(f"FRED fetch failed: {e}")
        return _MACRO_FALLBACK.copy()