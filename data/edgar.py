import requests
import pandas as pd
import yfinance as yf


def get_cik(ticker):
    headers = {"User-Agent": "forecaster app contact@example.com"}
    mapping_url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(mapping_url, headers=headers)
    data = response.json()
    for key in data:
        if data[key]["ticker"].upper() == ticker.upper():
            cik = str(data[key]["cik_str"]).zfill(10)
            return cik
    return None


def get_yfinance_quarterly(ticker):
    try:
        stock = yf.Ticker(ticker)
        income = stock.quarterly_income_stmt
        if income is None or income.empty:
            return pd.DataFrame()
        if "Total Revenue" not in income.index:
            return pd.DataFrame()
        revenue_row = income.loc["Total Revenue"]
        df = revenue_row.reset_index()
        df.columns = ["date", "revenue"]
        df["date"] = pd.to_datetime(df["date"])
        df["date"] = df["date"].dt.tz_localize(None)
        df["date"] = df["date"] + pd.offsets.QuarterEnd(0)
        df["frame"] = df["date"].apply(lambda d: f"CY{d.year}Q{d.quarter}-yf")
        df = df.dropna(subset=["revenue"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"yfinance fetch failed: {e}")
        return pd.DataFrame()


def get_quarterly_revenue(ticker):
    headers = {"User-Agent": "forecaster app contact@example.com"}

    cik = get_cik(ticker)
    if not cik:
        print(f"Could not find CIK for: {ticker}")
        return None
    print(f"Found CIK for {ticker}: {cik}")

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"SEC EDGAR error: {response.status_code}")
        return None
    facts = response.json()

    revenue_fields = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenuesNetOfInterestExpense",
    ]
    revenue_data = None
    for field in revenue_fields:
        try:
            revenue_data = facts["facts"]["us-gaap"][field]["units"]["USD"]
            print(f"Using revenue field: {field}")
            break
        except KeyError:
            continue
    if not revenue_data:
        print("Could not find revenue field")
        return None

    quarterly = []
    annual = []

    for entry in revenue_data:
        if "start" not in entry or "end" not in entry:
            continue
        start = pd.to_datetime(entry["start"])
        end = pd.to_datetime(entry["end"])
        duration = (end - start).days

        if 85 <= duration <= 99:
            quarterly.append({
                "date": entry["end"],
                "revenue": entry["val"],
                "frame": entry.get("frame", f"Q-{entry['end']}")
            })
        elif 360 <= duration <= 370:
            annual.append({
                "year": end.year,
                "annual_revenue": entry["val"]
            })

    df_edgar = pd.DataFrame(quarterly)
    df_edgar["date"] = pd.to_datetime(df_edgar["date"])
    df_edgar["date"] = df_edgar["date"] + pd.offsets.QuarterEnd(0)
    df_edgar = df_edgar.sort_values("date")
    df_edgar = df_edgar.drop_duplicates(subset="date", keep="last")
    print(f"Layer 1 - EDGAR direct: {len(df_edgar)} quarters")

    df_yf = get_yfinance_quarterly(ticker)
    if not df_yf.empty:
        edgar_dates = set(df_edgar["date"].dt.date)
        missing_yf = df_yf[~df_yf["date"].dt.date.isin(edgar_dates)]
        if not missing_yf.empty:
            print(f"Layer 2 - yfinance filled: {len(missing_yf)} quarters")
            df_edgar = pd.concat([df_edgar, missing_yf], ignore_index=True)
            df_edgar = df_edgar.sort_values("date")

    if annual:
        df_annual = pd.DataFrame(annual)
        df_annual = df_annual.drop_duplicates(subset="year", keep="last")
        annual_lookup = dict(zip(df_annual["year"], df_annual["annual_revenue"]))

        q4_rows = []
        for year, annual_rev in annual_lookup.items():
            year_data = df_edgar[df_edgar["date"].dt.year == year]
            has_q4 = any(year_data["date"].dt.month == 12)
            if has_q4:
                continue
            q123 = year_data[year_data["date"].dt.month != 12]
            if len(q123) == 3:
                q4_revenue = annual_rev - q123["revenue"].sum()
                q4_rows.append({
                    "date": pd.Timestamp(f"{year}-12-31"),
                    "revenue": q4_revenue,
                    "frame": f"CY{year}Q4-calc"
                })

        if q4_rows:
            df_q4 = pd.DataFrame(q4_rows)
            print(f"Layer 3 - calculated Q4s: {len(df_q4)} quarters")
            df_edgar = pd.concat([df_edgar, df_q4], ignore_index=True)

    df_final = df_edgar.sort_values("date").reset_index(drop=True)

    def clean_frame(row):
        if str(row["frame"]).startswith("Q-"):
            d = row["date"]
            return f"CY{d.year}Q{d.quarter}"
        return row["frame"]

    df_final["frame"] = df_final.apply(clean_frame, axis=1)
    print(f"Final dataset: {len(df_final)} quarters")
    return df_final
    

