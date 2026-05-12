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
    """
    Secondary data source: yfinance quarterly income statement.
    yfinance derives Q4 from annual - Q1+Q2+Q3, so it fills gaps
    that EDGAR's direct quarterly filings miss.
    """
    try:
        stock = yf.Ticker(ticker)
        income = stock.quarterly_income_stmt
        if income is None or income.empty:
            return pd.DataFrame()
        if "Total Revenue" not in income.index:
            return pd.DataFrame()
        revenue_row = income.loc["Total Revenue"]
        df = revenue_row.reset_index()
        df.columns = ["date_raw", "revenue"]
        df["date_raw"] = pd.to_datetime(df["date_raw"])
        df["date_raw"] = df["date_raw"].dt.tz_localize(None)
        # snap to nearest calendar quarter-end so dates align with EDGAR
        df["date"] = df["date_raw"] + pd.offsets.QuarterEnd(0)
        df["frame"] = df["date"].apply(lambda d: f"CY{d.year}Q{d.quarter}-yf")
        df = df.dropna(subset=["revenue"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"yfinance fetch failed: {e}")
        return pd.DataFrame()


def get_quarterly_revenue(ticker):
    """
    Three-layer revenue assembly:

    Layer 1 — SEC EDGAR XBRL direct: quarterly 10-Q filings (Q1, Q2, Q3).
              Companies do not file a standalone 10-Q for Q4.

    Layer 2 — yfinance: fills quarters that EDGAR misses. yfinance derives
              Q4 from annual − (Q1+Q2+Q3), so it covers fiscal-year companies.

    Layer 3 — Fiscal-year-aware Q4 derivation: for each annual filing,
              find its three quarterly entries by matching the original
              SEC start/end dates (not calendar-snapped dates).
              This correctly handles WMT (Jan), AAPL (Sep), MSFT (Jun),
              NVDA (Jan), and any other non-December fiscal year.

    The old Layer 3 logic used calendar year (df.dt.year == annual.year)
    which always found the wrong Q1+Q2+Q3 for fiscal-year companies, so
    Q4 was never calculated for them.
    """
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

    # Parse entries — keep original start/end dates for fiscal year matching
    quarterly = []   # 10-Q filings: Q1, Q2, Q3
    annual = []      # 10-K filings: full fiscal year

    for entry in revenue_data:
        if "start" not in entry or "end" not in entry:
            continue
        start = pd.to_datetime(entry["start"])
        end = pd.to_datetime(entry["end"])
        duration = (end - start).days

        if 85 <= duration <= 99:
            quarterly.append({
                "start_raw": start,   # original fiscal quarter start
                "end_raw": end,       # original fiscal quarter end
                "date": end + pd.offsets.QuarterEnd(0),   # snapped to calendar
                "revenue": entry["val"],
                "frame": entry.get("frame", f"Q-{entry['end']}")
            })
        elif 360 <= duration <= 370:
            annual.append({
                "start_raw": start,   # fiscal year start
                "end_raw": end,       # fiscal year end
                "annual_revenue": entry["val"]
            })

    if not quarterly:
        print("No quarterly entries found in EDGAR")
        df_edgar = pd.DataFrame(columns=["start_raw", "end_raw", "date", "revenue", "frame"])
    else:
        df_edgar = pd.DataFrame(quarterly)
        df_edgar = df_edgar.sort_values("date")
        # deduplicate by calendar-snapped date — keep the highest revenue
        # (some companies refile; the latest/highest is most accurate)
        df_edgar = df_edgar.sort_values("revenue", ascending=False)
        df_edgar = df_edgar.drop_duplicates(subset="date", keep="first")
        df_edgar = df_edgar.sort_values("date").reset_index(drop=True)
        print(f"Layer 1 - EDGAR direct: {len(df_edgar)} quarters")

    # ── Layer 2: yfinance fill ────────────────────────────────────────────
    # yfinance Q4 data fills the most common gap from Layer 1
    df_yf = get_yfinance_quarterly(ticker)
    if not df_yf.empty:
        edgar_dates = set(df_edgar["date"].dt.date) if not df_edgar.empty else set()
        missing_yf = df_yf[~df_yf["date"].dt.date.isin(edgar_dates)]
        if not missing_yf.empty:
            print(f"Layer 2 - yfinance filled: {len(missing_yf)} quarters")
            # add placeholder raw columns so concat works cleanly
            missing_yf = missing_yf.copy()
            if "start_raw" not in missing_yf.columns:
                missing_yf["start_raw"] = None
            if "end_raw" not in missing_yf.columns:
                missing_yf["end_raw"] = missing_yf.get("date_raw", missing_yf["date"])
            df_edgar = pd.concat([df_edgar, missing_yf[["start_raw", "end_raw", "date", "revenue", "frame"]]],
                                 ignore_index=True)
            df_edgar = df_edgar.sort_values("date").reset_index(drop=True)

    # ── Layer 3: fiscal-year-aware Q4 derivation ──────────────────────────
    # For each annual 10-K, find its Q1+Q2+Q3 by matching the original
    # SEC filing dates — not calendar years. This is the key fix for
    # companies like WMT/AAPL/MSFT whose fiscal year ≠ calendar year.
    if annual:
        # deduplicate annual entries: keep last filed per fiscal year end
        df_ann = pd.DataFrame(annual).drop_duplicates(subset="end_raw", keep="last")
        existing_dates = set(df_edgar["date"].dt.date) if not df_edgar.empty else set()

        q4_rows = []
        for _, ann in df_ann.iterrows():
            fy_start = ann["start_raw"]
            fy_end = ann["end_raw"]
            ann_rev = ann["annual_revenue"]

            # The fiscal Q4's calendar-snapped date
            q4_date = fy_end + pd.offsets.QuarterEnd(0)

            # Skip if we already have this quarter
            if q4_date.date() in existing_dates:
                continue

            # Find Q1+Q2+Q3: quarters whose original end date falls
            # strictly inside [fy_start, fy_end)
            if df_edgar.empty or "end_raw" not in df_edgar.columns:
                continue

            end_raw_col = pd.to_datetime(df_edgar["end_raw"], errors="coerce")
            mask = (end_raw_col >= fy_start) & (end_raw_col < fy_end)
            fy_quarters = df_edgar[mask]

            if len(fy_quarters) == 3:
                q4_rev = ann_rev - fy_quarters["revenue"].sum()
                if q4_rev > 0:  # sanity check
                    q4_rows.append({
                        "start_raw": None,
                        "end_raw": fy_end,
                        "date": q4_date,
                        "revenue": q4_rev,
                        "frame": f"CY{fy_end.year}Q4-calc"
                    })

        if q4_rows:
            df_q4 = pd.DataFrame(q4_rows)
            print(f"Layer 3 - calculated Q4s: {len(df_q4)} quarters")
            df_edgar = pd.concat([df_edgar, df_q4], ignore_index=True)

    # ── Final cleanup ─────────────────────────────────────────────────────
    df_final = df_edgar.sort_values("date").reset_index(drop=True)
    df_final = df_final.drop_duplicates(subset="date", keep="last")

    def clean_frame(row):
        if str(row["frame"]).startswith("Q-"):
            d = row["date"]
            return f"CY{d.year}Q{d.quarter}"
        return row["frame"]

    df_final["frame"] = df_final.apply(clean_frame, axis=1)
    df_final = df_final[["date", "revenue", "frame"]].copy()
    print(f"Final dataset: {len(df_final)} quarters")
    return df_final
