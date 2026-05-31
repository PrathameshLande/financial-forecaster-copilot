"""
Sales Rep Analytics — Data Cleaning & Merge Pipeline
=====================================================
Inputs  : employee_info.csv / mileage.csv / rep_perf.csv / sales_data.csv
Output  : merged_rep_analysis.csv  +  merged_rep_analysis.xlsx

Grain of final dataset: one row per rep per quarter

Run:
    python clean_and_merge.py

Edit FILE PATHS section below to point to your actual files.
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────────────────────────────────────
# FILE PATHS — edit these to match your file locations
# ─────────────────────────────────────────────────────────────────────────────
EMPLOYEE_INFO_FILE = "employee_info.csv"
MILEAGE_FILE       = "mileage.csv"
REP_PERF_FILE      = "rep_perf.csv"
SALES_FILE         = "sales_data.csv"
OUTPUT_CSV         = "merged_rep_analysis.csv"
OUTPUT_XLSX        = "merged_rep_analysis.xlsx"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load(path, label):
    """Load CSV or Excel, strip column name whitespace, print shape."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {label} at: {path}")
    if path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"  Loaded {label}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def normalize_yr_qtr(series):
    """
    Normalize quarter strings to a consistent 'YYYY QN' format.
    Handles common variants: '2024 Q1', '2024Q1', 'Q1 2024', '2024-Q1', '1Q2024'
    """
    s = series.astype(str).str.strip()
    # already "YYYY QN"
    mask1 = s.str.match(r"^\d{4} Q[1-4]$")
    # "YYYY-QN" or "YYYYQN"
    s = s.str.replace(r"(\d{4})[-]?Q([1-4])", r"\1 Q\2", regex=True)
    # "QN YYYY"
    s = s.str.replace(r"Q([1-4]) (\d{4})", r"\2 Q\1", regex=True)
    # "NQ YYYY"  (e.g. "1Q2024")
    s = s.str.replace(r"^([1-4])Q(\d{4})$", r"\2 Q\1", regex=True)
    return s


def attainment_tier(pct):
    """Classify quota attainment into human-readable tiers."""
    if pd.isna(pct):
        return "Unknown"
    if pct >= 120:
        return "Overachiever (≥120%)"
    if pct >= 100:
        return "At Quota (100–119%)"
    if pct >= 80:
        return "Near Quota (80–99%)"
    if pct >= 50:
        return "Below Quota (50–79%)"
    return "Far Below Quota (<50%)"


# ─────────────────────────────────────────────────────────────────────────────
# 1. EMPLOYEE INFO
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Employee Info ────────────────────────────────")

emp = load(EMPLOYEE_INFO_FILE, "Employee Info")

emp["Employee Number"]   = emp["Employee Number"].astype(str).str.strip()
emp["Original Hire Date"] = pd.to_datetime(emp["Original Hire Date"], errors="coerce")
emp["Termination Date"]   = pd.to_datetime(emp["Termination Date"],   errors="coerce")

# Is the rep currently active?
emp["Is Active"] = emp["Termination Date"].isna()

# Tenure: from hire to termination (or today if still active)
today = pd.Timestamp.today().normalize()
emp["Tenure Years"] = (
    emp["Termination Date"].fillna(today) - emp["Original Hire Date"]
).dt.days / 365.25
emp["Tenure Years"] = emp["Tenure Years"].round(2)

# Keep only what we need for the merge
emp_clean = emp[[
    "Employee Number",
    "Status Classification",
    "Position",
    "Original Hire Date",
    "Termination Date",
    "Is Active",
    "Tenure Years",
]].drop_duplicates(subset="Employee Number")   # one row per rep

print(f"  After dedup: {len(emp_clean):,} unique reps")


# ─────────────────────────────────────────────────────────────────────────────
# 2. MILEAGE
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Mileage ──────────────────────────────────────")

mil = load(MILEAGE_FILE, "Mileage")

mil["Employee Number"] = mil["Employee Number"].astype(str).str.strip()
mil["Date"]            = pd.to_datetime(mil["Date"], errors="coerce")
mil["Bus Miles"]       = pd.to_numeric(mil["Bus Miles"], errors="coerce").fillna(0)

# Drop rows with no date or employee
mil = mil.dropna(subset=["Date", "Employee Number"])

# Build Yr Qtr from the date (matches the format we'll use everywhere)
mil["Yr Qtr"] = (
    mil["Date"].dt.year.astype(str) + " Q" + mil["Date"].dt.quarter.astype(str)
)

# Aggregate to employee-quarter level (sum all trips within the quarter)
mil_agg = (
    mil.groupby(["Employee Number", "Yr Qtr"])
    .agg(
        Total_Miles   = ("Bus Miles", "sum"),
        Mileage_Trips = ("Bus Miles", "count"),   # how many entries = how many trips logged
    )
    .reset_index()
)

print(f"  Aggregated: {len(mil_agg):,} employee-quarter rows")


# ─────────────────────────────────────────────────────────────────────────────
# 3. REP PERFORMANCE BY QUARTER
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Rep Performance ──────────────────────────────")

rp = load(REP_PERF_FILE, "Rep Performance")

rp["Employee Number"] = rp["Employee Number"].astype(str).str.strip()
rp["Sales"]           = pd.to_numeric(rp["Sales"], errors="coerce")
rp["Goal"]            = pd.to_numeric(rp["Goal"],  errors="coerce")
rp["Yr Qtr"]          = normalize_yr_qtr(rp["Yr Qtr"])

# ── Calculated fields ──────────────────────────────────────────────────────
# Quota Attainment: what % of goal did the rep hit?
rp["Quota Attainment %"] = (rp["Sales"] / rp["Goal"] * 100).round(2)

# Raw gap: negative = missed quota, positive = beat quota
rp["Gap to Quota $"]     = (rp["Sales"] - rp["Goal"]).round(2)

# Human-readable tier
rp["Attainment Tier"]    = rp["Quota Attainment %"].apply(attainment_tier)

# Boolean flag for easy filtering
rp["Hit Quota"]          = rp["Quota Attainment %"] >= 100

print(f"  Quarters covered: {sorted(rp['Yr Qtr'].unique())}")
print(f"  Reps: {rp['Employee Number'].nunique():,} unique")
print(f"  Avg quota attainment: {rp['Quota Attainment %'].mean():.1f}%")

rp_clean = rp[[
    "Employee Number",
    "Yr Qtr",
    "Division",
    "DivisionName",
    "Territory",
    "TerritoryName",
    "Position",
    "Goal",
    "Sales",
    "Quota Attainment %",
    "Gap to Quota $",
    "Attainment Tier",
    "Hit Quota",
]]


# ─────────────────────────────────────────────────────────────────────────────
# 4. SALES DATA  (transaction level → aggregate before merging)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Sales Data ───────────────────────────────────")

sd = load(SALES_FILE, "Sales Data")

# ── 4a. Remove duplicate rows flagged in the data ──────────────────────────
dup_col = "Is Duplicate Row?"
if dup_col in sd.columns:
    before = len(sd)
    sd = sd[sd[dup_col].astype(str).str.strip().str.upper() != "TRUE"]
    print(f"  Dropped {before - len(sd):,} flagged duplicate rows")

# ── 4b. Keep only useful columns ───────────────────────────────────────────
SALES_KEEP = [
    # Join keys
    "Employee Number", "Current Employee Number",
    "CustomerNumber", "Current ParentNumber",
    "SalesOrderNumber", "InvoiceNumber",

    # Time
    "SalesOrderDate", "InvoiceDate", "Invoice Yr-Qtr",

    # Revenue metrics
    "NetSales", "GrossSales", "Quantity",

    # Product
    "Current ProductFamily", "Finance Product Family",
    "Current ProductLine_Chr", "ItemDescription", "ItemNumber",
    "ItemType", "Core20", "Active Product",
    "FxItem", "RxItem", "BundledItem", "Channel",

    # Customer
    "Current Territory", "Current Division",
    "Current CustomerCategoryType", "Current CustomerType",
    "Current CustomerLevel", "Current BusinessUnit",
    "Current State", "Current City",

    # Rep context
    "RepLevel", "Rep Type",

    # Flags
    "OPP", "Online Seller", "Potential Diverter",
    "Account Classification", "Clustered Account Type",
]

existing = [c for c in SALES_KEEP if c in sd.columns]
dropped  = [c for c in SALES_KEEP if c not in sd.columns]
if dropped:
    print(f"  Note: these expected columns weren't found → {dropped}")

sd = sd[existing].copy()
print(f"  Kept {len(existing)} of {len(SALES_KEEP)} target columns "
      f"(dropped from original: {sd.shape[1] - len(existing)} extra cols removed)")

# ── 4c. Type cleanup ───────────────────────────────────────────────────────
sd["Employee Number"]         = sd["Employee Number"].astype(str).str.strip()
sd["Current Employee Number"] = sd.get("Current Employee Number", sd["Employee Number"]).astype(str).str.strip()
sd["SalesOrderDate"]          = pd.to_datetime(sd.get("SalesOrderDate"), errors="coerce")
sd["InvoiceDate"]             = pd.to_datetime(sd.get("InvoiceDate"),    errors="coerce")
sd["NetSales"]                = pd.to_numeric(sd.get("NetSales"),         errors="coerce")
sd["GrossSales"]              = pd.to_numeric(sd.get("GrossSales"),       errors="coerce")
sd["Quantity"]                = pd.to_numeric(sd.get("Quantity"),         errors="coerce")

# Normalize quarter format to match rep_perf
if "Invoice Yr-Qtr" in sd.columns:
    sd["Invoice Yr-Qtr"] = normalize_yr_qtr(sd["Invoice Yr-Qtr"])

# ── 4d. Aggregate to employee-quarter level ────────────────────────────────
# Rep Perf is already at this grain; Sales is transaction-level.
# We roll Sales up so both can merge cleanly.
sales_agg = (
    sd.groupby(["Employee Number", "Invoice Yr-Qtr"])
    .agg(
        Txn_NetSales         = ("NetSales",        "sum"),
        Txn_GrossSales       = ("GrossSales",      "sum"),
        Txn_Quantity         = ("Quantity",         "sum"),
        Num_Orders           = ("SalesOrderNumber", "nunique"),
        Num_Customers        = ("CustomerNumber",   "nunique"),
        Avg_Order_Value      = ("NetSales",         "mean"),
        Num_Rx_Items         = ("RxItem",           lambda x: (x.astype(str).str.upper() == "TRUE").sum()),
        Num_Fx_Items         = ("FxItem",           lambda x: (x.astype(str).str.upper() == "TRUE").sum()),
        Num_Core20_Items     = ("Core20",           lambda x: (x.astype(str).str.upper() == "TRUE").sum()),
    )
    .reset_index()
    .rename(columns={"Invoice Yr-Qtr": "Yr Qtr"})
)

sales_agg["Avg_Order_Value"] = sales_agg["Avg_Order_Value"].round(2)
print(f"  Aggregated: {len(sales_agg):,} employee-quarter rows from {len(sd):,} transactions")


# ─────────────────────────────────────────────────────────────────────────────
# 5. MERGE — build one master dataset
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Merging ──────────────────────────────────────")

# Start with Rep Perf (the spine — one row per rep per quarter)
master = rp_clean.copy()
print(f"  Start (rep_perf):         {len(master):,} rows")

# + Employee Info
master = master.merge(
    emp_clean.drop(columns=["Position"], errors="ignore"),  # Position already in rep_perf
    on="Employee Number",
    how="left"
)
print(f"  After + Employee Info:    {len(master):,} rows")

# + Mileage (on rep + quarter)
master = master.merge(
    mil_agg,
    on=["Employee Number", "Yr Qtr"],
    how="left"
)
print(f"  After + Mileage:          {len(master):,} rows")
unmatched_mil = master["Total_Miles"].isna().sum()
if unmatched_mil:
    print(f"    → {unmatched_mil} rep-quarters had no mileage logged")

# + Sales aggregate (on rep + quarter)
master = master.merge(
    sales_agg,
    on=["Employee Number", "Yr Qtr"],
    how="left"
)
print(f"  After + Sales:            {len(master):,} rows")
unmatched_sales = master["Txn_NetSales"].isna().sum()
if unmatched_sales:
    print(f"    → {unmatched_sales} rep-quarters had no sales transactions")


# ─────────────────────────────────────────────────────────────────────────────
# 6. FINAL CALCULATED FIELDS ON MERGED DATASET
# ─────────────────────────────────────────────────────────────────────────────

# Sales efficiency: miles driven per $1,000 of revenue
# Lower = more efficient (more $ per mile driven)
master["Miles per $1K Revenue"] = (
    master["Total_Miles"] / (master["Txn_NetSales"] / 1000)
).replace([np.inf, -np.inf], np.nan).round(2)

# Revenue per customer (avg spend per account in the quarter)
master["Revenue per Customer"] = (
    master["Txn_NetSales"] / master["Num_Customers"]
).replace([np.inf, -np.inf], np.nan).round(2)

# Cross-check: does transaction-level NetSales agree with rep_perf Sales?
# (rep_perf Sales = what the CRM recorded; Txn_NetSales = invoiced transactions)
# A big gap here can flag data mismatches or returns not yet credited
master["Sales vs Txn Diff $"] = (master["Sales"] - master["Txn_NetSales"]).round(2)
master["Sales vs Txn Diff %"] = (
    master["Sales vs Txn Diff $"] / master["Sales"] * 100
).replace([np.inf, -np.inf], np.nan).round(2)


# ─────────────────────────────────────────────────────────────────────────────
# 7. COLUMN ORDER  (logical grouping for readability)
# ─────────────────────────────────────────────────────────────────────────────
COL_ORDER = [
    # Rep identity
    "Employee Number", "Position", "Status Classification", "Is Active",
    "Original Hire Date", "Termination Date", "Tenure Years",

    # Period / territory
    "Yr Qtr", "Division", "DivisionName", "Territory", "TerritoryName",

    # Quota & attainment (from rep_perf)
    "Goal", "Sales", "Quota Attainment %", "Gap to Quota $",
    "Attainment Tier", "Hit Quota",

    # Transaction sales (from sales_data)
    "Txn_NetSales", "Txn_GrossSales", "Txn_Quantity",
    "Num_Orders", "Num_Customers", "Avg_Order_Value",
    "Num_Rx_Items", "Num_Fx_Items", "Num_Core20_Items",

    # Cross-check
    "Sales vs Txn Diff $", "Sales vs Txn Diff %",

    # Revenue efficiency
    "Revenue per Customer",

    # Mileage
    "Total_Miles", "Mileage_Trips", "Miles per $1K Revenue",
]

# Add any columns that ended up in master but aren't in our order list
extra_cols = [c for c in master.columns if c not in COL_ORDER]
final_col_order = COL_ORDER + extra_cols
final_col_order = [c for c in final_col_order if c in master.columns]

master = master[final_col_order]


# ─────────────────────────────────────────────────────────────────────────────
# 8. EXPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Saving ───────────────────────────────────────")

master.to_csv(OUTPUT_CSV, index=False)
print(f"  ✅ {OUTPUT_CSV}")

master.to_excel(OUTPUT_XLSX, index=False, engine="openpyxl")
print(f"  ✅ {OUTPUT_XLSX}")

print(f"\n── Summary ──────────────────────────────────────")
print(f"  Final shape     : {master.shape[0]:,} rows × {master.shape[1]} columns")
print(f"  Unique reps     : {master['Employee Number'].nunique():,}")
print(f"  Quarters        : {sorted(master['Yr Qtr'].unique())}")
print(f"  Active reps     : {master['Is Active'].sum() if 'Is Active' in master else 'N/A'}")
print(f"  Avg attainment  : {master['Quota Attainment %'].mean():.1f}%")
print(f"  Reps at quota   : {master['Hit Quota'].sum()} / {len(master)}")
print(f"\nDone.")
