import csv
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RAW = HERE / "raw_data" / "current-daily-price-variation-mandi.csv"
OUT_DIR = HERE / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "onion_maharashtra_cleaned.csv"

def main():
    if not RAW.exists():
        print(f"Raw file not found: {RAW}")
        return 1

    df = pd.read_csv(RAW)

    # Normalize column names (the raw file uses odd names)
    df = df.rename(
        columns={
            "Arrival_Date": "Date",
            "Modal_x0020_Price": "Modal_Price",
            "Min_x0020_Price": "Min_Price",
            "Max_x0020_Price": "Max_Price",
        }
    )

    # Filter for Maharashtra + Onion
    df_filtered = df[
        (df["State"].str.contains("Maharashtra", na=False, case=False))
        & (df["Commodity"].str.contains("Onion", na=False, case=False))
    ].copy()

    if df_filtered.empty:
        print("No Onion rows for Maharashtra found in raw file.")
        return 1

    # Keep only expected columns
    cols = ["Date", "Market", "Modal_Price", "Min_Price", "Max_Price"]
    for c in cols:
        if c not in df_filtered.columns:
            df_filtered[c] = None

    df_filtered = df_filtered[cols]

    # Parse dates and coerce numeric
    df_filtered["Date"] = pd.to_datetime(df_filtered["Date"], dayfirst=True, errors="coerce")
    df_filtered["Modal_Price"] = pd.to_numeric(df_filtered["Modal_Price"], errors="coerce")
    df_filtered["Min_Price"] = pd.to_numeric(df_filtered["Min_Price"], errors="coerce")
    df_filtered["Max_Price"] = pd.to_numeric(df_filtered["Max_Price"], errors="coerce")

    df_filtered = df_filtered.dropna(subset=["Date"])  # drop unparseable rows

    df_filtered.to_csv(OUT, index=False)
    print(f"Wrote processed CSV to: {OUT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
