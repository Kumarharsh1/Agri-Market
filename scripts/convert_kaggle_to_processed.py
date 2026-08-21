"""
Convert the Kaggle 'Indian Agricultural Mandi Prices (2023-2025)' CSV
into the row-level processed format expected by src/data_prep.py's
load_clean_daily(): columns Date, Market, Min_Price, Max_Price, Modal_Price
(one row per market per day; load_clean_daily aggregates it further).

Usage:
    python scripts/convert_kaggle_to_processed.py
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "raw_data" / "onion_kaggle.csv"
OUT_PATH = BASE_DIR / "data" / "processed" / "onion_maharashtra_cleaned.csv"

COMMODITY = "Onion"
STATE = "Maharashtra"


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    df = df[(df["Commodity"] == COMMODITY) & (df["STATE"] == STATE)]
    if df.empty:
        raise ValueError(
            f"No rows found for Commodity={COMMODITY!r}, STATE={STATE!r}. "
            f"Check exact values in the raw CSV (case sensitivity, spelling)."
        )

    df["Date"] = pd.to_datetime(df["Price Date"], format="%m/%d/%Y", errors="coerce")
    before = len(df)
    df = df.dropna(subset=["Date"])
    dropped = before - len(df)
    if dropped:
        print(f"Warning: dropped {dropped} rows with unparseable dates")

    for col in ["Min_Price", "Max_Price", "Modal_Price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Min_Price", "Max_Price", "Modal_Price"])

    out = df[["Date", "Market Name", "Min_Price", "Max_Price", "Modal_Price"]].rename(
        columns={"Market Name": "Market"}
    )
    out = out.sort_values("Date")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(out)} rows to {OUT_PATH}")
    print(f"Date range: {out['Date'].min().date()} to {out['Date'].max().date()}")
    print(f"Distinct markets: {out['Market'].nunique()}")


if __name__ == "__main__":
    main()