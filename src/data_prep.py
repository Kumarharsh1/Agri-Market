from pathlib import Path

import pandas as pd

# Base path: project_root/data/processed
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_clean_daily() -> pd.DataFrame:
    """
    Load the cleaned Onion + Maharashtra data and aggregate
    to one row per date with average modal price.
    """
    csv_path = DATA_DIR / "onion_maharashtra_cleaned.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Required data file not found: {csv_path}\n"
            "Place `onion_maharashtra_cleaned.csv` under the project data/processed folder,\n"
            "or update DATA_DIR in src/data_prep.py to point to the correct location."
        )
    df = pd.read_csv(csv_path)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")

    daily = (
        df.groupby("Date", as_index=False)
        .agg(
            Avg_Modal_Price=("Modal_Price", "mean"),
            Min_Price=("Min_Price", "mean"),
            Max_Price=("Max_Price", "mean"),
            Num_Markets=("Market", "nunique"),
        )
        .sort_values("Date")
    )

    return daily