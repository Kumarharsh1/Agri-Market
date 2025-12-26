import pandas as pd
import numpy as np
from pathlib import Path

# Setup paths
HERE = Path(__file__).resolve().parents[1]
OUT_DIR = HERE / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "onion_maharashtra_cleaned.csv"

def generate_data():
    print("Generating synthetic data...")
    # Generate dates from 2023-01-01 to 2025-12-01
    dates = pd.date_range(start="2023-01-01", end="2025-12-01", freq="D")
    
    # Generate prices with seasonality and trend
    n = len(dates)
    trend = np.linspace(1000, 2500, n)  # General upward trend
    seasonality = 500 * np.sin(2 * np.pi * dates.dayofyear / 365) # Seasonal Pattern
    noise = np.random.normal(0, 50, n) # REDUCED VOLATILITY FOR SMOOTHER DEMO
    
    prices = trend + seasonality + noise
    prices = np.clip(prices, 500, 8000) # Clip to realistic values
    
    data = []
    markets = ["Pune", "Nashik", "Lasalgaon", "Solapur"]
    
    for date, price in zip(dates, prices):
        # Create 3-4 entries per day for different markets to simulate variance against mean
        daily_volatility = np.random.normal(0, 20, len(markets)) # Reduced daily variance too
        for i, market in enumerate(markets):
            p = price + daily_volatility[i]
            row = {
                "Date": date.strftime("%Y-%m-%d"),
                "Market": market,
                "Modal_Price": round(p, 2),
                "Min_Price": round(p * 0.8, 2),
                "Max_Price": round(p * 1.2, 2)
            }
            data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(OUT, index=False)
    print(f"Generated {len(df)} rows of synthetic data at: {OUT}")

if __name__ == "__main__":
    generate_data()
