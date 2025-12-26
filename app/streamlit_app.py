import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make src importable when running `streamlit run app/streamlit_app.py`
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Auto-prepare processed data on first run (useful for Streamlit Cloud)
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "onion_maharashtra_cleaned.csv"
CONVERT_SCRIPT = BASE_DIR / "scripts" / "convert_raw_to_processed.py"
if not PROCESSED_PATH.exists():
    try:
        st.info("Preparing data (running conversion script)...")
        subprocess.run([sys.executable, str(CONVERT_SCRIPT)], check=True)
    except Exception as e:
        # If data prep fails, show a friendly message but continue — app can still load if data is present elsewhere
        st.warning(f"Data preparation failed: {e}")

from src.data_prep import load_clean_daily
from src.forecast_service import make_forecast


def load_css_file(css_file_path):
    with open(css_file_path) as f:
        return st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(
    page_title="Mandi-Sense: Onion Price Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
css_path = BASE_DIR / "app" / "style.css"
if css_path.exists():
    load_css_file(css_path)

st.title("🧅 Mandi-Sense: Maharashtra Onion Forecast")

st.markdown(
    """
    <div style="text-align: center; margin-bottom: 0.5rem;">
        Forecast short-term onion prices in Maharashtra using historical mandi data and RandomForest regression.
    </div>
    """,
    unsafe_allow_html=True
)

# Load historical data
daily = load_clean_daily()
daily = daily.sort_values("Date")

# Sidebar
st.sidebar.header("⚙️ Forecast Controls")
horizon = st.sidebar.slider("Forecast horizon (days)", min_value=1, max_value=14, value=7)
st.sidebar.info(
    """
    **Model:** RandomForest Regressor  
    **Data:** Aggregated daily modal prices from Maharashtra mandis.
    """
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Decision Support:** Use these price trends to plan procurement or sales.")

# Key Metrics Area
last_date = daily["Date"].max()
last_price = daily.iloc[-1]["Avg_Modal_Price"]
prev_price = daily.iloc[-2]["Avg_Modal_Price"]
price_delta = last_price - prev_price

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Latest Date", last_date.strftime("%d %b %Y"))
with col2:
    st.metric("Current Price", f"₹{last_price:.2f}/q", delta=f"₹{price_delta:.2f}")
with col3:
    st.metric("Data Points", f"{len(daily)} days")

# Make forecast
forecast_df = make_forecast(horizon_days=horizon)

# Combine last N days of history + forecast for plotting
history_window_days = 90
history_tail = daily.tail(history_window_days).copy()
history_tail = history_tail[["Date", "Avg_Modal_Price"]]
history_tail = history_tail.rename(columns={"Avg_Modal_Price": "Historical"})

# Prepare plot data
plot_df = pd.merge(
    history_tail,
    forecast_df.rename(columns={"Predicted_Price": "Forecast"}),
    how="outer",
    on="Date",
)
plot_df = plot_df.set_index("Date")

st.subheader("📈 Price Trend & Forecast")
st.line_chart(plot_df, color=["#2E7D32", "#FF9800"]) # Green for history, Orange for forecast (Streamlit cycles colors, but we set preference in config or here)

col_chart, col_data = st.columns([2, 1])

with col_chart:
    st.markdown("##### Visual Trend")
    # (The chart is already above, or we could move it here. Leaving it full width above is better for mobile)
    st.info("The orange line (if customizable) or the second line represents the future forecast.")

with col_data:
    st.write("##### Forecast Values")
    st.dataframe(
        forecast_df.style.format({"Predicted_Price": "₹{:.2f}"}),
        use_container_width=True
    )

st.markdown("---")
st.caption(
    "Disclaimer: This is a predictive model for educational and portfolio demonstration purposes. "
    "Do not use for financial trading without validation."
)