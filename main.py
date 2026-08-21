from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.data_prep import load_clean_daily
from src.forecast_service import make_forecast

app = FastAPI(title="Agri-Market Price Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

_daily_series = None


def get_daily_series():
    global _daily_series
    if _daily_series is None:
        _daily_series = load_clean_daily()
    return _daily_series


@app.get("/")
def home():
    return {"status": "ok", "message": "Agri-Market forecast API is running"}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/history")
def history(days: int = 90):
    """Last N days of historical average price."""
    df = get_daily_series()
    recent = df.tail(days)
    return {
        "dates": recent["Date"].astype(str).tolist(),
        "avg_modal_price": recent["Avg_Modal_Price"].tolist(),
    }


@app.get("/forecast")
def forecast(horizon: int = 7):
    """Forecast the next `horizon` days of average onion price."""
    if horizon < 1 or horizon > 14:
        raise HTTPException(status_code=422, detail="horizon must be between 1 and 14 days")

    try:
        result = make_forecast(horizon_days=horizon)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Forecasting failed: {error}") from error

    return {
        "horizon": horizon,
        "dates": result["Date"].astype(str).tolist(),
        "predicted_price": result["Predicted_Price"].tolist(),
    }
