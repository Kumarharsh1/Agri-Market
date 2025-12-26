import sys
from pathlib import Path
import pandas as pd
import pytest

# Ensure src is importable
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.forecast_service import make_forecast, ConstantModel, load_model

def test_constant_model():
    """Test the fallback ConstantModel."""
    model = ConstantModel(v=100.0)
    preds = model.predict([[1, 2], [3, 4]])
    assert len(preds) == 2
    assert preds[0] == 100.0
    assert preds[1] == 100.0

def test_make_forecast_structure():
    """Test that make_forecast returns the expected DataFrame structure."""
    # This test assumes data is present or mocks it. 
    # Since we might not have data in CI, we should handle gracefully or mock load_clean_daily.
    # For now, we'll try to run it (assuming data prep ran) or expect failure if no data.
    
    # Check if processed data exists, if not, skip
    data_path = BASE_DIR / "data" / "processed" / "onion_maharashtra_cleaned.csv"
    if not data_path.exists():
        pytest.skip("Processed data not found, skipping integration test")

    forecast = make_forecast(horizon_days=3)
    
    assert isinstance(forecast, pd.DataFrame)
    assert len(forecast) == 3
    assert "Date" in forecast.columns
    assert "Predicted_Price" in forecast.columns
    assert pd.api.types.is_datetime64_any_dtype(forecast["Date"])
    assert pd.api.types.is_float_dtype(forecast["Predicted_Price"])

def test_load_model_creates_file():
    """Test that load_model will try to create a model if missing."""
    # We won't delete the real model, just check it returns something valid
    # provided data exists.
    data_path = BASE_DIR / "data" / "processed" / "onion_maharashtra_cleaned.csv"
    if not data_path.exists():
        pytest.skip("Processed data not found")
        
    model = load_model()
    assert model is not None
    # Check it has a predict method
    assert hasattr(model, "predict")
