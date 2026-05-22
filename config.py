"""Project configuration."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "sales_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "sales_cleaned.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

DATE_COL = "date"
TARGET_COL = "sales"
TEST_RATIO = 0.2
FORECAST_HORIZON = 30  # days ahead for business planning

# Feature engineering
LAG_FEATURES = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]

# ARIMA
ARIMA_SEASONAL_PERIOD = 7  # weekly seasonality for daily data
