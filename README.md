# AI-Based Sales Forecasting System

End-to-end pipeline to predict sales from historical time-series data using **ARIMA** and **machine learning**, evaluate with **RMSE/MAPE**, and generate **inventory & revenue planning** insights.

## Pipeline

```
Dataset → Data Cleaning → EDA + Trend Analysis → Feature Engineering
    → Train/Test Split → ARIMA + ML Models → Evaluation
    → Forecast Visualization → Business Insight Generation
```

## Quick start

```bash
cd sales_forecasting
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python main.py
```

Outputs land in `outputs/figures/` and `outputs/reports/`.

## Project structure

| Path | Purpose |
|------|---------|
| `main.py` | Runs the full pipeline |
| `config.py` | Paths, horizons, feature settings |
| `src/generate_dataset.py` | Synthetic daily sales data (replace with your CSV) |
| `src/data_cleaning.py` | Missing values, outliers, daily frequency |
| `src/eda.py` | Trends, seasonality, decomposition plots |
| `src/features.py` | Lags, rolling stats, calendar features |
| `src/models/arima_model.py` | Auto-ARIMA / SARIMA |
| `src/models/ml_model.py` | Gradient Boosting regressor |
| `src/evaluation.py` | RMSE & MAPE |
| `src/visualization.py` | Forecast & metric charts |
| `src/insights.py` | Business recommendations JSON/TXT |
| `notebooks/01_eda_exploration.ipynb` | Interactive EDA |

## Use your own dataset

Place a CSV at `data/raw/sales_data.csv` with columns:

- `date` — daily timestamps (`YYYY-MM-DD`)
- `sales` — numeric target (revenue or units)

Optional: `units_sold`, `region`. Then run `python main.py` (skip auto-generation if the file exists).

## Models

- **ARIMA**: `pmdarima.auto_arima` with weekly seasonality (SARIMA) for stable univariate forecasts.
- **ML**: Gradient Boosting on lag, rolling, and calendar features for non-linear patterns.

The pipeline selects the best model on the **test holdout** (lowest RMSE) for business insights.

## Resume / portfolio bullets

- Implemented ARIMA and ML models to predict sales trends from historical time-series data.
- Evaluated models using RMSE/MAPE and optimized for stable long-term forecasting.
- Converted forecasts into actionable business insights for inventory and revenue planning.

## Requirements

Python 3.10+. See `requirements.txt`.
