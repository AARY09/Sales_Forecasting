# Sales Forecasting

A complete sales forecasting pipeline for daily time-series data. This project cleans raw sales data, creates features, trains ARIMA and machine learning models, evaluates forecasting performance, and generates visual and business-ready insights.

## 🚀 Features

- End-to-end forecasting workflow from raw data to model evaluation
- Time-series data cleaning and missing value handling
- Exploratory data analysis and trend/seasonality visualization
- Feature engineering with lag, rolling, and calendar features
- ARIMA and Gradient Boosting regression models
- Forecast evaluation with RMSE and MAPE
- Output reports and charts for planning inventory and revenue

## 📁 Repository Structure

- `main.py` — Runs the full forecasting pipeline
- `config.py` — Configuration values, folder paths, forecast settings
- `requirements.txt` — Python dependencies
- `data/raw/` — Raw input dataset(s)
- `data/processed/` — Cleaned and prepared datasets
- `src/`
  - `generate_dataset.py` — Generates sample sales data for testing
  - `data_cleaning.py` — Cleans and preprocesses sales data
  - `eda.py` — Exploratory data analysis functions and charts
  - `features.py` — Creates lag, rolling, and calendar features
  - `models/`
    - `arima_model.py` — Time-series ARIMA forecasting
    - `ml_model.py` — Machine learning forecasting models
  - `evaluation.py` — Model evaluation metrics
  - `visualization.py` — Plots forecasts and results
  - `insights.py` — Generates business insight outputs
- `notebooks/` — Jupyter notebooks for exploratory analysis
- `outputs/` — Generated charts, figures, and reports

## ✅ Getting Started

1. Clone the repository

```bash
git clone https://github.com/AARY09/Sales_Forecasting.git
cd Sales_Forecasting
```

2. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the forecasting pipeline

```bash
python main.py
```

## 📥 Data Requirements

Place your dataset at `data/raw/sales_data.csv` and include at minimum:

- `date` — daily timestamps in `YYYY-MM-DD` format
- `sales` — numeric sales values

Optional columns:
- `units_sold`
- `region`

If `data/raw/sales_data.csv` is missing, the project may generate a sample dataset for testing.

## 📊 Output

Generated files are written into:

- `outputs/figures/` — charts and visualizations
- `outputs/reports/` — summary reports and business insights

## 🧠 Models

- **ARIMA** for univariate time-series forecasting
- **Gradient Boosting** for feature-based machine learning forecasting

The pipeline evaluates both approaches and reports performance using RMSE and MAPE.

## ✨ Notes

- Designed for daily sales forecasting and inventory planning
- Easy to extend with new datasets or additional model types
- Includes notebook support for exploratory analysis

## 📌 Requirements

- Python 3.10+
- See `requirements.txt` for package dependencies
