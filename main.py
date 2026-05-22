"""
AI-Based Sales Forecasting System
Pipeline: Dataset → Cleaning → EDA → Features → Train/Test → ARIMA + ML → Eval → Viz → Insights
"""
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    TEST_RATIO,
    FORECAST_HORIZON,
    DATE_COL,
    TARGET_COL,
    OUTPUT_DIR,
)
from src.generate_dataset import save_dataset
from src.data_cleaning import load_and_clean
from src.eda import run_eda
from src.features import engineer_features, train_test_split_ts
from src.models.arima_model import ARIMAForecaster
from src.models.ml_model import MLForecaster
from src.evaluation import evaluate_model, compare_models
from src.visualization import plot_forecasts, plot_metrics_comparison
from src.insights import generate_business_insights, save_insights


def main():
    print("\n" + "=" * 60)
    print("  AI-BASED SALES FORECASTING SYSTEM")
    print("=" * 60)

    # 1. Dataset
    print("\n[1/9] Generating / loading dataset...")
    if not RAW_DATA_PATH.exists():
        save_dataset()
    else:
        print(f"  Using existing: {RAW_DATA_PATH}")

    # 2. Data cleaning
    print("\n[2/9] Data cleaning...")
    cleaned = load_and_clean(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    print(f"  Cleaned rows: {len(cleaned)}")

    # 3. EDA + trend analysis
    print("\n[3/9] EDA + trend analysis...")
    summary = run_eda(cleaned)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # 4. Feature engineering
    print("\n[4/9] Feature engineering...")
    featured = engineer_features(cleaned)
    print(f"  Features: {featured.shape[1]} columns, {len(featured)} rows after lags")

    # 5. Train / test split (same cutoff date for ARIMA and ML)
    print("\n[5/9] Train/test split (chronological)...")
    train_clean, test_clean = train_test_split_ts(cleaned, TEST_RATIO)
    cutoff = test_clean[DATE_COL].min()
    train_feat = featured[featured[DATE_COL] < cutoff].copy()
    test_feat = featured[featured[DATE_COL] >= cutoff].copy()
    print(f"  Train: {len(train_clean)} | Test: {len(test_clean)} (ML test rows: {len(test_feat)})")

    # 6. Models — ARIMA + ML
    print("\n[6/9] Training models...")
    print("  - ARIMA (auto_arima)...")
    arima = ARIMAForecaster(seasonal=True)
    arima.fit(train_clean)
    arima_pred = arima.forecast_test(test_clean)
    arima_pred.index = test_clean.index

    print("  - ML (Gradient Boosting)...")
    ml = MLForecaster()
    ml.fit(train_feat)
    ml_pred = ml.predict(test_feat)

    # 7. Evaluation
    print("\n[7/9] Evaluation (RMSE / MAPE)...")
    results = [
        evaluate_model(test_clean[TARGET_COL], arima_pred, "ARIMA"),
        evaluate_model(test_feat[TARGET_COL], ml_pred, "ML (Gradient Boosting)"),
    ]
    metrics_df = compare_models(results)
    print(metrics_df.to_string(index=False))

    best_name = metrics_df.iloc[0]["model"]
    print(f"\n  Best model on test set: {best_name}")

    # Future forecasts
    print("\n[8/9] Forecast visualization...")
    arima_future = arima.forecast_future(test_clean[DATE_COL].max(), FORECAST_HORIZON)
    ml_future = ml.forecast_future(
        cleaned.tail(90),
        FORECAST_HORIZON,
        engineer_features,
    )

    predictions = {
        "ARIMA": arima_pred,
        "ML (Gradient Boosting)": ml_pred,
    }
    future_forecasts = {
        "ARIMA": arima_future,
        "ML (Gradient Boosting)": ml_future,
    }
    plot_forecasts(train_clean, test_clean, predictions, future_forecasts)
    plot_metrics_comparison(metrics_df)

    # 9. Business insights
    print("\n[9/9] Business insight generation...")
    if "ARIMA" in best_name:
        best_future = arima_future
    else:
        best_future = ml_future

    insights = generate_business_insights(
        cleaned, metrics_df, best_future, best_name, FORECAST_HORIZON
    )
    report_path = save_insights(insights)
    print(f"\n{insights['executive_summary']}")
    print("\n  Top recommendations:")
    for rec in insights["recommendations"][:3]:
        print(f"    - {rec}")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Figures:  {OUTPUT_DIR / 'figures'}")
    print(f"  Reports:  {report_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
