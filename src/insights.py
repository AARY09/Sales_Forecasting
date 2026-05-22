"""Convert forecasts into actionable business insights."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATE_COL, TARGET_COL, FORECAST_HORIZON, REPORTS_DIR


def generate_business_insights(
    cleaned_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    best_future_forecast: pd.DataFrame,
    best_model_name: str,
    horizon: int = FORECAST_HORIZON,
) -> dict:
    """
    Produce inventory and revenue planning recommendations from forecast output.
    """
    recent = cleaned_df.tail(30)
    avg_daily_sales = recent[TARGET_COL].mean()
    forecast_mean = best_future_forecast["forecast"].mean()
    forecast_total = best_future_forecast["forecast"].sum()

    growth_pct = ((forecast_mean - avg_daily_sales) / avg_daily_sales) * 100
    peak_day = best_future_forecast.loc[best_future_forecast["forecast"].idxmax()]
    low_day = best_future_forecast.loc[best_future_forecast["forecast"].idxmin()]

    # Simple inventory: safety stock ~ 7 days at forecasted mean + 15% buffer
    safety_stock_units = int((forecast_mean / 22) * 7 * 1.15)  # ~$22 avg unit price proxy
    reorder_point = int(safety_stock_units * 1.5)

    best_metrics = metrics_df.iloc[0]
    trend_label = "growth" if growth_pct > 2 else "decline" if growth_pct < -2 else "stable"

    insights = {
        "executive_summary": (
            f"Over the next {horizon} days, {best_model_name} projects {trend_label} "
            f"({growth_pct:+.1f}% vs last 30-day average). Expected revenue: ${forecast_total:,.0f}."
        ),
        "revenue_planning": {
            "forecast_horizon_days": horizon,
            "projected_total_revenue": round(forecast_total, 2),
            "projected_daily_avg": round(forecast_mean, 2),
            "recent_30d_daily_avg": round(avg_daily_sales, 2),
            "expected_growth_pct": round(growth_pct, 2),
            "peak_sales_date": str(peak_day[DATE_COL].date()),
            "peak_sales_amount": round(float(peak_day["forecast"]), 2),
            "lowest_sales_date": str(low_day[DATE_COL].date()),
            "lowest_sales_amount": round(float(low_day["forecast"]), 2),
        },
        "inventory_planning": {
            "recommended_safety_stock_units": safety_stock_units,
            "recommended_reorder_point_units": reorder_point,
            "action": (
                "Increase stock before peak demand window"
                if growth_pct > 5
                else "Maintain lean inventory; demand stable or softening"
            ),
        },
        "model_confidence": {
            "selected_model": best_model_name,
            "test_rmse": float(best_metrics["rmse"]),
            "test_mape_pct": float(best_metrics["mape_pct"]),
            "note": "Lower RMSE/MAPE indicates more reliable forecasts for planning.",
        },
        "recommendations": _build_recommendations(growth_pct, forecast_mean, avg_daily_sales),
    }
    return insights


def _build_recommendations(growth_pct: float, forecast_mean: float, recent_avg: float) -> list[str]:
    recs = []
    if growth_pct > 5:
        recs.append("Scale marketing spend on high-conversion days aligned with forecast peaks.")
        recs.append("Pre-order inventory 2–3 weeks ahead to avoid stockouts during demand uplift.")
    elif growth_pct < -5:
        recs.append("Run targeted promotions on predicted low-demand days to smooth revenue.")
        recs.append("Reduce purchase orders by 10–15% until trend reverses.")
    else:
        recs.append("Maintain current inventory policy; demand is relatively stable.")
    if forecast_mean > recent_avg * 1.1:
        recs.append("Allocate additional warehouse capacity for the upcoming 30-day window.")
    recs.append("Re-run forecasts weekly with new sales data to keep plans accurate.")
    return recs


def save_insights(insights: dict, path: Path | None = None) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = path or REPORTS_DIR / "business_insights.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2)
    _write_text_report(insights, REPORTS_DIR / "business_insights.txt")
    return path


def _write_text_report(insights: dict, path: Path) -> None:
    lines = [
        "=" * 60,
        "SALES FORECASTING — BUSINESS INSIGHTS REPORT",
        "=" * 60,
        "",
        insights["executive_summary"],
        "",
        "REVENUE PLANNING",
        "-" * 40,
    ]
    for k, v in insights["revenue_planning"].items():
        lines.append(f"  {k.replace('_', ' ').title()}: {v}")
    lines.extend(["", "INVENTORY PLANNING", "-" * 40])
    for k, v in insights["inventory_planning"].items():
        lines.append(f"  {k.replace('_', ' ').title()}: {v}")
    lines.extend(["", "MODEL CONFIDENCE", "-" * 40])
    for k, v in insights["model_confidence"].items():
        lines.append(f"  {k.replace('_', ' ').title()}: {v}")
    lines.extend(["", "RECOMMENDATIONS", "-" * 40])
    for i, rec in enumerate(insights["recommendations"], 1):
        lines.append(f"  {i}. {rec}")
    path.write_text("\n".join(lines), encoding="utf-8")
