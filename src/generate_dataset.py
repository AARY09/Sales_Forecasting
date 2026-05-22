"""Generate realistic synthetic daily sales time-series data."""
import numpy as np
import pandas as pd

from config import RAW_DATA_PATH, DATE_COL, TARGET_COL


def generate_sales_data(
    start_date: str = "2022-01-01",
    n_days: int = 730,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create daily sales with trend, weekly seasonality, monthly peaks, and noise.
    Mimics retail / e-commerce patterns for forecasting demos.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    t = np.arange(n_days)

    trend = 800 + 0.35 * t
    weekly = 120 * np.sin(2 * np.pi * t / 7)
    monthly = 80 * np.sin(2 * np.pi * t / 30)
    promo_spikes = rng.choice([0, 1], size=n_days, p=[0.92, 0.08]) * rng.integers(200, 500, n_days)
    noise = rng.normal(0, 45, n_days)

    sales = np.maximum(trend + weekly + monthly + promo_spikes + noise, 50).round(2)

    df = pd.DataFrame({DATE_COL: dates, TARGET_COL: sales})
    df["units_sold"] = (sales / rng.uniform(18, 28, n_days)).astype(int)
    df["region"] = rng.choice(["North", "South", "East", "West"], n_days)
    return df


def save_dataset(path=None) -> pd.DataFrame:
    path = path or RAW_DATA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_sales_data()
    df.to_csv(path, index=False)
    print(f"Dataset saved: {path} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    save_dataset()
