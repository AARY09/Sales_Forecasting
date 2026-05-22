"""Data cleaning and validation for sales time-series."""
import pandas as pd

from config import DATE_COL, TARGET_COL, PROCESSED_DATA_PATH


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize dates, remove duplicates, handle missing values and outliers.
    """
    data = df.copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce")
    data = data.dropna(subset=[DATE_COL, TARGET_COL])
    data = data.sort_values(DATE_COL).drop_duplicates(subset=[DATE_COL], keep="last")

    # Interpolate short gaps in daily series
    data = data.set_index(DATE_COL).asfreq("D")
    data[TARGET_COL] = data[TARGET_COL].interpolate(method="linear").ffill().bfill()
    if "units_sold" in data.columns:
        data["units_sold"] = data["units_sold"].interpolate(method="linear").ffill().bfill()
    data = data.reset_index()

    # Cap extreme outliers (beyond 3 IQR) — winsorize for stability
    q1, q3 = data[TARGET_COL].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
    data[TARGET_COL] = data[TARGET_COL].clip(lower=lower, upper=upper)

    data[TARGET_COL] = data[TARGET_COL].astype(float)
    return data.reset_index(drop=True)


def load_and_clean(raw_path, processed_path=None) -> pd.DataFrame:
    processed_path = processed_path or PROCESSED_DATA_PATH
    df = pd.read_csv(raw_path, parse_dates=[DATE_COL])
    cleaned = clean_sales_data(df)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(processed_path, index=False)
    return cleaned
