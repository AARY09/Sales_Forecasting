"""Feature engineering for ML forecasting models."""
import pandas as pd

from config import DATE_COL, TARGET_COL, LAG_FEATURES, ROLLING_WINDOWS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar, lag, and rolling features for supervised learning."""
    data = df.copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL])
    data = data.sort_values(DATE_COL).reset_index(drop=True)

    dt = data[DATE_COL]
    data["day_of_week"] = dt.dt.dayofweek
    data["day_of_month"] = dt.dt.day
    data["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    data["month"] = dt.dt.month
    data["quarter"] = dt.dt.quarter
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

    for lag in LAG_FEATURES:
        data[f"lag_{lag}"] = data[TARGET_COL].shift(lag)

    for window in ROLLING_WINDOWS:
        data[f"rolling_mean_{window}"] = (
            data[TARGET_COL].shift(1).rolling(window=window, min_periods=1).mean()
        )
        data[f"rolling_std_{window}"] = (
            data[TARGET_COL].shift(1).rolling(window=window, min_periods=1).std()
        )

    data = data.dropna().reset_index(drop=True)
    return data


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model input columns (exclude date, target, and non-numeric)."""
    exclude = {DATE_COL, TARGET_COL, "region"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def train_test_split_ts(
    df: pd.DataFrame, test_ratio: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split — no shuffle for time series."""
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
