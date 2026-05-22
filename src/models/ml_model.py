"""Machine learning forecaster (Gradient Boosting on engineered features)."""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from config import DATE_COL, TARGET_COL
from src.features import get_feature_columns


class MLForecaster:
    """Supervised ML model using lag and calendar features."""

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []

    def fit(self, train_df: pd.DataFrame) -> "MLForecaster":
        self.feature_cols = get_feature_columns(train_df)
        X = train_df[self.feature_cols]
        y = train_df[TARGET_COL]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        X = df[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        return pd.Series(self.model.predict(X_scaled), index=df.index)

    def forecast_future(
        self, history: pd.DataFrame, horizon: int, feature_fn
    ) -> pd.DataFrame:
        """
        Recursive multi-step forecast: append predictions and re-engineer features.
        feature_fn: callable that takes history df and returns featured df.
        """
        extended = history.copy()
        future_dates = []
        future_preds = []

        for _ in range(horizon):
            next_date = extended[DATE_COL].max() + pd.Timedelta(days=1)
            stub = pd.DataFrame({DATE_COL: [next_date], TARGET_COL: [np.nan]})
            if "units_sold" in extended.columns:
                stub["units_sold"] = extended["units_sold"].iloc[-1]
            if "region" in extended.columns:
                stub["region"] = extended["region"].iloc[-1]

            extended = pd.concat([extended, stub], ignore_index=True)
            hist_for_features = extended.copy()
            hist_for_features[TARGET_COL] = hist_for_features[TARGET_COL].ffill()
            featured = feature_fn(hist_for_features)
            row = featured.iloc[[-1]]
            pred = float(self.predict(row).iloc[0])
            extended.loc[extended.index[-1], TARGET_COL] = pred
            future_dates.append(next_date)
            future_preds.append(pred)

        return pd.DataFrame({DATE_COL: future_dates, "forecast": future_preds})
