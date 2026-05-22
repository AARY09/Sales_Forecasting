"""ARIMA / SARIMA forecasting with auto parameter search."""
import warnings

import numpy as np
import pandas as pd
from pmdarima import auto_arima

from config import DATE_COL, TARGET_COL, ARIMA_SEASONAL_PERIOD


class ARIMAForecaster:
    """Auto-ARIMA wrapper for stable univariate time-series forecasting."""

    def __init__(self, seasonal: bool = True, seasonal_period: int = ARIMA_SEASONAL_PERIOD):
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self.model = None
        self.fitted_values_ = None

    def fit(self, train_df: pd.DataFrame) -> "ARIMAForecaster":
        y = train_df.set_index(DATE_COL)[TARGET_COL]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = auto_arima(
                y,
                seasonal=self.seasonal,
                m=self.seasonal_period if self.seasonal else 1,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                max_p=3,
                max_q=3,
                max_P=2,
                max_Q=2,
                d=None,
                D=None,
                trace=False,
            )
        self.fitted_values_ = pd.Series(
            self.model.predict_in_sample(),
            index=y.index[-len(self.model.predict_in_sample()) :],
        )
        return self

    def predict(self, steps: int, index: pd.DatetimeIndex | None = None) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        forecast = self.model.predict(n_periods=steps)
        if index is not None:
            return pd.Series(forecast, index=index[:steps])
        return pd.Series(forecast)

    def forecast_test(self, test_df: pd.DataFrame) -> pd.Series:
        steps = len(test_df)
        idx = test_df[DATE_COL].values
        return self.predict(steps, index=pd.DatetimeIndex(idx))

    def forecast_future(
        self, last_date: pd.Timestamp, horizon: int
    ) -> pd.DataFrame:
        future_idx = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
        preds = self.predict(horizon, index=future_idx)
        return pd.DataFrame({DATE_COL: future_idx, "forecast": preds.values})
