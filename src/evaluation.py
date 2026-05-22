"""Model evaluation metrics: RMSE and MAPE."""
import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_model(
    y_true: pd.Series, y_pred: pd.Series, model_name: str
) -> dict:
    metrics = {
        "model": model_name,
        "rmse": round(rmse(y_true, y_pred), 2),
        "mape_pct": round(mape(y_true, y_pred), 2),
    }
    return metrics


def compare_models(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results).sort_values("rmse")
    df["rank_rmse"] = range(1, len(df) + 1)
    return df
