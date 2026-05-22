"""Exploratory data analysis and trend decomposition."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

from config import DATE_COL, TARGET_COL, FIGURES_DIR


def run_eda(df: pd.DataFrame, save_plots: bool = True) -> dict:
    """Compute summary stats, trends, seasonality; optionally save EDA figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ts = df.set_index(DATE_COL)[TARGET_COL]

    summary = {
        "n_observations": len(df),
        "date_range": f"{df[DATE_COL].min().date()} to {df[DATE_COL].max().date()}",
        "mean_sales": round(ts.mean(), 2),
        "std_sales": round(ts.std(), 2),
        "min_sales": round(ts.min(), 2),
        "max_sales": round(ts.max(), 2),
        "mom_growth_pct": round(ts.pct_change(30).iloc[-1] * 100, 2)
        if len(ts) > 30
        else None,
    }

    if save_plots:
        _plot_time_series(df)
        _plot_distribution(ts)
        _plot_decomposition(ts)

    return summary


def _plot_time_series(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df[DATE_COL], df[TARGET_COL], linewidth=0.9, color="#2563eb")
    ax.set_title("Daily Sales Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales ($)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_time_series.png", dpi=120)
    plt.close(fig)


def _plot_distribution(ts: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(ts, kde=True, ax=axes[0], color="#059669")
    axes[0].set_title("Sales Distribution")
    df_tmp = ts.reset_index()
    df_tmp.columns = [DATE_COL, TARGET_COL]
    df_tmp["dow"] = pd.to_datetime(df_tmp[DATE_COL]).dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    sns.boxplot(data=df_tmp, x="dow", y=TARGET_COL, order=order, ax=axes[1], color="#3b82f6")
    axes[1].set_title("Sales by Day of Week")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_distribution_seasonality.png", dpi=120)
    plt.close(fig)


def _plot_decomposition(ts: pd.Series) -> None:
    period = 7 if len(ts) >= 14 else max(2, len(ts) // 2)
    result = seasonal_decompose(ts, model="additive", period=period, extrapolate_trend="freq")
    fig = result.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle("Trend / Seasonal / Residual Decomposition", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_decomposition.png", dpi=120)
    plt.close(fig)
