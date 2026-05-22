"""Forecast visualization: actual vs predicted, future outlook."""
import matplotlib.pyplot as plt
import pandas as pd

from config import DATE_COL, TARGET_COL, FIGURES_DIR


def plot_forecasts(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictions: dict[str, pd.Series],
    future_forecasts: dict[str, pd.DataFrame] | None = None,
    save_path=None,
) -> None:
    """Overlay train, test actuals, model predictions, and optional future horizon."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = save_path or FIGURES_DIR / "04_model_forecasts.png"

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train_df[DATE_COL], train_df[TARGET_COL], label="Train", color="#94a3b8", linewidth=1)
    ax.plot(test_df[DATE_COL], test_df[TARGET_COL], label="Test (Actual)", color="#1e293b", linewidth=1.5)

    colors = {"ARIMA": "#dc2626", "ML (Gradient Boosting)": "#7c3aed"}
    for name, preds in predictions.items():
        color = colors.get(name, "#059669")
        ax.plot(test_df[DATE_COL].values, preds.values, label=f"{name} (Test Pred)", linestyle="--", color=color)

    if future_forecasts:
        last_test_date = test_df[DATE_COL].max()
        for name, fdf in future_forecasts.items():
            color = colors.get(name, "#059669")
            ax.plot(
                fdf[DATE_COL],
                fdf["forecast"],
                label=f"{name} (Future)",
                linestyle=":",
                linewidth=2,
                color=color,
            )
        ax.axvline(last_test_date, color="gray", linestyle="-", alpha=0.5, label="Forecast Start")

    ax.set_title("Sales Forecast: ARIMA vs Machine Learning")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales ($)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_metrics_comparison(metrics_df: pd.DataFrame, save_path=None) -> None:
    save_path = save_path or FIGURES_DIR / "05_metrics_comparison.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    models = metrics_df["model"]
    axes[0].bar(models, metrics_df["rmse"], color=["#dc2626", "#7c3aed"])
    axes[0].set_title("RMSE (lower is better)")
    axes[0].set_ylabel("RMSE")
    axes[1].bar(models, metrics_df["mape_pct"], color=["#dc2626", "#7c3aed"])
    axes[1].set_title("MAPE % (lower is better)")
    axes[1].set_ylabel("MAPE (%)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
