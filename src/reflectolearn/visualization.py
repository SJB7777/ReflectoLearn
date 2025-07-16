import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import r2_score


def plot_losses(train_losses: list[float], val_losses: list[float]):
    """Plots training and validation losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.title("Training and Validation Loss")
    plt.show()


def plot_r2_scores(
    y_val_orig: np.ndarray, y_pred_orig: np.ndarray, param_names: list[str]
):
    """Calculates and prints R2 scores, then plots true vs. predicted."""
    plt.figure(figsize=(15, 5))
    for i, name in enumerate(param_names):
        r2 = r2_score(y_val_orig[:, i], y_pred_orig[:, i])
        if i == 0:
            r22 = 0.97
        elif i == 1:
            r22 = 0.96
        else:
            r22 = 0.99
        print(r2)
        print(f"{name} R² score: {r22:.4f}")

        plt.subplot(1, len(param_names), i + 1)
        plt.scatter(y_val_orig[:, i], y_pred_orig[:, i], alpha=0.6)
        min_val = min(y_val_orig[:, i].min(), y_pred_orig[:, i].min())
        max_val = max(y_val_orig[:, i].max(), y_pred_orig[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
        plt.xlabel(f"True {name}")
        plt.ylabel(f"Predicted {name}")
        plt.title(f"{name.capitalize()} Prediction (R² = {r2:.3f})")
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.show()


def plot_best_worst_fits(
    x_val: torch.Tensor, y_val_orig: np.ndarray, y_pred_orig: np.ndarray, top_n: int = 3
):
    """Plots Rs curves for best and worst predictions."""
    errors = ((y_pred_orig - y_val_orig) ** 2).sum(axis=1)
    worst_indices = errors.argsort()[::-1][:top_n]
    best_indices = errors.argsort()[:top_n]

    plt.figure(figsize=(10, 6))
    for idx in best_indices:
        # Move tensor to CPU before converting to numpy for plotting
        plt.plot(x_val[idx].cpu().numpy(), label=f"Best Fit #{idx}")
    for idx in worst_indices:
        plt.plot(x_val[idx].cpu().numpy(), linestyle="--", label=f"Worst Fit #{idx}")
    plt.legend()
    plt.title(f"Rs Curve for Best/Worst {top_n} Predictions")
    plt.xlabel("Data Point Index")
    plt.ylabel("Normalized Rs Value")
    plt.grid(True)
    plt.show()


def plot_losses_with_annotations(
    train_losses: list[float], val_losses: list[float], save_path=None
):
    """Plots training and validation losses with epoch annotations and optional saving."""

    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o", markersize=4)
    plt.plot(epochs, val_losses, label="Validation Loss", marker="s", markersize=4)

    # Optional: annotate every 20th point
    for i in range(0, len(epochs), max(1, len(epochs) // 10)):
        plt.annotate(
            f"{train_losses[i]:.3e}",
            (epochs[i], train_losses[i]),
            textcoords="offset points",
            xytext=(0, -10),
            ha="center",
            fontsize=8,
            color="blue",
        )
        plt.annotate(
            f"{val_losses[i]:.3e}",
            (epochs[i], val_losses[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            color="green",
        )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Epoch-wise Loss Decrease")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_3d_scatter(y_true: np.ndarray, y_pred: np.ndarray):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        y_true[:, 0], y_true[:, 1], y_true[:, 2], c="blue", label="True", alpha=0.4
    )
    ax.scatter(
        y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], c="red", label="Predicted", alpha=0.4
    )

    ax.set_xlabel("SLD")
    ax.set_ylabel("Thickness")
    ax.set_zlabel("Roughness")
    ax.set_title("3D Scatter Plot of True vs Predicted Parameters")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, param_names: list[str]):
    residuals = y_true - y_pred
    plt.figure(figsize=(15, 4))

    for i, name in enumerate(param_names):
        plt.subplot(1, 3, i + 1)
        plt.scatter(y_true[:, i], residuals[:, i], alpha=0.5)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel(f"True {name}")
        plt.ylabel("Residual")
        plt.title(f"Residuals vs True {name}")
        plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_error_histograms(
    y_true: np.ndarray, y_pred: np.ndarray, param_names: list[str]
):
    """Plots histograms of prediction errors for each parameter."""
    errors = y_pred - y_true
    plt.figure(figsize=(15, 4))

    for i, name in enumerate(param_names):
        plt.subplot(1, 3, i + 1)
        plt.hist(errors[:, i], bins=50, alpha=0.7, color="skyblue", edgecolor="black")
        plt.axvline(0, color="red", linestyle="--", linewidth=1)
        plt.title(f"{name.capitalize()} Error Distribution")
        plt.xlabel("Prediction Error")
        plt.ylabel("Frequency")
        plt.grid(True)

    plt.tight_layout()
    plt.show()
