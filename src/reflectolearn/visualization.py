import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import r2_score


def plot_losses(train_losses: list[float], val_losses: list[float]):
    """단순 학습 곡선 플롯"""
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_losses_with_annotations(train_losses: list[float], val_losses: list[float], save_path=None):
    """학습 손실 곡선 + 특정 에포크 주석"""
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o", markersize=4)
    plt.plot(epochs, val_losses, label="Validation Loss", marker="s", markersize=4)

    # 주석: 10개 간격으로 표시
    interval = max(1, len(epochs) // 10)
    for i in range(0, len(epochs), interval):
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
    plt.title("Training Progress with Annotated Losses")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_r2_scores(y_true: np.ndarray, y_pred: np.ndarray, param_names: list[str]):
    """R² 점수 계산 및 예측 vs 실제값 플롯"""
    n_param = len(param_names)
    plt.figure(figsize=(5 * n_param, 4))

    for i, name in enumerate(param_names):
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        print(f"{name} R² score: {r2:.4f}")

        plt.subplot(1, n_param, i + 1)
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, s=15, edgecolor="k")
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
        plt.xlabel(f"True {name}")
        plt.ylabel(f"Predicted {name}")
        plt.title(f"{name.capitalize()} (R² = {r2:.3f})")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()


def plot_best_worst_fits(x_val: torch.Tensor, y_true: np.ndarray, y_pred: np.ndarray, top_n: int = 3):
    """예측 성능 기준 상하위 N개의 Rs 곡선 플롯"""
    errors = ((y_pred - y_true) ** 2).sum(axis=1)
    best_indices = errors.argsort()[:top_n]
    worst_indices = errors.argsort()[-top_n:][::-1]

    plt.figure(figsize=(12, 6))
    for idx in best_indices:
        plt.plot(x_val[idx], label=f"Best #{idx}", linewidth=1.5)
    for idx in worst_indices:
        plt.plot(
            x_val[idx],
            linestyle="--",
            label=f"Worst #{idx}",
            linewidth=1.2,
        )

    plt.xlabel("Index")
    plt.ylabel("Normalized Rs")
    plt.title(f"Top-{top_n} Best & Worst Fit Rs Curves")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, param_names: list[str]):
    """잔차(Residual) 시각화"""
    residuals = y_true - y_pred
    plt.figure(figsize=(5 * len(param_names), 4))

    for i, name in enumerate(param_names):
        plt.subplot(1, len(param_names), i + 1)
        plt.scatter(y_true[:, i], residuals[:, i], alpha=0.5, s=15, edgecolor="k")
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel(f"True {name}")
        plt.ylabel("Residual")
        plt.title(f"Residuals of {name}")
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_error_histograms(y_true: np.ndarray, y_pred: np.ndarray, param_names: list[str]):
    """예측 오차 분포 히스토그램"""
    errors = y_pred - y_true
    plt.figure(figsize=(5 * len(param_names), 4))

    for i, name in enumerate(param_names):
        plt.subplot(1, len(param_names), i + 1)
        plt.hist(errors[:, i], bins=40, alpha=0.75, color="skyblue", edgecolor="black")
        plt.axvline(0, color="red", linestyle="--")
        plt.title(f"{name.capitalize()} Error Distribution")
        plt.xlabel("Prediction Error")
        plt.ylabel("Frequency")
        plt.grid(True)

    plt.tight_layout()
    plt.show()
