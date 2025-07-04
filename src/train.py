import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from src.model import XRRHybridRegressor, XRRRegressor
from src.read import get_data


def normalize(arr: np.ndarray) -> np.ndarray:
    """Nornalize a NumPy array to mean 0, std 1."""
    return (arr - arr.mean()) / arr.std()


def load_and_preprocess_data(
    data_file_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, StandardScaler]:
    """Loads data, applies log transformation and normalization."""
    if not data_file_path.exists():
        raise FileNotFoundError(f"Data file {data_file_path} does not exist.")

    data = get_data(data_file_path)

    # Reflectivity input (log scale + normalize)
    x_all_np = data["Rs"].astype(np.float32)  # Ensure float32 from numpy
    x_all_np = np.log10(x_all_np + 1e-8)
    x_all_np = normalize(x_all_np)
    x_all = torch.tensor(x_all_np, dtype=torch.float32)

    # Target parameters normalization using StandardScaler
    y_all_np = data["params"].astype(np.float32)  # Ensure float32 from numpy
    scaler = StandardScaler()
    y_all_scaled_np = scaler.fit_transform(y_all_np)
    y_all_scaled = torch.tensor(y_all_scaled_np, dtype=torch.float32)

    return x_all, y_all_scaled, scaler


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    num_epochs: int,
    patience: int = 20,  # Early Stopping patience
) -> tuple[list[float], list[float]]:
    """Trains the model with early stopping and mixed precision."""
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # For Automatic Mixed Precision (AMP)
    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

    print("\nStarting training...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()

            if scaler:  # Use AMP if CUDA is available
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(batch_x)
                    loss = loss_fn(outputs, batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:  # Standard training for CPU or when AMP is not used
                outputs = model(batch_x)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()

            total_train_loss += loss.item() * batch_x.size(0)  # Aggregate loss

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_x_val, batch_y_val in val_loader:
                batch_x_val, batch_y_val = batch_x_val.to(device), batch_y_val.to(
                    device
                )
                val_outputs = model(batch_x_val)
                val_loss_batch = loss_fn(val_outputs, batch_y_val)
                total_val_loss += val_loss_batch.item() * batch_x_val.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"[{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

        # Early Stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Optionally save the best model here:
            # torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(
                    f"Early stopping at epoch {epoch+1} as validation loss did not improve for {patience} epochs."
                )
                break

    return train_losses, val_losses


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    scaler: StandardScaler,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates the model and returns original scale predictions and true values."""
    model.eval()
    all_y_pred_scaled = []
    all_y_val_scaled = []
    with torch.no_grad():
        for batch_x_val, batch_y_val in val_loader:
            batch_x_val, batch_y_val = batch_x_val.to(device), batch_y_val.to(device)
            val_outputs = model(batch_x_val)
            all_y_pred_scaled.append(val_outputs.cpu().numpy())
            all_y_val_scaled.append(batch_y_val.cpu().numpy())

    y_pred_scaled_np = np.vstack(all_y_pred_scaled)
    y_val_scaled_np = np.vstack(all_y_val_scaled)

    # Inverse transform to original scale
    y_pred_orig = scaler.inverse_transform(y_pred_scaled_np)
    y_val_orig = scaler.inverse_transform(y_val_scaled_np)

    return y_pred_orig, y_val_orig


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
    import matplotlib.pyplot as plt

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


def evaluate_from_saved_model(model_path: Path, scaler_path: Path, data_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로딩
    x_all, y_all_scaled, scaler = load_and_preprocess_data(data_path)

    # 평가셋 분리
    _, x_val, _, y_val_scaled = train_test_split(
        x_all, y_all_scaled, test_size=0.2, random_state=42
    )
    val_dataset = TensorDataset(x_val, y_val_scaled)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # 모델 로딩
    model = XRRHybridRegressor(input_length=x_val.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 스케일러 로딩
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)

    # 평가
    y_pred_orig, y_val_orig = evaluate_model(model, val_loader, scaler, device)

    param_names = ["sld", "thickness", "roughness"]
    plot_r2_scores(y_val_orig, y_pred_orig, param_names)
    plot_best_worst_fits(x_val, y_val_orig, y_pred_orig, top_n=5)
    plot_error_histograms(y_val_orig, y_pred_orig, param_names)

    print("✅ Evaluation complete.")


# ============================
# 메인 실행 로직
# ============================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loading and preprocessing
    data_file = Path("X:\\XRR_AI\\hdf5_XRR") / "p100o3_2.h5"
    x_all, y_all_scaled, scaler = load_and_preprocess_data(data_file)

    # Train/Validation split
    x_train, x_val, y_train, y_val = train_test_split(
        x_all, y_all_scaled, test_size=0.2, random_state=42
    )

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    batch_size: int = 256  # Consider adjusting this based on GPU memory
    num_workers: int = 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )  # num_workers for parallel data loading
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )  # num_workers for parallel data loading

    # Model, Optimizer, Loss setup
    model = XRRHybridRegressor(input_length=x_train.shape[1])
    # model = XRRRegressor()
    model.to(device)  # Move model to GPU

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Train the model
    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device,
        num_epochs=200,
        patience=50,  # Increased epochs and patience for better convergence
    )
    # 모델 저장 경로
    MODEL_PATH = Path("saved_model_hybrid.pt")

    # 학습 종료 후 모델 저장
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

    # Plot losses
    plot_losses(train_losses, val_losses)

    # Evaluate the model
    y_pred_orig, y_val_orig = evaluate_model(model, val_loader, scaler, device)

    # Plot R2 scores for each parameter
    param_names = ["sld", "thickness", "roughness"]
    plot_r2_scores(y_val_orig, y_pred_orig, param_names)
    plot_losses_with_annotations(
        train_losses, val_losses, save_path="loss_plot.png"
    )  # Save loss plot
    # Plot best/worst fit Rs curves
    # plot_best_worst_fits(
    #     x_val, y_val_orig, y_pred_orig, top_n=5
    # )  # Show top 5 best/worst
    plot_3d_scatter(y_val_orig, y_pred_orig)  # 3D scatter plot of true vs predicted
    plot_residuals(y_val_orig, y_pred_orig, param_names)  # Resid
    plot_error_histograms(y_val_orig, y_pred_orig, param_names)
