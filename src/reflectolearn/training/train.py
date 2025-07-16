from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from ..math_utils import normalize, q_fourier_transform_multisample
from ..io.read import get_data


def load_and_preprocess_data(
    data_file_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, StandardScaler]:
    """Loads data, applies log transformation and normalization."""
    if not data_file_path.exists():
        raise FileNotFoundError(f"Data file {data_file_path} does not exist.")

    data = get_data(data_file_path)

    # Reflectivity input (log scale + normalize)
    Rs = data["Rs"].astype(np.float32)  # Ensure float32 from numpy
    FT = q_fourier_transform_multisample(Rs, data["q"], np.linspace(0, 500, 2048))
    rho_z = np.abs(FT) ** 2
    x_all_np = np.log10(rho_z + 1e-8)
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
    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(batch_x)
                    loss = loss_fn(outputs, batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_x)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
            total_train_loss += loss.item() * batch_x.size(0)  # Aggregate loss

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_x_val, batch_y_val in val_loader:
                batch_x_val = batch_x_val.to(device)
                batch_y_val = batch_y_val.to(device)
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
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(
                    f"Early stopping at epoch {epoch+1} as validation loss did not improve for {patience} epochs."
                )
                break

    return train_losses, val_losses


def evaluate_model(model, val_loader, scaler, device):
    model.eval()
    y_pred_list, y_true_list = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            y_pred_list.append(outputs.cpu().numpy())
            y_true_list.append(batch_y.cpu().numpy())

    y_pred = scaler.inverse_transform(np.vstack(y_pred_list))
    y_true = scaler.inverse_transform(np.vstack(y_true_list))
    return y_pred, y_true
