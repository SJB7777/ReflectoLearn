import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


# TODO: Develop this function
def loss_with_prior(y_pred, y_true, prior_values=None, prior_mask=None, lambda_prior=0.1):
    # 기본 MSE
    loss_data = F.mse_loss(y_pred, y_true)

    prior_loss = 0.0
    if prior_values is not None and prior_mask is not None:
        for idx in prior_mask:
            prior_loss += F.mse_loss(y_pred[:, idx], prior_values[:, idx])
        prior_loss /= len(prior_mask)

    return loss_data + lambda_prior * prior_loss


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    num_epochs: int,
    patience: int = 20,
) -> tuple[list[float], list[float]]:
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    stagnation_counter = 0
    amp_scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            if amp_scaler:
                with torch.amp.autocast(device_type=device.type):
                    predictions = model(inputs)
                    loss = loss_fn(predictions, targets)
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                predictions = model(inputs)
                loss = loss_fn(predictions, targets)
                loss.backward()
                optimizer.step()

            epoch_train_loss += loss.item() * inputs.size(0)

        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs)
                loss = loss_fn(predictions, targets)
                epoch_val_loss += loss.item() * inputs.size(0)

        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"[Epoch {epoch + 1:03d}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            if stagnation_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}: no improvement for {patience} epochs.")
                break

    return train_losses, val_losses


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    scaler: StandardScaler,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_preds = []
    y_trues = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            y_preds.append(outputs.cpu().numpy())
            y_trues.append(targets.cpu().numpy())

    y_pred_np = scaler.inverse_transform(np.vstack(y_preds))
    y_true_np = scaler.inverse_transform(np.vstack(y_trues))

    return y_pred_np, y_true_np
