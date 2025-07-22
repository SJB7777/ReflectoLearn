import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from reflectolearn.data_processing.preprocess import load_and_preprocess_data
from reflectolearn.io.save import save_model
from reflectolearn.models.model import get_model
from reflectolearn.training.train import train_model
from reflectolearn.types import ModelType
from scripts.config import load_config


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device_and_workers() -> tuple[torch.device, int]:
    if torch.backends.mps.is_available():
        return torch.device("mps"), 0
    elif torch.cuda.is_available():
        return torch.device("cuda"), 4
    else:
        return torch.device("cpu"), 0


def prepare_dataloaders(x_all, y_all, batch_size: int, seed: int, num_workers: int):
    x_train, x_val, y_train, y_val = train_test_split(
        x_all, y_all, test_size=0.2, random_state=seed
    )

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def main():
    logger.info("Starting training script")
    config = load_config(Path("config.yml"))
    logger.info(f"Config: {config}")
    seed = config["training"]["random_seed"]
    set_seed(seed)

    device, num_workers = get_device_and_workers()
    logger.info(f"Using device: {device}")

    raw_name = Path(config["data"]["file_name"]).stem
    data_version = config["data"]["version"]

    data_file = Path(config["data"]["data_dir"]) / f"{raw_name}_{data_version}.h5"

    x_all, y_all_scaled, scaler = load_and_preprocess_data(
        data_file, config["data"]["version"]
    )

    train_loader, val_loader = prepare_dataloaders(
        x_all,
        y_all_scaled,
        batch_size=config["training"]["batch_size"],
        seed=seed,
        num_workers=num_workers,
    )

    model_name = config["model"]["type"]
    lr = config["training"]["learning_rate"]
    model = get_model(
        model_type=ModelType.from_str(model_name),
        input_length=x_all.shape[1],
        output_length=y_all_scaled.shape[1],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    logger.info("Starting model training...")
    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device,
        num_epochs=config["training"]["num_epochs"],
        patience=config["training"]["patience"],
    )
    logger.info("Model training finished.")

    # === Naming Convention ===
    tag = f"{model_name}__{data_version}__seed{seed}__lr{lr:.0e}"
    # === Directories ===
    model_dir = Path(config["results"]["model_dir"])
    scaler_dir = Path(config["results"]["scaler_dir"])
    stats_dir = Path(config["results"]["stats_dir"])

    for d in [model_dir, scaler_dir, stats_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # === Save model ===
    model_path = model_dir / f"{tag}.pt"
    save_model(model.state_dict(), model_path)
    logger.info(f"Best model saved to {model_path}")

    # === Save scaler ===
    scaler_path = scaler_dir / f"{tag}.scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")

    # === Save training curves ===
    stats_path = stats_dir / f"{tag}.stats.npz"
    np.savez(
        stats_path,
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
        config=config,
    )
    logger.info(f"Training statistics saved to {stats_path}")


if __name__ == "__main__":
    main()
