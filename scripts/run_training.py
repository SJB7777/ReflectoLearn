from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import joblib

from reflectolearn.models.model import get_model
from reflectolearn.training.train import train_model, load_and_preprocess_data
from reflectolearn.io.save import save_model


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    config = load_config(Path("config.yml"))
    logger.info(f"Config: {config}")
    seed = config["training"]["random_seed"]
    set_seed(seed)

    device, num_workers = get_device_and_workers()
    logger.info(f"Using device: {device}")

    raw_name = Path(config["data"]["file_name"]).stem
    data_version = config["data"]["version"]
    data_file = Path(config["data"]["processed_data_dir"]) / f"{raw_name}_{data_version}.h5"

    x_all, y_all_scaled, scaler = load_and_preprocess_data(data_file)

    train_loader, val_loader = prepare_dataloaders(
        x_all,
        y_all_scaled,
        batch_size=config["training"]["batch_size"],
        seed=seed,
        num_workers=num_workers,
    )

    model = get_model(
        name=config["model"]["type"],
        input_length=x_all.shape[1],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"]
    )
    loss_fn = nn.MSELoss()

    logger.info("Starting model training...")
    model_states, val_losses = train_model(
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

    model_dir = Path(config["results"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    model_name = config["model"]["type"]
    lr = config["training"]["learning_rate"]
    model_filename = f"{model_name}_seed{seed}_lr{lr}_lose{min(val_losses):.4f}_data-{data_version}.pt"
    model_path = model_dir / model_filename
    save_model(model_states, model_path)
    joblib.dump(scaler, model_dir / "scaler.pkl")

    logger.info(f"Best model saved to {model_path}")
    logger.info(f"Scaler saved to {model_dir / 'scaler.pkl'}")


if __name__ == "__main__":
    main()
