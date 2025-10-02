import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from reflectolearn.config import ConfigManager
from reflectolearn.device import get_device_and_workers
from reflectolearn.io import append_timestamp, read_xrr_hdf5, save_model
from reflectolearn.logger import setup_logger
from reflectolearn.math_utils import normalize
from reflectolearn.models.general import get_model
from reflectolearn.training.train import train_model


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_dataloaders(x_all, y_all, batch_size: int, seed: int, num_workers: int):
    x_train, x_val, y_train, y_val = train_test_split(x_all, y_all, test_size=0.2, random_state=seed)

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
    ConfigManager.initialize("config.yaml")
    config = ConfigManager.load_config()

    logger.info("Starting training script")
    logger.info(f"Config: \n{config.model_dump_json(indent=2)}")
    seed = config.training.seed
    set_seed(seed)

    device, num_workers = get_device_and_workers()
    logger.info(f"Using device: {device}")
    logger.info(f"Number of workers: {num_workers}")

    # Data Loading
    data = read_xrr_hdf5(config.path.data_file)
    q = data["q"]
    x_array = data["R"]
    y_array = np.concatenate([data["roughness"], data["thickness"], data["sld"]], axis=1).astype(np.float32)

    logger.info(f"Shape of q: {q.shape}")
    logger.info(f"Shape of x_array: {x_array.shape}")
    logger.info(f"Shape of y_array: {y_array.shape}")

    # Thickness prior estimation
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y_array)
    y_all = torch.tensor(y_scaled, dtype=torch.float32)

    reflectivity = torch.tensor(data["R"], dtype=torch.float32)
    x_all = normalize(reflectivity)
    train_loader, val_loader = prepare_dataloaders(
        x_all,
        y_all,
        batch_size=config.training.batch_size,
        seed=seed,
        num_workers=num_workers,
    )

    input_length: int = x_all.shape[1]
    output_length: int = y_all.shape[1]
    logger.info(f"Input length: {input_length}")
    logger.info(f"Input length: {output_length}")
    model = get_model(
        model_type=config.training.type,
        input_length=input_length,
        output_length=output_length,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    loss_fn = nn.MSELoss()

    logger.info("Starting model training...")
    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device,
        num_epochs=config.training.epochs,
        patience=config.training.patience,
    )
    logger.info("Model training finished.")
    # === Directories ===
    result_dir = append_timestamp(config.path.output_dir / "model")
    result_dir.mkdir(parents=True, exist_ok=True)

    # === Save model ===
    model_path = result_dir / "model.pt"
    save_model(model.state_dict(), model_path)
    logger.info(f"Best model saved to {model_path}")

    # === Save scaler ===
    scaler_path = result_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")

    # === Save training curves ===
    stats_path = result_dir / "stats.npz"
    np.savez(
        stats_path,
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
        config=config,
    )
    logger.info(f"Training statistics saved to {stats_path}")
    logger.info(f"Save model to {result_dir}")


if __name__ == "__main__":
    logger = setup_logger()
    try:
        main()
    except Exception:
        logger.exception("Application failed with error")
        raise
