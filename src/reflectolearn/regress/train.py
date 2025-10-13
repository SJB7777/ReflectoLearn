from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ..logger import setup_logger
from .data import ThicknessDataset
from .model import ThicknessRegressor


def train_regressor(
    dataset: ThicknessDataset,
    save_file: str,
    n_layer: int,
    max_epoch: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
):
    """
    Train regression model for thickness prediction given n_layer.
    Returns dict with model state and training history (for evaluate.py).
    """
    logger = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Train/Val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    input_dim = dataset.input_dim
    model = ThicknessRegressor(input_dim=input_dim, n_layer=n_layer).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "val_loss": []
    }

    # Training loop
    for epoch in range(1, max_epoch + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for R, y in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            R, y = R.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(R)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for R, y in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                R, y = R.to(device), y.to(device)
                preds = model(R)
                loss = criterion(preds, y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        logger.info(
            f"[n={n_layer}] Epoch {epoch:02d}/{max_epoch} "
            f"| Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

    # Save checkpoint
    save_path = Path(save_file)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "n_layer": n_layer,
        "input_dim": input_dim,
        "history": history
    }
    torch.save(checkpoint, save_path)
    logger.info(f"✅ Model saved at {save_path}")

    return checkpoint


if __name__ == "__main__":
    from .data import ThicknessDataset


    n_layer:int = 3
    dataset_path = r"D:\03_Resources\Data\XRR_AI\data\250929.h5"
    save_file = rf"results/regress_thickness_n{n_layer}_2.pt"
    dataset = ThicknessDataset(dataset_path, n_layer)
    checkpoint = train_regressor(dataset, save_file, n_layer)
    # print(checkpoint)
