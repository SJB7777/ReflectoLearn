from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.model_manager import MLPManager, CNNModelManager
from src.loader import parse_out_data, parse_con_parameter
from src.preprocess import fill_nan_with_interp


def load_data(
    data_path: Path, sample_nums: Iterable[int]
) -> tuple[np.ndarray, np.ndarray]:
    X_data = []
    y_data = []
    for num in tqdm(sample_nums, desc="Load Data"):
        out_file = data_path / f"d{num:05}" / "1.out"
        con_file = data_path / f"d{num:05}" / "1.con"
        data = parse_out_data(out_file)
        x = data[["XOBS", "YCALC"]].values
        x = fill_nan_with_interp(x)
        X_data.append(x)
        y = np.asarray(parse_con_parameter(con_file))[
            [4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        ]
        y_data.append(y)

    return np.stack(X_data, axis=0), np.stack(y_data, axis=0)


def train_mlp(
    X_data: np.ndarray, y_data: np.ndarray, model_path: str | Path
) -> dict[str, list[float]]:
    """Train the MLP model"""
    input_dim: int = X_data.shape[1] * X_data.shape[2]
    output_dim: int = y_data.shape[1]

    manager = MLPManager(input_dim=input_dim, output_dim=output_dim, learning_rate=1e-3)

    return manager.train(
        X_data=X_data,
        y_data=y_data,
        model_path=model_path,
        epochs=5,
        batch_size=64,
        val_ratio=0.1,
    )


def train_cnn(
    X_data: np.ndarray, y_data: np.ndarray, model_path: str | Path
) -> dict[str, list[float]]:

    epochs = 15  # CNN might need more/different epochs than MLP
    batch_size = 64
    val_ratio = 0.1
    learning_rate = 1e-4  # CNNs often benefit from smaller learning rates initially
    model_path = Path("models") / "cnn_simulation_model.pth"
    height = X_data.shape[1]  # e.g., 313
    width = X_data.shape[2]  # e.g., 2
    channels = 1  # Treat as a single-channel input
    cnn_input_shape: tuple[int, int, int] = (channels, height, width)
    output_dim: int = y_data.shape[1]
    manager = CNNModelManager(
        input_shape=cnn_input_shape, output_dim=output_dim, learning_rate=learning_rate
    )
    return manager.train(
        X_data=X_data,  # Manager's _prepare_data should handle shape (N, H, W') -> (N, C, H, W)
        y_data=y_data,
        model_path=model_path,
        epochs=epochs,
        batch_size=batch_size,
        val_ratio=val_ratio,
    )


def main():
    """Main function to train using the XRRMLPManager"""
    # model_path = Path("models") / "mlp_model.pth"
    data_path = Path("..\\data\\simulation_data")

    X_data, y_data = load_data(data_path, range(1, 10001))
    # history = train_cnn(X_data, y_data, model_path="models/cnn_simulation_model.pth")
    history = train_mlp(X_data, y_data, model_path="models/mlp_model.pth")

    epochs = range(1, len(history["train_loss"]) + 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
