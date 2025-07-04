from pathlib import Path

import numpy as np

from src.preprocess import fill_nan_with_interp
from src.model_manager import MLPManager, CNNModelManager
from src.loader import parse_out_data, parse_con_parameter


def load_data(data_path: Path) -> tuple[np.ndarray, np.ndarray]:

    out_file = data_path / "1.out"
    con_file = data_path / "1.con"
    data = parse_out_data(out_file)

    X_test = data[["XOBS", "YOBS"]].values[:314]
    X_test = fill_nan_with_interp(X_test)[np.newaxis, :, :]

    y_test = np.asarray(parse_con_parameter(con_file))
    return X_test, y_test


def predict_mlp(
    model_path: str | Path, X_test: np.ndarray, y_test: np.ndarray
) -> np.ndarray:
    """Predict using the MLP model"""
    input_dim: int = X_test.shape[1] * X_test.shape[2]
    output_dim: int = y_test.shape[0]
    manager = MLPManager(input_dim=input_dim, output_dim=output_dim, learning_rate=1e-3)

    return manager.predict(X_test, model_path=model_path)


def predict_cnn(
    model_path: str | Path, X_test: np.ndarray, y_test: np.ndarray
) -> np.ndarray:
    """Predict using the MLP model"""
    height = X_test.shape[1]  # e.g., 313
    width = X_test.shape[2]  # e.g., 2
    channels = 1  # Treat as a single-channel input

    input_shape: tuple[int, int, int] = (channels, height, width)
    output_dim: int = y_test.shape[0]

    manager = CNNModelManager(
        input_shape=input_shape, output_dim=output_dim, learning_rate=1e-3
    )

    return manager.predict(X_test, model_path=model_path)


def main():
    """Main function to predict using the CNNModelManager"""
    data_path = Path(".\\lsfit_software")
    X_test, y_test = load_data(data_path)

    predictions = predict_mlp("models\\mlp_model.pth", X_test, y_test)
    # predictions = predict_cnn("models\\cnn_simulation_model.pth", X_test, y_test)

    idxs = [i + 1 for i in [4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
    for idx, val in zip(idxs, predictions[0]):
        print(idx, val)


if __name__ == "__main__":
    main()
