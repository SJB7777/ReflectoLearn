from pathlib import Path

import joblib
import numpy as np
import torch
from loguru import logger

from reflectolearn import visualization as viz
from reflectolearn.config import ConfigManager
from reflectolearn.data_processing.preprocess import preprocess_features
from reflectolearn.io import get_data
from reflectolearn.models.model import get_model


def load_evaluation_assets():
    """모델, 스케일러, 평가 데이터, 학습 곡선을 불러옵니다."""

    ConfigManager.initialize("config.yaml")
    config = ConfigManager.load_config()

    output_root: Path = config.project.output_dir
    model_file: Path = output_root / "model.pt"
    scaler_file: Path = output_root / "scaler.pkl"
    stats_file: Path = output_root / "stats.npz"
    data_file: Path = config.data.data_file

    logger.info(f"[INFO] Loading model from: {model_file}")
    logger.info(f"[INFO] Loading scaler from: {scaler_file}")
    logger.info(f"[INFO] Loading stats from: {stats_file}")
    logger.info(f"[INFO] Loading data from: {data_file}")

    # Load scaler
    scaler = joblib.load(scaler_file)

    # Load training statistics
    stats = np.load(stats_file)
    train_losses = stats["train_losses"]
    val_losses = stats["val_losses"]

    # Load evaluation data
    data = get_data(data_file)
    x_val: torch.Tensor = preprocess_features(config.project.version, data)
    y_val_orig: np.ndarray = data["params"]

    # Load model
    input_length: int = x_val.shape[1]
    output_length: int = y_val_orig.shape[1]
    logger.info(f"Input Length: {input_length}")
    logger.info(f"Input Length: {output_length}")
    model = get_model(
        config.model.type,
        input_length=input_length,
        output_length=output_length,
    )
    model.load_state_dict(
        torch.load(model_file, map_location="cpu", weights_only=True),
    )
    model.eval()

    return model, scaler, x_val, y_val_orig, train_losses, val_losses


def evaluate_and_visualize():
    model, scaler, x_val, y_val_orig, train_losses, val_losses = load_evaluation_assets()

    # 추정된 출력 파라미터 수
    n_outputs = y_val_orig.shape[1]
    if n_outputs % 3 != 0:
        raise ValueError(f"출력 파라미터 수({n_outputs})가 3의 배수가 아닙니다.")

    n_layer = n_outputs // 3
    param_names = [f"{param}_{i}" for i in range(n_layer) for param in ("roughness", "sld", "thickness")]

    with torch.no_grad():
        y_pred_normalized = model(x_val).numpy()
    y_pred_orig = scaler.inverse_transform(y_pred_normalized)

    viz.plot_losses(train_losses, val_losses)
    viz.plot_r2_scores(y_val_orig, y_pred_orig, param_names)
    viz.plot_best_worst_fits(x_val, y_val_orig, y_pred_orig, top_n=3)
    viz.plot_residuals(y_val_orig, y_pred_orig, param_names)
    viz.plot_error_histograms(y_val_orig, y_pred_orig, param_names)


if __name__ == "__main__":
    evaluate_and_visualize()
