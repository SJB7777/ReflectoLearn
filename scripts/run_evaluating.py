from pathlib import Path

import joblib
import numpy as np
import torch
from loguru import logger

from reflectolearn import visualization as viz
from reflectolearn.data_processing.preprocess import preprocess_features
from reflectolearn.io import get_data
from reflectolearn.models.model import get_model
from reflectolearn.config import load_config


def load_evaluation_assets():
    """모델, 스케일러, 평가 데이터, 학습 곡선을 불러옵니다."""

    config = load_config()
    model_name = config.model

    output_root: Path = config.project.output_dir
    model_file: Path = output_root / "model.pt"
    scaler_file: Path = output_root / "scaler.pkl"
    stats_file: Path = output_root / "stats.npz"
    data_file: Path = config.data.input_file

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
    x_val = preprocess_features(config.project.type, data)
    y_val_orig = data["params"]

    # Load model
    model = get_model(model_name, input_length=x_val.shape[1])
    model.load_state_dict(
        torch.load(model_file, map_location="cpu"),
    )
    model.eval()

    return model, scaler, x_val, y_val_orig, train_losses, val_losses


def evaluate_and_visualize():
    PARAM_NAMES = ["roughness", "sld", "thickness"]

    model, scaler, x_val, y_val_orig, train_losses, val_losses = (
        load_evaluation_assets()
    )

    with torch.no_grad():
        y_pred_normalized = model(x_val).numpy()
    y_pred_orig = scaler.inverse_transform(y_pred_normalized)

    viz.plot_losses(train_losses, val_losses)
    viz.plot_r2_scores(y_val_orig, y_pred_orig, PARAM_NAMES)
    viz.plot_best_worst_fits(x_val, y_val_orig, y_pred_orig, top_n=3)
    viz.plot_residuals(y_val_orig, y_pred_orig, PARAM_NAMES)
    viz.plot_error_histograms(y_val_orig, y_pred_orig, PARAM_NAMES)


if __name__ == "__main__":
    evaluate_and_visualize()
