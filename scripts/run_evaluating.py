from pathlib import Path

import joblib
import numpy as np
import torch
from loguru import logger

from reflectolearn import visualization as viz
from reflectolearn.config import ConfigManager
from reflectolearn.io import get_data
from reflectolearn.models.model import get_model
from reflectolearn.processing.preprocess import preprocess_features


def load_evaluation_assets(device: torch.device):
    """모델, 스케일러, 평가 데이터, 학습 곡선을 불러옵니다."""

    ConfigManager.initialize("config.yaml")
    config = ConfigManager.load_config()

    output_root: Path = config.path.output_dir
    model_file: Path = output_root / "model.pt"
    scaler_file: Path = output_root / "scaler.pkl"
    stats_file: Path = output_root / "stats.npz"
    data_file: Path = config.path.data_file

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
    x_val: torch.Tensor = preprocess_features(config.project.version, data).float().to(device)
    y_val_orig: np.ndarray = data["params"]

    # Load model
    input_length: int = x_val.shape[1]
    output_length: int = y_val_orig.shape[1]
    logger.info(f"Input Length: {input_length}")
    logger.info(f"Output Length: {output_length}")
    model = get_model(
        config.training.type,
        input_length=input_length,
        output_length=output_length,
    )
    model.load_state_dict(
        torch.load(model_file, map_location=device, weights_only=True),
    )
    model.to(device)
    model.eval()

    return model, scaler, x_val, y_val_orig, train_losses, val_losses


def evaluate_and_visualize(batch_size: int = 64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model, scaler, x_val, y_val_orig, train_losses, val_losses = load_evaluation_assets(device)

    n_outputs = y_val_orig.shape[1]
    if n_outputs % 3 != 0:
        raise ValueError(f"출력 파라미터 수({n_outputs})가 3의 배수가 아닙니다.")

    n_layer = n_outputs // 3
    param_names = [f"{param}_{i}" for i in range(n_layer) for param in ("roughness", "sld", "thickness")]

    # 배치 처리
    y_pred_normalized_list = []
    with torch.no_grad():
        for i in range(0, len(x_val), batch_size):
            x_batch = x_val[i:i+batch_size]
            y_batch = model(x_batch)
            y_pred_normalized_list.append(y_batch.cpu().numpy())

    y_pred_normalized = np.concatenate(y_pred_normalized_list, axis=0)
    y_pred_orig = scaler.inverse_transform(y_pred_normalized)

    # 시각화
    viz.plot_losses(train_losses, val_losses)
    viz.plot_r2_scores(y_val_orig, y_pred_orig, param_names)
    viz.plot_best_worst_fits(x_val.cpu().numpy(), y_val_orig, y_pred_orig, top_n=3)
    viz.plot_residuals(y_val_orig, y_pred_orig, param_names)
    viz.plot_error_histograms(y_val_orig, y_pred_orig, param_names)


if __name__ == "__main__":
    evaluate_and_visualize(batch_size=64)
