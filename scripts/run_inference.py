import tkinter as tk
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from tkinter import filedialog

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit
from scipy.signal import argrelmax
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from reflectolearn.config import ConfigManager
from reflectolearn.device import get_device_and_workers
from reflectolearn.io import read_xrr_hdf5
from reflectolearn.logger import setup_logger
from reflectolearn.math_utils import normalize
from reflectolearn.models.general import get_model
from reflectolearn.processing.fitting import estimate_q, func_gauss3_with_noise_ver2, s_vector_transform_q, xrr_fft
from reflectolearn.types import ModelType


@torch.no_grad()
def blend_with_prior_scaled(
    y_pred_scaled: torch.Tensor,                     # (N, D) in scaled space
    prior_raw: np.ndarray | None,                 # (N, D) in raw space, np.nan where no prior
    scaler: StandardScaler,
    prior_indices: Iterable[int] = (2, 3),
    alpha_model: float = 0.7,                        # weight for model in [0,1]
) -> torch.Tensor:
    """
    최종 = alpha * y_pred + (1-alpha) * prior  (scaled 공간에서 수행)
    prior가 없는 위치(np.nan)는 그대로 y_pred 유지.
    """
    if prior_raw is None:
        return y_pred_scaled

    y_pred = y_pred_scaled.clone()
    prior_scaled = np.full_like(prior_raw, np.nan, dtype=np.float64)

    # raw prior -> scaled prior (지정 인덱스만)
    for j in prior_indices:
        prior_scaled[:, j] = (prior_raw[:, j] - scaler.mean_[j]) / scaler.scale_[j]

    # blend
    for j in prior_indices:
        mask = ~np.isnan(prior_scaled[:, j])
        if np.any(mask):
            p = torch.tensor(prior_scaled[:, j], dtype=y_pred.dtype, device=y_pred.device)
            y_pred[mask, j] = alpha_model * y_pred[mask, j] + (1.0 - alpha_model) * p[mask]
    return y_pred


def gaussian3_fitting(x_fit, y_fit, p0):
    bounds = (0, np.inf)
    return curve_fit(func_gauss3_with_noise_ver2, x_fit, y_fit, p0=p0, bounds=bounds)[0]


def guess(args):
    q, R, x_upper_bound = args
    try:
        crit_q = estimate_q(q, R)
        dat = np.stack([q, R], axis=1)
        xproc, yproc = s_vector_transform_q(dat, crit_q)
        x_fft, y_fft = xrr_fft(xproc, yproc, window=2, n=10000)
        x_fft *= 2 * np.pi
        y_fft_norm = y_fft / y_fft[0]

        y_diff = np.diff(y_fft_norm)
        idx_lb = np.where((y_diff >= -0.01) & (x_fft[1:] > 2))[0][0] + 1
        idx_ub = np.where(x_fft > x_upper_bound)[0][0]

        x_fit, y_fit = x_fft[idx_lb: idx_ub + 1], y_fft_norm[idx_lb: idx_ub + 1]

        idx_local_max = argrelmax(y_fit[x_fit < x_upper_bound])
        y_local = y_fit[idx_local_max]
        x_local = x_fit[idx_local_max]

        if y_local.size < 2:
            return None

        top2_idx = np.argsort(y_local)[-2:]
        top2_x = np.sort(x_local[top2_idx])
        pmax2, pmax3 = top2_x[0], top2_x[1]

        p0 = [0.1, 5, 0.1, pmax2, 5, 0.1, pmax3, 5, 1, 10, 0.001]
        popt = gaussian3_fitting(x_fit, y_fit, p0)

        pos2 = popt[3]
        pos3 = popt[6]
        pos1 = pos3 - pos2
        return pos1, pos2
    except Exception:
        return None


@torch.no_grad()
def inference(ckpt_path, scaler_path, q, Rs, model_type: ModelType, alpha_model=0.7, prior_indices=(1, 3), batch_size=64, device=None):
    # --- load scaler & data
    scaler: StandardScaler = joblib.load(scaler_path)

    x = torch.tensor(Rs, dtype=torch.float32)
    N, L = x.shape
    D = scaler.mean_.shape[0]

    # --- load model
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_type, input_length=L, output_length=D).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # --- batch model inference
    dataset = TensorDataset(x)
    loader = DataLoader(dataset, batch_size=batch_size)
    y_pred_list = []
    for (xb,) in loader:
        y_pred_list.append(model(xb.to(device)).cpu())
    y_pred_scaled = torch.cat(y_pred_list, dim=0)

    # --- parallel prior calculation
    logger.info("Start Guessing...")
    x_upper_bound = 200
    iterable = [(q, R, x_upper_bound) for R in Rs]
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        prior_results = list(tqdm(executor.map(guess, iterable), total=len(iterable), desc="Processing"))

    prior_raw = np.full((N, D), np.nan, dtype=np.float64)
    for i, g in enumerate(prior_results):
        if g is None:
            continue
        pos1, pos2 = g
        prior_raw[i, 2] = pos1
        prior_raw[i, 3] = pos2

    # --- blend with prior
    y_blended_scaled = y_pred_scaled.clone()
    for j in prior_indices:
        mask = ~np.isnan(prior_raw[:, j])
        if np.any(mask):
            p = torch.tensor((prior_raw[:, j] - scaler.mean_[j]) / scaler.scale_[j], dtype=y_pred_scaled.dtype)
            y_blended_scaled[mask, j] = alpha_model * y_pred_scaled[mask, j] + (1 - alpha_model) * p[mask]

    # --- inverse transform
    y_pred_raw = scaler.inverse_transform(y_pred_scaled.numpy())
    y_blended_raw = scaler.inverse_transform(y_blended_scaled.numpy())

    return y_pred_raw, y_blended_raw, prior_raw


def ask_dir(initialdir="./results") -> Path:
    """
    Shows a directory selection dialog and returns the selected folder path.
    """
    root = tk.Tk()
    root.withdraw()

    folder_path = filedialog.askdirectory(
        initialdir=initialdir,
        title="Select a model root folder"
    )

    if folder_path:
        return Path(folder_path)
    else:
        # User canceled the dialog
        return None


# === 평가 함수 ===
def safe_rmse(y_true, y_pred):
    mask = ~np.isnan(y_pred).any(axis=1)
    return np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2))


def safe_r2_score(y_true, y_pred, multioutput="raw_values"):
    """
    NaN-safe r2_score 계산. feature 단위로 NaN이 있는 샘플을 제외하고 계산.
    """
    results = []
    n_features = y_true.shape[1]

    for j in range(n_features):
        mask = ~np.isnan(y_pred[:, j]) & ~np.isnan(y_true[:, j])
        if np.sum(mask) < 2:
            # 샘플이 너무 적으면 R² 계산 불가 → NaN 반환
            results.append(np.nan)
        else:
            score = r2_score(y_true[mask, j], y_pred[mask, j])
            results.append(score)

    return np.array(results) if multioutput == "raw_values" else np.nanmean(results)


def main() -> None:

    ConfigManager.initialize("config.yaml")
    config = ConfigManager.load_config()

    device, num_workers = get_device_and_workers()
    logger.info(f"Using device: {device}")
    logger.info(f"Number of workers: {num_workers}")

    # === ground truth 로드 ===
    data = read_xrr_hdf5(config.path.data_file)
    q = data["q"]
    x_array = normalize(data["R"])
    y_true = np.concatenate([data["roughness"], data["thickness"], data["sld"]], axis=1).astype(np.float32)

    model_root: Path = ask_dir()

    logger.info(f"Shape of q: {q.shape}")
    logger.info(f"Shape of x_array: {x_array.shape}")
    logger.info(f"Shape of y_true: {y_true.shape}")
    logger.info(f"Model files root: {model_root}")

    # y_hat: 순수 모델 예측 (raw)
    # y_hat_blend: prior 보정 결과 (raw)
    # prior: 추정된 prior (raw; 없는 곳은 NaN)
    y_hat, y_hat_blend, prior = inference(
        ckpt_path = model_root / "model.pt",
        scaler_path= model_root / "scaler.pkl",
        q = q,
        Rs = x_array,
        model_type = ModelType.HYBRID,
        alpha_model = 0.7,                  # 모델 70%, prior 30%
        prior_indices = (2, 3),
        device=device
    )

    rmse_model = safe_rmse(y_true, y_hat)
    rmse_blended = safe_rmse(y_true, y_hat_blend)
    rmse_prior = safe_rmse(y_true, prior)

    logger.info(f"RMSE Model only: {rmse_model:.4f}")
    logger.info(f"RMSE Model+Prior: {rmse_blended:.4f}")
    logger.info(f"RMSE Prior only: {rmse_prior:.4f}")

    # === R² 계산 (각 파라미터별) ===
    r2_model = safe_r2_score(y_true, y_hat, multioutput='raw_values')
    r2_blended = safe_r2_score(y_true, y_hat_blend, multioutput='raw_values')
    r2_prior = safe_r2_score(y_true, prior, multioutput='raw_values')

    logger.info("\nR² per parameter")
    for i in range(y_true.shape[1]):
        logger.info(f"y[{i}]: model={r2_model[i]:.3f}, blended={r2_blended[i]:.3f}, prior={r2_prior[i]:.3f}")

    # === 시각화 ===
    prior_indices = (1, 3)
    for j in prior_indices:
        plt.figure()
        plt.scatter(y_true[:, j], y_hat[:, j], label="Model", alpha=0.2)
        plt.scatter(y_true[:, j], y_hat_blend[:, j], label="Blended", alpha=0.2)
        plt.scatter(y_true[:, j], prior[:, j], label="Prior", alpha=0.2)
        plt.plot([y_true[:, j].min(), y_true[:, j].max()],
                [y_true[:, j].min(), y_true[:, j].max()],
                'k--', lw=1)
        plt.xlabel("Ground Truth")
        plt.ylabel(f"y[{j}] Prediction")
        plt.title(f"Prediction comparison for parameter {j}")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    logger = setup_logger()
    try:
        main()
    except Exception:
        logger.exception("Application failed with error")
        raise
