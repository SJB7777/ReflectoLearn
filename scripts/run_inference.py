from collections.abc import Iterable

import joblib
import numpy as np
import torch
from scipy.optimize import curve_fit
from scipy.signal import argrelmax
from sklearn.preprocessing import StandardScaler

from reflectolearn.io import get_data, read_xrr_hdf5
from reflectolearn.math_utils import normalize
from reflectolearn.models.model import get_model
from reflectolearn.processing.fitting import estimate_q, func_gauss3_with_noise_ver2, s_vector_transform_q, xrr_fft
from reflectolearn.types import ModelType


@torch.no_grad()
def blend_with_prior_scaled(
    y_pred_scaled: torch.Tensor,                     # (N, D) in scaled space
    prior_raw: np.ndarray | None,                 # (N, D) in raw space, np.nan where no prior
    scaler: StandardScaler,
    prior_indices: Iterable[int] = (1, 3),
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


def guessing(q, R) -> None | tuple[float, float]:

    x_upper_bound = 200
    crit_q = estimate_q(q, R)

    # FFT
    dat = np.stack([q, R], axis=1)
    xproc, yproc = s_vector_transform_q(dat, crit_q)
    x_fft, y_fft = xrr_fft(xproc, yproc, window=2, n=10000)
    x_fft = x_fft * 2 * np.pi
    y_fft_norm = y_fft / y_fft[0]

    # First increasing index
    y_diff = np.diff(y_fft_norm)
    idx_lb = np.where((y_diff >= -0.01) & (x_fft[1:] > 2))[0][0] + 1
    idx_ub = np.where(x_fft > x_upper_bound)[0][0]

    # fitting range
    x_fit, y_fit = x_fft[idx_lb: idx_ub + 1], y_fft_norm[idx_lb: idx_ub + 1]

    # Find Peaks
    idx_local_max = argrelmax(y_fit[x_fit < x_upper_bound])

    # local maxima들의 인덱스와 값
    y_local = y_fit[idx_local_max]
    x_local = x_fit[idx_local_max]

    if y_local.size == 1:
        return None
    if y_local.size == 1:
        top2_x = np.array([x_local[0], x_local[0]])            # 그 두 점의 x 위치
    else:
        top2_idx = np.argsort(y_local)[-2:]          # 상위 2개 (y 값 기준)
        top2_x = np.sort(x_local[top2_idx])

    pmax2, pmax3 = top2_x[0], top2_x[1]

    # params: list[str]= ["a1", "w1", "a2", "pmax2", "w2", "a3", "pmax3", "w3", "a4", "w4", "z0"]
    p0: list[float] = [0.1, 5, 0.1, pmax2, 5, 0.1, pmax3, 5, 1, 10, 0.001]
    try:
        popt = gaussian3_fitting(x_fit, y_fit, p0)
    except RuntimeError:
        return None

    pos2 = popt[3]
    pos3 = popt[6]
    pos1 = pos3-pos2
    return pos1, pos2  # thickness1->y[1], thickness2->y[3]


@torch.no_grad()
def infer(
    ckpt_path: str,
    scaler_path: str,
    data_file: str,
    model_type: ModelType,
    alpha_model: float = 0.7,
    prior_indices=(1, 3),
):
    # --- load
    scaler: StandardScaler = joblib.load(scaler_path)
    data = read_xrr_hdf5(data_file)
    q = data["q"]                      # (Lq,)
    Rs = data["R"]                    # (N, L)
    x = normalize(torch.tensor(Rs, dtype=torch.float32))  # (N, L)

    # --- model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = scaler.mean_.shape[0]
    model = get_model(model_type, input_length=x.shape[1], output_length=D).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # --- raw prediction (scaled 공간)
    y_pred_scaled = model(x.to(device))  # (N, D)

    # --- build prior (raw 공간). prior 없는 위치는 np.nan
    prior_raw = np.full((len(Rs), D), np.nan, dtype=np.float64)
    for i, R in enumerate(Rs):
        g = guessing(q, R)
        if g is None:
            continue
        pos1, pos2 = g
        prior_raw[i, 2] = pos1
        prior_raw[i, 3] = pos2

    # --- blend (scaled 공간에서)
    y_blended_scaled = blend_with_prior_scaled(
        y_pred_scaled, prior_raw, scaler, prior_indices=prior_indices, alpha_model=alpha_model
    )

    # --- inverse transform to raw
    y_pred_raw = scaler.inverse_transform(y_pred_scaled.cpu().numpy())
    y_blended_raw  = scaler.inverse_transform(y_blended_scaled.cpu().numpy())

    return y_pred_raw, y_blended_raw, prior_raw

if __name__ == "__main__":
    from pathlib import Path

    import joblib
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import r2_score

    from reflectolearn.config import ConfigManager
    from reflectolearn.io import get_data
    from reflectolearn.logger import setup_logger

    logger = setup_logger()
    ConfigManager.initialize("config.yaml")
    config = ConfigManager.load_config()

    model_root: Path = Path("./results/model_20250902_104441")
    y_hat, y_hat_blend, prior = infer(
        ckpt_path = model_root / "model.pt",
        scaler_path= model_root / "scaler.pkl",
        data_file = config.path.data_file,
        model_type = ModelType.HYBRID,         # get_model 내부 이름에 맞추세요
        alpha_model = 0.7,                  # 모델 70%, prior 30%
        prior_indices = (2, 3),
    )
    # y_hat: 순수 모델 예측 (raw)
    # y_hat_blend: prior 보정 결과 (raw)
    # prior: 추정된 prior (raw; 없는 곳은 NaN)

    # === ground truth 로드 ===
    data = read_xrr_hdf5(config.path.data_file)
    q = data["q"]
    x_array = data["R"]
    y_true = np.concatenate([data["roughness"], data["thickness"], data["sld"]], axis=1).astype(np.float32)

    # === 평가 함수 ===
    def rmse(y_true, y_pred):
        return np.sqrt(np.nanmean((y_true - y_pred)**2))

    rmse_model = rmse(y_true, y_hat)
    rmse_blended = rmse(y_true, y_hat_blend)
    rmse_prior = rmse(y_true, prior)

    logger.info(f"RMSE Model only: {rmse_model:.4f}")
    logger.info(f"RMSE Model+Prior: {rmse_blended:.4f}")
    logger.info(f"RMSE Prior only: {rmse_prior:.4f}")

    # === R² 계산 (각 파라미터별) ===
    r2_model = r2_score(y_true, y_hat, multioutput='raw_values')
    r2_blended = r2_score(y_true, y_hat_blend, multioutput='raw_values')
    r2_prior = r2_score(y_true, prior, multioutput='raw_values')

    logger.info("\nR² per parameter")
    for i in range(y_true.shape[1]):
        logger.info(f"y[{i}]: model={r2_model[i]:.3f}, blended={r2_blended[i]:.3f}, prior={r2_prior[i]:.3f}")

    # === 시각화 ===
    prior_indices = (1, 3)
    for j in prior_indices:
        plt.figure()
        plt.scatter(y_true[:, j], y_hat[:, j], label="Model", alpha=0.6)
        plt.scatter(y_true[:, j], y_hat_blend[:, j], label="Blended", alpha=0.6)
        plt.scatter(y_true[:, j], prior[:, j], label="Prior", alpha=0.6)
        plt.plot([y_true[:, j].min(), y_true[:, j].max()],
                [y_true[:, j].min(), y_true[:, j].max()],
                'k--', lw=1)
        plt.xlabel("Ground Truth")
        plt.ylabel(f"y[{j}] Prediction")
        plt.title(f"Prediction comparison for parameter {j}")
        plt.legend()
        plt.show()
