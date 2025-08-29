import numpy as np
import torch
from loguru import logger
from scipy.optimize import curve_fit
from scipy.signal import argrelmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from reflectolearn.config import ConfigManager
from reflectolearn.io import get_data
from reflectolearn.math_utils import normalize
from reflectolearn.models.model import get_model
from reflectolearn.processing.fitting import (
    estimate_q,
    func_gauss3_with_noise_ver2,
    s_vector_transform_q,
    xrr_fft,
)


def multi_gaussian_fitting2(x_fit, y_fit, p0):
    # ----------------------- Fit -----------------------
    bounds = (0, np.inf)
    popt, pcov = curve_fit(func_gauss3_with_noise_ver2, x_fit, y_fit, p0=p0, bounds=bounds)

    return popt, pcov


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
        popt, pcov = multi_gaussian_fitting2(x_fit, y_fit, p0)
    except RuntimeError:
        return None

    pos2 = popt[3]
    pos3 = popt[6]
    pos1 = pos3-pos2
    return pos1, pos2  # thickness1->y[1], thickness2->y[3]


def main():
    logger.info("Starting training script")
    ConfigManager.initialize("config.yaml")
    config = ConfigManager.load_config()
    logger.info(f"Config: {config}")
    seed = config.training.seed
    set_seed(seed)

    device, num_workers = get_device_and_workers()
    logger.info(f"Using device: {device}")
    logger.info(f"Number of workers: {num_workers}")

    data_file = config.path.data_file

    # Data Loading
    data = get_data(data_file)
    q = data["q"]
    y_array = data["params"].astype(np.float32)  # Shape: (n_samples, n_params)
    guesses: list[None | tuple[float, float]] = [guessing(q, R) for R in data["Rs"]]
    print(guesses.count(None))
    # Thickness prior estimation
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y_array)
    y_all = torch.tensor(y_scaled, dtype=torch.float32)

    reflectivity = torch.tensor(data["Rs"], dtype=torch.float32)
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


    # optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    # loss_fn = nn.MSELoss()

    # logger.info("Starting model training...")
    # train_losses, val_losses = train_model(
    #     model,
    #     train_loader,
    #     val_loader,
    #     optimizer,
    #     loss_fn,
    #     device,
    #     num_epochs=config.training.epochs,
    #     patience=config.training.patience,
    # )
    # logger.info("Model training finished.")
    # # === Directories ===
    # result_dir = config.path.output_dir

    # for d in [result_dir, result_dir, result_dir]:
    #     d.mkdir(parents=True, exist_ok=True)

    # # === Save model ===
    # model_path = result_dir / "model.pt"
    # save_model(model.state_dict(), model_path)
    # logger.info(f"Best model saved to {model_path}")

    # # === Save scaler ===
    # scaler_path = result_dir / "scaler.pkl"
    # joblib.dump(scaler, scaler_path)
    # logger.info(f"Scaler saved to {scaler_path}")

    # # === Save training curves ===
    # stats_path = result_dir / "stats.npz"
    # np.savez(
    #     stats_path,
    #     train_losses=np.array(train_losses),
    #     val_losses=np.array(val_losses),
    #     config=config,
    # )
    # logger.info(f"Training statistics saved to {stats_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application failed with error: {e}", exc_info=True)
        raise
