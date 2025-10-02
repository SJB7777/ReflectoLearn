from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import argrelmax
from torch.utils.data import DataLoader, Dataset

from reflectolearn.config import ConfigManager
from reflectolearn.device import get_device_and_workers
from reflectolearn.io import read_xrr_hdf5
from reflectolearn.logger import setup_logger
from reflectolearn.math_utils import normalize
from reflectolearn.models.multi_branch import XRRMultiBranchRegressor
from reflectolearn.processing.fitting import estimate_q, s_vector_transform_q, xrr_fft


# ========================
# Physics Feature Extractor
# ========================
def extract_physics_features(q, R, x_upper_bound=200):
    crit_q = estimate_q(q, R)
    dat = np.stack([q, R], axis=1)
    xproc, yproc = s_vector_transform_q(dat, crit_q)
    x_fft, y_fft = xrr_fft(xproc, yproc, window=2, n=10000)
    x_fft *= 2 * np.pi
    y_fft_norm = y_fft / y_fft[0] if y_fft[0] != 0 else y_fft

    features = {}
    try:
        y_diff = np.diff(y_fft_norm)
        idx_lb = np.where((y_diff >= -0.01) & (x_fft[1:] > 2))[0][0] + 1
        idx_ub = np.where(x_fft > x_upper_bound)[0][0]
        x_fit, y_fit = x_fft[idx_lb:idx_ub+1], y_fft_norm[idx_lb:idx_ub+1]

        idx_local_max = argrelmax(y_fit[x_fit < x_upper_bound])
        x_local, y_local = x_fit[idx_local_max], y_fit[idx_local_max]

        if len(x_local) >= 2:
            features["fft_peak_pos2"] = float(x_local[-2])
            features["fft_peak_pos3"] = float(x_local[-1])
            features["fft_peak_spacing"] = float(x_local[-1] - x_local[-2])
            features["fft_peak_ratio"] = float(y_local[-2] / y_local[-1])
    except Exception:
        features["fft_peak_pos2"] = 0.0
        features["fft_peak_pos3"] = 0.0
        features["fft_peak_spacing"] = 0.0
        features["fft_peak_ratio"] = 0.0

    features["crit_q"] = crit_q
    return np.array(list(features.values()), dtype=np.float32)


# ========================
# Dataset
# ========================
class XRRDataset(Dataset):
    def __init__(self, q, R, y_true):
        self.q = q
        self.R = R
        self.y_true = y_true

    def __len__(self):
        return len(self.R)

    def __getitem__(self, idx):
        q = self.q
        R = self.R[idx]
        y = self.y_true[idx]

        # raw reflectivity curve
        curve_tensor = torch.tensor(R, dtype=torch.float32)

        # physics features
        feat = extract_physics_features(q, R)
        feat_tensor = torch.tensor(feat, dtype=torch.float32)

        return curve_tensor, feat_tensor, torch.tensor(y, dtype=torch.float32)


# ========================
# Training Loop
# ========================
def train():
    ConfigManager.initialize("config.yaml")
    config = ConfigManager.load_config()

    device, num_workers = get_device_and_workers()
    logger.info(f"Using device: {device}")
    logger.info(f"Number of workers: {num_workers}")

    # === Load data ===
    data = read_xrr_hdf5(config.path.data_file)
    q = data["q"]
    x_array = normalize(data["R"])
    y_true = np.concatenate([data["roughness"], data["thickness"], data["sld"]], axis=1).astype(np.float32)

    dataset = XRRDataset(q, x_array, y_true)
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=num_workers)

    # === Init model ===
    dummy_feat = extract_physics_features(q, x_array[0])
    num_phys_features = dummy_feat.shape[0]

    model = XRRMultiBranchRegressor(
        input_length=x_array.shape[1],
        num_phys_features=num_phys_features,
        output_length=y_true.shape[1]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    criterion = nn.MSELoss()

    # === Training ===
    for epoch in range(config.training.epochs):
        model.train()
        total_loss = 0
        for curve, feat, target in loader:
            curve, feat, target = curve.to(device), feat.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(curve, feat)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.info(f"Epoch {epoch+1}/{config.training.epochs}, Loss={total_loss/len(loader):.4f}")

    # save model
    torch.save(model.state_dict(), Path(config.path.output_dir) / "model_multibranch.pt")


if __name__ == "__main__":
    logger = setup_logger()
    try:
        train()
    except Exception:
        logger.exception("Training failed with error")
        raise
