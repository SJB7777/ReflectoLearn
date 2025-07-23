from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ..io import get_data
from ..math_utils import normalize, q_fourier_transform_multisample
from ..models.model import XRRHybridRegressor
from ..visualization import (
    plot_3d_scatter,
    plot_best_worst_fits,
    plot_error_histograms,
    plot_r2_scores,
    plot_residuals,
)
from .train import evaluate_model


def load_and_preprocess_data(data_file_path: Path):
    if not data_file_path.exists():
        raise FileNotFoundError(f"Data file {data_file_path} does not exist.")

    data = get_data(data_file_path)

    Rs = data["Rs"].astype("float32")
    FT = q_fourier_transform_multisample(Rs, data["q"], np.linspace(0, 500, 2048))
    rho_z = np.abs(FT) ** 2
    x_all_np = np.log10(rho_z + 1e-8)
    x_all_np = normalize(x_all_np)
    x_all = torch.tensor(x_all_np, dtype=torch.float32)

    y_all_np = data["params"].astype("float32")
    scaler = StandardScaler()
    y_all_scaled_np = scaler.fit_transform(y_all_np)
    y_all_scaled = torch.tensor(y_all_scaled_np, dtype=torch.float32)

    return x_all, y_all_scaled, scaler


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Paths
    model_path = Path("saved_model_hybrid_fourier.pt")
    data_path = Path("X:\\XRR_AI\\hdf5_XRR") / "raw_data\\p100o3_2.h5"

    # Load data
    x_all, y_all_scaled, scaler = load_and_preprocess_data(data_path)
    _, x_val, _, y_val_scaled = train_test_split(
        x_all, y_all_scaled, test_size=0.2, random_state=42
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val_scaled), batch_size=256, shuffle=False
    )

    # Load model
    model = XRRHybridRegressor(input_length=x_val.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Evaluate
    y_pred_orig, y_val_orig = evaluate_model(model, val_loader, scaler, device)

    # Visualization
    param_names = ["sld", "thickness", "roughness"]
    plot_r2_scores(y_val_orig, y_pred_orig, param_names)
    plot_best_worst_fits(x_val, y_val_orig, y_pred_orig, top_n=5)
    plot_error_histograms(y_val_orig, y_pred_orig, param_names)
    plot_3d_scatter(y_val_orig, y_pred_orig)
    plot_residuals(y_val_orig, y_pred_orig, param_names)

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
