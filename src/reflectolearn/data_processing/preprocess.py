from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from ..io.read import get_data
from ..io.save import xrd2hdf5
from ..math_utils import normalize, q_fourier_transform_multisample_gpu


def preprocess_features(version: str, raw_data: dict) -> torch.Tensor:
    reflectivity = torch.tensor(raw_data["Rs"], dtype=torch.float32)

    match version:
        case "raw":
            features = torch.log10(reflectivity + 1e-8)
        case "q4":
            q_values = torch.tensor(raw_data["q"], dtype=torch.float32)
            features = q_values**4 * reflectivity
            features = torch.log10(features + 1e-8)
        case "fourier":
            z_axis = torch.linspace(0, 500, 2048, dtype=torch.float32)
            ft_result = q_fourier_transform_multisample_gpu(
                reflectivity, raw_data["q"], z_axis
            )
            features = torch.abs(ft_result) ** 2
            features = torch.log10(features + 1e-8)
        case _:
            raise ValueError(f"Unsupported preprocessing version: {version}")

    return normalize(features)


def load_and_preprocess_data(
    data_file_path: Path, data_version: str = "raw"
) -> tuple[torch.Tensor, torch.Tensor, StandardScaler]:
    if not data_file_path.exists():
        raise FileNotFoundError(f"Data file {data_file_path} does not exist.")

    raw_data = get_data(data_file_path)
    x_tensor = preprocess_features(data_version, raw_data)

    y_array = raw_data["params"].astype(np.float32)
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y_array)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    return x_tensor, y_tensor, scaler


def preprocess_q4(raw_file: Path, preprocessed_file: Path):
    """Preprocesses the raw XRR data and saves it in a new HDF5 file."""
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw data file {raw_file} does not exist.")

    # Load raw data
    raw_data = get_data(raw_file)

    preprocessed_arr = raw_data["Rs"] * raw_data["q"] ** 4
    preprocessed_arr = np.log(preprocessed_arr + 1e-10)
    Rs = normalize(preprocessed_arr)
    preprocessed_data = {
        "q": raw_data["q"],
        "Rs": Rs,
        "params": raw_data["params"],
    }

    xrd2hdf5(preprocessed_data, preprocessed_file)


def preprocess_file_gpu(raw_file: Path, preprocessed_file: Path):
    """Preprocesses the raw XRR data and saves it in a new HDF5 file."""
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw data file {raw_file} does not exist.")

    # Load raw data
    raw_data = get_data(raw_file)

    # Perform Fourier transform and normalization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ft_arr = q_fourier_transform_multisample_gpu(
        torch.from_numpy(raw_data["Rs"]).to(device),
        torch.from_numpy(raw_data["q"]).to(device),
        torch.linspace(0, 500, 2048, device=device),
    )
    rho_z = torch.abs(ft_arr) ** 2
    log_rho_z = torch.log(rho_z + 1e-10)  # Avoid log(0)
    preprocessed_arr = normalize(log_rho_z)
    Rs = preprocessed_arr.cpu().numpy().astype("float32")

    # Prepare data for saving
    preprocessed_data = {
        "q": raw_data["q"],
        "Rs": Rs,
        "params": raw_data["params"],
    }

    # Save preprocessed data to HDF5 file
    xrd2hdf5(preprocessed_data, preprocessed_file)


def preprocess_file(raw_file: Path, preprocessed_file: Path):
    """Preprocesses the raw XRR data and saves it in a new HDF5 file."""
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw data file {raw_file} does not exist.")

    # Load raw data
    raw_data = get_data(raw_file)

    Rs = preprocess_features(raw_data["version"], raw_data)

    # Prepare data for saving
    preprocessed_data = {
        "q": raw_data["q"],
        "Rs": Rs.numpy().astype("float32"),
        "params": raw_data["params"],
    }

    # Save preprocessed data to HDF5 file
    xrd2hdf5(preprocessed_data, preprocessed_file)
