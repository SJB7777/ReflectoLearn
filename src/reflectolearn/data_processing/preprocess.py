from pathlib import Path

import numpy as np
import torch

from ..math_utils import (
    normalize,
    q_fourier_transform_multisample,
    q_fourier_transform_multisample_gpu,
)
from ..io.read import get_data
from ..io.save import xrd2hdf5


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

    # Perform Fourier transform and normalization
    ft_arr = q_fourier_transform_multisample(
        raw_data["Rs"], raw_data["q"], np.linspace(0, 500, 2048)
    )
    rho_z = np.abs(ft_arr) ** 2
    preprocessed_arr = normalize(rho_z)

    # Prepare data for saving
    preprocessed_data = {
        "q": raw_data["q"],
        "Rs": preprocessed_arr,
        "params": raw_data["params"],
    }

    # Save preprocessed data to HDF5 file
    xrd2hdf5(preprocessed_data, preprocessed_file)
