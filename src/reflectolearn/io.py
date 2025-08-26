from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm


def xrd2hdf5(data: dict[str, np.ndarray], save_file: Path):
    if not {"q", "Rs", "params"}.issubset(data.keys()):
        raise ValueError("Data must contain 'q', 'Rs', and 'params' keys.")

    with h5py.File(save_file, "w") as f:
        f.create_dataset("q", data=data["q"].astype("f4"))
        samples_group = f.create_group("samples")

        for i in tqdm(range(data["Rs"].shape[0]), desc="Saving samples"):
            R = data["Rs"][i]
            thickness, roughness, sld = data["params"][i]

            sample_name = f"sample_{i:06d}"
            g = samples_group.create_group(sample_name)

            g.create_dataset("R", data=R.astype("f4"))
            g.attrs["thickness"] = thickness
            g.attrs["roughness"] = roughness
            g.attrs["sld"] = sld


def save_model(model_state_dict: dict, save_path: Path):
    """Saves the model's state dictionary to the specified path."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_state_dict, save_path)


def get_data(path: Path):
    """
    Read XRR simulation data from the HDF5 file created by make_xrr_hdf5.

    Parameters
    ----------
    path : Path
        The path to the HDF5 file.

    Returns
    -------
    dict
        A dictionary containing:
        - 'q': np.ndarray, the global Q values.
        - 'Rs': np.ndarray, a 2D array of reflectivity (R) values for all samples.
                Shape: (n_sample, len(q))
        - 'params': np.ndarray, a 2D array of parameters for all samples.
                    Shape: (n_sample, n_layer * 3) where each row is
                    [t0, r0, sld0, t1, r1, sld1, ...]
    """
    Rs = []
    params = []

    with h5py.File(path, "r") as f:
        # Read global q values
        q = f["q"][:]

        # Determine n_layer from the first sample's attributes
        # This assumes all samples have the same n_layer
        # And that thickness_0, thickness_1, ... are present
        first_sample_name = list(f["samples"].keys())[0]
        first_sample_group = f["samples"][first_sample_name]

        # Count how many 'thickness_X' attributes exist to infer n_layer
        n_layer = 0
        while f"thickness_{n_layer}" in first_sample_group.attrs:
            n_layer += 1

        if n_layer == 0:
            raise ValueError("Could not determine n_layer from HDF5 attributes. 'thickness_0' not found.")

        # Iterate through each sample group
        for sample_name in tqdm(f["samples"], desc="Loading samples"):
            sample_group = f["samples"][sample_name]
            R = sample_group["R"][:]
            Rs.append(R)

            # Extract parameters for the current sample
            current_sample_params = []
            for i in range(n_layer):
                current_sample_params.append(sample_group.attrs[f"thickness_{i}"])
                current_sample_params.append(sample_group.attrs[f"roughness_{i}"])
                current_sample_params.append(sample_group.attrs[f"sld_{i}"])
            params.append(current_sample_params)

    # Convert lists to numpy arrays for consistent output
    return {"q": np.array(q), "Rs": np.array(Rs), "params": np.array(params)}
