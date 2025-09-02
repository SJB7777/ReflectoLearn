import re
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from refnx.reflect.structure import Structure
from tqdm import tqdm

from reflectolearn.processing.simulate import add_xrr_noise, make_n_layer_structure, make_parameters, structure_to_R


def append_timestamp(path: str | Path) -> Path:
    """
    Append a timestamp to a path (file or directory) to make it unique.
    The exact naming format is not fixed.
    """
    path = Path(path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if path.suffix:
        new_name = f"{path.stem}_{timestamp}{path.suffix}"
    else:
        new_name = f"{path.name}_{timestamp}"

    return path.parent / new_name


def next_unique_file(path: Path | str) -> Path:
    """
    Generate a unique file path.
    - If no file exists with the same name, return the original path.
    - If files exist, append (n) where n = max existing number + 1.

    Example:
        file.txt       -> file.txt        (if doesn't exist)
        file.txt       -> file (1).txt     (if file.txt exists)
        file (1).txt    -> file (2).txt     (if file(1).txt exists)
    """
    path = Path(path)
    parent = path.parent
    stem = path.stem
    suffix = path.suffix

    # Regular expression: Find '(number)' end of the stem.
    pattern = re.compile(rf"^{re.escape(stem)}(?:\((\d+)\))?$")

    # Find same patterns
    existing = [f for f in parent.glob(f"{stem}*{suffix}") if pattern.match(f.stem)]

    if not existing:
        return path

    max_n = 0
    for f in existing:
        m = pattern.match(f.stem)
        if m and m.group(1):
            n = int(m.group(1))
            if n > max_n:
                max_n = n

    return parent / f"{stem}({max_n + 1}) {suffix}"


def simulate_one(idx: int, n_layer: int, q: np.ndarray, has_noise: bool):
    """Make one sample"""
    thicknesses, roughnesses, slds = make_parameters(n_layer)
    structure: Structure = make_n_layer_structure(thicknesses, roughnesses, slds)
    R = structure_to_R(structure, q)
    if has_noise:
        R = add_xrr_noise(R)
    return (
        idx,
        R.astype("f4"),
        np.array(thicknesses, dtype="f4"),
        np.array(roughnesses, dtype="f4"),
        np.array(slds, dtype="f4"),
    )


def make_xrr_hdf5(
    save_file: Path,
    n_layer: int,
    q: np.ndarray,
    n_sample: int,
    has_noise: bool = True,
    n_workers: int | None = None,
    batch_size: int = 1000,
    chunksize: int = 100
):
    """
    Generate large XRR HDF5 Dataset
    """
    N = len(q)

    with h5py.File(save_file, "w") as f:
        # Save q
        f.create_dataset("q", data=q.astype("f4"))

        # Make large dataset
        dR = f.create_dataset(
            "R",
            shape=(n_sample, N),
            dtype="f4",
            compression="lzf",
            chunks=(batch_size, N),
        )
        dT = f.create_dataset(
            "thickness",
            shape=(n_sample, n_layer),
            dtype="f4",
            compression="lzf",
            chunks=(batch_size, n_layer),
        )
        dRough = f.create_dataset(
            "roughness",
            shape=(n_sample, n_layer),
            dtype="f4",
            compression="lzf",
            chunks=(batch_size, n_layer),
        )
        dSLD = f.create_dataset(
            "sld",
            shape=(n_sample, n_layer),
            dtype="f4",
            compression="lzf",
            chunks=(batch_size, n_layer),
        )

        # Parallel processing
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            pbar = tqdm(total=n_sample)
            for batch_start in range(0, n_sample, batch_size):
                batch_end = min(batch_start + batch_size, n_sample)
                batch_indices = range(batch_start, batch_end)

                results = list(
                    executor.map(
                        simulate_one,
                        batch_indices,
                        [n_layer] * len(batch_indices),
                        [q] * len(batch_indices),
                        [has_noise] * len(batch_indices),
                        chunksize=chunksize,
                    )
                )

                # Save batch result
                for idx, R, T, Rough, SLD in results:
                    dR[idx] = R
                    dT[idx] = T
                    dRough[idx] = Rough
                    dSLD[idx] = SLD

                pbar.update(len(batch_indices))


def read_xrr_hdf5(file: str | Path) -> dict:
    with h5py.File(file, "r") as f:
        data = {}
        for key, val in f.items():
            data[key] = val[()]
        return data


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
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File {str(path)} not found")

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


if __name__ == "__main__":
    from reflectolearn.config import ConfigManager

    ConfigManager.initialize("config.yaml")
    config = ConfigManager.load_config()
    data = read_xrr_hdf5(config.path.data_file)
    print(data)
