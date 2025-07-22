import random
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from ..data_processing.simulate import (
    make_one_layer_structure,
    make_structure_2l,
    simulate_xrr,
    simulate_xrr_with_noise,
)


def main_hdf5(save_file: Path, total: int):
    N: int = 100
    q = np.linspace(0.005, 0.3, N)

    # Create HDF5 file
    with h5py.File(save_file, "w") as f:
        # Save q globally at root
        f.create_dataset("q", data=q.astype("f4"))
        # Create "samples" group
        samples_group = f.create_group("samples")

        for i in tqdm(range(total)):
            thickness = random.uniform(20, 1000)
            roughness = max(random.uniform(0, 100), thickness * 0.4)
            sld = random.uniform(1.0, 14.0)

            # structure = make_one_layer_structue(thickness, roughness, sld)
            thickness2 = random.uniform(20, 1000)
            roughness2 = max(random.uniform(0, 100), thickness * 0.4)
            sld2 = random.uniform(1.0, 14.0)
            structure = make_structure_2l(
                thickness, roughness, sld, thickness2, roughness2, sld2
            )
            R = simulate_xrr(structure, q)

            sample_name = f"sample_{i:06d}"
            g = samples_group.create_group(sample_name)

            # Create dataset for R
            g.create_dataset("R", data=R.astype("f4"))

            # Save metadata as attributes
            g.attrs["thickness"] = thickness
            g.attrs["roughness"] = roughness
            g.attrs["sld"] = sld

            g.attrs["thickness2"] = thickness2
            g.attrs["roughness2"] = roughness2
            g.attrs["sld2"] = sld2


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


def main_hdf5_nlayer(save_file: Path, total: int, n_layers: int):
    from ..data_processing.simulate import simulate_xrr
    N = 100
    q = np.linspace(0.005, 0.3, N)

    with h5py.File(save_file, "w") as f:
        f.create_dataset("q", data=q.astype("f4"))
        samples_group = f.create_group("samples")

        for i in tqdm(range(total), desc="Generating samples"):
            thicknesses, roughnesses, slds = random_layer_parameters(n_layers)
            structure = make_n_layer_structure(thicknesses, roughnesses, slds)
            R = simulate_xrr(structure, q)

            g = samples_group.create_group(f"sample_{i:06d}")
            g.create_dataset("R", data=R.astype("f4"))

            # metadata 저장
            g.attrs["n_layers"] = n_layers
            for j in range(n_layers):
                g.attrs[f"thickness_{j}"] = thicknesses[j]
                g.attrs[f"roughness_{j}"] = roughnesses[j]
                g.attrs[f"sld_{j}"] = slds[j]



def save_model(model_state_dict: dict, save_path: Path):
    """Saves the model's state dictionary to the specified path."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_state_dict, save_path)


def main() -> None:
    save_dir = Path("X:/XRR_AI/hdf5_XRR")
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    main_hdf5(save_dir / "p100o6_2_raw.h5", total=500_000)


if __name__ == "__main__":
    main()
