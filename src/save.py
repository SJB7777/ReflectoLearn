from pathlib import Path
import random

import h5py
import numpy as np
from tqdm import tqdm

from simulate import simulate_xrr_with_noise, make_one_layer_structue, simulate_xrr


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

            structure = make_one_layer_structue(thickness, roughness, sld)
            R = simulate_xrr(structure, q)

            sample_name = f"sample_{i:06d}"
            g = samples_group.create_group(sample_name)

            # Create dataset for R
            g.create_dataset("R", data=R.astype("f4"))

            # Save metadata as attributes
            g.attrs["thickness"] = thickness
            g.attrs["roughness"] = roughness
            g.attrs["sld"] = sld


def main() -> None:
    save_dir = Path("X:/XRR_AI/hdf5_XRR")
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    main_hdf5(save_dir / "100p_paper.h5", total=10)


if __name__ == "__main__":
    main()
