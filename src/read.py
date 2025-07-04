from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_data(path: Path):
    """Read data from the HDF5 file."""
    with h5py.File(path, "r") as f:
        q = f["q"][:]
        Rs = []
        params = []
        for sample_name in tqdm(f["samples"]):
            sample_group = f["samples"][sample_name]
            R = sample_group["R"][:]
            thickness = sample_group.attrs["thickness"]
            roughness = sample_group.attrs["roughness"]
            sld = sample_group.attrs["sld"]

            Rs.append(R)
            params.append([thickness, roughness, sld])

    return {"q": np.array(q), "Rs": np.array(Rs), "params": np.array(params)}


if __name__ == "__main__":

    file = Path("X:\\XRR_AI\\hdf5_XRR\\100p.h5")

    with h5py.File(file, "r") as f:
        # Read q from the root
        q = f["q"][:]

        # Read all samples
        samples = []
        for sample_name in tqdm(f["samples"]):
            sample_group = f["samples"][sample_name]
            R = sample_group["R"][:]
            thickness = sample_group.attrs["thickness"]
            roughness = sample_group.attrs["roughness"]
            sld = sample_group.attrs["sld"]

            samples.append(
                {
                    "R": R,
                    "thickness": thickness,
                    "roughness": roughness,
                    "sld": sld,
                }
            )
    print(f"Total samples: {len(samples)}")

    num: int = 4
    sample = samples[num]
    R = sample["R"]
    thickness = sample["thickness"]
    roughness = sample["roughness"]
    sld = sample["sld"]

    print(f"Thickness: {thickness}")
    print(f"Roughness: {roughness}")
    print(f"SLD: {sld}")

    plt.plot(q, R)
    plt.semilogy(base=10)
    plt.show()
