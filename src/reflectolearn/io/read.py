from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ..math_utils import q_fourier_transform

def get_data(path: Path):
    """Read data from the HDF5 file."""
    with h5py.File(path, "r") as f:
        q = f["q"][:]
        Rs = []
        params = []
        for sample_name in tqdm(f["samples"], desc="Loading samples"):
            sample_group = f["samples"][sample_name]
            R = sample_group["R"][:]
            thickness = sample_group.attrs["thickness"]
            roughness = sample_group.attrs["roughness"]
            sld = sample_group.attrs["sld"]

            Rs.append(R)
            params.append([thickness, roughness, sld])

    return {"q": np.array(q), "Rs": np.array(Rs), "params": np.array(params)}


if __name__ == "__main__":

    file = Path("X:\\XRR_AI\\hdf5_XRR\\p100o3_paper.h5")

    data = get_data(file)

    num: int = 4
    q = data["q"]
    R = data["Rs"][num]
    thickness, roughness, sld = data["params"][num]

    print(f"Thickness: {thickness}")
    print(f"Roughness: {roughness}")
    print(f"SLD: {sld}")

    plt.plot(q, R)
    plt.semilogy(base=10)
    plt.show()

    z = np.linspace(0, 500, 2048)
    FT = q_fourier_transform(q, R, z)
    rho_z = np.abs(FT) ** 2

    # 정규화 후 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(z, rho_z)
    plt.semilogy(base=10)
    plt.title("Depth Profile from XRR (|FT|^2)")
    plt.xlabel("z (Å)")
    plt.ylabel("Normalized Density Profile")
    plt.grid(True)
    plt.show()
