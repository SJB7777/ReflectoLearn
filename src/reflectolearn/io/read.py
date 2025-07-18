from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def get_data(path: Path):
    """Read data from the HDF5 file."""
    with h5py.File(path, "r") as f:
        q = f["q"][:]
        Rs = []
        params = []
        for sample_name in tqdm(f["samples"], desc="Loading samples"):
            sample_group = f["samples"][sample_name]
            R = sample_group["R"][:]

            # Extract parameters
            base_attrs = [
                sample_group.attrs["thickness"],
                sample_group.attrs["roughness"],
                sample_group.attrs["sld"],
            ]

            if len(sample_group.attrs) == 6:
                base_attrs.extend(
                    [
                        sample_group.attrs["thickness2"],
                        sample_group.attrs["roughness2"],
                        sample_group.attrs["sld2"],
                    ]
                )

            params.append(base_attrs)
            Rs.append(R)

    return {"q": np.array(q), "Rs": np.array(Rs), "params": np.array(params)}


if __name__ == "__main__":

    file = Path("X:\\XRR_AI\\hdf5_XRR\\data") / "p100o6_raw.h5"

    data = get_data(file)
