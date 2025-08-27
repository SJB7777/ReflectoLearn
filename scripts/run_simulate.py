from pathlib import Path

import h5py
import numpy as np
from loguru import logger
from refnx.reflect.structure import Structure
from tqdm import tqdm

from reflectolearn.config import ConfigManager
from reflectolearn.processing.simulate import add_xrr_noise, make_n_layer_structure, make_parameters, structure_to_R


def make_xrr_hdf5(save_file: Path, n_layer: int, q: np.ndarray, n_sample: int):
    # Create HDF5 file
    with h5py.File(save_file, "w") as f:
        # Save q globally at root
        f.create_dataset("q", data=q.astype("f4"))
        # Create "samples" group
        samples_group = f.create_group("samples")

        for i in tqdm(range(n_sample)):
            thicknesses, roughnesses, slds = make_parameters(n_layer)
            structure: Structure = make_n_layer_structure(thicknesses=thicknesses, roughnesses=roughnesses, slds=slds)
            R = structure_to_R(structure, q)
            R = add_xrr_noise(R)
            sample_name: str = f"sample_{i:06d}"
            g = samples_group.create_group(sample_name)

            # Create dataset for R
            g.create_dataset("R", data=R.astype("f4"))
            # Save metadata as attributes
            for i, (thickness, roughness, sld) in enumerate(zip(thicknesses, roughnesses, slds, strict=False)):
                g.attrs[f"thickness_{i}"] = thickness
                g.attrs[f"roughness_{i}"] = roughness
                g.attrs[f"sld_{i}"] = sld


def main():
    logger.info("Starting XRR simulation")
    ConfigManager.initialize("config.yaml")
    config = ConfigManager.load_config()
    logger.info("Configuration loaded successfully")

    N: int = 100
    n_sample: int = 1_000_000
    n_layer: int = 2
    q = np.linspace(0.03, 0.3, N)

    logger.info(f"Simulation parameters: N={N}, n_sample={n_sample:_}, n_layer={n_layer}")
    logger.info(f"Q range: {q[0]:.3f} to {q[-1]:.3f}")

    data_file: Path = config.data.data_file
    logger.info(f"Output file: {data_file}")

    logger.info("Generating XRR data...")
    make_xrr_hdf5(save_file=data_file, n_layer=n_layer, q=q, n_sample=n_sample)

    logger.info("XRR simulation data saved successfully")


if __name__ == "__main__":
    main()
