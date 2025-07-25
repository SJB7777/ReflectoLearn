from pathlib import Path

from loguru import logger
import numpy as np
import h5py
from tqdm import tqdm
from refnx.reflect.structure import Structure

from reflectolearn.data_processing.simulate import (
    make_n_layer_structure,
    simulate_xrr_with_noise,
    make_parameters,
)
from reflectolearn.config import load_config


def make_xrr_hdf5(save_file: Path, n_layer: int, q: np.ndarray, n_sample: int):

    # Create HDF5 file
    with h5py.File(save_file, "w") as f:
        # Save q globally at root
        f.create_dataset("q", data=q.astype("f4"))
        # Create "samples" group
        samples_group = f.create_group("samples")

        for i in tqdm(range(n_sample)):
            thicknesses, roughnesses, slds = make_parameters(n_layer)
            structure: Structure = make_n_layer_structure(
                thicknesses=thicknesses, roughnesses=roughnesses, slds=slds
            )
            R = simulate_xrr_with_noise(structure, q)

            sample_name: str = f"sample_{i:06d}"
            g = samples_group.create_group(sample_name)

            # Create dataset for R
            g.create_dataset("R", data=R.astype("f4"))
            # Save metadata as attributes
            for i, (thickness, roughness, sld) in enumerate(
                zip(thicknesses, roughnesses, slds)
            ):
                g.attrs[f"thickness_{i}"] = thickness
                g.attrs[f"roughness_{i}"] = roughness
                g.attrs[f"sld_{i}"] = sld


def main():

    logger.info("Starting XRR simulation")

    config = load_config()
    logger.info("Configuration loaded successfully")

    N: int = 100
    n_sample: int = 1_000_000
    n_layer: int = 3
    q = np.linspace(0.03, 0.3, N)

    logger.info(
        f"Simulation parameters: N={N}, n_sample={n_sample:_}, n_layer={n_layer}"
    )
    logger.info(f"Q range: {q[0]:.3f} to {q[-1]:.3f}")

    data_file: Path = config.data.data_file
    logger.info(f"Output file: {data_file}")

    logger.info("Generating XRR data...")
    make_xrr_hdf5(save_file=data_file, n_layer=n_layer, q=q, n_sample=n_sample)

    logger.info("XRR simulation data saved successfully")


if __name__ == "__main__":
    main()
