from pathlib import Path

import numpy as np

from reflectolearn.config import ConfigManager
from reflectolearn.io import make_xrr_hdf5, next_unique_file


def main():
    import psutil

    from reflectolearn.logger import setup_logger


    logger = setup_logger()
    logger.info("Starting XRR simulation")
    ConfigManager.initialize("config.yaml")
    config = ConfigManager.load_config()
    logger.info(f"Configuration: \n{config.model_dump_json(indent=2)}")

    N: int = config.simulate.n_q
    n_sample: int = config.simulate.n_sample
    n_layer: int = 2
    q = np.linspace(0.03, 0.3, N)

    logger.info(f"Simulation parameters: N={N}, n_sample={n_sample:_}, n_layer={n_layer}")
    logger.info(f"Q range: {q[0]:.3f} to {q[-1]:.3f}")

    n_workers: int = psutil.cpu_count(logical=True) - 1
    batch_size: int = 500
    chunk_size: int = max(int(n_sample / batch_size / n_workers / 8), 30)
    logger.info(f"Number of worker: {n_workers}")
    logger.info(f"Batch size: {n_workers}")
    logger.info(f"Chunk size: {chunk_size}")

    data_file: Path = config.path.data_file
    data_file: Path = next_unique_file(data_file)
    logger.info(f"Output file: {data_file}")

    logger.info("Generating XRR data...")
    make_xrr_hdf5(
        save_file=data_file, n_layer=n_layer, q=q, n_sample=n_sample, has_noise=False,
        n_workers=n_workers, batch_size=batch_size, chunksize=chunk_size
    )

    logger.info("XRR simulation data saved successfully")


if __name__ == "__main__":
    main()
