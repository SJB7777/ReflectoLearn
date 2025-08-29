from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from loguru import logger
from refnx.reflect.structure import Structure
from tqdm import tqdm

from reflectolearn.config import ConfigManager
from reflectolearn.io import next_unique_file
from reflectolearn.processing.simulate import add_xrr_noise, make_n_layer_structure, make_parameters, structure_to_R


def simulate_one(idx: int, n_layer: int, q: np.ndarray, has_noise: bool):
    """한 샘플 생성"""
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
):
    """
    대규모 XRR HDF5 데이터셋 생성

    Args:
        save_file: 저장할 HDF5 경로
        n_layer: 층 개수
        q: q 벡터 (길이 N)
        n_sample: 총 샘플 수
        has_noise: 노이즈 추가 여부
        n_workers: 병렬 워커 수 (기본값 = CPU 코어 수)
        batch_size: 저장 배치 크기 (기본=1000)
    """
    N = len(q)

    with h5py.File(save_file, "w") as f:
        # q 저장
        f.create_dataset("q", data=q.astype("f4"))

        # 대규모 dataset 생성 (압축, chunking 적용)
        dR = f.create_dataset(
            "R",
            shape=(n_sample, N),
            dtype="f4",
            compression="lzf",
            chunks=(batch_size, N),
        )
        dT = f.create_dataset(
            "thicknesses",
            shape=(n_sample, n_layer),
            dtype="f4",
            compression="lzf",
            chunks=(batch_size, n_layer),
        )
        dRough = f.create_dataset(
            "roughnesses",
            shape=(n_sample, n_layer),
            dtype="f4",
            compression="lzf",
            chunks=(batch_size, n_layer),
        )
        dSLD = f.create_dataset(
            "slds",
            shape=(n_sample, n_layer),
            dtype="f4",
            compression="lzf",
            chunks=(batch_size, n_layer),
        )

        # 병렬 실행
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for batch_start in tqdm(range(0, n_sample, batch_size)):
                batch_end = min(batch_start + batch_size, n_sample)
                batch_indices = range(batch_start, batch_end)

                results = list(
                    executor.map(
                        simulate_one,
                        batch_indices,
                        [n_layer] * len(batch_indices),
                        [q] * len(batch_indices),
                        [has_noise] * len(batch_indices),
                        chunksize=50,  # 워커당 작업 단위
                    )
                )

                # batch 결과 저장
                for idx, R, T, Rough, SLD in results:
                    dR[idx] = R
                    dT[idx] = T
                    dRough[idx] = Rough
                    dSLD[idx] = SLD

def main():
    import psutil

    logger.info("Starting XRR simulation")
    ConfigManager.initialize("config.yaml")
    config = ConfigManager.load_config()
    logger.info("Configuration loaded successfully")

    N: int = 300
    n_sample: int = 3_000_000
    n_layer: int = 2
    q = np.linspace(0.03, 0.3, N)

    n_worker: int = psutil.cpu_count(logical=True) // 2
    batch_size: int = 1000
    logger.info(f"Simulation parameters: N={N}, n_sample={n_sample:_}, n_layer={n_layer}")
    logger.info(f"Q range: {q[0]:.3f} to {q[-1]:.3f}")

    data_file: Path = config.path.data_file
    data_file = next_unique_file(data_file)
    logger.info(f"Output file: {data_file}")

    logger.info(f"Number of worker: {n_worker}")
    logger.info("Generating XRR data...")
    make_xrr_hdf5(
        save_file=data_file, n_layer=n_layer, q=q, n_sample=n_sample, has_noise=False,
        n_workers=n_worker, batch_size=batch_size
    )

    logger.info("XRR simulation data saved successfully")


if __name__ == "__main__":
    main()
