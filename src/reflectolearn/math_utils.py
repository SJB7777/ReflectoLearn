import numpy as np
import torch
from tqdm import tqdm

from .functional import batch_indices


def normalize(arr: np.ndarray) -> np.ndarray:
    """Nornalize a NumPy array to mean 0, std 1."""
    return (arr - arr.mean()) / arr.std()


def q_fourier_transform(
    qs: np.ndarray, reflections: np.ndarray, zs: np.ndarray
) -> np.ndarray:
    Z = zs[:, np.newaxis]
    Q = qs[np.newaxis, :]

    phase = np.exp(-1j * Q * Z)
    integrand = np.pow(qs, 4) * reflections
    integrand = integrand[np.newaxis, :]

    return np.trapezoid(integrand * phase, qs, axis=1)


def q_fourier_transform_multisample_gpu(
    R_all, q, z, batch_size=1024, show_progress=True
):
    N_samples, N_q = R_all.shape
    N_z = z.shape[0]

    # Compute q^4 and phase
    q4 = q**4
    phase = torch.exp(-1j * torch.outer(z, q))

    # Initialize output tensor
    FT_all = torch.empty((N_samples, N_z), dtype=torch.complex128)

    # Process in batches
    batches = batch_indices(N_samples, batch_size)
    pbar = tqdm(
        batches,
        total=N_samples // batch_size,
        desc="Fourier Transform",
        disable=not show_progress,
    )

    for start, end in pbar:
        R_batch = R_all[start:end, :]

        # Compute integrand and phase multiplication
        integrand = R_batch * q4
        integrand_phase = integrand[:, None, :] * phase

        # Perform trapezoid integration using torch.trapz
        FT_batch = torch.trapz(integrand_phase, q, dim=-1)

        # Store results
        FT_all[start:end, :] = FT_batch

    return FT_all


def q_fourier_transform_multisample(R_all, q, z, batch_size=1024, show_progress=True):
    N_samples, N_q = R_all.shape
    N_z = z.shape[0]

    q4 = q**4
    phase = np.exp(-1j * np.outer(z, q))

    FT_all = np.empty((N_samples, N_z), dtype=np.complex64)

    batches = batch_indices(N_samples, batch_size)
    pbar = tqdm(
        batches,
        total=N_samples // batch_size,
        desc="Fourier Transform",
        disable=not show_progress,
    )
    for start, end in pbar:
        R_batch = R_all[start:end, :]

        integrand = R_batch * q4
        integrand_phase = integrand[:, np.newaxis, :] * phase
        FT_batch = np.trapezoid(integrand_phase, q, axis=-1)

        # 결과를 원래 배열의 해당 위치에 저장
        FT_all[start:end, :] = FT_batch

    return FT_all


def apply_poisson_noise(arr: np.ndarray, s: float) -> np.ndarray:
    """Apply Poisson noise to an array."""
    expected_counts = s * arr
    noisy_counts = np.random.poisson(expected_counts)
    return noisy_counts / s


def get_background_noise(count: int, b_min: float, b_max: float) -> np.ndarray:
    """Generate background noise."""
    b = pow(10, np.random.uniform(b_min, b_max))
    return np.random.normal(b, 0.1 * b, count)
