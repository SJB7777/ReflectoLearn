import numpy as np


def apply_poisson_noise(arr: np.ndarray, s: float) -> np.ndarray:
    """Apply Poisson noise to an array."""
    expected_counts = s * arr
    noisy_counts = np.random.poisson(expected_counts)
    return noisy_counts / s


def get_background_noise(count: int, b_min: float, b_max: float) -> np.ndarray:
    """Generate background noise."""
    b = pow(10, np.random.uniform(b_min, b_max))
    return np.random.normal(b, 0.1 * b, count)

