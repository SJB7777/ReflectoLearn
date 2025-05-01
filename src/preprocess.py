import numpy as np
from scipy.interpolate import interp1d


def fill_nan_with_interp(x: np.ndarray) -> np.ndarray:
    """
    Fills NaN values in each column of a 2D array using linear interpolation.

    Args:
        x: 2D NumPy array with potential NaN values.

    Returns:
        2D NumPy array with NaN values replaced by interpolated values.
    """
    x = x.copy()  # Avoid modifying input array
    for col in range(x.shape[1]):
        y = x[:, col]
        nans = np.isnan(y)
        if nans.any():
            not_nan = ~nans
            indices = np.arange(len(y))
            interp_fn = interp1d(
                indices[not_nan],
                y[not_nan],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            x[nans, col] = interp_fn(indices[nans])
    return x
