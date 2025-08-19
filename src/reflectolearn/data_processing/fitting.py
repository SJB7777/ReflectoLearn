import matplotlib.pyplot as plt
import numpy as np
import scipy as scp
import scipy.optimize as optimize
import scipy.signal.windows as fft_windows

hc = 12.398  # Planck constant * speed of light [keV·Å]


# ---------------- Core transformation functions ---------------- #
def tth2qz_by_energy(tth_deg: float | np.ndarray, energy: float) -> np.ndarray:
    """Convert 2θ (deg) to qz (1/nm) using beam energy in keV."""
    th_rad = np.deg2rad(tth_deg / 2)
    return 4 * np.pi * np.sin(th_rad) * energy / hc


def XRR_q(data_q: np.ndarray, q_crit: float) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess XRR reflectivity data for FFT analysis (q, R)."""
    q_nm, R = data_q[:, 0], data_q[:, 1]
    s_cor = np.sqrt(q_crit**2 - q_nm**2)
    mask = np.isfinite(s_cor)
    s_cor, intensity = s_cor[mask], (s_cor[mask] ** 4) * R[mask]

    s_cor, unique_idx = np.unique(s_cor, return_index=True)
    intensity = intensity[unique_idx]

    if len(s_cor) < 4:
        raise ValueError(f"Not enough points ({len(s_cor)}) for cubic interpolation.")

    interp_kind = "cubic" if len(s_cor) >= 4 else "linear"
    x = np.linspace(s_cor.min(), s_cor.max(), 1000)
    f = scp.interpolate.interp1d(s_cor, intensity, kind=interp_kind)
    return x, f(x)


def FFT(x: np.ndarray, y: np.ndarray, d: float | None = None, window: int = 2, n: int | None = None):
    """Compute (real) FFT with multiple window options."""
    if d is None:
        d = x[1] - x[0]
    N = len(y)

    # Select window function
    if window == 0:
        win = np.ones(N)
    elif window == 1:
        win = fft_windows.hann(N)
    elif window == 2:
        win = fft_windows.hamming(N)
    else:
        win = fft_windows.flattop(N)

    if n is None:
        n = N

    # FFT with normalization
    yf = 2 / N * np.abs(scp.fftpack.fft(win * y / np.mean(win), n=n))
    xf = scp.fftpack.fftfreq(n, d=d)
    return xf[: n // 2], yf[: n // 2]


# ---------------- Fit model functions ---------------- #
def funcNoise(x, amp, ex):
    return amp / np.power(x, ex)


def funcGauss(p, a, pmax, w):
    return a * np.exp(-np.log(2) * ((pmax - p) / (w / 2)) ** 2)


def funcGauss2(p, a1, a2, pmax1, pmax2, w1, w2):
    return funcGauss(p, a1, pmax1, w1) + funcGauss(p, a2, pmax2, w2)


def funcGauss3(p, a1, w1, a2, pmax2, w2, a3, pmax3, w3, amp, ex, z0):
    pmax1 = pmax3 - pmax2
    return funcGauss2(p, a1, a2, pmax1, pmax2, w1, w2) \
           + funcGauss(p, a3, pmax3, w3) \
           + funcNoise(p, amp, ex) + z0


# ---------------- Higher-level analysis ---------------- #
def estimate_qc(q: np.ndarray, R: np.ndarray, smooth_window=5) -> float:
    """Estimate critical momentum transfer qc from reflectivity curve."""
    logR = np.log(R)
    logR_smooth = np.convolve(logR, np.ones(smooth_window) / smooth_window, mode="same")
    dlogR = np.gradient(logR_smooth, q)
    return q[np.argmax(np.abs(dlogR))]


def analyze_xrr_fft(data: np.ndarray, crit_q: float):
    """Perform FFT-based analysis on reflectivity data."""
    q_uniform, intensity_uniform = XRR_q(data, crit_q)
    fft_x, fft_y = FFT(q_uniform, intensity_uniform, window=2, n=10000)
    fft_y_norm = fft_y / fft_y[0]

    mask_bg = np.logical_or(
        np.logical_and(fft_x > 1, fft_x < 5),
        np.logical_and(fft_x > 26, fft_x < 80)
    )
    mask_full = np.logical_and(fft_x > 1.1, fft_x < 80)

    popt_noise, _ = optimize.curve_fit(funcNoise, fft_x[mask_bg], fft_y_norm[mask_bg])
    p0 = [0.2, 0.3, 0.2, 7, 0.3, 0.2, 13, 0.3, 1, 2, 2e-3]
    bounds = (0, np.inf)
    popt_gauss3, _ = optimize.curve_fit(funcGauss3, fft_x[mask_full], fft_y_norm[mask_full], p0=p0, bounds=bounds)

    return fft_x, fft_y_norm, popt_noise, popt_gauss3


def plot_xrr_fft(fft_x, fft_y_norm, popt_gauss3):
    """Plot FFT spectrum with Gaussian component fits."""
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(8, 6))
    ax[0].plot(fft_x, fft_y_norm, "o-", ms=1, lw=0.7, color="dimgrey", label="FFT data")
    ax[1].plot(fft_x, fft_y_norm, "o-", ms=1, lw=0.7, color="dimgrey")

    ax[0].plot(fft_x, funcGauss3(fft_x, *popt_gauss3), "-", color="goldenrod", label="Multi-Gaussian fit")

    ax[1].plot(fft_x, funcGauss(fft_x, *popt_gauss3[2:5]), "--", color="darkblue", label="Gaussian 1")
    ax[1].plot(fft_x, funcGauss(fft_x, popt_gauss3[0], popt_gauss3[6]-popt_gauss3[3], popt_gauss3[1]),
               "--", color="purple", label="Gaussian 2")
    ax[1].plot(fft_x, funcGauss(fft_x, *popt_gauss3[5:8]), "--", color="teal", label="Gaussian 3")
    ax[1].plot(fft_x, funcNoise(fft_x, *popt_gauss3[8:10]), "--", color="chocolate", label="1/f noise")

    ax[0].set_ylabel("Normalized FFT amplitude")
    ax[1].set_xlabel("Thickness (nm)")
    ax[0].set_xlim(0, 80)
    ax[0].set_ylim(0, 1.1)
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    plt.show()
