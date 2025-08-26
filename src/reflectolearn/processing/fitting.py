import matplotlib.pyplot as plt
import numpy as np
import scipy as scp
import scipy.signal.windows as fft_windows


def tth2qz_by_energy(tth_deg: np.ndarray, energy_keV: float) -> np.ndarray:
    """
    Convert 2theta angles (deg) to qz (1/Å) given X-ray energy.

    Parameters
    ----------
    tth_deg : array_like
        2theta angles in degrees.
    energy_keV : float
        X-ray energy in keV.

    Returns
    -------
    qz : np.ndarray
        Momentum transfer qz values (1/Å).
    """
    # keV -> Å
    wavelength = 12.39842 / energy_keV  # [Å]
    # 2θ (deg) -> θ (rad)
    theta_rad = np.radians(tth_deg / 2.0)
    # qz [1/Å]
    qz = (4 * np.pi / wavelength) * np.sin(theta_rad)
    return qz

def estimate_q(q: np.ndarray, R: np.ndarray, threshold: float = 0.99):
    """
    Estimate critical scattering vector qc from reflectivity curve.
    Parameters
    ----------
    q : np.ndarray
        scattering vector values.
    R : np.ndarray
        Reflectivity values.
    threshold : float, optional
        Reflectivity cutoff (default = 0.99).
    Returns
    -------
    qc : float
        Estimated critical q value.
    """
    # normalize
    R = R / np.max(R)

    # 방법 1: R이 threshold 아래로 내려가기 시작하는 첫 q
    mask = np.where(R < threshold)[0]
    if len(mask) > 0:
        thc_est = q[mask[0]]
    else:
        thc_est = q[np.argmin(R)]  # 전반사 없으면 최소값 반환

    # 방법 2: 기울기 최대점도 같이 구해볼 수 있음
    dR = np.gradient(R, q)
    slope_q = q[np.argmin(dR)]

    # 둘 다 비슷하다면 평균
    return (thc_est + slope_q) / 2


# ---------------------------
# XRR 전처리 + FFT
# ---------------------------
def preprocess_xrr(data, crit_ang, wave_length: float = 0.152):
    """
    Preprocess XRR dataset for FFT.
    :param data: numpy array with [2θ angle, intensity]
    :param crit_ang: critical angle (deg)
    :param wave_length: wavelength (nm)
    :return: (x, y) -> rescaled evenly spaced dataset
    """
    s_cor = 2 * np.sqrt(
        (np.cos(np.pi * crit_ang / 2 / 180)) ** 2
        - (np.cos(np.pi * data[:, 0] / 2 / 180)) ** 2
    ) / wave_length

    mask = np.logical_not(np.isnan(s_cor))
    s_cor = s_cor[mask]
    intensity = s_cor**4 * data[mask, 1]

    x = np.linspace(s_cor.min(), s_cor.max(), 1000)
    f = scp.interpolate.interp1d(s_cor, intensity, kind="cubic")
    return x, f(x)

def preprocess_xrr_q(data_q: np.ndarray, q_crit: float, step_num: int = 1000):
    """
    Preprocess XRR dataset when input is already qz.
    :param data_q: numpy array with [qz (1/nm), intensity]
    :param q_crit: critical q (1/nm)
    :return: (x, y) -> rescaled evenly spaced dataset
    """
    qz = data_q[:, 0]
    intensity = data_q[:, 1]

    q_cor, mask = np.unique(np.sqrt(np.clip(qz**2 - q_crit**2, 0, None)), return_index=True)
    intensity = intensity[mask]
    # Fresnel normalization (보정)
    intensity_corr = q_cor**4 * intensity

    # 등간격 보간
    x = np.linspace(q_cor.min(), q_cor.max(), step_num)
    f = scp.interpolate.interp1d(q_cor, intensity_corr, kind="cubic")
    return x, f(x)


def xrr_fft(x, y, d=None, window=2, n=None):
    """
    FFT 변환 (XRR 분석 전용)
    :param x: delta inverse q
    :param y: intensity
    """
    if d is None:
        d = x[1] - x[0]

    N = len(y)
    if window == 0:
        w = np.ones(N)
    elif window == 1:
        w = fft_windows.hann(N)
    elif window == 2:
        w = fft_windows.hamming(N)
    else:
        w = fft_windows.flattop(N)

    if n is None:
        n = N

    yf = 2 / N * np.abs(scp.fftpack.fft(w * y / np.mean(w), n=n))
    xf = scp.fftpack.fftfreq(n, d=d)
    return xf[: n // 2], yf[: n // 2]

# ---------------------------
# Fitting 모델 정의
# ---------------------------
def func_noise(x, amp, ex):
    """1/f noise background"""
    return amp / np.power(x, ex)

def func_gauss(x, a, pmax, w):
    """Single Gaussian"""
    return a * np.exp(-np.log(2) * ((pmax - x) / (w / 2)) ** 2)

def func_gauss2(x, a1, a2, pmax1, pmax2, w1, w2):
    """Sum of two Gaussians"""
    return func_gauss(x, a1, pmax1, w1) + func_gauss(x, a2, pmax2, w2)

def func_gauss3_with_noise(x, a1, w1, a2, pmax2, w2, a3, pmax3, w3, amp, ex, z0):
    """Multi-Gaussian + noise"""
    pmax1 = pmax3 - pmax2
    return (
        func_gauss2(x, a1, a2, pmax1, pmax2, w1, w2)
        + func_gauss(x, a3, pmax3, w3)
        + func_noise(x, amp, ex)
        + z0
    )

def func_noise2(x, a, w):
    """Noise Gaussian"""
    return a * np.exp(-np.log(2) * (x / (w / 2)) ** 2)

def func_gauss3_with_noise_ver2(p, a1, w1, a2, pmax2, w2, a3, pmax3, w3, a4, w4, z0):
    """Multi-Gaussian + noise"""
    pmax1 = pmax3 - pmax2
    return (
        func_gauss2(p, a1, a2, pmax1, pmax2, w1, w2)
        + func_gauss(p, a3, pmax3, w3)
        + func_noise2(p, a4, w4)
        + z0
    )


# ---------------------------
# 시각화
# ---------------------------
def plot_fft_fit(FFTpadx, FFTpady_n, xmask, ymask, poptGauss3):
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)

    # Raw data
    ax[0].plot(FFTpadx, FFTpady_n, "o-", ms=1, lw=0.7, c="dimgrey", label="data")
    ax[1].plot(FFTpadx, FFTpady_n, "o-", ms=1, lw=0.7, c="dimgrey", label="data")

    # Combined fit
    ax[0].plot(xmask, func_gauss3_with_noise(xmask, *poptGauss3), "-", c="goldenrod", label="multi-Gaussian fit")

    # Components
    ax[1].plot(xmask, func_gauss(xmask, *poptGauss3[2:5]), "--", c="darkblue", label="Gaussian 1")
    ax[1].plot(xmask, func_gauss(xmask, poptGauss3[0], poptGauss3[6]-poptGauss3[3], poptGauss3[1]), "--", c="purple", label="Gaussian 2")
    ax[1].plot(xmask, func_gauss(xmask, *poptGauss3[5:8]), "--", c="teal", label="Gaussian 3")
    ax[1].plot(xmask, func_noise(xmask, *poptGauss3[8:10]), "--", c="chocolate", label="1/f^α")

    ax[0].set_ylabel("normalized FFT amplitude")
    ax[1].set_xlabel("thickness (nm)")
    ax[0].set_xlim(0, 35)
    ax[0].set_ylim(0, 0.45)

    ax[0].legend()
    ax[1].legend()
    plt.show()
