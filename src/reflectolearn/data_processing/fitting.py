import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy import fftpack


def XRR_from_q(q, intensity, crit_q=None, num_points=1000):
    # 1. Apply critical_q mask if given
    if crit_q is not None:
        mask = q >= crit_q
        q = q[mask]
        intensity = intensity[mask]

    # 2. q⁴ background correction
    intensity_corrected = (q ** 4) * intensity

    # 3. Generate evenly spaced q-grid
    q_uniform = np.linspace(q.min(), q.max(), num_points)

    # 4. Interpolate to uniform grid
    interp_func = interpolate.interp1d(q, intensity_corrected, kind="cubic")
    intensity_uniform = interp_func(q_uniform)

    return q_uniform, intensity_uniform


def estimate_critical_q(q, R, window=5):
    """
    반사율 곡선에서 임계 q 값을 추정.
    이동 평균 필터로 스무딩 후 기울기 변화를 이용.
    """
    logR = np.log10(R + 1e-12)
    smooth_logR = np.convolve(logR, np.ones(window) / window, mode='same')
    grad = np.gradient(smooth_logR, q)
    critical_idx = np.argmin(grad)  # 가장 급격한 감소 지점
    return q[critical_idx]

def subtract_baseline(y):
    """
    데이터의 평균값을 빼서 baseline 제거.
    """
    return y - np.mean(y)

def fft_thickness_analysis(q, R, crit_q=None):
    """
    q-R 데이터로부터 주기성 분석.
    critical_q 이후 데이터를 사용.
    """
    mask = np.full_like(q, True, dtype=bool) if crit_q is None else q > crit_q
    q_sel, R_sel = q[mask], R[mask]

    y = subtract_baseline(np.log10(R_sel + 1e-12))
    y_fft = fftpack.fft(y)
    freq = fftpack.fftfreq(len(q_sel), d=(q_sel[1] - q_sel[0]))

    thickness = 2 * np.pi / freq  # nm 단위로 변환 시 scaling 필요
    return q_sel, y, thickness, np.abs(y_fft)

def plot_fft_results(q_sel, y, thickness, fft_amp):
    """
    FFT 분석 결과 플롯.
    """
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(q_sel, y, label="Log(R) - Baseline")
    plt.xlabel("q (Å⁻¹)")
    plt.ylabel("Log(R)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(thickness, fft_amp, label="FFT Amplitude")
    plt.xlabel("Thickness (Å)")
    plt.ylabel("Amplitude")
    plt.xlim(0, np.max(thickness) / 2)  # Nyquist 제한
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    from reflectolearn.data_processing.simulate import make_multifilm


    q = np.linspace(0.02, 0.3, 500)
    R = make_multifilm(3, q, add_noise=True)['R']
    q_sel, y, thickness, fft_amp = fft_thickness_analysis(q, R, crit_q=None)
    plot_fft_results(q_sel, y, thickness, fft_amp)
