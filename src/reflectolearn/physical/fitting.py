import matplotlib.pyplot as plt
import numpy as np

from reflectolearn.processing.fitting import (
    estimate_q,
    func_gauss,
    func_gauss3_with_noise,
    func_noise,
    s_vector_transform_q,
    tth2qz_by_energy,
    xrr_fft,
)
from reflectolearn.processing.preprocess import remove_q4_decay
from reflectolearn.processing.simulate import add_xrr_noise, make_n_layer_structure, make_parameters, structure_to_R


from itertools import combinations_with_replacement

# -----------------------
# 1. 데이터 생성
# -----------------------
beam_energy = 8.04751  # keV
tth = np.linspace(0.2, 6, 300)  # rad
q = tth2qz_by_energy(tth, beam_energy)

# 2층 박막 시뮬레이션 (기판 + 2개 필름)
thicknesses, roughnesses, slds = make_parameters(2)
structure = make_n_layer_structure(thicknesses, roughnesses, slds)
print(f"{thicknesses = }")
print(f"{roughnesses = }")
print(f"{slds = }")

real_thicks = thicknesses
real_thick_combs = [sum(set(comb)) for comb in combinations_with_replacement(real_thicks, r=len(real_thicks))]
print(f"{real_thick_combs = }")
R = structure_to_R(structure, q)
R_noise = R.copy()
R_noise = add_noise(R)

# Plot
fig, axs = plt.subplots(1, 1, figsize=(8, 5))

axs.semilogy(tth, R, "--", label="Simulated Reflectivity")
axs.semilogy(tth, R_noise, "-", label="Simulated Reflectivity with Noise")

axs.set_title("XRR Simulation")
axs.legend()
plt.show()