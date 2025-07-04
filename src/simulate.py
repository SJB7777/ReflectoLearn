from typing import Final
import random

import numpy as np
import matplotlib.pyplot as plt
from refnx.reflect import ReflectModel, SLD
from refnx.reflect.structure import Structure

from src.noise import apply_poisson_noise, get_background_noise


def make_one_layer_structue(thickness: float, roughness: float, sld: float):
    air = SLD(0.0, name="Air")
    oxide = SLD(2.5, name="Oxide")
    film = SLD(sld, name="Thin Film")
    substrate = SLD(2.0, name="Substrate")

    structure = air(0, 0) | oxide(20, 2) | film(thickness, roughness) | substrate(0, 3)
    return structure


def simulate_xrr(structure: Structure, q: np.ndarray):
    model = ReflectModel(structure)
    R = model(q)
    return R


def simulate_xrr_with_noise(structure: Structure, q: np.ndarray):
    N = len(q)
    R = simulate_xrr(structure, q)
    R_poisson = apply_poisson_noise(R, s=10 ** random.uniform(6, 8))
    uniform_noise = np.random.uniform(0.7, 1.3, N)
    background_noise = get_background_noise(N, -7, -4)
    curve_scaling = np.random.uniform(0.9, 1.1)
    R_noise = R_poisson * uniform_noise * curve_scaling + background_noise
    return R_noise


if __name__ == "__main__":

    # thickness = random.uniform(20, 1000)
    # roughness = max(random.uniform(0, 100), thickness * 0.4)
    # sld = random.uniform(1.0, 14.0)
    thickness: float = 193.7
    roughness: float = 77.48
    sld: float = 3.41
    print(f"Thickness: {thickness}")
    print(f"Roughness: {roughness}")
    print(f"SLD: {sld}")

    structure = make_one_layer_structue(thickness, roughness, sld)

    N: Final[int] = 100
    q = np.linspace(0.005, 0.3, N)
    R = simulate_xrr(structure, q)
    R_poisson = apply_poisson_noise(R, s=10 ** random.uniform(6, 8))

    uniform_noise = np.random.uniform(0.7, 1.3, N)
    b = pow(10, np.random.uniform(-7, -4))
    get_background_noise = np.random.normal(b, 0.1 * b, N)
    curve_scaling = np.random.uniform(0.9, 1.1)
    R_all_noise = R_poisson * uniform_noise * curve_scaling + get_background_noise

    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    ax.plot(q, R, label="Ground Truth", color="black")
    ax.set_title("Simulated XRR Reflectivity", fontsize=16)
    ax.set_xlabel("q (1/Å)")
    ax.set_ylabel("Reflectivity")
    ax.semilogy(base=10)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle("Simulated XRR with Noise", fontsize=16)
    axs[0].plot(q, R_poisson, label="Poisson Noise")
    axs[0].plot(q, get_background_noise, label="Background Noise")
    axs[0].plot(q, R, label="Ground Truth", color="black")

    axs[1].plot(q, R * curve_scaling, label="Curve Scaling", color="green")
    axs[1].plot(q, R * uniform_noise, label="Uniform Noise", color="purple")
    axs[1].plot(q, R, label="Ground Truth", color="black")

    axs[2].plot(q, R_all_noise, label="All Noise Combined", color="red")
    axs[2].plot(q, R, label="Ground Truth", color="black")

    for ax in axs.flat:
        ax.set_xlabel("q (1/Å)")
        ax.set_ylabel("Reflectivity")
        ax.semilogy(base=10)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
