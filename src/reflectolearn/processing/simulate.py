import random

import numpy as np
from refnx.reflect import SLD, ReflectModel, Stack
from refnx.reflect.structure import Structure

from ..math_utils import apply_poisson_noise, get_background_noise


def make_parameters(n: int):
    thicknesses: list[float] = []
    roughnesses: list[float] = []
    slds: list[float] = []
    for _ in range(n):
        max_thick = 100
        thickness = random.uniform(20, max_thick)
        roughness = random.uniform(0, thickness * 0.06)
        sld = random.uniform(1.0, 3.0)

        thicknesses.append(thickness)
        roughnesses.append(roughness)
        slds.append(sld)

    return thicknesses, roughnesses, slds


def make_multifilm(n_layer: int, q: np.ndarray, add_noise=True):
    thicknesses, roughnesses, slds = make_parameters(n_layer)
    structure: Structure = make_n_layer_structure(thicknesses=thicknesses, roughnesses=roughnesses, slds=slds)
    R = simulate_xrr_with_noise(structure, q) if add_noise else structure2R(structure, q)

    return {
        "structure": structure,
        "thickness": thicknesses,
        "roughness": roughnesses,
        "sld": slds,
        "R": R,
    }


def make_one_layer_structure(thickness: float, roughness: float, sld: float):
    air = SLD(0.0, name="Air")
    oxide = SLD(2.5, name="Oxide")
    film = SLD(sld, name="Thin Film")
    substrate = SLD(2.0, name="Substrate")

    structure = air(0, 0) | oxide(20, 2) | film(thickness, roughness) | substrate(0, 3)
    return structure


def make_n_layer_structure(
    thicknesses: list[float],
    roughnesses: list[float],
    slds: list[float],
) -> Structure:
    # 기본 매질 정의
    air = SLD(0.0, name="Air")
    substrate = SLD(2.0, name="Substrate")

    # Stack을 이용한 반복 구조 생성
    multilayer = Stack(name="Multilayer", repeats=len(thicknesses))
    for t, r, s in zip(thicknesses, roughnesses, slds, strict=False):
        film = SLD(s, name=f"Film SLD={s}")
        multilayer.append(film(t, r))

    # Structure 조립
    structure = air(0, 0) | multilayer | substrate(0, 3)
    return structure


def make_structure_2l(
    thickness1: float,
    roughness1: float,
    sld1: float,
    thickness2: float,
    roughness2: float,
    sld2: float,
):
    air = SLD(0.0, name="Air")
    oxide = SLD(2.5, name="Oxide")
    film1 = SLD(sld1, name="Thin Film")
    film2 = SLD(sld2, name="Thin Film2")
    substrate = SLD(2.0, name="Substrate")

    structure = (
        air(0, 0) | oxide(20, 2) | film2(thickness2, roughness2) | film1(thickness1, roughness1) | substrate(0, 3)
    )
    return structure


def structure2R(structure: Structure, q: np.ndarray):
    if q[0] < 0.01:
        raise ValueError("Initial value of q is too close to 0.")
    model = ReflectModel(structure)
    R = model(q)
    return R


def add_xrr_noise(R: np.ndarray) -> np.ndarray:
    N = len(R)
    R_poisson = apply_poisson_noise(R, s=10 ** random.uniform(2, 5))
    uniform_noise = np.random.uniform(0.3, 0.9, N)
    background_noise = get_background_noise(N, -9, -7)
    curve_scaling = np.random.uniform(0.99, 1.01)
    R_noise = R_poisson * uniform_noise * curve_scaling + background_noise
    return R_noise


def simulate_xrr_with_noise(structure: Structure, q: np.ndarray):
    R = structure2R(structure, q)
    return add_xrr_noise(R)
