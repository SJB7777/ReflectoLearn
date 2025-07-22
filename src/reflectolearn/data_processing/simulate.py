import random

import numpy as np
from refnx.reflect import SLD, ReflectModel, Stack
from refnx.reflect.structure import Structure

from ..math_utils import apply_poisson_noise, get_background_noise


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
    repeats: int = 1
) -> Structure:
    # 기본 매질 정의
    air = SLD(0.0, name="Air")
    substrate = SLD(2.0, name="Substrate")

    # Stack을 이용한 반복 구조 생성
    multilayer = Stack(name="Multilayer", repeats=repeats)
    for t, r, s in zip(thicknesses, roughnesses, slds):
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
        air(0, 0)
        | oxide(20, 2)
        | film2(thickness2, roughness2)
        | film1(thickness1, roughness1)
        | substrate(0, 3)
    )
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
