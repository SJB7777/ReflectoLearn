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
        thickness = random.uniform(30, max_thick)
        roughness = random.uniform(0, thickness * 0.03)
        sld = random.uniform(0.5, 20.0)

        thicknesses.append(thickness)
        roughnesses.append(roughness)
        slds.append(sld)

    return thicknesses, roughnesses, slds


def make_n_layer_structure(
    thicknesses: list[float],
    roughnesses: list[float],
    slds: list[float],
) -> Structure:
    # 기본 매질 정의
    air = SLD(0.0, name="Air")
    sio2 = SLD(19.2, name="SiO2")

    # Stack을 이용한 반복 구조 생성
    multilayer = Stack(name="Multilayer", repeats=len(thicknesses))
    for t, r, s in zip(thicknesses, roughnesses, slds, strict=False):
        film = SLD(s, name=f"Film SLD={s}")
        multilayer.append(film(t, r))

    # Structure 조립
    structure = air(0, 0) | multilayer | sio2(0, 3)
    return structure


def structure_to_R(structure: Structure, q: np.ndarray):
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
