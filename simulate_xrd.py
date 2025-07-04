import random
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from tqdm import tqdm

from src.lsfit_wrapper import run_lsfit


layer_param_name = {
    "sld": "disp / n*b layer",
    "roughness": "sigma layer in A",
    "density": "di_nb/beta layer",
    "thickness": "layer thickness",
}

substate_param_name = {
    "sld": "disp / n*b substrate",
    "roughness": "sigma substrate in A",
    "density": "di_nb/beta substrate",
}


@dataclass
class Substrate:
    name: str = None
    sld: float = None  # disp / n*b substrate
    roughness: float = None  # sigma layer in A
    density: float = None  # di_nb/beta layer

    def __post_init__(self):
        if self.roughness == 0:
            raise ValueError("SLD cannot be zero.")


@dataclass
class Layer:
    name: str = None
    sld: float = None
    roughness: float = None
    density: float = None
    thickness: float = None

    def __post_init__(self):
        if self.roughness == 0:
            raise ValueError("SLD cannot be zero.")


def generate_multilayer(
    min_layers: int = 1, max_layers: int = 5
) -> dict[str, Substrate | list[Layer]]:

    substrate = Substrate(
        name="Si",
        sld=4.899598,  # Si SLD in 10^-6 Å^-2
        roughness=0.1,
        density=66.116276,
    )

    sio2_thickness = 8
    sio2_layer = Layer(
        name="SiO2",
        sld=4.610959, # * #random.uniform(0.7, 1.3),
        roughness=3, # random.uniform(0.1, sio2_thickness * 0.3),
        density=117.329199, #* random.uniform(0.7, 1.3),
        thickness=sio2_thickness,
    )

    hfo2_thickness = random.randint(1, 30)
    hfo2_layer = Layer(
        name="HfO2",
        sld=15.12988 * random.uniform(0.7, 1.3),
        roughness=random.uniform(0.1, hfo2_thickness * 0.3),
        density=7.8903 * random.uniform(0.7, 1.3),
        thickness=random.randint(1, 30),
    )

    layers = [sio2_layer, hfo2_layer]

    return {"substrate": substrate, "layers": layers}


def make_con_file(global_params, substrate: Substrate, layers: list[Layer]):
    """Make con file for lsfit."""

    def smart_format(val, precision=6):
        """적절한 자리수 표현: 너무 크거나 작으면 과학적 표기법 사용"""

        if isinstance(val, (float, int)):
            if val and abs(val) < 1e-2 or abs(val) >= 1e4:
                return f"{val:.{precision}E}"  # 과학적 표기법
            return f"{val:.{precision}f}"  # 일반 소수
        return str(val)

    lines = []

    # Header
    lines.append("Parameter and refinement control file produced by  program LSFIT")
    lines.append("DBI G/N Text for X-axis(A20) Text for Y-axis(A20) REP")
    lines.append("I   N   z  [\\AA]             log(|FT\\{Int\\cdotq_{   1")
    lines.append("### name of parameter.............  Value          Increment")

    # Body
    idx = 1
    # Global
    for name, (val, inc) in global_params.items():
        lines.append(
            f"{idx:3d} {name:<29}   {smart_format(val):15}{smart_format(inc):12}"
        )
        idx += 1
    # Substrate
    sub = asdict(substrate)
    for field in ["sld", "density", "roughness"]:
        val = sub[field]
        inc = val * 0.05  # 5% 증분 기본
        name = f"{substate_param_name[field]:<21}0 part {len(layers)}"
        lines.append(f"{idx:3d} {name}   {smart_format(val):15}{smart_format(inc):12}")
        idx += 1
    lines.append(
        f"{idx:3d} intensity offset                0.107172E-01  0.8971571E-02"
    )
    idx += 1
    # Layers
    for i, layer in enumerate(layers, start=1):
        lay = asdict(layer)
        for field in ["sld", "density", "roughness", "thickness"]:
            val = lay[field]
            inc = val * 0.05 if field != "thickness" else val * 0.01
            name = f"{layer_param_name[field]:<21}{i} part {len(layers)}"
            lines.append(
                f"{idx:3d} {name}   {smart_format(val):15}{smart_format(inc):12}"
            )
            idx += 1

    # Tail
    lines.append("Parameter Variation pattern  /  selected files :  1111")
    lines.append(
        "0        1         2         3         4         5         6         7"
    )
    lines.append(
        "1234567890123456789012345678901234567890123456789012345678901234567890123456789"
    )

    return [line + "\n" for line in lines]


def save_metadata(
    save_dir: Path, multilayer: dict[str, Substrate | list[Layer]]
) -> None:
    """다층 구조의 메타데이터를 JSON 형식으로 저장하는 함수."""
    metadata = {
        "substrate": {
            "name": multilayer["substrate"].name,
            "sld": multilayer["substrate"].sld,
            "roughness": multilayer["substrate"].roughness,
            "density": multilayer["substrate"].density,
        },
        "layers": [
            {
                "name": layer.name,
                "sld": layer.sld,
                "roughness": layer.roughness,
                "density": layer.density,
                "thickness": layer.thickness,
            }
            for layer in multilayer["layers"]
        ],
    }

    with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def main() -> None:
    """메인 함수"""
    global_params = {
        "footprint in deg": (0.2025540, 0.0082232),
        "background (-log value)": (0.004, 0.002053),
        "diffractometer resolution": (5.0000, 0.0977063),
        "[disp,n*b] reference material": (0.000000, 0.3053987e-2),
    }
    software_dir = Path(".\\lsfit_software").resolve()
    save_root = Path("..\\data\\onelayer").resolve()
    save_root.mkdir(exist_ok=True, parents=True)

    for run_n in tqdm(range(100001, 1000001), desc="Generating multilayers"):
        multilayer = generate_multilayer(min_layers=1, max_layers=1)
        new_lines = make_con_file(
            global_params,
            multilayer["substrate"],
            multilayer["layers"],
        )
        save_dir = save_root / f"d{run_n:05}"
        # NOTE: More than 8 characters will make error in lsfit
        if len(save_dir.stem) > 8:
            raise ValueError(
                f"Directory name {save_dir.stem} is too long. Max length is 8 characters."
            )
        save_dir.mkdir(exist_ok=True, parents=True)
        save_metadata(save_dir, multilayer)
        run_lsfit(software_dir, save_dir, new_lines, capture_output=True)


if __name__ == "__main__":
    main()
