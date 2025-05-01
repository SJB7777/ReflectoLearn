import random
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import json

# import numpy as np
from tqdm import tqdm


@dataclass
class Substrate:
    name: str = None
    sld: float = None  # disp / n*b substrate
    roughness: float = None  # sigma layer in A
    density: float = None  # di_nb/beta layer


@dataclass
class Layer:
    name: str = None
    sld: float = None
    roughness: float = None
    density: float = None
    thickness: float = None


# 실제 물질 데이터베이스 (물질명, SLD, 밀도)
# SLD 단위: 10^-6 Å^-2, 밀도 단위: g/cm^3
MATERIAL_DATABASE = {
    # 기판 물질
    "substrates": {
        "Si": (20.0700, 2.3290),  # 실리콘
        "SiO2": (18.9000, 2.2000),  # 이산화규소 (석영)
        "Al2O3": (33.5000, 3.9700),  # 알루미나 (사파이어)
        "GaAs": (19.5800, 5.3170),  # 갈륨비소
        "MgO": (17.7000, 3.5800),  # 산화마그네슘
    },
    # 층 물질
    "layers": {
        # 금속
        "Al": (21.3000, 2.7000),  # 알루미늄
        "Ti": (18.9000, 4.5060),  # 티타늄
        "Cr": (28.0200, 7.1400),  # 크롬
        "Fe": (56.1300, 7.8740),  # 철
        "Co": (60.7000, 8.9000),  # 코발트
        "Ni": (68.7200, 8.9020),  # 니켈
        "Cu": (56.4700, 8.9600),  # 구리
        "Zn": (48.8000, 7.1330),  # 아연
        "Pt": (96.5000, 21.4500),  # 백금
        "Au": (130.300, 19.3200),  # 금
        "Ag": (37.3000, 10.4900),  # 은
        # 산화물
        "SiO2": (18.9000, 2.2000),  # 이산화규소
        "TiO2": (30.5800, 4.2300),  # 이산화티타늄
        "Fe2O3": (41.1400, 5.2400),  # 산화철
        "Fe3O4": (37.0100, 5.1750),  # 자철석
        "ZnO": (42.3800, 5.6100),  # 산화아연
        "Al2O3": (33.5000, 3.9700),  # 알루미나
        "HfO2": (45.1600, 9.6800),  # 이산화하프늄
        # 반도체
        "Si": (20.0700, 2.3290),  # 실리콘
        "Ge": (28.3000, 5.3230),  # 게르마늄
        "GaAs": (19.5800, 5.3170),  # 갈륨비소
        "InP": (20.6600, 4.8100),  # 인화인듐
        # 자성체
        "CoFe": (58.4100, 8.3870),  # 코발트-철 합금
        "CoFeB": (56.0000, 8.2000),  # 코발트-철-붕소 합금
        "NiFe": (64.7100, 8.7000),  # 니켈-철 합금 (퍼멀로이)
        # 다층 자성 메모리 물질
        "MgO": (17.7000, 3.5800),  # 산화마그네슘
        "Ta": (55.0800, 16.6540),  # 탄탈륨
        "IrMn": (65.0000, 10.1800),  # 이리듐-망간 합금
    },
}

# 물질별 표면 거칠기 범위 (Å)
ROUGHNESS_RANGES = {
    "Si": (2.0, 8.0),
    "SiO2": (0, 50.0),
    "Al2O3": (3.0, 12.0),
    "GaAs": (2.5, 10.0),
    "MgO": (2.0, 9.0),
    "Al": (5.0, 20.0),
    "Ti": (4.0, 15.0),
    "Cr": (3.0, 12.0),
    "Fe": (4.0, 15.0),
    "Co": (3.5, 12.0),
    "Ni": (3.0, 12.0),
    "Cu": (3.5, 14.0),
    "Zn": (5.0, 18.0),
    "Pt": (2.0, 10.0),
    "Au": (2.0, 10.0),
    "Ag": (2.5, 11.0),
    "TiO2": (3.0, 12.0),
    "Fe2O3": (4.0, 15.0),
    "Fe3O4": (4.0, 15.0),
    "ZnO": (3.5, 14.0),
    "HfO2": (3.0, 11.0),
    "Ge": (3.0, 12.0),
    "InP": (2.5, 10.0),
    "CoFe": (3.0, 12.0),
    "CoFeB": (3.0, 12.0),
    "NiFe": (3.0, 12.0),
    "Ta": (3.0, 12.0),
    "IrMn": (3.0, 12.0),
    "default": (2.0, 15.0),  # 기본 범위
}

# 물질별 층 두께 범위 (Å)
THICKNESS_RANGES = {
    "Al": (50, 500),
    "Ti": (30, 300),
    "Cr": (30, 300),
    "Fe": (30, 400),
    "Co": (20, 200),
    "Ni": (30, 300),
    "Cu": (50, 500),
    "Zn": (50, 400),
    "Pt": (20, 200),
    "Au": (20, 300),
    "Ag": (30, 400),
    "SiO2": (50, 1000),
    "TiO2": (30, 500),
    "Fe2O3": (30, 400),
    "Fe3O4": (30, 400),
    "ZnO": (30, 500),
    "Al2O3": (50, 800),
    "HfO2": (30, 300),
    "Si": (100, 1000),
    "Ge": (50, 500),
    "GaAs": (50, 500),
    "InP": (50, 500),
    "CoFe": (20, 150),
    "CoFeB": (20, 150),
    "NiFe": (20, 200),
    "MgO": (10, 100),
    "Ta": (30, 200),
    "IrMn": (30, 200),
    "default": (0, 500),  # 기본 범위
}

SLD_DENSITY = {"SiO2": (4.88, 117), "TiN": (10.33, 22.78), "HfO2": (15.1, 8.45)}

# 특정 다층 구조 템플릿 정의
MULTILAYER_TEMPLATES = [
    # 자기 터널 접합(MTJ) 구조
    {"name": "MTJ", "layers": ["Ta", "CoFeB", "MgO", "CoFeB", "Ta"]},
    # 거대자기저항(GMR) 구조
    {"name": "GMR", "layers": ["Ta", "NiFe", "Cu", "CoFe", "IrMn", "Ta"]},
    # 반도체 게이트 구조
    {"name": "Gate_Stack", "layers": ["Si", "SiO2", "HfO2", "Ti", "Al"]},
    # 광학 반사 코팅
    {"name": "Optical_Coating", "layers": ["SiO2", "TiO2", "SiO2", "TiO2"]},
    # 금속-산화물-반도체 구조
    {"name": "MOS", "layers": ["Si", "SiO2", "Al"]},
]


def get_roughness_range(material_name: str) -> tuple[float, float]:
    """물질에 따른 거칠기 범위를 반환하는 함수"""
    return ROUGHNESS_RANGES.get(material_name, ROUGHNESS_RANGES["default"])


def get_thickness_range(material_name: str) -> tuple[float, float]:
    """물질에 따른 두께 범위를 반환하는 함수"""
    return THICKNESS_RANGES.get(material_name, THICKNESS_RANGES["default"])


def generate_multilayer(
    min_layers: int = 1, max_layers: int = 5, use_templates: bool = True
) -> dict[str, Substrate | list[Layer]]:
    """랜덤한 다층 구조를 생성하는 함수."""
    substrate_name = random.choice(list(MATERIAL_DATABASE["substrates"].keys()))
    substrate_sld, substrate_density = MATERIAL_DATABASE["substrates"][substrate_name]
    substrate_roughness_range = get_roughness_range(substrate_name)

    substrate = Substrate(
        name=substrate_name,
        sld=substrate_sld,
        roughness=random.uniform(*substrate_roughness_range),
        density=substrate_density,
    )

    layers = []
    if use_templates and random.random() < 0.7:  # 70% 확률로 템플릿 사용
        template = random.choice(MULTILAYER_TEMPLATES)
        for name in template["layers"][:max_layers]:
            sld, density = MATERIAL_DATABASE["layers"].get(name, (None, None))
            roughness_range = get_roughness_range(name)
            thickness_range = get_thickness_range(name)

            layer = Layer(
                name=name,
                sld=sld,
                roughness=random.uniform(*roughness_range),
                density=density,
                thickness=random.uniform(*thickness_range),
            )
            layers.append(layer)
    else:
        num_layers = random.randint(min_layers, max_layers)
        for _ in range(num_layers):
            name = random.choice(list(MATERIAL_DATABASE["layers"].keys()))
            sld, density = MATERIAL_DATABASE["layers"][name]
            roughness_range = get_roughness_range(name)
            thickness_range = get_thickness_range(name)
        layers.append(layer)
    return {"substrate": substrate, "layers": layers}

    # substrate_name = "SiO2"
    # substrate = Substrate(
    #     name=substrate_name,
    #     sld=SLD_DENSITY[substrate_name][0] * random.uniform(0.7, 1.3),
    #     density=SLD_DENSITY[substrate_name][1] * random.uniform(0.7, 1.3),
    #     roughness=random.uniform(*ROUGHNESS_RANGES[substrate_name]),
    # )
    # name1 = "TiN"
    # thickness1 = random.uniform(*THICKNESS_RANGES["default"])
    # roughness1 = thickness1 * random.uniform(0, 0.7)
    # layer1 = Layer(
    #     name=substrate_name,
    #     sld=SLD_DENSITY[name1][0] * random.uniform(0.7, 1.3),
    #     density=SLD_DENSITY[name1][1] * random.uniform(0.7, 1.3),
    #     roughness=roughness1,
    #     thickness=thickness1,
    # )
    # name2 = "HfO2"
    # thickness2 = random.uniform(*THICKNESS_RANGES["default"])
    # roughness2 = thickness2 * random.uniform(0, 0.7)
    # layer2 = Layer(
    #     name=substrate_name,
    #     sld=SLD_DENSITY[name2][0] * random.uniform(0.7, 1.3),
    #     density=SLD_DENSITY[name2][1] * random.uniform(0.7, 1.3),
    #     roughness=roughness2,
    #     thickness=thickness2,
    # )
    # name3 = "TiN"
    # thickness3 = random.uniform(*THICKNESS_RANGES["default"])
    # roughness3 = thickness3 * random.uniform(0, 0.7)
    # layer3 = Layer(
    #     name=substrate_name,
    #     sld=SLD_DENSITY[name3][0] * random.uniform(0.7, 1.3),
    #     density=SLD_DENSITY[name3][1] * random.uniform(0.7, 1.3),
    #     roughness=roughness3,
    #     thickness=thickness3,
    # )

    # layers = [layer1, layer2, layer3]
    # return {"substrate": substrate, "layers": layers}


def rewrite_con_file(
    con_lines: list[str], multilayer: dict[str, Substrate | list[Layer]]
) -> list[str]:
    """con 파일의 내용을 multilayer에 맞게 수정하는 함수."""

    def replace_value(line: str, new_value: float) -> str:
        new_str = str(new_value).ljust(7, "0")[:8]
        return f"{line[:37]}{new_str}      {line.split()[-1]}\n"

    new_lines = con_lines.copy()
    substrate = multilayer["substrate"]
    if substrate.sld is not None:
        new_lines[8] = replace_value(new_lines[8], substrate.sld)
    if substrate.roughness is not None:
        new_lines[9] = replace_value(new_lines[9], substrate.roughness)
    if substrate.density is not None:
        new_lines[10] = replace_value(new_lines[10], substrate.density)

    for i, layer in enumerate(multilayer["layers"]):
        if layer.sld is not None:
            new_lines[12 + i * 4] = replace_value(new_lines[12 + i * 4], layer.sld)
        if layer.roughness is not None:
            new_lines[13 + i * 4] = replace_value(
                new_lines[13 + i * 4], layer.roughness
            )
        if layer.density is not None:
            new_lines[14 + i * 4] = replace_value(new_lines[14 + i * 4], layer.density)
        if layer.thickness is not None:
            new_lines[15 + i * 4] = replace_value(
                new_lines[15 + i * 4], layer.thickness
            )

    return new_lines


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


def run_lsfit(
    software_dir: Path,
    save_dir: Path,
    new_lines: list[str],
    capture_output: bool = True,
):
    """lsfit 소프트웨어를 실행하는 함수"""
    shutil.copy(software_dir / "1.dat", save_dir / "1.dat")
    with open(save_dir / "1.con", "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    start = (save_dir / "1").as_posix().encode() + b"\n\n"
    save = b"e\ny\ny\ny\n"
    end = b"q"
    command = start + save + end

    subprocess.run(
        [software_dir / "lsfit.exe"],
        cwd=software_dir,
        input=command,
        capture_output=capture_output,
        check=True,
    )


def main() -> None:
    """메인 함수"""
    software_dir = Path(".\\lsfit_software")
    save_root = Path("..\\data\\simulation_data").absolute()
    save_root.mkdir(exist_ok=True, parents=True)

    with open(software_dir / "1.con", "r", encoding="utf-8") as f:
        con_lines = f.readlines()
    for run_n in tqdm(range(1, 10001)):
        multilayer = generate_multilayer(min_layers=1, max_layers=2)
        new_lines = rewrite_con_file(con_lines, multilayer)

        save_dir = save_root / f"d{run_n:05}"
        # More than 8 characters will make error in lsfit
        if len(save_dir.stem) > 8:
            raise ValueError(
                f"Directory name {save_dir.stem} is too long. Max length is 8 characters."
            )
        save_dir.mkdir(exist_ok=True, parents=True)
        save_metadata(save_dir, multilayer)
        run_lsfit(software_dir, save_dir, new_lines, capture_output=True)


if __name__ == "__main__":
    main()
