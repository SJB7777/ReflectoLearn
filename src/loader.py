from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_dat_file(dat_file: str | Path) -> list[str]:
    return pd.read_csv(
        dat_file,
        sep=r"\s+",
        skipinitialspace=True,
        encoding="utf-8",
        header=None,
        names=["angle", "intensity"],  # TODO: Check the header's real name
    )


def parse_out_qualities(out_file: str | Path):
    """Read the quality section from the out file."""
    quality_section: list[str] = []
    start_reading_quality = False
    start_qualtity_section = False
    with open(out_file, "r", encoding="utf-8") as f:
        for line in f:
            if "C quality of the agreement data vs. model" in line:
                start_reading_quality = True
                continue

            if start_reading_quality:
                # print(line)
                if not start_qualtity_section and line.strip() == "C":
                    start_qualtity_section = True
                    continue
                quality_section.append(line.strip())
                if start_qualtity_section and line.strip() == "C":
                    break

    qualities = {}
    # print(quality_section)
    for line in quality_section:
        line.strip()
        if line == "C":
            continue
        factor, value = line.rsplit(None, 1)
        factor = " ".join(factor[2:].split())
        qualities[factor] = float(value.strip())
    return qualities


def parse_out_data(out_file: str | Path) -> pd.DataFrame:
    """Read the out file and return a DataFrame."""
    return pd.read_csv(
        out_file,
        sep=r"\s+",
        skipinitialspace=True,
        encoding="utf-8",
        index_col=0,
        names=["#", "XOBS", "YOBS", "YCALC", "DELTA", "DELTA/SIGMA"],
        comment="C",
    )


def parse_con_parameter(file):
    parameters = []

    with open(file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # '### name of parameter' 다음 줄부터 시작
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("### name of parameter"):
            start_idx = i + 1
            break

    if start_idx is None:
        raise ValueError("No parameter section found in file.")

    # 읽기 시작
    for line in lines[start_idx:]:
        if line.strip() == "" or line.strip().startswith("Parameter Variation pattern"):
            break  # 파라미터 끝났으면 멈춤

        parts = line.strip().split()
        if len(parts) < 4:
            continue  # 파싱 실패한 라인은 스킵

        # parts[2] = 파라미터 이름 일부일 수 있으니, Value는 뒤에서 2번째
        try:
            value = float(parts[-2])
            parameters.append(value)
        except ValueError:
            continue  # 혹시 float 변환 실패하면 스킵

    return parameters


if __name__ == "__main__":
    root_dir = Path(
        "C:\\dev\\science\\xray_reflection\\XRRmaker\\data\\rl_data\\sample"
    )
    print("root_dir", root_dir)

    out_file = root_dir / "1.out"
    dat_file = root_dir / "1.dat"

    out_qualities = parse_out_qualities(out_file)
    out_df = parse_out_data(out_file)
    dat_df = read_dat_file(dat_file)

    filepath = root_dir / "1.con"
    params = parse_con_parameter(filepath)
    print(f"Number of Parameter: {len(params)}개")
    print(params)

    for key, value in out_qualities.items():
        print(f"{key}: {value}")

    plt.plot(out_df["XOBS"], out_df["YOBS"], label="YOBS")
    plt.plot(out_df["XOBS"], out_df["YCALC"], label="YCALC")
    plt.legend()
    plt.show()

    plt.plot(dat_df["angle"], np.log(dat_df["intensity"]), label="dat")
    plt.legend()
    plt.show()
