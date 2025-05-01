from pathlib import Path
import shutil
import subprocess

from tqdm import tqdm


software_dir: Path = Path("C:\\dev\\science\\xray_reflection\\XRRmaker\\lsfit_software")
save_dir: Path = Path("results").absolute()

for run_n in tqdm(range(1, 101)):
    save_file = save_dir / f"data{run_n}"
    save_file.mkdir(exist_ok=True)

    confile: Path = software_dir / "1.con"
    datfile: Path = software_dir / "1.dat"
    shutil.copy(confile, save_file / "1.con")
    shutil.copy(datfile, save_file / "1.dat")

    start: bytes = (save_file / "1").as_posix().encode() + b"\n\n"
    params: bytes = b"0000000000000001\n\n"
    save: bytes = b"e\ny\ny\ny\n"
    end: bytes = b"q"
    command: bytes = start + params + save + end

    subprocess.run(
        [software_dir / "lsfit.exe"],
        cwd=software_dir,
        input=command,
        capture_output=True,
        check=True,
    )
