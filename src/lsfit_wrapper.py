from pathlib import Path
import shutil
import subprocess
from subprocess import CompletedProcess


def run_lsfit(
    software_dir: Path,
    save_dir: Path,
    new_lines: list[str],
    capture_output: bool = True,
) -> CompletedProcess[str]:
    """lsfit 소프트웨어를 실행하는 함수"""
    shutil.copy(software_dir / "1.dat", save_dir / "1.dat")
    with open(save_dir / "1.con", "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    # Commands
    start = [str(save_dir / "1")]
    save = ["e", "y", "y", "y"]
    end = ["q"]
    command = start + save + end
    # Run lsfit.exe
    result = subprocess.run(
        [software_dir / "lsfit.exe"],
        cwd=save_dir,
        input="\n".join(command),
        text=True,
        capture_output=capture_output,
        check=True,
        encoding="utf-8",
        timeout=5,
    )

    return result.stdout, result.stderr
    # TODO: To handle the error, we need to check the output and error messages.
    # process = subprocess.Popen(
    #     [software_dir / "lsfit.exe"],
    #     cwd=save_dir,
    #     stdin=subprocess.PIPE,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    #     text=True
    # )

    # # 줄 단위로 보내기
    # for line in [str(save_dir / "111"), "e", "y", "y", "y", "q"]:
    #     process.stdin.write(line + "\n")
    #     process.stdin.flush()
    # stdout, stderr = process.communicate()
    # print(stdout)
    # print(stderr)
