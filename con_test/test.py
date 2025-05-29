from pathlib import Path
from subprocess import CalledProcessError

from src.lsfit_wrapper import run_lsfit


def generate_lsfit_test_cases_and_run(software_dir: Path, test_root: Path):
    """여러 포맷의 수치를 넣은 con 파일을 만들고, 각각 lsfit을 실행해 통과 여부를 기록"""
    test_cases = {
        "ok_float_short": ("0.123456", "0.012345"),
        "ok_scientific_small": ("4.000000E-03", "2.053000E-03"),
        "ok_scientific_neg_exp": ("0.107172E-01", "0.8971571E-02"),
        "ok_large_float": ("9999999.9999", "99999.99"),
        "too_large_float": ("123456789012345.0", "1234567890.0"),
        "too_long_scientific": ("1.23456789012345E+123", "1.2E+120"),
        "tiny_float": ("1.0E-50", "1.0E-51"),
    }

    header = [
        "Parameter and refinement control file produced by  program LSFIT",
        "DBI G/N Text for X-axis(A20) Text for Y-axis(A20) REP",
        "I   N   z  [\\AA]             log(|FT\\{Int\\cdotq_{   1",
        "### name of parameter.............  Value          Increment",
    ]

    valid_body = [
        "  1 footprint in deg                0.202554        8.223200E-03",
        "  2 background (-log value)         4.000000E-03    2.053000E-03",
        "  3 diffractometer resolution       5.000000        0.097706",
        "  4 [disp,n*b] reference material   0.000000        3.053987E-03",
        "  5 disp / n*b substrate 0 part 1   4.899598        0.244980",
        "  6 di_nb/beta substrate 0 part 1   66.116276       3.305814",
        "  7 sigma substrate in A 0 part 1   0.100000        5.000000E-03",
        "  8 intensity offset                0.107172E-01  0.8971571E-02",
        "  9 disp / n*b layer     1 part 1   4.020025        0.201001",
        " 10 di_nb/beta layer     1 part 1   84.936562       4.246828",
        " 11 sigma layer in A     1 part 1   1.404134        0.070207",
        " 12 layer thickness      1 part 1   8.000000        0.080000",
    ]

    footer = [
        "Parameter Variation pattern  /  selected files :  1111",
        "0        1         2         3         4         5         6         7",
        "1234567890123456789012345678901234567890123456789012345678901234567890123456789",
    ]

    results = {}

    for case_name, (val, inc) in test_cases.items():
        case_dir = test_root / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        test_line = f" 13 test param {case_name:<13} {val:>16}{inc:>12}"
        con_lines = [
            line + "\n" for line in (header + valid_body + [test_line] + footer)
        ]

        try:
            out, err = run_lsfit(software_dir, case_dir, con_lines)
            results[case_name] = "✅ PASS"
        except CalledProcessError as e:
            results[case_name] = f"❌ FAIL (exit code {e.returncode})"
        except Exception as e:
            results[case_name] = f"❌ FAIL ({type(e).__name__}: {e})"

    print("\n📋 LSFIT 테스트 결과:")
    for name, result in results.items():
        print(f"{name:<25}: {result}")


if __name__ == "__main__":
    software_path = Path(r"C:\dev\science\xray_reflection\ReflectoLearn\lsfit_software")
    test_output_root = Path(r"C:\dev\science\xray_reflection\ReflectoLearn\con_test")

    generate_lsfit_test_cases_and_run(software_path, test_output_root)
