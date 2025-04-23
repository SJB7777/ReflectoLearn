from pathlib import Path
import shutil
import random
import subprocess
import time

# ---------------- 설정 ----------------

# 랜덤으로 선택할 9번/10번 파라미터 값들 (쌍으로 선택됨)
pair_list = [
    (10.330000, 22.780000),
    (15.130000, 8.450000),
    (28.480000, 15.044000),
    (10.000000, 47.000000)
    # 원하는 값들 계속 추가 가능
]

# 11번 sigma 파라미터 값의 랜덤 범위
range_11 = (0.0000, 30.0000)

# 12번 thickness 파라미터 값의 랜덤 범위
range_12 = (30.0000, 400.0000)

# 실행할 외부 프로그램 명령어 (예: exe 파일 이름 또는 경로)
program_command = Path("lsfit.exe")  # <- 실제 실행 파일명으로 수정

# 결과 저장 디렉토리
base_dir = Path("results")
base_dir.mkdir(exist_ok=True)


print("자동 반복 시작! 중지하려면 Ctrl + C 를 누르세요.\n")
# ---------------- 무한 반복 루프 ----------------
i = 1
try:
    while i < 10001:
        # 새 폴더 생성 (예: results/data1)
        folder_name = base_dir / f"data{i}"
        folder_name.mkdir(exist_ok=True)

        # 원본 1.con.txt 파일을 읽어서 라인 리스트로 저장
        with open("1.con", "r", encoding='utf-8') as f:
            lines = f.readlines()

        # 랜덤 파라미터 생성
        val9, val10 = random.choice(pair_list)
        val11 = round(random.uniform(*range_11), 5)
        val12 = round(random.uniform(*range_12), 4)
        val13, val14 = random.choice(pair_list)
        val15 = round(random.uniform(*range_11), 5)
        val16 = round(random.uniform(*range_12), 4)

        # 수정할 줄 번호 인덱스 (파일 줄 번호는 1부터지만, 인덱스는 0부터 시작)
        param_idx = (9, 10, 11, 12, 13, 14, 15, 16)
        param_idx = [idx + 3 for idx in param_idx]
        # 값만 바꿔서 포맷 유지하는 함수
        def replace_value(line, new_value):
            new_str = str(new_value).ljust(7, '0')[:8]
            return f"{line[:37]}{new_str}      {line.split()[-1]}\n"

        # 9~12번 파라미터 줄 수정
        lines[param_idx[0]] = replace_value(lines[param_idx[0]], val9)
        lines[param_idx[1]] = replace_value(lines[param_idx[1]], val10)
        lines[param_idx[2]] = replace_value(lines[param_idx[2]], val11)
        lines[param_idx[3]] = replace_value(lines[param_idx[3]], val12)
        lines[param_idx[4]] = replace_value(lines[param_idx[4]], val13)
        lines[param_idx[5]] = replace_value(lines[param_idx[5]], val14)
        lines[param_idx[6]] = replace_value(lines[param_idx[6]], val15)
        lines[param_idx[7]] = replace_value(lines[param_idx[7]], val16)

        # 수정된 설정 파일을 1.con 으로 저장
        with open("1.con", "w", encoding='utf-8') as f:
            f.writelines(lines)

        # 프로그램 실행 후 e y y y 입력 전달
        proc = subprocess.Popen(program_command, stdin=subprocess.PIPE)
        proc.communicate(input=b"\n\ne\ny\ny\ny\n")
        time.sleep(0.4)
        # 결과 파일을 폴더에 저장
        shutil.copy("1.con", folder_name / "1.con")
        shutil.copy("1.out", folder_name / "1.out")

        # 진행 상황 출력
        print(f"[{time.strftime('%H:%M:%S')}] {folder_name} 완료")

        i += 1
        time.sleep(0.2)

except KeyboardInterrupt:
    print("\n자동 반복을 중지했습니다. 수고하셨습니다!")
