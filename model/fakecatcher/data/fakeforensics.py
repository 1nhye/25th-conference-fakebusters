import os
import csv
import pandas as pd
from pathlib import Path

def get_video_files(base_dir, csv_path):
    """
    하위 디렉토리를 탐색해 동영상 파일의 정보를 CSV에 저장합니다.
    - video_name: 파일 이름
    - video_path: 파일 경로
    - label: manipulated_sequences(1) 또는 original_sequences(0)
    """
    # 저장할 데이터를 담을 리스트
    video_data = []

    # 지원하는 동영상 확장자 (필요시 확장 가능)
    video_extensions = (".mp4", ".avi", ".mov", ".mkv")

    # 하위 디렉토리 탐색
    for root, _, files in os.walk(base_dir):
        for file in files:
            # 동영상 파일 확인
            if file.endswith(video_extensions):
                # 경로 생성 및 변환 (Path 사용)
                file_path = Path(root).joinpath(file).as_posix()
                # 라벨 설정
                if "manipulated_sequences" in root:
                    label = 0
                elif "original_sequences" in root:
                    label = 1
                else:
                    continue  # 지정되지 않은 디렉토리는 무시

                # 데이터 저장
                video_data.append({
                    "video_name": file,
                    "video_path": file_path,
                    "label": label
                })

    # CSV 파일에 데이터 저장
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["video_name", "video_path", "label"])
        writer.writeheader()
        writer.writerows(video_data)

    print(f"CSV 파일이 저장되었습니다: {csv_path}")

def load_fakeforensics_data(csv_path: str):
    data = pd.read_csv(csv_path)
    video_paths = data['video_path']
    labels = data['label']
    return video_paths, labels

# 실행 예제
if __name__ == "__main__":
    # 작업 디렉토리와 저장할 CSV 파일 경로
    base_directory = "D:/2024년/4-1/산학/data/ff_data"  # 탐색할 루트 디렉토리
    output_csv = "video_list.csv"  # 저장할 CSV 파일 이름

    # 함수 실행
    get_video_files(base_directory, output_csv)
