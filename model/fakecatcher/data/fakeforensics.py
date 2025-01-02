import os
import csv
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def get_video_files(base_dir, csv_path, split_num):
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

def split_and_save_videos(csv_path, train_csv, test_csv, test_ratio=0.3):
    """
    CSV 파일을 로드하여 훈련 데이터와 테스트 데이터로 나누고 저장합니다.
    - csv_path: 전체 데이터셋이 저장된 CSV 파일 경로
    - train_csv: 훈련 데이터셋 CSV 파일 경로
    - test_csv: 테스트 데이터셋 CSV 파일 경로
    - test_ratio: 테스트 데이터 비율 (기본값 0.3)
    """
    # 데이터 로드
    data = pd.read_csv(csv_path)

    # 라벨별로 데이터 나누기
    train_data = []
    test_data = []
    for label in data['label'].unique():
        label_data = data[data['label'] == label]
        train, test = train_test_split(label_data, test_size=test_ratio, random_state=42)
        train_data.append(train)
        test_data.append(test)

    # 병합
    train_data = pd.concat(train_data)
    test_data = pd.concat(test_data)

    # 파일 저장
    train_data.to_csv(train_csv, index=False, encoding="utf-8")
    test_data.to_csv(test_csv, index=False, encoding="utf-8")

    print(f"Train CSV 파일이 저장되었습니다: {train_csv}")
    print(f"Test CSV 파일이 저장되었습니다: {test_csv}")

# 실행 예제
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base_dirc", type=str, required=True, help="Path to the base directory")
    parser.add_argument("-s", "--split_number", type=int, required=False)
    args = parser.parse_args()

    # 작업 디렉토리와 저장할 CSV 파일 경로
    base_directory = args.base_dirc  # 탐색할 루트 디렉토리
    split_num = args.split_number
    output_csv = os.path.join(base_directory,"video_list.csv")  # 전체 데이터 저장할 CSV 파일 이름
    train_csv = os.path.join(base_directory,"train_video_list.csv")  # 훈련 데이터 저장 파일
    test_csv = os.path.join(base_directory,"test_video_list.csv")    # 테스트 데이터 저장 파일

    # 모든 비디오 정보를 수집하여 CSV로 저장
    get_video_files(base_directory, output_csv, split_num)

    # 데이터를 train/test로 나누어 각각 저장
    split_and_save_videos(output_csv, train_csv, test_csv)
