import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import deque

import sys
sys.path.append('/root/LipForensics')
from preprocessing.utils import warp_img, apply_transform, cut_patch  # 필요한 유틸리티 함수 import

STD_SIZE = (256, 256)
STABLE_POINTS = [33, 36, 39, 42, 45]


class CropMouthProcessor:
    def __init__(self, mean_face_path, crop_width=96, crop_height=96, start_idx=48, stop_idx=68, window_margin=12):
        """
        CropMouthProcessor 초기화
        :param mean_face_path: Mean face landmarks 파일 경로
        :param crop_width: 자를 입의 폭
        :param crop_height: 자를 입의 높이
        :param start_idx: 입 랜드마크 시작 인덱스
        :param stop_idx: 입 랜드마크 끝 인덱스
        :param window_margin: 랜드마크 스무딩에 사용할 창 크기
        """
        self.mean_face_landmarks = np.load(mean_face_path)
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.start_idx = start_idx
        self.stop_idx = stop_idx
        self.window_margin = window_margin

    def crop_frames_and_save(self, frames_dir, landmarks_dir):
        """
        비디오 디렉토리를 처리하여 입 모양을 정렬하고 자릅니다.
        :param frames_dir: 입력 비디오 프레임 디렉토리
        :param landmarks_dir: 랜드마크 디렉토리
        :return: 결과 저장 디렉토리 경로
        """
        # target_dir 자동 생성: frames_dir 이름에 "_cropped_mouths"
        if not frames_dir.endswith("_frames"):
            raise ValueError("frames_dir must end with '_frames'")
        target_dir = frames_dir.replace("_frames", "_cropped_mouths")
        os.makedirs(target_dir, exist_ok=True)

        frame_names = sorted(os.listdir(frames_dir))
        q_frames, q_landmarks, q_name = deque(), deque(), deque()

        for frame_name in tqdm(frame_names, desc="Processing frames", unit="frame"):
            # 프레임 읽기
            frame_path = os.path.join(frames_dir, frame_name)
            landmark_path = os.path.join(landmarks_dir, f"{frame_name[:-4]}.npy")

            if not os.path.exists(landmark_path):
                print(f"Landmark file missing for frame {frame_name}. Skipping.")
                continue

            with Image.open(frame_path) as pil_img:
                img = np.array(pil_img)
            landmarks = np.load(landmark_path)

            q_frames.append(img)
            q_landmarks.append(landmarks)
            q_name.append(frame_name)

            # 스무딩 및 자르기
            if len(q_frames) == self.window_margin:
                smoothed_landmarks = np.mean(q_landmarks, axis=0)
                cur_landmarks = q_landmarks.popleft()
                cur_landmarks = np.squeeze(cur_landmarks)
                cur_frame = q_frames.popleft()
                cur_name = q_name.popleft()

                smoothed_landmarks = np.squeeze(smoothed_landmarks)
                if smoothed_landmarks.ndim != 2 or smoothed_landmarks.shape[0] != 68:
                    print(f"Skipping frame {cur_name}: unexpected landmarks shape.")
                    continue

                # 이미지 변환
                trans_frame, trans = warp_img(
                    smoothed_landmarks[STABLE_POINTS, :],
                    self.mean_face_landmarks[STABLE_POINTS, :],
                    cur_frame,
                    STD_SIZE,
                )

                try:

                    trans_landmarks = trans(cur_landmarks)
                except Exception as e:
                    print(f"Error transforming landmarks for frame {cur_name}: {e}")
                    continue

                # 입 영역 자르기
                cropped_frame = cut_patch(
                    trans_frame,
                    trans_landmarks[self.start_idx : self.stop_idx],
                    self.crop_height // 2,
                    self.crop_width // 2,
                )

                # 결과 저장
                target_path = os.path.join(target_dir, cur_name)
                Image.fromarray(cropped_frame.astype(np.uint8)).save(target_path)

        print("Processing complete for video directory.")
        return target_dir

    def process(self, frames_dir, landmarks_dir):
        """
        비디오 디렉토리를 처리합니다.
        :param frames_dir: 비디오 프레임 디렉토리
        :param landmarks_dir: 랜드마크 디렉토리
        :return: 결과 저장 디렉토리 경로
        """
        return self.crop_frames_and_save(frames_dir, landmarks_dir)


if __name__ == "__main__":
    # 기본 실행
    frames_dir = "./example_frames"
    landmarks_dir = "./example_FAN"
    mean_face_path = "/root/LipForensics/preprocessing/20words_mean_face.npy"

    processor = CropMouthProcessor(mean_face_path)
    cropped_mouths_dir = processor.process(frames_dir, landmarks_dir)
    print(f"Cropped mouths saved to: {cropped_mouths_dir}")
