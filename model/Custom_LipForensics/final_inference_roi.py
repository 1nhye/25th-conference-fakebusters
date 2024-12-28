import argparse
import dlib
import yaml
import numpy as np

import sys
sys.path.insert(0, "/root/25th-conference-fakebusters/Custom_LipForensics/LipForensics")
from inference import evaluate_lipforensics
sys.path.insert(0, "/root/25th-conference-fakebusters/Custom_LipForensics/roi_extractor/utils")
from preprocess import VideoROIExtractor


def main():
    # argparse 설정
    parser = argparse.ArgumentParser(description="Extract frames from a video and process with VideoROIExtractor.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration YAML file.")
    args = parser.parse_args()

    # Config 파일 읽기
    with open(args.config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # dlib 모델 초기화
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(config['face_predictor_path'])
    mean_face = np.load(config['mean_face_path'])

    # VideoROIExtractor 초기화
    processor = VideoROIExtractor(
        input_video_path=args.video_path,
        detector=detector,
        predictor=predictor,
        mean_face=mean_face,
        config=config
    )

    # 1~4. 비디오 처리 (프레임 추출 및 ROI 크롭)
    rois = processor.preprocess_video()  # 비디오에서 크롭된 입 영역 추출
    cropped_mouth_images_path = processor.save_frames(rois)
    cropped_mouth_video_path = processor.save(rois)
    if cropped_mouth_images_path is None:
        print("Failed to process video with VideoROIExtractor.")
        return
    print("Cropped mouths successfully extracted.")
    print('cropped_mouth_images_path:', cropped_mouth_images_path)
    print('cropped_mouth_video_path:', cropped_mouth_video_path)
    
    # 5. LipForensics
    prediction = evaluate_lipforensics(cropped_mouths_path=cropped_mouth_images_path)
    print(f"Final Prediction: {prediction:.4f}")


if __name__ == "__main__":
    main()
