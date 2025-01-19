#!/usr/bin/env python
# inference.py
import os
import argparse
import torch
import cv2
from einops import rearrange
from torchvision import transforms
from PIL import Image
import numpy as np
import imageio
import logging

# -------------------------
# 사용자 정의 모듈 import
# -------------------------
from inference.inference import inference  # 기존에 사용하시던 inference 함수
from models import VectorQuantizedVAE  # VQ-VAE 모델

# ---------------------------
# 전역 상수 (경로 등 설정)
# ---------------------------
UPLOAD_DIR = './uploads'
MM_REPRESENTATION_DIR = './inference/mm_representation'
RECONSTRUCTION_DIR = './inference/reconstruction'

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MM_REPRESENTATION_DIR, exist_ok=True)
os.makedirs(RECONSTRUCTION_DIR, exist_ok=True)

# ------------------
# config 딕셔너리
# ------------------
config = {
    'data_root': 'MM_Det/',
    'ckpt': './weights/MM-Det/current_model.pth',
    'lmm_ckpt': 'sparklexfantasy/llava-1.5-7b-rfrd',
    'lmm_base': None,
    'st_ckpt': './weights/ViT/vit_base_r50_s16_224.orig_in21k/jx_vit_base_resnet50_224_in21k-6f7c7740.pth',
    'st_pretrained': True,
    'model_name': 'MMDet',
    'expt': 'MMDet_01',
    'window_size': 10,
    'conv_mode': 'llava_v1',
    'new_tokens': 64,
    'selected_layers': [-1],
    'interval': 200,
    'load_8bit': False,
    'load_4bit': True,
    'seed': 42,
    'gpus': 1,
    'cache_mm': True,
    'mm_root': '',
    'debug': False,
    'output_dir': MM_REPRESENTATION_DIR,
    'output_fn': 'mm_representation.pth',
    'mode': 'inference',
    'num_workers': 1,
    'sample_size': -1,
    'classes': ['inference'],
    'bs': 1
}

def main():
    """
    명령줄 인자를 받아 1회 추론을 수행:
    1) VQ-VAE 복원
    2) 영상 트림
    3) Inference 수행
    """

    parser = argparse.ArgumentParser(description="Run single inference from command line")
    parser.add_argument('-v', '--video_path', type=str, required=True,
                        help='입력 동영상 경로')
    parser.add_argument('-r', '--reconstruction_path', type=str, default=RECONSTRUCTION_DIR,
                        help='복원 및 트리밍 후 결과 저장 경로')
    args = parser.parse_args()

    video_path = args.video_path
    reconstruction_path = args.reconstruction_path

    

    os.makedirs(reconstruction_path, exist_ok=True)

    # 1. VQVAE 복원
    prefix = os.path.splitext(os.path.basename(video_path))[0]
    reconstructed_video_path = os.path.join(reconstruction_path, f"{prefix}_reconstructed.mp4")
    create_reconstructed_video_vqvae(video_path, reconstructed_video_path)
    print(f"[1] Reconstructed video saved at: {reconstructed_video_path}")

    # 2. 영상 트림 (해상도 축소)
    trimmed_prefix = f"{prefix}_trimmed"
    trimmed_video_path = os.path.join(UPLOAD_DIR, f"{trimmed_prefix}.mp4")
    trim_success = trim_video(reconstructed_video_path, trimmed_video_path)
    if not trim_success:
        print("[2] Error: Unable to trim video")
        return

    print(f"[2] Trimmed video saved at: {trimmed_video_path}")

    # 3. config 업데이트 & Inference 호출
    data_root_path = os.path.join(reconstruction_path, trimmed_prefix, "0_real")
    mm_root_path = os.path.join(MM_REPRESENTATION_DIR, trimmed_prefix, "0_real")
    config['data_root'] = data_root_path
    #config['mm_root'] = os.path.join(mm_root_path, "original", "mm_representation.pth")
    try:
        print("[3] Running inference...")
        results = inference(
            _video_dir=trimmed_video_path,
            _mother_dir=reconstruction_path,
            _image_dir=config['data_root'],
            _mm_representation_path=MM_REPRESENTATION_DIR,
            _reconstruction_path=reconstruction_path,
            _config=config,
        )
        print("[3] Inference complete.")
        print("Results:", results)

    except Exception as e:
        print(f"[3] Error during inference: {str(e)}")
    finally:
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleared.")

def create_reconstructed_video_vqvae(input_video: str, output_video: str):
    """
    VQVAE로 복원된 동영상을 생성 (imageio 사용).
    (async -> 일반 함수로 수정)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VectorQuantizedVAE(3, 256, 512)

    # VQVAE 모델 파라미터 로드
    model_ckpt_path = './weights/vqvae/model.pt'  # 필요시 수정
    if not os.path.exists(model_ckpt_path):
        raise FileNotFoundError(f"VQVAE 모델 체크포인트를 찾을 수 없습니다: {model_ckpt_path}")

    state_dict = torch.load(model_ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # 동영상 불러오기
    vc = cv2.VideoCapture(input_video)
    if not vc.isOpened():
        raise ValueError("Error: Unable to open input video for reconstruction.")

    fps = int(vc.get(cv2.CAP_PROP_FPS))
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = imageio.get_writer(output_video, fps=fps, codec="libx264", format="FFMPEG")
    recons_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    frame_count = 0
    while True:
        rval, frame = vc.read()
        if not rval:
            break

        frame_count += 1
        logging.info(f"Reconstructing frame {frame_count}...")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = recons_transform(Image.fromarray(frame_rgb)).unsqueeze(0).to(device)

        with torch.no_grad():
            recons, _, _ = model(img_tensor)
            recons = recons * 0.5 + 0.5  # Denormalize
            recons_np = rearrange(recons.squeeze(0).cpu().numpy(), 'c h w -> h w c') * 255.0
            recons_np = np.clip(recons_np, 0, 255).astype(np.uint8)

        writer.append_data(recons_np)

    vc.release()
    writer.close()

def trim_video(input_video: str, output_video: str, target_width: int = 320, target_height: int = 180):
    """
    입력 동영상을 트림하여 새로운 동영상 생성 (해상도 변경).
    (async -> 일반 함수로 수정)
    """
    vc = cv2.VideoCapture(input_video)
    if not vc.isOpened():
        logging.error("Error: Unable to open video for trimming.")
        return False

    fps = 16
    total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logging.info(f"Original resolution: {original_width}x{original_height}")
    logging.info(f"Target resolution: {target_width}x{target_height}")

    writer = imageio.get_writer(output_video, fps=fps, codec="libx264", format="FFMPEG")
    
    frame_count = 0
    while frame_count < total_frames:
        rval, frame = vc.read()
        if not rval:
            break

        resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        frame_count += 1
        writer.append_data(resized_frame_rgb)

    vc.release()
    writer.close()
    logging.info(f"Trimmed video saved at: {output_video}")
    return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
