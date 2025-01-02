from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import uvicorn
import asyncio
import os
import torch
import cv2
from einops import rearrange
from torchvision import transforms
from PIL import Image
import numpy as np
import imageio
import cv2
import glob

# ------------------------
# 사용자 정의 모듈 import
# ------------------------
from inference import inference
from models import VectorQuantizedVAE

# ------------------------
# 전역 상수 정의
# ------------------------
UPLOAD_DIR = '/root/frame_diffusion_detection/MM_Det/inference/uploads'
MM_REPRESENTATION_DIR = '/root/frame_diffusion_detection/MM_Det/inference/mm_representation'
RECONSTRUCTION_DIR = '/root/frame_diffusion_detection/MM_Det/inference/reconstruction'

os.makedirs(UPLOAD_DIR, exist_ok=True)            # 업로드 경로 생성
os.makedirs(MM_REPRESENTATION_DIR, exist_ok=True) # mm_representation 경로 생성
os.makedirs(RECONSTRUCTION_DIR, exist_ok=True)    # reconstruction 경로 생성

app = FastAPI()

# --------------------------------------------------
# config 딕셔너리 (필요에 따라 경로 또는 옵션 수정)
# --------------------------------------------------
config = {
    'data_root': '',  # 아래 run_inference에서 재설정됨
    'ckpt': '/root/frame_diffusion_detection/MM_Det/weights/MM-Det/current_model.pth',
    'lmm_ckpt': 'sparklexfantasy/llava-1.5-7b-rfrd',
    'lmm_base': None,
    'st_ckpt': 'weights/ViT/vit_base_r50_s16_224.orig_in21k/jx_vit_base_resnet50_224_in21k-6f7c7740.pth',
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
    'cache_mm': True,  # 아래 run_inference에서 mm_root 설정 시 사용
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


# ------------------------------------
# Inference를 비동기로 실행하는 함수
# ------------------------------------
import cv2
import glob

async def run_inference(video_path: str, reconstruction_path: str, config: dict):
    """Inference 함수 호출 및 결과 반환 (VQVAE 복원 -> 트림 -> Inference -> 동영상 저장)"""
    prefix = os.path.splitext(os.path.basename(video_path))[0]
    reconstructed_video_path = os.path.join(RECONSTRUCTION_DIR, f"{prefix}_reconstructed.mp4")

    # 1. VQVAE 복원된 동영상 생성
    print("Starting VQVAE reconstruction...")
    await create_reconstructed_video_vqvae(video_path, reconstructed_video_path)

    # 2. 트림된 비디오 생성
    trimmed_prefix = f"{prefix}_trimmed"
    trimmed_video_path = os.path.join(UPLOAD_DIR, f"{trimmed_prefix}.mp4")
    print("Starting video trimming...")

    # 트림 작업
    trim_success = await trim_video(reconstructed_video_path, trimmed_video_path)
    if not trim_success:
        return {"message": "Error: Unable to trim video", "status": "error"}

    # config 업데이트
    data_root_path = os.path.join(reconstruction_path, trimmed_prefix, "0_real")
    mm_root_path = os.path.join(MM_REPRESENTATION_DIR, trimmed_prefix, "0_real")
    config['data_root'] = data_root_path
    config['mm_root'] = os.path.join(mm_root_path, "original", "mm_representation.pth")
    print(f"Updated config: data_root={config['data_root']}, mm_root={config['mm_root']}")

    try:
        # 3. Inference 실행
        results = inference(
            _video_dir=trimmed_video_path,
            _mother_dir=reconstruction_path,
            _image_dir=config['data_root'],
            _mm_representation_path=MM_REPRESENTATION_DIR,
            _reconstruction_path=reconstruction_path,
            _config=config,
        )
        return {
            "message": "Inference complete",
            "status": "success",
            "results": results,
            "reconstructed_video_path": reconstructed_video_path,
        }
    except Exception as e:
        return {"message": f"Error during inference: {str(e)}", "status": "error"}
    finally:
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleared.")


async def create_reconstructed_video_vqvae(input_video: str, output_video: str):
    """VQVAE로 복원된 동영상을 생성 (imageio 사용)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # VQVAE 모델 로드
    model = VectorQuantizedVAE(3, 256, 512)
    state_dict = torch.load('/root/frame_diffusion_detection/MM_Det/weights/vqvae/model.pt', map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    # 동영상 처리
    vc = cv2.VideoCapture(input_video)
    if not vc.isOpened():
        raise ValueError("Error: Unable to open input video for reconstruction.")

    fps = int(vc.get(cv2.CAP_PROP_FPS))
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # imageio로 동영상 작성기 초기화
    writer = imageio.get_writer(
        output_video, fps=fps, codec="libx264", format="FFMPEG"
    )

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
        print(f"Reconstructing frame {frame_count}...")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = recons_transform(Image.fromarray(frame_rgb)).unsqueeze(0).to(device)

        with torch.no_grad():
            recons, _, _ = model(img_tensor)
            recons = recons * 0.5 + 0.5  # Denormalize
            recons_np = rearrange(recons.squeeze(0).cpu().numpy(), 'c h w -> h w c') * 255.0
            recons_np = np.clip(recons_np, 0, 255).astype(np.uint8)

        # Write reconstructed frame to the video using imageio
        writer.append_data(recons_np)

    vc.release()
    writer.close()
    print(f"Reconstructed video saved at: {output_video}")

async def trim_video(input_video: str, output_video: str, target_width: int = 320, target_height: int = 180):
    """입력 동영상을 트림하여 새로운 동영상 생성 (해상도 낮춤)"""
    vc = cv2.VideoCapture(input_video)
    if not vc.isOpened():
        print("Error: Unable to open video for trimming.")
        return False

    fps = 16
    total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Original resolution: {original_width}x{original_height}")
    print(f"Target resolution: {target_width}x{target_height}")

    writer = imageio.get_writer(
        output_video, fps=fps, codec="libx264", format="FFMPEG"
    )
    
    frame_count = 0
    while frame_count < total_frames:  # 16개의 프레임만 유지
        rval, frame = vc.read()
        if not rval:
            break

        # 프레임 해상도 조정
        resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # 프레임 저장
        frame_count += 1
        writer.append_data(resized_frame_rgb)

    vc.release()
    writer.close()
    print(f"Trimmed video with resized frames saved at: {output_video}")
    return True



# ------------------------------------------
# 업로드된 비디오 파일 처리, 스트리밍 + 추론
# ------------------------------------------
@app.get("/how_are_you/")
async def how_are_you():
    """
    간단한 테스트 엔드포인트
    서버가 배포된 뒤, http://<host>:8000/how_are_you/ 로 접속해보세요.
    """
    return {"message": "I am fine. Thank you!"}
from fastapi.responses import JSONResponse
import json

@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    video_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(video_path, "wb") as f:
        f.write(await file.read())

    # Inference 실행
    inference_result = await run_inference(video_path, RECONSTRUCTION_DIR, config)
    reconstructed_video_path = inference_result.get("reconstructed_video_path")
    inference_data = inference_result.get("results", {})

    if reconstructed_video_path:
        # Inference 결과를 JSON 형식으로 헤더에 포함
        inference_json = json.dumps(inference_data)
        headers = {
            "X-Inference-Result": inference_json
        }
        print(inference_json)
        return StreamingResponse(
            open(reconstructed_video_path, "rb"),
            media_type="video/mp4",
            headers=headers
        )
    else:
        # 오류 발생 시 JSON 응답 반환
        return JSONResponse(
            content={"message": "Error processing video", "details": inference_result},
            status_code=500
        )

# ----------------------------
# python main.py 로 서버 실행
# ----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

