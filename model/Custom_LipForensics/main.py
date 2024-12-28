from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
#from starlette.responses import Response
from pathlib import Path
import shutil
import subprocess
import os

app = FastAPI()

# 비디오 저장 디렉토리 설정
VIDEO_UPLOAD_DIR = Path("uploaded_videos")
VIDEO_UPLOAD_DIR.mkdir(exist_ok=True)  # 디렉터리가 없으면 생성

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    """
    비디오 파일을 업로드받아 서버에 저장하는 API
    """
    # 비디오 파일 MIME 타입 확인
    if file.content_type not in ["video/mp4", "video/avi", "video/mov", "video/mkv"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a valid video file.")

    # 저장 경로 설정
    file_path = VIDEO_UPLOAD_DIR / file.filename

    # 비디오 파일 저장
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    config_path = "/root/25th-conference-fakebusters/Custom_LipForensics/roi_extractor/config.yaml"

    # final_inference_roi.py 실행
    process = subprocess.run(
            ["python", "/root/25th-conference-fakebusters/Custom_LipForensics/final_inference_roi.py", 
             "--video_path", str(file_path), 
             "--config_path", config_path],
            capture_output=True,
            text=True
        )
    
    # print("STDOUT:", process.stdout)
    # print("STDERR:", process.stderr)

    
    if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Error running final_inference.py: {process.stderr}")

    # 결과 처리
    output = process.stdout
    cropped_mouth_video_path = None
    prediction_score = None

    # 결과 파싱
    for line in output.splitlines():
        if "cropped_mouth_video_path" in line:
            cropped_mouth_video_path = line.split(":")[-1].strip()
        elif "Final Prediction" in line:
            prediction_score = float(line.split(":")[-1].strip().split()[0])

    if not cropped_mouth_video_path or prediction_score is None:
         raise HTTPException(status_code=500, detail="Failed to parse results from final_inference.py.")
    
    
    # 처리된 비디오 파일 열기
    video_file = open(cropped_mouth_video_path, "rb")

    # 스트리밍 응답 생성
    response = StreamingResponse(video_file, media_type="video/mp4")
    response.headers["Score"] = f"{prediction_score}"
    return response