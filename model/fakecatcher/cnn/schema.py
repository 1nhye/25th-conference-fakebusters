from pydantic import BaseModel

# GET "/" 엔드포인트
class RootResponse(BaseModel):
    """
    Root 엔드포인트의 출력 스키마
    """
    message: str

# POST "/upload-video" 엔드포인트
class UploadVideoOutput(BaseModel):
    """
    업로드된 비디오의 출력 스키마
    """
    message: str
    file_name: str
    file_path: str
    score: str

class ErrorResponse(BaseModel):
    """
    에러 응답의 출력 스키마
    """
    detail: str