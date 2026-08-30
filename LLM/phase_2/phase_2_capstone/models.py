from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    dense_model: str
    sparse_model: str

class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int

class DocumentMetadata(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    upload_time: str

class AskRequest(BaseModel):
    question: str = Field(..., description="Câu hỏi của người dùng")
    doc_id: Optional[str] = Field(None, description="Tùy chọn filter theo doc_id cụ thể")

class SourceSnippet(BaseModel):
    text: str
    doc_name: str
    page_number: Optional[int] = None
    score: float

class AskResponse(BaseModel):
    answer: str
    sources: List[SourceSnippet]
