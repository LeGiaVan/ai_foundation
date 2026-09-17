from typing import Optional, List
from pydantic import BaseModel, Field

class ResearchResult(BaseModel):
    """Cấu trúc dữ liệu đầu ra chuẩn (Structured Output) theo yêu cầu Capstone."""
    answer: str = Field(..., description="Câu trả lời đầy đủ, chi tiết và được định dạng đẹp mắt.")
    sources: List[str] = Field(default_factory=list, description="Danh sách các nguồn trích dẫn từ tài liệu RAG hoặc Web Search.")
    agents_used: List[str] = Field(default_factory=list, description="Danh sách các Worker Agent đã tham gia giải quyết tác vụ.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Độ tự tin của câu trả lời từ 0.0 đến 1.0.")

class ResearchRequest(BaseModel):
    user_id: str = Field(default="user_default", description="Mã định danh người dùng để cá nhân hóa và quản lý Memory.")
    question: str = Field(..., description="Câu hỏi hoặc yêu cầu nghiên cứu.")

class ResearchJobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class PendingAction(BaseModel):
    tool: str
    input: dict
    reason: Optional[str] = None

class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "running" | "waiting_approval" | "completed" | "error"
    pending_action: Optional[PendingAction] = None
    result: Optional[ResearchResult] = None
    error: Optional[str] = None

class ApproveRequest(BaseModel):
    approve: bool = Field(default=True, description="True để phê duyệt chạy tiếp, False để từ chối hành động.")
    feedback: Optional[str] = Field(default=None, description="Lời nhắn hoặc chỉ dẫn bổ sung của người dùng.")
