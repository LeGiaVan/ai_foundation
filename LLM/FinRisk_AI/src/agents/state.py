"""
state.py — Định nghĩa State dùng chung trong toàn bộ LangGraph StateGraph.

State là "bộ nhớ" được truyền qua từng Node. Mỗi Node đọc State,
thực hiện công việc của mình, rồi trả về phần State được cập nhật.

Nguyên tắc thiết kế State tốt:
  - Đủ để mỗi Node tự đọc mà không cần gọi sang Node khác.
  - Không lưu dữ liệu cực lớn (bytes của file) — chỉ lưu metadata/path.
  - Tường minh: mỗi field có docstring rõ ai ghi, ai đọc.

Scale-up hint:
  - Phase 2: Thêm retrieved_context (list[str]) khi tích hợp Qdrant RAG.
  - Phase 3: Thêm langfuse_trace_id để link trace vào report cuối.
"""

from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class FinancialMetrics(TypedDict):
    """Kết quả từ Math Agent — các chỉ số tài chính được tính bằng Python Tools."""
    altman_z_score: float | None
    dscr: float | None                        # Debt Service Coverage Ratio
    debt_to_equity: float | None
    quick_ratio: float | None
    gross_margin: float | None                # Biên lợi nhuận gộp
    net_cash_from_operations: float | None    # Dòng tiền thuần từ HĐKD


class RiskAssessment(TypedDict):
    """Kết quả từ Risk Agent — phán quyết tín dụng theo Khung 5C."""
    risk_score: float                         # 0-100, càng cao càng rủi ro
    risk_level: str                           # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    recommendation: str                       # "APPROVE" | "REVIEW" | "REJECT"
    red_flags: list[str]                      # Danh sách cảnh báo tìm được
    five_c_summary: dict[str, str]            # Đánh giá từng chữ C: Capacity, Capital, ...


class FinRiskState(TypedDict):
    """
    State chính của toàn bộ pipeline FinRisk AI.

    Lifecycle:
        [User Input] → Supervisor → (RAGAgent | MathAgent | RiskAgent) → HITL? → Output
    """
    # ── Input ────────────────────────────────────────────────────────────
    session_id: str           # ID phiên — dùng cho Checkpointer để resume
    user_id: str              # ID người dùng — dùng cho Cost tracking (Langfuse Phase 3)
    question: str             # Câu hỏi / yêu cầu từ chuyên viên tín dụng
    company_name: str         # Tên doanh nghiệp đang thẩm định
    document_paths: list[str] # Đường dẫn các file BCTC đã upload (Phase 2: tải vào Qdrant)

    # ── Routing (do Supervisor ghi) ──────────────────────────────────────
    next_agent: str           # Node tiếp theo: "rag_agent"|"math_agent"|"risk_agent"|"END"
    iteration: int            # Đếm số vòng lặp, chặn vô hạn khi > max_agent_iterations

    # ── Dữ liệu trung gian ───────────────────────────────────────────────
    # Phase 2: RAG Agent sẽ ghi vào đây sau khi search Qdrant
    retrieved_context: list[str]              # Các đoạn văn bản tìm được từ BCTC

    # Phase 1: Math Agent ghi kết quả tính toán vào đây
    raw_financials: dict[str, Any]            # Số liệu thô từ BCTC (nhập tay ở Phase 1)
    financial_metrics: FinancialMetrics       # Các chỉ số tài chính đã tính

    # Phase 1: Risk Agent ghi phán quyết vào đây
    risk_assessment: RiskAssessment

    # ── Human-in-the-Loop ────────────────────────────────────────────────
    requires_human_approval: bool             # True khi risk_score vượt ngưỡng
    hitl_reason: str                          # Giải thích tại sao cần Human review
    human_decision: str | None               # "APPROVED" | "REJECTED" | None (chưa quyết)
    human_comment: str | None               # Ghi chú của Giám đốc Tín dụng

    # ── Output ───────────────────────────────────────────────────────────
    final_report: str                         # Báo cáo thẩm định hoàn chỉnh

    # ── Conversation history (dùng add_messages để tự động merge) ────────
    messages: Annotated[list[BaseMessage], add_messages]
