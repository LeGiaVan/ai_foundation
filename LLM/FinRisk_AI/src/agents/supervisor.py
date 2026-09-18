"""
supervisor.py — Node Supervisor: Bộ não điều phối của toàn bộ pipeline.

Supervisor là node đầu tiên được gọi và cũng được gọi lại sau mỗi Agent.
Nhiệm vụ: Đọc State hiện tại → Quyết định Agent nào chạy tiếp theo.

Routing logic (có thể điều chỉnh prompt để thay đổi hành vi):
  1. Nếu chưa có retrieved_context  → dispatch sang rag_agent (Phase 2)
  2. Nếu chưa có financial_metrics  → dispatch sang math_agent
  3. Nếu chưa có risk_assessment    → dispatch sang risk_agent
  4. Nếu đã có tất cả              → "END" (sinh báo cáo cuối)
  5. Nếu vòng lặp vượt ngưỡng      → "END" với cảnh báo (an toàn)

Scale-up hint:
  - Thêm Agent mới: thêm điều kiện routing ở đây + đăng ký node trong graph.py
  - Phase 3: Thêm Langfuse trace context để đo latency của từng routing step.
"""

import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

from src.agents.state import FinRiskState
from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Prompt cho Supervisor — lưu ở đây để dễ chuyển lên Langfuse Registry (Phase 3)
SUPERVISOR_SYSTEM_PROMPT = """
Bạn là Supervisor điều phối quy trình thẩm định tín dụng doanh nghiệp.
Nhiệm vụ: Phân tích State hiện tại và quyết định bước tiếp theo.

Các Agent có sẵn:
- "rag_agent": Tìm kiếm thông tin từ Báo cáo Tài chính (BCTC). Gọi TRƯỚC nếu chưa có context.
- "math_agent": Tính toán chỉ số tài chính (Z-Score, DSCR, D/E, Quick Ratio).
- "risk_agent": Đánh giá rủi ro tổng thể và đưa ra phán quyết tín dụng.
- "END": Kết thúc pipeline và sinh báo cáo thẩm định.

Trả về JSON: {"next": "<agent_name>", "reason": "<lý do ngắn gọn>"}
""".strip()


def supervisor_node(state: FinRiskState, llm: BaseChatModel) -> dict:
    """
    Node Supervisor — quyết định agent tiếp theo dựa trên State.

    Args:
        state: State hiện tại của pipeline
        llm: LLM instance (được inject từ graph.py)

    Returns:
        dict cập nhật State: {"next_agent": ..., "iteration": ...}
    """
    iteration = state.get("iteration", 0)

    # Chặn vòng lặp vô hạn — safety net quan trọng
    if iteration >= settings.max_agent_iterations:
        logger.warning(
            "Max iterations (%d) reached for session=%s. Forcing END.",
            settings.max_agent_iterations,
            state.get("session_id"),
        )
        return {"next_agent": "END", "iteration": iteration + 1}

    # Routing thủ công (deterministic) — không cần LLM cho Phase 1
    # Ưu điểm: Nhanh hơn, rẻ hơn, dễ debug hơn.
    # Phase 2+: Có thể chuyển sang LLM-based routing nếu logic phức tạp hơn.

    # Bước 1: Cần context từ BCTC chưa? (Phase 2 sẽ kích hoạt rag_agent)
    # Tạm thời bỏ qua ở Phase 1, context được cung cấp thủ công qua raw_financials
    # if not state.get("retrieved_context"):
    #     return {"next_agent": "rag_agent", "iteration": iteration + 1}

    # Bước 2: Đã tính chỉ số tài chính chưa?
    metrics = state.get("financial_metrics") or {}
    if not metrics.get("altman_z_score") and not metrics.get("dscr"):
        logger.info("Routing → math_agent (no financial_metrics yet)")
        return {"next_agent": "math_agent", "iteration": iteration + 1}

    # Bước 3: Đã có phán quyết rủi ro chưa?
    risk = state.get("risk_assessment") or {}
    if not risk.get("recommendation"):
        logger.info("Routing → risk_agent (no risk_assessment yet)")
        return {"next_agent": "risk_agent", "iteration": iteration + 1}

    # Bước 4: Hoàn thành pipeline
    logger.info("Routing → END (pipeline complete)")
    return {"next_agent": "END", "iteration": iteration + 1}


def route_after_supervisor(state: FinRiskState) -> str:
    """
    Conditional Edge function: Đọc state.next_agent và trả về tên node đích.
    Được gọi bởi LangGraph để quyết định edge nào đi tiếp.
    """
    return state["next_agent"]
