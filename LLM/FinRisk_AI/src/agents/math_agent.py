"""
math_agent.py — Node Math Agent: Tính toán chỉ số tài chính.

Đây là node QUAN TRỌNG NHẤT về độ chính xác:
  - LLM KHÔNG tự tính, chỉ quyết định tool nào cần gọi và với tham số nào.
  - Mọi phép tính thực hiện bởi Python tools trong financial_tools.py.
  - Pattern: ReAct loop (Reason → Action → Observe → Reason...)

Flow của Math Agent (ReAct):
  1. LLM đọc context → Quyết định gọi tool nào
  2. ToolNode execute tool → Trả kết quả
  3. LLM đọc kết quả → Quyết định gọi tool tiếp hoặc summarize
  4. Khi đủ chỉ số → Tổng hợp vào financial_metrics

Scale-up hint:
  - Thêm chỉ số mới: Tạo @tool trong financial_tools.py + thêm vào FINANCIAL_TOOLS.
  - Phase 3: Wrap toàn bộ node trong @observe() của Langfuse để track token cost.
"""

import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.language_models import BaseChatModel

from src.agents.state import FinRiskState, FinancialMetrics
from src.tools.financial_tools import FINANCIAL_TOOLS
from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

MATH_AGENT_SYSTEM_PROMPT = """
Bạn là chuyên gia phân tích tài chính. Nhiệm vụ của bạn là tính toán chính xác các chỉ số tài chính.

QUAN TRỌNG: LUÔN sử dụng tools Python để tính toán. KHÔNG TỰ nhẩm tính.

Các tools có sẵn:
- calculate_altman_z_score: Dự báo nguy cơ phá sản
- calculate_dscr: Hệ số khả năng trả nợ
- calculate_debt_to_equity: Đòn bẩy tài chính  
- calculate_quick_ratio: Thanh khoản tức thì

Lấy số liệu từ context được cung cấp và gọi đúng tool với đúng tham số.
Sau khi tính xong, tổng hợp kết quả dưới dạng JSON với key "financial_metrics".
""".strip()


def math_agent_node(state: FinRiskState, llm: BaseChatModel) -> dict:
    """
    Node Math Agent — gọi Python Tools để tính chỉ số tài chính.

    Dùng ReAct pattern: LLM quyết định tool → ToolNode chạy → LLM đọc kết quả.
    """
    logger.info("Math Agent invoked for session=%s", state.get("session_id"))

    # Bind tools vào LLM (LLM sẽ biết tools nào có sẵn để gọi)
    llm_with_tools = llm.bind_tools(FINANCIAL_TOOLS)

    context = "\n\n".join(state.get("retrieved_context", []))
    messages = [
        SystemMessage(content=MATH_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Dữ liệu tài chính doanh nghiệp:
{context}

Câu hỏi phân tích: {state.get("question", "Phân tích toàn diện")}

Hãy tính toán đầy đủ các chỉ số tài chính có thể tính được từ dữ liệu trên.
""".strip()),
    ]

    # ReAct loop — tối đa 5 vòng tool calls
    tool_map = {t.name: t for t in FINANCIAL_TOOLS}
    metrics_results = {}

    for step in range(5):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # Nếu LLM không gọi tool nữa → đã xong
        if not response.tool_calls:
            logger.info("Math Agent completed in %d steps", step + 1)
            break

        # Thực thi từng tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            if tool_name not in tool_map:
                result = {"error": f"Tool '{tool_name}' không tồn tại"}
            else:
                try:
                    result = tool_map[tool_name].invoke(tool_args)
                    metrics_results[tool_name] = result
                    logger.debug("Tool %s executed: result=%s", tool_name, result)
                except Exception as e:
                    result = {"error": str(e)}
                    logger.error("Tool %s failed: %s", tool_name, e)

            messages.append(ToolMessage(
                content=json.dumps(result, ensure_ascii=False),
                tool_call_id=tool_id,
            ))

    # Map kết quả tools vào FinancialMetrics TypedDict
    financial_metrics: FinancialMetrics = {
        "altman_z_score": (metrics_results.get("calculate_altman_z_score") or {}).get("z_score"),
        "dscr": (metrics_results.get("calculate_dscr") or {}).get("dscr"),
        "debt_to_equity": (metrics_results.get("calculate_debt_to_equity") or {}).get("debt_to_equity"),
        "quick_ratio": (metrics_results.get("calculate_quick_ratio") or {}).get("quick_ratio"),
        "gross_margin": None,            # Phase 2: Tính từ P&L trích xuất từ BCTC
        "net_cash_from_operations": None, # Phase 2: Tính từ Báo cáo LCTT
    }

    return {
        "financial_metrics": financial_metrics,
        "messages": messages[2:],  # Chỉ lưu messages mới (bỏ system + human gốc)
    }
