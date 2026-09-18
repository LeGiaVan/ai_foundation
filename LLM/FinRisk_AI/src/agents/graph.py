"""
graph.py — Lắp ráp toàn bộ LangGraph StateGraph.

Đây là file "dây điện" kết nối tất cả các Node và Edge.
Mọi thứ khác (State, Nodes, Tools) là module độc lập — graph.py là nơi duy nhất
biết về toàn bộ kiến trúc pipeline.

Sơ đồ luồng (xem project_proposal_fintech.md để có diagram đầy đủ):

    START
      │
      ▼
  [supervisor] ──conditional──► [rag_agent]  (Phase 2)
      │                              │
      │◄─────────────────────────────┘
      │
      ├──► [math_agent] ──► [supervisor]
      │
      ├──► [risk_agent] ──► [supervisor]
      │
      └──► [hitl_node]  ──► (chờ Human input) ──► [supervisor]
      │
      └──► END ──► [report_node] ──► FINAL

Checkpointer (SQLite ở Phase 1, PostgreSQL ở Phase 4+):
  Mỗi bước được lưu vào DB → Hệ thống có thể resume sau khi HITL phê duyệt.

Scale-up hint:
  - Phase 4: Đổi MemorySaver → AsyncPostgresSaver khi deploy lên VPS.
  - Phase 3: Wrap compile() với LangfuseCallbackHandler để auto-trace.
"""

import logging
import functools
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from src.agents.state import FinRiskState
from src.agents.supervisor import supervisor_node, route_after_supervisor
from src.agents.rag_agent import rag_agent_node
from src.agents.math_agent import math_agent_node
from src.agents.risk_agent import risk_agent_node
from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ── LLM Factory ─────────────────────────────────────────────────────────────

def _build_llm() -> BaseChatModel:
    """
    Khởi tạo LLM dựa trên config.
    Pattern Factory: Dễ switch provider mà không sửa code Agent.

    Phase học  : groq (miễn phí, dùng model openai/gpt-oss-120b)
    Phase production: anthropic (Claude Haiku) hoặc openai
    """
    if settings.llm_provider == "groq":
        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    elif settings.llm_provider == "anthropic":
        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    elif settings.llm_provider == "openai":
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


# ── HITL Node ────────────────────────────────────────────────────────────────

def hitl_node(state: FinRiskState) -> dict:
    """
    Human-in-the-Loop Node — tạm dừng pipeline chờ Giám đốc Tín dụng phê duyệt.

    Cách hoạt động:
      1. Node này được gọi khi risk_assessment.requires_human_approval = True
      2. Gọi interrupt() → LangGraph lưu State vào Checkpointer → trả về cho API
      3. API gửi thông báo cho người dùng (qua webhook/email)
      4. Người dùng vào Web UI, nhập quyết định → API gọi graph.invoke() với state mới
      5. Pipeline tiếp tục từ điểm dừng

    Scale-up hint:
      - Phase 3: Thêm webhook notification khi interrupt được kích hoạt.
      - Phase 4: Lưu pending approvals vào PostgreSQL để track SLA.
    """
    logger.info(
        "HITL triggered for session=%s. Reason: %s",
        state.get("session_id"),
        state.get("hitl_reason"),
    )

    # interrupt() = điểm dừng của LangGraph. Không dùng Python input() trong production!
    # Payload này được gửi về cho API để hiển thị cho người dùng.
    human_input = interrupt({
        "message": "Yêu cầu phê duyệt từ Giám đốc Tín dụng",
        "reason": state.get("hitl_reason"),
        "risk_score": state.get("risk_assessment", {}).get("risk_score"),
        "risk_level": state.get("risk_assessment", {}).get("risk_level"),
        "red_flags": state.get("risk_assessment", {}).get("red_flags", []),
        "company": state.get("company_name"),
        "instruction": "Nhập 'APPROVED' hoặc 'REJECTED' và comment (tùy chọn)",
    })

    # Sau khi Human resume, human_input chứa quyết định
    decision = human_input.get("decision", "").upper()
    comment = human_input.get("comment", "")

    logger.info("HITL decision received: decision=%s comment=%s", decision, comment)

    return {
        "human_decision": decision,
        "human_comment": comment,
        "requires_human_approval": False,  # Reset flag sau khi đã có quyết định
    }


def should_trigger_hitl(state: FinRiskState) -> Literal["hitl_node", "supervisor"]:
    """Conditional edge: Sau risk_agent, kiểm tra có cần HITL không."""
    if state.get("requires_human_approval", False):
        return "hitl_node"
    return "supervisor"


# ── Report Node ──────────────────────────────────────────────────────────────

def report_node(state: FinRiskState, llm: BaseChatModel) -> dict:
    """
    Node cuối cùng — tổng hợp báo cáo thẩm định hoàn chỉnh.
    LLM tổng hợp ngôn ngữ đẹp từ dữ liệu có cấu trúc trong State.
    """
    metrics = state.get("financial_metrics", {})
    risk = state.get("risk_assessment", {})
    human_decision = state.get("human_decision")

    # Xác định trạng thái phán quyết cuối
    if human_decision:
        final_verdict = f"PHÁN QUYẾT CUỐI (Giám đốc Tín dụng): {human_decision}"
        if state.get("human_comment"):
            final_verdict += f"\nGhi chú: {state['human_comment']}"
    else:
        final_verdict = f"PHÁN QUYẾT TỰ ĐỘNG: {risk.get('recommendation', 'N/A')}"

    prompt = f"""
Hãy viết báo cáo thẩm định tín dụng chuyên nghiệp cho doanh nghiệp: {state.get("company_name")}.

## Dữ liệu đầu vào:
- Altman Z-Score: {metrics.get("altman_z_score")}
- DSCR: {metrics.get("dscr")}
- Debt-to-Equity: {metrics.get("debt_to_equity")}
- Quick Ratio: {metrics.get("quick_ratio")}

## Kết quả đánh giá rủi ro:
- Risk Score: {risk.get("risk_score")}/100
- Risk Level: {risk.get("risk_level")}
- Red Flags: {risk.get("red_flags", [])}
- Đánh giá 5C: {risk.get("five_c_summary", {})}

## {final_verdict}

Viết báo cáo ngắn gọn, chuyên nghiệp, có cấu trúc rõ ràng bằng tiếng Việt.
Bắt buộc có: Tóm tắt, Phân tích 5C, Phán quyết & Điều kiện cho vay (nếu approve).
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    logger.info("Final report generated for session=%s", state.get("session_id"))

    return {"final_report": response.content}


# ── Graph Assembly ────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    """
    Lắp ráp StateGraph hoàn chỉnh.

    Args:
        checkpointer: LangGraph checkpointer (MemorySaver mặc định cho Phase 1,
                      AsyncPostgresSaver cho Phase 4+ production)

    Returns:
        Compiled LangGraph app (có thể gọi .invoke() hoặc .stream())
    """
    llm = _build_llm()

    # functools.partial để inject llm vào node functions (vì LangGraph node chỉ nhận state)
    _supervisor = functools.partial(supervisor_node, llm=llm)
    _rag = functools.partial(rag_agent_node, llm=llm)
    _math = functools.partial(math_agent_node, llm=llm)
    _risk = functools.partial(risk_agent_node, llm=llm)
    _report = functools.partial(report_node, llm=llm)

    # Khởi tạo StateGraph với State schema
    graph = StateGraph(FinRiskState)

    # Đăng ký tất cả Nodes
    graph.add_node("supervisor", _supervisor)
    graph.add_node("rag_agent", _rag)
    graph.add_node("math_agent", _math)
    graph.add_node("risk_agent", _risk)
    graph.add_node("hitl_node", hitl_node)       # HITL: không cần LLM
    graph.add_node("report_node", _report)

    # Đăng ký Edges
    graph.add_edge(START, "supervisor")           # Luôn bắt đầu từ Supervisor

    # Supervisor → Agents (Conditional Edge)
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,                   # Đọc state.next_agent
        {
            "rag_agent": "rag_agent",
            "math_agent": "math_agent",
            "risk_agent": "risk_agent",
            "END": "report_node",                 # END → sinh báo cáo cuối
        },
    )

    # Agents → Supervisor (quay lại để Supervisor quyết định bước tiếp)
    graph.add_edge("rag_agent", "supervisor")
    graph.add_edge("math_agent", "supervisor")

    # Risk Agent → HITL check (Conditional)
    graph.add_conditional_edges(
        "risk_agent",
        should_trigger_hitl,
        {"hitl_node": "hitl_node", "supervisor": "supervisor"},
    )

    # HITL → Supervisor (sau khi Human quyết định, quay về Supervisor để tiếp tục)
    graph.add_edge("hitl_node", "supervisor")

    # Report → END
    graph.add_edge("report_node", END)

    # Compile với Checkpointer (bắt buộc cho HITL interrupt/resume)
    if checkpointer is None:
        checkpointer = MemorySaver()              # In-memory cho dev/testing

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("FinRisk AI graph compiled successfully.")
    return compiled


# ── Singleton cho toàn bộ application ───────────────────────────────────────

_graph_instance = None

def get_graph():
    """
    Singleton — chỉ build graph một lần, tái sử dụng.
    Phase 4: Đổi checkpointer từ MemorySaver → AsyncPostgresSaver.
    """
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = build_graph()
    return _graph_instance
