"""
rag_agent.py — Node RAG Agent: Tìm kiếm context từ Báo cáo Tài chính.

Phase 1 (STUB): Đọc raw_financials từ State (người dùng nhập tay).
Phase 2 (CURRENT): Qdrant Hybrid Search + Cross-Encoder Rerank.

Luồng Phase 2:
  state["question"]
    → search_and_rerank() → Qdrant (Dense + Sparse RRF) → Cross-Encoder
    → list[RetrievedDoc] (Top-4 parent chunks)
    → state["retrieved_context"] (list[str])

Scale-up hint:
  - Thêm filter_payload theo company/year khi hệ thống multi-tenant.
  - Phase 3: Bọc hàm này bằng @observe() decorator Langfuse.
  - Phase 4: Chuyển sang async def để FastAPI không block event loop.
"""

import logging

from langchain_core.language_models import BaseChatModel

from src.agents.state import FinRiskState
from src.config import get_settings
from src.retriever.qdrant_client import ensure_collection
from src.retriever.searcher import RetrievedDoc, search_and_rerank

logger = logging.getLogger(__name__)


def rag_agent_node(state: FinRiskState, llm: BaseChatModel) -> dict:
    """
    Node RAG Agent — Hybrid Search + Rerank để lấy context tài chính.

    Args:
        state: LangGraph State hiện tại.
        llm: BaseChatModel (chưa dùng ở Phase 2, giữ signature nhất quán).
             Phase 3+ sẽ dùng để Contextual Retrieval (gắn prefix context).

    Returns:
        dict với key "retrieved_context": list[str] các đoạn văn liên quan.
    """
    session_id = state.get("session_id", "unknown")
    question = state.get("question", "")
    company = state.get("company_name", "")

    logger.info("RAG Agent invoked for session=%s, question='%s...'", session_id, question[:60])

    if not question:
        logger.warning("RAG Agent: question rỗng, trả về context trống.")
        return {"retrieved_context": []}

    # Tạo collection nếu chưa tồn tại (idempotent, an toàn khi gọi nhiều lần)
    try:
        ensure_collection()
    except Exception as e:
        logger.warning("RAG Agent: Không kết nối được Qdrant (%s). Fallback về raw_financials.", e)
        return _fallback_to_raw(state)

    # Xây dựng filter nếu có company name (tránh lấy dữ liệu của công ty khác)
    filter_payload = None
    if company:
        filter_payload = {"company": company}

    # Hybrid Search + Cross-Encoder Rerank
    try:
        docs: list[RetrievedDoc] = search_and_rerank(
            query=question,
            filter_payload=filter_payload,
        )
    except Exception as e:
        logger.error("RAG Agent: search_and_rerank thất bại: %s. Fallback về raw_financials.", e)
        return _fallback_to_raw(state)

    if not docs:
        logger.warning(
            "RAG Agent: Không tìm thấy tài liệu trong Qdrant cho query='%s'. "
            "Kiểm tra: (1) Qdrant đang chạy? (2) Đã index PDF chưa? (3) Đúng collection?",
            question[:60],
        )
        # Fallback: dùng raw_financials để pipeline không bị dừng hoàn toàn
        return _fallback_to_raw(state)

    # Format docs thành list[str] để downstream agents dùng
    context = []
    for doc in docs:
        meta = doc.metadata
        source_info = (
            f"[Nguồn: {meta.get('company', 'N/A')} | "
            f"Năm: {meta.get('year', 'N/A')} | "
            f"Loại: {meta.get('doc_type', 'N/A')} | "
            f"Trang: {meta.get('page', 'N/A')} | "
            f"Relevance: {doc.score:.2f}]"
        )
        context.append(f"{source_info}\n{doc.page_content}")

    logger.info("RAG Agent: Retrieved %d docs from Qdrant for session=%s", len(context), session_id)
    return {"retrieved_context": context}


def _fallback_to_raw(state: FinRiskState) -> dict:
    """
    Fallback khi Qdrant không có dữ liệu hoặc không kết nối được.
    Chuyển raw_financials (nhập tay) thành text context.
    Giữ pipeline hoạt động ngay cả khi chưa có PDF nào được index.
    """
    raw = state.get("raw_financials", {})
    company = state.get("company_name", "Doanh nghiệp")

    if not raw:
        logger.warning("Fallback: raw_financials cũng rỗng. Context hoàn toàn trống.")
        return {"retrieved_context": []}

    context = [
        f"[Dữ liệu nhập thủ công — chưa có PDF trong Qdrant]\n"
        f"Dữ liệu tài chính của {company}:\n"
        + "\n".join(
            f"  {k}: {v:,.0f} VND" if isinstance(v, (int, float)) else f"  {k}: {v}"
            for k, v in raw.items()
        )
    ]
    return {"retrieved_context": context}
