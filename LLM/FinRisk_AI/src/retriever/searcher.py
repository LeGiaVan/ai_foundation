"""
searcher.py — Hybrid Search (Dense + Sparse) và Cross-Encoder Reranking.

Pipeline:
  query text
    → embed_dense(query) + embed_sparse(query)
    → Qdrant Hybrid Search với RRF (Reciprocal Rank Fusion) merger
    → Top-K candidates (default 20)
    → Cross-Encoder rerank (BGE-Reranker-v2-m3) → score > threshold
    → Top-N results (default 4) trả về cho LLM

Tại sao Cross-Encoder sau Hybrid Search?
  - Hybrid Search nhanh nhưng không hoàn hảo (vector approximate).
  - Cross-Encoder chấm điểm lại (query, doc) cùng nhau → chính xác hơn nhiều.
  - Trade-off: chậm hơn, nhưng N nhỏ (4) nên chấp nhận được.

Scale-up hint:
  - Khi cần filter theo công ty/năm: thêm qdrant_filter param vào search_hybrid().
  - LLMLingua: Nén parent_text từ 1000 token → 300 token trước khi gửi LLM.
"""

import logging
from dataclasses import dataclass, field
from functools import lru_cache

from qdrant_client.http.models import (
    FusionQuery,
    Prefetch,
    SparseVector,
)
from sentence_transformers import CrossEncoder

from src.config import get_settings
from src.retriever.embedder import embed_dense, embed_sparse
from src.retriever.qdrant_client import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    get_qdrant_client,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDoc:
    """Kết quả search — cấu trúc này được truyền vào LLM context."""
    page_content: str          # parent_text — ngữ cảnh đầy đủ để LLM đọc
    child_text: str            # child_text — đoạn nhỏ đã match (dùng để debug)
    score: float               # Cross-Encoder score (0.0–1.0)
    metadata: dict = field(default_factory=dict)  # company, year, doc_type, page


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    """
    Singleton Cross-Encoder reranker.
    Lần đầu tải model ~1.1GB từ HuggingFace Hub.
    """
    settings = get_settings()
    logger.info("Loading reranker model: %s (first run: ~1.1GB download)", settings.reranker_model)
    return CrossEncoder(settings.reranker_model)


def search_hybrid(
    query: str,
    collection_name: str | None = None,
    top_k: int | None = None,
    filter_payload: dict | None = None,
) -> list[dict]:
    """
    Hybrid Search: Dense (cosine) + Sparse (BM25) → RRF fusion trong Qdrant.

    Args:
        query: Câu hỏi của người dùng.
        collection_name: Override collection. None → dùng settings.
        top_k: Số candidates trả về. None → dùng settings.retrieval_top_k.
        filter_payload: Qdrant filter dict (ví dụ: {"company": "Vinamilk"}).
                        None = không filter.

    Returns:
        List payload dict từ Qdrant (chứa child_text, parent_text, metadata).
    """
    settings = get_settings()
    collection = collection_name or settings.qdrant_collection
    k = top_k or settings.retrieval_top_k
    client = get_qdrant_client()

    # Embed query với cả 2 model
    dense_vec = embed_dense([query])[0]
    sparse_dict = embed_sparse([query])[0]

    # Qdrant Query API: Prefetch Dense + Prefetch Sparse → RRF merge
    results = client.query_points(
        collection_name=collection,
        prefetch=[
            Prefetch(
                query=dense_vec,
                using=DENSE_VECTOR_NAME,
                limit=k,
            ),
            Prefetch(
                query=SparseVector(
                    indices=sparse_dict["indices"],
                    values=sparse_dict["values"],
                ),
                using=SPARSE_VECTOR_NAME,
                limit=k,
            ),
        ],
        query=FusionQuery(fusion="rrf"),  # Reciprocal Rank Fusion
        limit=k,
        with_payload=True,
    )

    payloads = [point.payload for point in results.points if point.payload]
    logger.info("Hybrid search returned %d candidates for query='%s...'", len(payloads), query[:50])
    return payloads


def rerank(
    query: str,
    candidates: list[dict],
    top_n: int | None = None,
    threshold: float | None = None,
) -> list[RetrievedDoc]:
    """
    Cross-Encoder reranking: chấm điểm lại từng cặp (query, child_text).

    Args:
        query: Câu hỏi gốc.
        candidates: Output của search_hybrid() — list payload dict.
        top_n: Số kết quả trả về sau rerank. None → dùng settings.
        threshold: Score tối thiểu để giữ lại. None → dùng settings.

    Returns:
        List RetrievedDoc đã sort theo score giảm dần.
    """
    if not candidates:
        return []

    settings = get_settings()
    n = top_n or settings.retrieval_top_n
    min_score = threshold or settings.reranker_threshold
    reranker = _get_reranker()

    # Cross-Encoder chấm điểm (query, child_text) cặp
    child_texts = [c.get("child_text", c.get("parent_text", "")) for c in candidates]
    pairs = [(query, text) for text in child_texts]
    scores = reranker.predict(pairs)  # numpy array

    # Kết hợp score với payload
    scored = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    results = []
    for score, payload in scored[:n]:
        if float(score) < min_score:
            logger.debug("Reranker: skipped doc score=%.3f < threshold=%.3f", score, min_score)
            continue
        results.append(RetrievedDoc(
            page_content=payload.get("parent_text", ""),
            child_text=payload.get("child_text", ""),
            score=float(score),
            metadata={k: v for k, v in payload.items() if k not in ("child_text", "parent_text")},
        ))

    logger.info("Reranked %d candidates → %d results (threshold=%.2f)", len(candidates), len(results), min_score)
    return results


def search_and_rerank(
    query: str,
    collection_name: str | None = None,
    filter_payload: dict | None = None,
) -> list[RetrievedDoc]:
    """
    High-level API: Hybrid Search + Cross-Encoder Rerank trong 1 lần gọi.
    Đây là hàm được gọi từ rag_agent_node.

    Args:
        query: Câu hỏi người dùng.
        collection_name: Override collection. None → dùng settings.
        filter_payload: Qdrant filter (ví dụ: {"company": "Vinamilk", "year": 2023}).

    Returns:
        List RetrievedDoc sẵn sàng gửi LLM.
    """
    candidates = search_hybrid(query, collection_name, filter_payload=filter_payload)
    if not candidates:
        logger.warning("search_and_rerank: Qdrant trả về 0 candidates. Collection có dữ liệu chưa?")
        return []
    return rerank(query, candidates)
