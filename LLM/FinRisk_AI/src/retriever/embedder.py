"""
embedder.py — Dense (BGE-M3) và Sparse (BM25) embedding wrapper.

Tại sao 2 loại embedding?
  - Dense (BGE-M3): Hiểu ngữ nghĩa. "doanh thu" ≈ "revenue" → match được.
  - Sparse (BM25): Khớp từ khoá chính xác. "DSCR 1.2" → phải có từ "DSCR".
  Hybrid = Dense + Sparse → vừa hiểu nghĩa, vừa khớp số liệu chính xác.

Scale-up hint:
  - fastembed batch encode tự động, an toàn với GPU nếu có.
  - Khi dùng multi-tenant, thêm metadata filter vào vector payload.
"""

import logging
from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_dense_embedder() -> TextEmbedding:
    """
    Singleton Dense Embedder dùng BAAI/bge-m3.
    Lần đầu tải model ~600MB từ HuggingFace Hub (cache ở ~/.cache/fastembed).
    """
    logger.info("Loading dense embedding model: BAAI/bge-m3 (first run: ~600MB download)")
    return TextEmbedding("BAAI/bge-m3")


@lru_cache(maxsize=1)
def get_sparse_embedder() -> SparseTextEmbedding:
    """
    Singleton Sparse Embedder dùng Qdrant/bm25.
    BM25 được tích hợp sẵn trong fastembed — không cần index riêng.
    """
    logger.info("Loading sparse embedding model: Qdrant/bm25")
    return SparseTextEmbedding("Qdrant/bm25")


def embed_dense(texts: list[str]) -> list[list[float]]:
    """
    Encode list text thành dense vectors.

    Args:
        texts: Danh sách văn bản cần embed.

    Returns:
        List dense vectors, mỗi vector có 1024 chiều (BGE-M3).
    """
    embedder = get_dense_embedder()
    # fastembed trả về generator → chuyển thành list
    return [vec.tolist() for vec in embedder.embed(texts)]


def embed_sparse(texts: list[str]) -> list[dict]:
    """
    Encode list text thành sparse vectors (BM25).

    Returns:
        List dict {"indices": [...], "values": [...]}, chuẩn Qdrant SparseVector.
    """
    embedder = get_sparse_embedder()
    results = []
    for sparse_vec in embedder.embed(texts):
        results.append({
            "indices": sparse_vec.indices.tolist(),
            "values": sparse_vec.values.tolist(),
        })
    return results
