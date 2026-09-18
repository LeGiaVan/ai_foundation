"""
qdrant_client.py — Singleton Qdrant connection và collection setup.

Tại sao Singleton?
  - Kết nối Qdrant tốn thời gian khởi tạo.
  - Dùng chung 1 instance trong toàn bộ ứng dụng (thread-safe với GIL).

Scale-up hint (Phase 4):
  - Khi deploy Docker, đổi QDRANT_HOST=qdrant (tên service trong docker-compose).
  - Khi dùng Qdrant Cloud, điền QDRANT_API_KEY và đổi QDRANT_HOST thành endpoint cloud.
"""

import logging
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    SparseVectorParams,
    VectorParams,
)

from src.config import get_settings

logger = logging.getLogger(__name__)

# Chiều vector của BAAI/bge-m3 dense embedding
DENSE_VECTOR_SIZE = 1024

# Tên vector trong Qdrant collection (phải nhất quán khi upsert & search)
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """
    Singleton Qdrant client.
    Kết nối tới Qdrant local hoặc cloud tùy settings.

    Scale-up hint: Khi Phase 4 dùng async FastAPI, đổi sang AsyncQdrantClient.
    """
    settings = get_settings()
    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key or None,  # None = không auth (local)
        timeout=30,
    )
    logger.info("Qdrant client connected to %s:%s", settings.qdrant_host, settings.qdrant_port)
    return client


def ensure_collection(collection_name: str | None = None) -> None:
    """
    Tạo Qdrant collection nếu chưa tồn tại.
    Schema: Dense vector (cosine) + Sparse vector (BM25 dot product).

    Args:
        collection_name: Tên collection. None → dùng settings.qdrant_collection.
    """
    settings = get_settings()
    name = collection_name or settings.qdrant_collection
    client = get_qdrant_client()

    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        logger.info("Collection '%s' already exists — skip creation.", name)
        return

    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(
                size=DENSE_VECTOR_SIZE,
                distance=Distance.COSINE,  # BGE-M3 chuẩn cosine
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams()  # BM25 dùng dot product ngầm định
        },
    )
    logger.info("Created collection '%s' with Dense(%d) + Sparse(BM25).", name, DENSE_VECTOR_SIZE)
