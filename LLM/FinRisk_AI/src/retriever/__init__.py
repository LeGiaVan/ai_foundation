"""
retriever/__init__.py — Public API của module RAG retriever.

Các component chính:
  - qdrant_client : Singleton Qdrant connection + collection setup
  - embedder      : Dense (BGE-M3) + Sparse (BM25) embedding
  - indexer       : PDF parse + chunk + upsert vào Qdrant
  - searcher      : Hybrid search + Cross-Encoder rerank
"""

from src.retriever.searcher import RetrievedDoc, search_and_rerank  # noqa: F401
from src.retriever.indexer import index_pdf  # noqa: F401
