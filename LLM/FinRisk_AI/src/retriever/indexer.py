"""
indexer.py — PDF parse, chunking (Parent-Document strategy), và upsert vào Qdrant.

Pipeline:
  PDF file
    → pdfplumber bóc tách text + bảng biểu (chuyển thành Markdown table)
    → Chia parent chunks (~1000 tokens) để lưu context đầy đủ
    → Chia child chunks (~150 tokens) để embed & search (chính xác hơn)
    → Upsert cặp (child_embedding, parent_text) vào Qdrant

Tại sao Parent-Document Retriever?
  - Search trên child nhỏ → precision cao (embed ngắn, focus).
  - Trả về parent lớn → LLM nhận context đủ thông tin để trả lời.

Scale-up hint:
  - Thêm `doc_type` vào metadata: "bctc" | "thong_tu" | "kiem_toan"
  - Dùng Celery để index PDF bất đồng bộ (Phase 4).
  - Contextual Retrieval (Anthropic): Dùng LLM gắn prefix context trước khi embed.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
import tiktoken
from qdrant_client.http.models import PointStruct, SparseVector

from src.config import get_settings
from src.retriever.embedder import embed_dense, embed_sparse
from src.retriever.qdrant_client import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    ensure_collection,
    get_qdrant_client,
)

logger = logging.getLogger(__name__)
_TOKENIZER = tiktoken.get_encoding("cl100k_base")  # Chuẩn GPT-4/Claude


@dataclass
class DocumentChunk:
    """Đại diện cho 1 child chunk + parent context của nó."""
    child_text: str          # Văn bản nhỏ dùng để embed & search
    parent_text: str         # Văn bản lớn gửi LLM khi retrieved
    metadata: dict = field(default_factory=dict)  # company, year, doc_type, page, ...


# ─── PDF Parsing ───────────────────────────────────────────────────────────────

def _table_to_markdown(table: list[list]) -> str:
    """Chuyển bảng biểu pdfplumber thành Markdown table."""
    if not table or not table[0]:
        return ""
    rows = []
    header = [str(cell or "").strip() for cell in table[0]]
    rows.append("| " + " | ".join(header) + " |")
    rows.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in table[1:]:
        cells = [str(cell or "").strip() for cell in row]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def parse_pdf(pdf_path: str | Path) -> list[dict]:
    """
    Bóc tách văn bản và bảng biểu từ file PDF.

    Returns:
        List dict mỗi trang: {"page": int, "text": str}
        Text đã gộp cả bảng biểu (dưới dạng Markdown table).
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF không tìm thấy: {path}")

    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            # Bóc tách text thường
            text = page.extract_text() or ""

            # Bóc tách bảng biểu → chuyển thành Markdown
            tables = page.extract_tables()
            table_md = "\n\n".join(_table_to_markdown(t) for t in tables if t)

            full_page_text = text
            if table_md:
                full_page_text += f"\n\n### Bảng biểu (trang {i}):\n{table_md}"

            if full_page_text.strip():
                pages.append({"page": i, "text": full_page_text.strip()})

    logger.info("Parsed PDF '%s': %d pages with content.", path.name, len(pages))
    return pages


# ─── Chunking ──────────────────────────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _split_by_tokens(text: str, max_tokens: int) -> list[str]:
    """Chia text thành các đoạn không vượt quá max_tokens."""
    tokens = _TOKENIZER.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i: i + max_tokens]
        chunks.append(_TOKENIZER.decode(chunk_tokens))
    return chunks


def chunk_documents(pages: list[dict], metadata: dict) -> list[DocumentChunk]:
    """
    Parent-Document chunking strategy:
      1. Mỗi trang → chia thành parent chunks (chunk_parent_tokens tokens).
      2. Mỗi parent → chia thành child chunks (chunk_child_tokens tokens).
      3. Child dùng để embed & search; parent dùng để gửi LLM.

    Args:
        pages: Output của parse_pdf().
        metadata: Dict với company, year, doc_type, filename, ...

    Returns:
        List DocumentChunk.
    """
    settings = get_settings()
    result: list[DocumentChunk] = []

    for page_data in pages:
        page_num = page_data["page"]
        full_text = page_data["text"]

        # Chia trang thành parent chunks
        parents = _split_by_tokens(full_text, settings.chunk_parent_tokens)
        for parent_text in parents:
            if not parent_text.strip():
                continue
            # Chia parent thành child chunks nhỏ hơn để embed
            children = _split_by_tokens(parent_text, settings.chunk_child_tokens)
            for child_text in children:
                if not child_text.strip():
                    continue
                result.append(DocumentChunk(
                    child_text=child_text,
                    parent_text=parent_text,
                    metadata={**metadata, "page": page_num},
                ))

    logger.info("Chunked %d pages → %d DocumentChunks.", len(pages), len(result))
    return result


# ─── Indexing ──────────────────────────────────────────────────────────────────

def _make_chunk_id(child_text: str, metadata: dict) -> str:
    """Tạo UUID ổn định từ nội dung chunk (idempotent upsert)."""
    raw = f"{metadata.get('filename', '')}-{metadata.get('page', 0)}-{child_text[:100]}"
    return hashlib.md5(raw.encode()).hexdigest()


def index_documents(chunks: list[DocumentChunk], collection_name: str | None = None) -> int:
    """
    Upsert list DocumentChunk vào Qdrant với Dense + Sparse vectors.

    Args:
        chunks: Output của chunk_documents().
        collection_name: Override collection. None → dùng settings.

    Returns:
        Số chunks đã upsert thành công.
    """
    if not chunks:
        logger.warning("index_documents: Không có chunk nào để index.")
        return 0

    settings = get_settings()
    collection = collection_name or settings.qdrant_collection
    ensure_collection(collection)
    client = get_qdrant_client()

    # Batch embed tất cả child text cùng lúc (hiệu quả hơn từng cái một)
    child_texts = [c.child_text for c in chunks]
    logger.info("Embedding %d chunks (dense + sparse)...", len(child_texts))
    dense_vecs = embed_dense(child_texts)
    sparse_vecs = embed_sparse(child_texts)

    points = []
    for chunk, dense, sparse in zip(chunks, dense_vecs, sparse_vecs):
        point_id = _make_chunk_id(chunk.child_text, chunk.metadata)
        points.append(PointStruct(
            id=point_id,
            vector={
                DENSE_VECTOR_NAME: dense,
                SPARSE_VECTOR_NAME: SparseVector(
                    indices=sparse["indices"],
                    values=sparse["values"],
                ),
            },
            payload={
                "child_text": chunk.child_text,
                "parent_text": chunk.parent_text,
                **chunk.metadata,
            },
        ))

    # Upsert theo batch 100 để tránh timeout
    BATCH_SIZE = 100
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i: i + BATCH_SIZE]
        client.upsert(collection_name=collection, points=batch)
        logger.debug("Upserted batch %d/%d", i // BATCH_SIZE + 1, -(-len(points) // BATCH_SIZE))

    logger.info("Indexed %d chunks into collection '%s'.", len(points), collection)
    return len(points)


def index_pdf(
    pdf_path: str | Path,
    company: str,
    year: int,
    doc_type: str = "bctc",
    collection_name: str | None = None,
) -> int:
    """
    High-level API: Parse PDF → Chunk → Index vào Qdrant.

    Args:
        pdf_path: Đường dẫn file PDF.
        company: Tên công ty (ví dụ: "Vinamilk").
        year: Năm tài chính (ví dụ: 2023).
        doc_type: Loại tài liệu: "bctc" | "thong_tu" | "kiem_toan".
        collection_name: Override collection. None → dùng settings.

    Returns:
        Số chunks đã index.

    Example:
        >>> index_pdf("vinamilk_q2_2023.pdf", company="Vinamilk", year=2023)
    """
    path = Path(pdf_path)
    metadata = {
        "company": company,
        "year": year,
        "doc_type": doc_type,
        "filename": path.name,
    }
    pages = parse_pdf(path)
    chunks = chunk_documents(pages, metadata)
    return index_documents(chunks, collection_name)
