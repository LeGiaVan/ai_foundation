"""
indexer.py — PDF parse, chunking (Parent-Document strategy), và upsert vào Qdrant.

Pipeline:
  PDF file
    → pdfplumber bóc tách text + bảng biểu (chuyển thành Markdown table)
    → Tách khối bảng Markdown thành Atomic Units (không bao giờ bị cắt vụn)
    → Chia parent chunks (~1000 tokens) bằng Recursive Separator Splitter
    → Chia child chunks (~150 tokens) với Overlap (~30 tokens)
    → Upsert cặp (child_embedding, parent_text) vào Qdrant

Tại sao Parent-Document Retriever?
  - Search trên child nhỏ → precision cao (embed ngắn, focus).
  - Trả về parent lớn → LLM nhận context đủ thông tin để trả lời.

Chunking Strategy (Upgraded):
  - Recursive Separator Splitter: Ưu tiên ngắt theo thứ tự ngữ pháp tự nhiên
    ("\n\n" → "\n" → ". " → "; " → ", " → " ") thay vì cắt cứng theo token count.
  - Table Preservation: Khối Markdown Table (bắt đầu bằng "|") được giữ nguyên
    thành 1 chunk duy nhất, không bao giờ bị cắt vụn.
  - Chunk Overlap: Child chunks có 30 tokens overlap để tránh đứt mạch ý nghĩa
    tại ranh giới chunk.

Scale-up hint:
  - Thêm `doc_type` vào metadata: "bctc" | "thong_tu" | "kiem_toan"
  - Dùng Celery để index PDF bất đồng bộ (Phase 4).
  - Contextual Retrieval (Anthropic): Dùng LLM gắn prefix context trước khi embed.
"""

import hashlib
import logging
import re
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

# Thứ tự ưu tiên separator: từ "ranh giới tự nhiên nhất" đến "cắt bắt buộc"
_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " "]

# Regex phát hiện khối Markdown Table (bắt đầu bằng "|", kết thúc bằng "|")
_TABLE_BLOCK_RE = re.compile(
    r"((?:^[ \t]*\|.+\|[ \t]*$\n?)+)",
    re.MULTILINE,
)


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
    """Đếm số tokens trong text (dùng tokenizer cl100k_base)."""
    return len(_TOKENIZER.encode(text))


def _split_by_tokens(text: str, max_tokens: int) -> list[str]:
    """
    Legacy splitter — giữ lại cho backward compatibility.
    Chia text thành các đoạn không vượt quá max_tokens (cắt cứng).
    """
    tokens = _TOKENIZER.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i: i + max_tokens]
        chunks.append(_TOKENIZER.decode(chunk_tokens))
    return chunks


def _extract_table_blocks(text: str) -> tuple[list[str], list[str]]:
    """
    Tách văn bản thành 2 loại khối: text thường và Markdown Table.

    Markdown Table là Atomic Unit — KHÔNG BAO GIỜ bị cắt vụn.
    Khi 1 bảng dài hơn max_tokens, nó vẫn được giữ nguyên thành 1 chunk.

    Returns:
        (blocks, block_types):
          - blocks: list các đoạn text/table.
          - block_types: list tương ứng "text" hoặc "table".
    """
    blocks = []
    block_types = []
    last_end = 0

    for match in _TABLE_BLOCK_RE.finditer(text):
        # Text trước bảng
        before = text[last_end:match.start()]
        if before.strip():
            blocks.append(before.strip())
            block_types.append("text")

        # Khối bảng (atomic)
        table_block = match.group(0).strip()
        if table_block:
            blocks.append(table_block)
            block_types.append("table")

        last_end = match.end()

    # Text sau bảng cuối cùng
    after = text[last_end:]
    if after.strip():
        blocks.append(after.strip())
        block_types.append("text")

    # Nếu không có bảng nào, trả về toàn bộ text
    if not blocks:
        blocks = [text]
        block_types = ["text"]

    return blocks, block_types


def _recursive_split(
    text: str,
    max_tokens: int,
    separators: list[str] | None = None,
) -> list[str]:
    """
    Recursive Character Text Splitter — chuẩn Production.

    Thuật toán:
      1. Thử chia text bằng separator đầu tiên (ưu tiên "\n\n").
      2. Gộp các đoạn nhỏ lại cho đến khi vượt max_tokens → flush thành 1 chunk.
      3. Nếu 1 đoạn đơn lẻ vẫn dài hơn max_tokens → đệ quy với separator tiếp theo.
      4. Fallback cuối cùng: cắt cứng theo token (an toàn, không mất dữ liệu).

    Args:
        text: Văn bản cần chia.
        max_tokens: Kích thước tối đa mỗi chunk (tính bằng tokens).
        separators: Danh sách separator theo thứ tự ưu tiên.

    Returns:
        List các text chunk, mỗi chunk ≤ max_tokens.
    """
    if separators is None:
        separators = _DEFAULT_SEPARATORS

    # Base case: text đã đủ ngắn
    if _count_tokens(text) <= max_tokens:
        return [text] if text.strip() else []

    # Nếu hết separator → fallback cắt cứng theo token
    if not separators:
        return _split_by_tokens(text, max_tokens)

    current_sep = separators[0]
    remaining_seps = separators[1:]

    # Split bằng separator hiện tại
    parts = text.split(current_sep)

    chunks: list[str] = []
    current_chunk = ""

    for part in parts:
        # Thử gộp part vào chunk hiện tại
        candidate = (current_sep.join([current_chunk, part]) if current_chunk else part)

        if _count_tokens(candidate) <= max_tokens:
            # Vẫn vừa → tiếp tục gộp
            current_chunk = candidate
        else:
            # Vượt quá → flush chunk hiện tại
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # Kiểm tra: part đơn lẻ có vượt max_tokens?
            if _count_tokens(part) > max_tokens:
                # Đệ quy với separator tiếp theo (nhỏ hơn)
                sub_chunks = _recursive_split(part, max_tokens, remaining_seps)
                chunks.extend(sub_chunks)
                current_chunk = ""
            else:
                current_chunk = part

    # Flush chunk cuối cùng
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _add_overlap(chunks: list[str], overlap_tokens: int) -> list[str]:
    """
    Thêm Chunk Overlap: lấy ~overlap_tokens cuối của chunk trước
    gắn vào đầu chunk sau.

    Mục đích: Tránh đứt mạch ý nghĩa tại ranh giới chunk.
    Ví dụ: overlap=30 tokens → 30 tokens cuối chunk_i sẽ xuất hiện
    ở đầu chunk_{i+1}.

    Args:
        chunks: List text chunks (đã chia xong).
        overlap_tokens: Số tokens overlap giữa 2 chunk liên tiếp.

    Returns:
        List chunks đã có overlap. Chunk đầu tiên giữ nguyên.
    """
    if overlap_tokens <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tokens = _TOKENIZER.encode(chunks[i - 1])
        # Lấy overlap_tokens cuối cùng của chunk trước
        overlap_part = _TOKENIZER.decode(prev_tokens[-overlap_tokens:])
        result.append(overlap_part + " " + chunks[i])

    return result


def chunk_documents(pages: list[dict], metadata: dict) -> list[DocumentChunk]:
    """
    Parent-Document chunking strategy (Recursive Splitter + Table Preservation + Overlap):

      1. Mỗi trang → tách Markdown Table thành Atomic Units (không bao giờ cắt vụn).
      2. Text thường → Recursive Separator Split thành parent chunks (~1000 tokens).
      3. Mỗi parent → Recursive Split thành child chunks (~150 tokens) + Overlap (~30 tokens).
      4. Child dùng để embed & search; parent dùng để gửi LLM.

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

        # Bước 1: Tách text thường vs Markdown Table blocks
        blocks, block_types = _extract_table_blocks(full_text)

        # Bước 2: Xử lý từng block
        parent_chunks: list[str] = []
        for block, btype in zip(blocks, block_types):
            if btype == "table":
                # Bảng = Atomic Unit → giữ nguyên thành 1 parent chunk
                parent_chunks.append(block)
            else:
                # Text thường → Recursive Split thành parent chunks
                parents = _recursive_split(block, settings.chunk_parent_tokens)
                parent_chunks.extend(parents)

        # Bước 3: Mỗi parent → chia thành child chunks + overlap
        for parent_text in parent_chunks:
            if not parent_text.strip():
                continue

            # Child chunks: Recursive Split + Overlap
            raw_children = _recursive_split(parent_text, settings.chunk_child_tokens)
            children = _add_overlap(raw_children, settings.chunk_overlap_tokens)

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
