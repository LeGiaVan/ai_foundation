"""
test_retriever.py — Unit tests cho Phase 2 RAG components.

Test strategy:
  - test_embed_*   : Test embedding không cần Qdrant (chỉ cần models).
  - test_chunking  : Test pure Python, không cần network.
  - test_pdf_parse : Test pdfplumber với file PDF nhỏ.
  - test_search_*  : Cần Qdrant chạy (docker) — đánh dấu @pytest.mark.integration.

Chạy unit tests (nhanh, không cần Qdrant):
  pytest tests/test_retriever.py -v -m "not integration"

Chạy integration tests (cần Qdrant docker):
  pytest tests/test_retriever.py -v -m integration
"""

import pytest

# ─── Recursive Splitter Tests ──────────────────────────────────────────────────

class TestRecursiveSplit:
    """Test _recursive_split — Recursive Separator Splitter."""

    def test_short_text_no_split(self):
        """Text ngắn hơn max_tokens → trả về nguyên 1 chunk."""
        from src.retriever.indexer import _recursive_split

        text = "Tổng doanh thu năm 2023 đạt 59,956 tỷ đồng."
        chunks = _recursive_split(text, max_tokens=500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_splits_on_double_newline_first(self):
        """Ưu tiên ngắt tại '\\n\\n' trước khi thử separator khác."""
        from src.retriever.indexer import _recursive_split

        text = "Đoạn 1 ngắn.\n\nĐoạn 2 ngắn.\n\nĐoạn 3 ngắn."
        chunks = _recursive_split(text, max_tokens=10)
        # Mỗi đoạn ~5 tokens → 3 chunks khi max_tokens=10
        assert len(chunks) >= 2
        assert "Đoạn 1" in chunks[0]

    def test_falls_back_to_sentence_split(self):
        """Khi không có '\\n\\n' hay '\\n', split theo '. '."""
        from src.retriever.indexer import _recursive_split

        text = "Câu một rất ngắn. Câu hai cũng ngắn. Câu ba kết thúc."
        chunks = _recursive_split(text, max_tokens=10)
        assert len(chunks) >= 2
        # Mỗi chunk phải là câu hoàn chỉnh (hoặc gần hoàn chỉnh)
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_empty_text_returns_empty(self):
        """Text rỗng → trả về list rỗng."""
        from src.retriever.indexer import _recursive_split

        assert _recursive_split("", max_tokens=100) == []
        assert _recursive_split("   ", max_tokens=100) == []

    def test_long_text_produces_bounded_chunks(self):
        """Mỗi chunk phải ≤ max_tokens (trừ trường hợp 1 từ dài hơn max)."""
        from src.retriever.indexer import _recursive_split, _count_tokens

        long_text = "Doanh thu quý 2 tăng trưởng mạnh. " * 100
        chunks = _recursive_split(long_text, max_tokens=50)
        assert len(chunks) > 1
        for chunk in chunks:
            # Cho phép sai lệch nhỏ do separator
            assert _count_tokens(chunk) <= 55  # 50 + buffer nhỏ


class TestLegacySplitByTokens:
    """Test _split_by_tokens — legacy splitter (backward compatibility)."""

    def test_split_by_tokens_short_text(self):
        """Text ngắn hơn max_tokens → chỉ có 1 chunk."""
        from src.retriever.indexer import _split_by_tokens

        text = "Tổng doanh thu năm 2023 đạt 59,956 tỷ đồng."
        chunks = _split_by_tokens(text, max_tokens=500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_by_tokens_long_text(self):
        """Text dài hơn max_tokens → phải có nhiều chunks."""
        from src.retriever.indexer import _split_by_tokens

        long_text = "Đây là câu ngắn. " * 200  # ~600 tokens
        chunks = _split_by_tokens(long_text, max_tokens=100)
        assert len(chunks) > 1


# ─── Table Extraction Tests ───────────────────────────────────────────────────

class TestTableExtraction:
    """Test _extract_table_blocks — tách Markdown Table thành Atomic Unit."""

    def test_no_table_returns_full_text(self):
        """Text không có bảng → 1 block loại 'text'."""
        from src.retriever.indexer import _extract_table_blocks

        text = "Doanh thu 2023 đạt 59,956 tỷ đồng."
        blocks, types = _extract_table_blocks(text)
        assert len(blocks) == 1
        assert types[0] == "text"

    def test_table_extracted_as_atomic(self):
        """Khối Markdown Table phải là 1 block riêng loại 'table'."""
        from src.retriever.indexer import _extract_table_blocks

        text = (
            "Phần text trước bảng.\n\n"
            "| Chỉ tiêu | 2023 | 2022 |\n"
            "|---|---|---|\n"
            "| Doanh thu | 59,956 | 58,100 |\n"
            "| Lợi nhuận | 9,042 | 8,567 |\n\n"
            "Phần text sau bảng."
        )
        blocks, types = _extract_table_blocks(text)
        assert "table" in types
        table_idx = types.index("table")
        # Bảng phải chứa đầy đủ các hàng
        assert "Doanh thu" in blocks[table_idx]
        assert "Lợi nhuận" in blocks[table_idx]

    def test_multiple_tables(self):
        """Nhiều bảng → nhiều block 'table' riêng biệt."""
        from src.retriever.indexer import _extract_table_blocks

        text = (
            "Text.\n\n"
            "| A | B |\n|---|---|\n| 1 | 2 |\n\n"
            "Giữa.\n\n"
            "| C | D |\n|---|---|\n| 3 | 4 |\n"
        )
        blocks, types = _extract_table_blocks(text)
        table_count = types.count("table")
        assert table_count == 2


# ─── Overlap Tests ─────────────────────────────────────────────────────────────

class TestOverlap:
    """Test _add_overlap — thêm token overlap giữa 2 child chunk liền kề."""

    def test_single_chunk_no_overlap(self):
        """Chỉ 1 chunk → không thay đổi."""
        from src.retriever.indexer import _add_overlap

        chunks = ["Chunk duy nhất."]
        result = _add_overlap(chunks, overlap_tokens=30)
        assert result == chunks

    def test_overlap_adds_suffix_of_previous(self):
        """Chunk thứ 2 phải bắt đầu bằng phần cuối của chunk thứ 1."""
        from src.retriever.indexer import _add_overlap

        chunks = [
            "Doanh thu tăng trưởng 5% so với năm ngoái nhờ mở rộng thị trường nội địa.",
            "Chi phí hoạt động giảm 3% giúp cải thiện biên lợi nhuận đáng kể.",
        ]
        result = _add_overlap(chunks, overlap_tokens=5)
        assert len(result) == 2
        # Chunk đầu giữ nguyên
        assert result[0] == chunks[0]
        # Chunk 2 phải dài hơn chunk gốc (vì có overlap prepend)
        assert len(result[1]) > len(chunks[1])

    def test_zero_overlap_no_change(self):
        """Overlap = 0 → không thay đổi."""
        from src.retriever.indexer import _add_overlap

        chunks = ["A", "B"]
        assert _add_overlap(chunks, overlap_tokens=0) == chunks


# ─── Chunking Integration Tests ───────────────────────────────────────────────

class TestChunking:
    """Test chunk_documents end-to-end."""

    def test_chunk_documents_empty_pages(self):
        """Pages rỗng → trả về list rỗng, không lỗi."""
        from src.retriever.indexer import chunk_documents

        result = chunk_documents([], metadata={"company": "Test"})
        assert result == []

    def test_chunk_documents_creates_parent_child_pairs(self):
        """Mỗi chunk phải có cả child_text lẫn parent_text."""
        from src.retriever.indexer import chunk_documents

        pages = [{"page": 1, "text": "Doanh thu 2023: 59,956 tỷ đồng. " * 50}]
        chunks = chunk_documents(pages, metadata={"company": "Test", "year": 2023})

        assert len(chunks) > 0
        for c in chunks:
            assert c.child_text  # không rỗng
            assert c.parent_text  # không rỗng
            assert len(c.child_text) <= len(c.parent_text)  # child ≤ parent

    def test_chunk_metadata_propagated(self):
        """Metadata phải được propagate vào mỗi chunk."""
        from src.retriever.indexer import chunk_documents

        pages = [{"page": 3, "text": "Nội dung BCTC Vinamilk. " * 20}]
        chunks = chunk_documents(pages, metadata={"company": "Vinamilk", "year": 2023})

        for c in chunks:
            assert c.metadata["company"] == "Vinamilk"
            assert c.metadata["year"] == 2023
            assert c.metadata["page"] == 3

    def test_table_preserved_as_atomic(self):
        """Markdown Table trong text không bị cắt vụn giữa chừng."""
        from src.retriever.indexer import chunk_documents

        table_text = (
            "| Chỉ tiêu | 2023 | 2022 |\n"
            "|---|---|---|\n"
            "| Doanh thu | 59,956 | 58,100 |\n"
            "| Lợi nhuận | 9,042 | 8,567 |\n"
            "| Chi phí | 14,890 | 14,200 |"
        )
        pages = [{"page": 1, "text": f"Intro text.\n\n{table_text}\n\nText kết luận."}]
        chunks = chunk_documents(pages, metadata={"company": "Test"})

        # Phải có ít nhất 1 chunk mà parent_text chứa toàn bộ bảng
        table_found = any(
            "Doanh thu" in c.parent_text and "Lợi nhuận" in c.parent_text and "Chi phí" in c.parent_text
            for c in chunks
        )
        assert table_found, "Bảng bị cắt vụn — vi phạm Atomic Unit!"


class TestTableToMarkdown:
    """Test bóc tách bảng biểu."""

    def test_empty_table(self):
        """Bảng rỗng → trả về chuỗi rỗng, không lỗi."""
        from src.retriever.indexer import _table_to_markdown

        assert _table_to_markdown([]) == ""
        assert _table_to_markdown([[]]) == ""

    def test_simple_table(self):
        """Bảng đơn giản → Markdown format đúng."""
        from src.retriever.indexer import _table_to_markdown

        table = [
            ["Chỉ tiêu", "2023", "2022"],
            ["Doanh thu", "59,956", "58,100"],
            ["Lợi nhuận", "9,042", "8,567"],
        ]
        result = _table_to_markdown(table)
        assert "| Chỉ tiêu | 2023 | 2022 |" in result
        assert "| Doanh thu | 59,956 | 58,100 |" in result
        assert "---" in result  # Separator row

    def test_table_with_none_cells(self):
        """Ô None → chuyển thành chuỗi rỗng, không crash."""
        from src.retriever.indexer import _table_to_markdown

        table = [["Col1", None], [None, "Value"]]
        result = _table_to_markdown(table)  # Không nên raise exception
        assert "Col1" in result


class TestChunkId:
    """Test idempotent chunk ID generation."""

    def test_same_input_same_id(self):
        """Cùng input → cùng ID (idempotent, an toàn khi re-index)."""
        from src.retriever.indexer import _make_chunk_id

        id1 = _make_chunk_id("text content", {"filename": "test.pdf", "page": 1})
        id2 = _make_chunk_id("text content", {"filename": "test.pdf", "page": 1})
        assert id1 == id2

    def test_different_input_different_id(self):
        """Nội dung khác → ID khác."""
        from src.retriever.indexer import _make_chunk_id

        id1 = _make_chunk_id("text A", {"filename": "test.pdf", "page": 1})
        id2 = _make_chunk_id("text B", {"filename": "test.pdf", "page": 1})
        assert id1 != id2


# ─── Integration Tests (cần Qdrant Docker) ─────────────────────────────────────

@pytest.mark.integration
class TestQdrantIntegration:
    """Cần Qdrant đang chạy tại localhost:6333."""

    def test_ensure_collection_creates_if_not_exists(self):
        """ensure_collection() phải tạo collection nếu chưa có."""
        from src.retriever.qdrant_client import ensure_collection, get_qdrant_client

        test_collection = "test_integration_temp"
        client = get_qdrant_client()

        # Xoá collection test nếu đã tồn tại (cleanup từ lần trước)
        try:
            client.delete_collection(test_collection)
        except Exception:
            pass

        ensure_collection(test_collection)

        existing = {c.name for c in client.get_collections().collections}
        assert test_collection in existing

        # Cleanup
        client.delete_collection(test_collection)

    def test_index_and_search_roundtrip(self):
        """
        End-to-end: Index demo chunks → search → verify kết quả.
        Dùng collection riêng để không ảnh hưởng production data.
        """
        from src.retriever.indexer import DocumentChunk, index_documents
        from src.retriever.searcher import search_hybrid
        from src.retriever.qdrant_client import get_qdrant_client, ensure_collection

        test_collection = "test_integration_search"
        client = get_qdrant_client()

        # Cleanup
        try:
            client.delete_collection(test_collection)
        except Exception:
            pass

        ensure_collection(test_collection)

        # Index 1 chunk mẫu
        chunks = [
            DocumentChunk(
                child_text="DSCR Vinamilk 2023 là 2.8, vùng an toàn.",
                parent_text="DSCR Vinamilk 2023 là 2.8. Công ty có khả năng trả nợ tốt.",
                metadata={"company": "Vinamilk", "year": 2023, "doc_type": "bctc", "page": 1, "filename": "test.txt"},
            )
        ]
        count = index_documents(chunks, collection_name=test_collection)
        assert count == 1

        # Search và verify
        results = search_hybrid("DSCR Vinamilk", collection_name=test_collection, top_k=5)
        assert len(results) > 0
        assert any("DSCR" in r.get("child_text", "") for r in results)

        # Cleanup
        client.delete_collection(test_collection)


@pytest.mark.integration
class TestSearchAndRerank:
    """Test reranker — cần model BAAI/bge-reranker-v2-m3 đã tải."""

    def test_rerank_returns_scored_docs(self):
        """rerank() phải trả về RetrievedDoc với score hợp lệ."""
        from src.retriever.searcher import rerank

        candidates = [
            {
                "child_text": "DSCR Vinamilk 2023 là 2.8",
                "parent_text": "DSCR chi tiết: hệ số khả năng trả nợ 2.8",
                "company": "Vinamilk", "year": 2023, "page": 1,
            },
            {
                "child_text": "Thời tiết hôm nay đẹp",
                "parent_text": "Không liên quan tới tài chính",
                "company": "N/A", "year": 0, "page": 0,
            },
        ]

        results = rerank("DSCR Vinamilk là bao nhiêu?", candidates, top_n=2, threshold=0.0)

        # Phải có ít nhất 1 kết quả
        assert len(results) >= 1
        # Kết quả đầu tiên phải có score cao nhất
        if len(results) > 1:
            assert results[0].score >= results[1].score

    def test_rerank_filters_by_threshold(self):
        """Docs dưới threshold phải bị loại."""
        from src.retriever.searcher import rerank

        candidates = [
            {
                "child_text": "Nội dung hoàn toàn không liên quan",
                "parent_text": "...",
                "company": "N/A", "year": 0, "page": 0,
            }
        ]
        # Threshold cực cao → tất cả bị loại
        results = rerank("DSCR cụ thể của Vinamilk 2023?", candidates, threshold=0.99)
        assert len(results) == 0

    def test_rerank_deduplicates_same_parent(self):
        """Nhiều child chunk cùng parent → chỉ giữ 1 (score cao nhất)."""
        from src.retriever.searcher import rerank

        same_parent = "DSCR Vinamilk 2023 = 2.8. Biên an toàn. Công ty có khả năng trả nợ tốt."
        candidates = [
            {
                "child_text": "DSCR Vinamilk 2023 = 2.8",
                "parent_text": same_parent,
                "company": "Vinamilk", "year": 2023, "page": 1,
            },
            {
                "child_text": "Biên an toàn. Công ty có khả năng trả nợ tốt.",
                "parent_text": same_parent,
                "company": "Vinamilk", "year": 2023, "page": 1,
            },
        ]

        results = rerank("DSCR Vinamilk là bao nhiêu?", candidates, top_n=5, threshold=0.0)
        # Dù có 2 child khác nhau, cùng parent → chỉ giữ 1
        assert len(results) == 1
