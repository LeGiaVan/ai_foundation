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

# ─── Chunking Tests (pure Python, không cần model/network) ─────────────────────

class TestChunking:
    """Test chunking logic độc lập."""

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
