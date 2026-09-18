"""
index_sample.py — Script CLI để index file PDF mẫu vào Qdrant.

Dùng để test pipeline indexing trước khi tích hợp vào production.
KHÔNG phải production code — chỉ dùng để thực hành.

Cách dùng:
  # Index 1 file PDF
  python scripts/index_sample.py --pdf path/to/bctc.pdf --company "Vinamilk" --year 2023

  # Dùng file PDF mẫu được tạo tự động (không cần PDF thật)
  python scripts/index_sample.py --demo
"""

import argparse
import logging
import sys
from pathlib import Path

# Thêm project root vào Python path để import src.*
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_demo_mode() -> None:
    """
    Demo mode: Tạo file text giả lập BCTC và index vào Qdrant.
    Dùng khi chưa có PDF thật để test pipeline.
    """
    from src.retriever.qdrant_client import ensure_collection
    from src.retriever.indexer import DocumentChunk, index_documents

    logger.info("=== DEMO MODE: Tạo dữ liệu mẫu BCTC ===")
    ensure_collection()

    # Tạo các chunk mẫu giả lập BCTC
    sample_chunks = [
        DocumentChunk(
            child_text="Tổng doanh thu năm 2023 của Vinamilk đạt 59,956 tỷ đồng, tăng 3.2% so với năm 2022.",
            parent_text=(
                "KẾT QUẢ KINH DOANH NĂM 2023 — CÔNG TY CỔ PHẦN SỮA VIỆT NAM (VINAMILK)\n\n"
                "Tổng doanh thu năm 2023 của Vinamilk đạt 59,956 tỷ đồng, tăng 3.2% so với năm 2022. "
                "Lợi nhuận gộp đạt 22,158 tỷ đồng, biên lợi nhuận gộp 36.96%. "
                "Chi phí bán hàng và quản lý doanh nghiệp tổng cộng 14,890 tỷ đồng. "
                "Lợi nhuận trước thuế đạt 9,042 tỷ đồng."
            ),
            metadata={"company": "Vinamilk", "year": 2023, "doc_type": "bctc", "page": 5, "filename": "demo_bctc.txt"},
        ),
        DocumentChunk(
            child_text="DSCR của Vinamilk năm 2023 ước tính 2.8, cho thấy khả năng trả nợ tốt.",
            parent_text=(
                "PHÂN TÍCH CHỈ SỐ TÀI CHÍNH — VINAMILK 2023\n\n"
                "Hệ số khả năng trả nợ (DSCR) của Vinamilk năm 2023 ước tính 2.8, cho thấy khả năng trả nợ tốt. "
                "Nợ vay dài hạn: 2,340 tỷ đồng. Nợ vay ngắn hạn: 1,120 tỷ đồng. "
                "Tổng nợ vay: 3,460 tỷ đồng. Hệ số D/E = 0.47 (thấp, lành mạnh). "
                "Quick Ratio = 1.32 (trên ngưỡng an toàn 0.8)."
            ),
            metadata={"company": "Vinamilk", "year": 2023, "doc_type": "bctc", "page": 8, "filename": "demo_bctc.txt"},
        ),
        DocumentChunk(
            child_text="Altman Z-Score của Vinamilk năm 2023 ước tính 3.42, vùng an toàn (Z > 2.99).",
            parent_text=(
                "ĐÁNH GIÁ RỦI RO PHÁ SẢN — ALTMAN Z-SCORE 2023\n\n"
                "Altman Z-Score của Vinamilk năm 2023 ước tính 3.42, thuộc vùng AN TOÀN (Z > 2.99). "
                "Working Capital: 8,920 tỷ. Total Assets: 42,630 tỷ. "
                "Retained Earnings: 18,540 tỷ. EBIT: 9,850 tỷ. "
                "Market Cap: 89,200 tỷ. Total Liabilities: 12,450 tỷ. Revenue: 59,956 tỷ."
            ),
            metadata={"company": "Vinamilk", "year": 2023, "doc_type": "bctc", "page": 12, "filename": "demo_bctc.txt"},
        ),
        DocumentChunk(
            child_text="Thông tư 41/2016/TT-NHNN quy định tỷ lệ an toàn vốn tối thiểu 8% đối với ngân hàng thương mại.",
            parent_text=(
                "THÔNG TƯ 41/2016/TT-NHNN — TỶ LỆ AN TOÀN VỐN\n\n"
                "Thông tư 41/2016/TT-NHNN quy định tỷ lệ an toàn vốn (CAR) tối thiểu 8% đối với ngân hàng thương mại "
                "và chi nhánh ngân hàng nước ngoài. CAR = Vốn tự có / (Tổng tài sản có rủi ro). "
                "Đây là chuẩn Basel II được áp dụng tại Việt Nam từ 01/01/2020."
            ),
            metadata={"company": "NHNN", "year": 2016, "doc_type": "thong_tu", "page": 3, "filename": "demo_thong_tu.txt"},
        ),
    ]

    count = index_documents(sample_chunks)
    logger.info("✅ Demo: Đã index %d chunks vào Qdrant.", count)
    logger.info("Mở http://localhost:6333/dashboard để xem collection.")


def run_pdf_mode(pdf_path: str, company: str, year: int, doc_type: str) -> None:
    """Index file PDF thật vào Qdrant."""
    from src.retriever.indexer import index_pdf

    logger.info("=== INDEX PDF: %s ===", pdf_path)
    count = index_pdf(
        pdf_path=pdf_path,
        company=company,
        year=year,
        doc_type=doc_type,
    )
    logger.info("✅ Đã index %d chunks từ '%s'.", count, pdf_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index tài liệu tài chính vào Qdrant")
    parser.add_argument("--demo", action="store_true", help="Chạy demo mode với dữ liệu mẫu (không cần PDF)")
    parser.add_argument("--pdf", type=str, help="Đường dẫn file PDF cần index")
    parser.add_argument("--company", type=str, default="Unknown", help="Tên công ty")
    parser.add_argument("--year", type=int, default=2023, help="Năm tài chính")
    parser.add_argument("--doc-type", type=str, default="bctc", choices=["bctc", "thong_tu", "kiem_toan"])
    args = parser.parse_args()

    if args.demo:
        run_demo_mode()
    elif args.pdf:
        run_pdf_mode(args.pdf, args.company, args.year, args.doc_type)
    else:
        parser.print_help()
        print("\n⚠️  Dùng --demo để chạy nhanh không cần PDF, hoặc --pdf để index file thật.")
        sys.exit(1)


if __name__ == "__main__":
    main()
