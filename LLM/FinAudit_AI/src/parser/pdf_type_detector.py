"""
pdf_type_detector.py — Bộ nhận diện loại trang PDF (Digital vs Scanned).
Quyết định định tuyến xử lý:
  - char_count >= 50: Trang văn bản số (Digital Text) -> Chuyển sang TextParser (pdfplumber)
  - char_count < 50: Trang ảnh chụp (Scanned Image) -> Chuyển sang OCRPipeline
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pdfplumber

logger = logging.getLogger(__name__)


@dataclass
class PageClassification:
    """Kết quả phân loại cho từng trang trong tài liệu PDF."""
    page: int
    page_type: Literal["text", "scanned"]
    char_count: int
    images_count: int
    has_tables: bool

    @property
    def is_scanned(self) -> bool:
        return self.page_type == "scanned"

    @property
    def is_text(self) -> bool:
        return self.page_type == "text"


@dataclass
class DocumentClassification:
    """Đánh giá tổng quan về toàn bộ tài liệu PDF."""
    total_pages: int
    text_pages_count: int
    scanned_pages_count: int
    is_predominantly_scanned: bool
    pages: list[PageClassification]


class PDFTypeDetector:
    """Bộ phân loại định tuyến trang PDF BCTC."""

    def __init__(self, min_char_threshold: int = 50) -> None:
        self.min_char_threshold = min_char_threshold

    def detect_pages(self, pdf_path: str | Path) -> list[PageClassification]:
        """
        Quét và phân loại từng trang trong file PDF.

        Args:
            pdf_path: Đường dẫn file PDF

        Returns:
            list[PageClassification]: Danh sách kết quả phân loại từng trang
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file PDF: {path.resolve()}")

        results: list[PageClassification] = []

        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text() or ""
                char_count = len(raw_text.strip())
                images_count = len(page.images)
                tables = page.find_tables()
                has_tables = len(tables) > 0

                if char_count >= self.min_char_threshold:
                    page_type = "text"
                else:
                    page_type = "scanned"

                results.append(
                    PageClassification(
                        page=i,
                        page_type=page_type,
                        char_count=char_count,
                        images_count=images_count,
                        has_tables=has_tables,
                    )
                )

        logger.info(
            "Đã quét %d trang file '%s': %d trang text, %d trang scanned.",
            len(results),
            path.name,
            sum(1 for p in results if p.is_text),
            sum(1 for p in results if p.is_scanned),
        )
        return results

    def classify_document(self, pdf_path: str | Path) -> DocumentClassification:
        """Phân loại tổng thể toàn bộ tài liệu PDF."""
        pages = self.detect_pages(pdf_path)
        total = len(pages)
        scanned_count = sum(1 for p in pages if p.is_scanned)
        text_count = sum(1 for p in pages if p.is_text)

        return DocumentClassification(
            total_pages=total,
            text_pages_count=text_count,
            scanned_pages_count=scanned_count,
            is_predominantly_scanned=(scanned_count > text_count),
            pages=pages,
        )


def detect_pdf_pages(pdf_path: str | Path) -> list[dict]:
    """Hàm tiện ích tương thích với API đặc tả trong proposal.md."""
    detector = PDFTypeDetector()
    results = detector.detect_pages(pdf_path)
    return [
        {
            "page": r.page,
            "type": r.page_type,
            "char_count": r.char_count,
            "images_count": r.images_count,
        }
        for r in results
    ]
