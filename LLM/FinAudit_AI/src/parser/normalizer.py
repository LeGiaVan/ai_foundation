"""
normalizer.py — Module 3: Output Normalization cho FinAudit AI.
Hợp nhất kết quả từ 2 nhánh (Digital Text Parser và OCR Pipeline) thành danh sách
ParsedBlock thống nhất, chuẩn hóa thứ tự đọc, gắn nhãn source, xử lý bảng nối trang
và phân bổ đơn vị tiền tệ.
"""

import logging
from collections.abc import Sequence

from src.models import ParsedBlock

logger = logging.getLogger(__name__)


class OutputNormalizer:
    """Bộ chuẩn hóa và hợp nhất đầu ra từ TextParser và OCRPipeline."""

    def normalize(
        self,
        text_blocks: Sequence[ParsedBlock] | None = None,
        ocr_blocks: Sequence[ParsedBlock] | None = None,
        doc_unit: str | None = None,
    ) -> list[ParsedBlock]:
        """
        Hợp nhất và chuẩn hóa blocks từ 2 nguồn:
          1. Đảm bảo thuộc tính source ("pdfplumber" vs "ocr")
          2. Sắp xếp theo thứ tự đọc tự nhiên: (page, y0, x0)
          3. Đánh số lại block_id liên tục và duy nhất theo từng trang (p{page}_b{index})
          4. Xử lý ghép bảng nối trang (Multi-page spanning tables)
          5. Lan truyền đơn vị tiền tệ mặc định cho toàn tài liệu

        Args:
            text_blocks: Danh sách blocks trích xuất từ digital text (pdfplumber)
            ocr_blocks: Danh sách blocks trích xuất từ scan (OCR)
            doc_unit: Đơn vị tiền tệ nhận diện được ở cấp tài liệu (ví dụ: 'VND')

        Returns:
            list[ParsedBlock]: Danh sách khối văn bản và bảng biểu đã chuẩn hóa
        """
        combined: list[ParsedBlock] = []

        if text_blocks:
            for b in text_blocks:
                b.source = "pdfplumber"
                combined.append(b)

        if ocr_blocks:
            for b in ocr_blocks:
                b.source = "ocr"
                combined.append(b)

        if not combined:
            return []

        # 1. Phát hiện doc_unit nếu chưa được chỉ định
        if not doc_unit:
            for b in combined:
                u = b.metadata.get("unit")
                if u:
                    doc_unit = u
                    break

        # 2. Sắp xếp theo trang và tọa độ (y0 từ trên xuống, x0 từ trái sang)
        def sort_key(b: ParsedBlock) -> tuple[int, float, float]:
            if b.bbox and len(b.bbox) >= 2:
                return (b.page, float(b.bbox[1]), float(b.bbox[0]))
            return (b.page, 0.0, 0.0)

        combined.sort(key=sort_key)

        # 3. Đánh lại block_id theo thứ tự đọc trong từng trang
        page_counters: dict[int, int] = {}
        for b in combined:
            p = b.page
            page_counters[p] = page_counters.get(p, 0) + 1
            b.block_id = f"p{p}_b{page_counters[p]}"

        # 4. Ghép các bảng kéo dài qua nhiều trang (Table Spanning across consecutive pages)
        merged_blocks = self._merge_spanning_tables(combined)

        # 5. Gán đơn vị tiền tệ mặc định cho các block chưa có
        if doc_unit:
            for b in merged_blocks:
                if not b.metadata.get("unit"):
                    b.metadata["unit"] = doc_unit

        logger.info(
            "OutputNormalizer: Đã chuẩn hóa %d blocks (Text: %d, OCR: %d, Sau ghép bảng: %d).",
            len(combined),
            len(text_blocks or []),
            len(ocr_blocks or []),
            len(merged_blocks),
        )
        return merged_blocks

    def _merge_spanning_tables(self, blocks: list[ParsedBlock]) -> list[ParsedBlock]:
        """Phát hiện và nối các bảng kéo dài qua 2 trang liên tiếp (Table Spanning)."""
        if len(blocks) < 2:
            return blocks

        merged: list[ParsedBlock] = []
        i = 0
        while i < len(blocks):
            curr = blocks[i]
            if curr.is_table and i + 1 < len(blocks):
                nxt = blocks[i + 1]
                if (
                    nxt.is_table
                    and nxt.page == curr.page + 1
                    and curr.num_cols == nxt.num_cols
                    and curr.num_cols > 1
                ):
                    nxt_lines = nxt.content.splitlines()
                    # Bỏ dòng header của bảng trang sau nếu trùng số cột
                    if len(nxt_lines) > 2:
                        data_addition = "\n" + "\n".join(nxt_lines[2:])
                        curr.content += data_addition
                        curr.metadata["num_rows"] = curr.num_rows + nxt.num_rows
                        curr.metadata["spans_pages"] = [curr.page, nxt.page]
                        logger.info(
                            "OutputNormalizer: Đã nối bảng kéo dài qua 2 trang: Trang %d -> Trang %d (Block %s)",
                            curr.page,
                            nxt.page,
                            curr.block_id,
                        )
                        merged.append(curr)
                        i += 2
                        continue

            merged.append(curr)
            i += 1

        return merged


def normalize_output(
    text_blocks: list[ParsedBlock] | None = None,
    ocr_blocks: list[ParsedBlock] | None = None,
    doc_unit: str | None = None,
) -> list[ParsedBlock]:
    """Hàm tiện ích chuẩn hóa và hợp nhất đầu ra theo proposal.md."""
    normalizer = OutputNormalizer()
    return normalizer.normalize(text_blocks=text_blocks, ocr_blocks=ocr_blocks, doc_unit=doc_unit)
