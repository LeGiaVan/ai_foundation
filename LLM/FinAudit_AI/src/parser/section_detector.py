"""
section_detector.py — Nhận diện tiêu đề và phân đoạn Section cho Báo cáo tài chính.
Áp dụng Kiến trúc Chuẩn Hóa Phân Cấp 4 Tầng (Hierarchical Section & Heading Engine):
  - Tầng 1: Canonical Taxonomy (Hệ thống khái niệm chuẩn hóa TT 200/VAS/IFRS)
  - Tầng 2: Multi-Strategy Pattern Matcher (Prefix + Fuzzy Dictionary + TOC Alignment)
  - Tầng 3: Monotonic Hierarchy State Machine (Cây phân cấp một chiều chống nhảy cóc)
  - Tầng 4: Đóng gói Metadata phục vụ Parent-Child RAG (Child = Breadcrumb & Ref; Parent = Full Content)
"""

from dataclasses import dataclass
import logging
import re
import unicodedata
from typing import Any

from src.models import ClassifiedBlock, Section
from src.parser.ocr_postprocess import strip_boilerplate_lines

logger = logging.getLogger(__name__)

# ==============================================================================
# TẦNG 1: CANONICAL TAXONOMY & CONCEPT ONTOLOGY (TT 200 / VAS / IFRS)
# ==============================================================================

# Báo cáo cốt lõi (Level 3 - H3)
CORE_STATEMENT_CANONICAL = {
    "CORE_BALANCE_SHEET": "BẢNG CÂN ĐỐI KẾ TOÁN",
    "CORE_INCOME_STATEMENT": "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH",
    "CORE_CASH_FLOW": "BÁO CÁO LƯU CHUYỂN TIỀN TỆ",
}

# 8 Phần La Mã chuẩn trong Bản Thuyết minh BCTC (Level 3 - H3)
ROMAN_CANONICAL = {
    "I": ("NOTE_SEC_GENERAL_INFO", "I. ĐẶC ĐIỂM HOẠT ĐỘNG CỦA DOANH NGHIỆP", 1),
    "II": ("NOTE_SEC_ACCOUNTING_PERIOD", "II. KỲ KẾ TOÁN VÀ ĐƠN VỊ TIỀN TỆ SỬ DỤNG TRONG KẾ TOÁN", 2),
    "III": ("NOTE_SEC_ACCOUNTING_STANDARDS", "III. CHUẨN MỰC VÀ CHẾ ĐỘ KẾ TOÁN ÁP DỤNG", 3),
    "IV": ("NOTE_SEC_ACCOUNTING_POLICIES", "IV. CÁC CHÍNH SÁCH KẾ TOÁN ÁP DỤNG", 4),
    "V": ("NOTE_SEC_BALANCE_SHEET", "V. THÔNG TIN BỔ SUNG CHO CÁC KHOẢN MỤC TRÌNH BÀY TRONG BẢNG CÂN ĐỐI KẾ TOÁN", 5),
    "VI": ("NOTE_SEC_INCOME_STATEMENT", "VI. THÔNG TIN BỔ SUNG CHO CÁC KHOẢN MỤC TRÌNH BÀY TRONG BÁO CÁO KẾT QUẢ KINH DOANH", 6),
    "VII": ("NOTE_SEC_CASH_FLOW", "VII. THÔNG TIN BỔ SUNG CHO CÁC KHOẢN MỤC TRÌNH BÀY TRONG BÁO CÁO LƯU CHUYỂN TIỀN TỆ", 7),
    "VIII": ("NOTE_SEC_OTHER_INFO", "VIII. NHỮNG THÔNG TIN KHÁC", 8),
}

# Danh sách tiêu đề BCTC cấp 1 phổ biến tại Việt Nam (hỗ trợ legacy matching)
_PRIMARY_STATEMENTS = [
    ("bảng cân đối kế toán", "financial_statements"),
    ("báo cáo tình hình tài chính", "financial_statements"),
    ("báo cáo kết quả hoạt động kinh doanh", "financial_statements"),
    ("báo cáo kết quả kinh doanh", "financial_statements"),
    ("báo cáo lưu chuyển tiền tệ", "financial_statements"),
    ("thuyết minh báo cáo tài chính", "notes"),
    ("bản thuyết minh báo cáo tài chính", "notes"),
    ("báo cáo của công ty kiểm toán độc lập", "auditor_report"),
    ("báo cáo kiểm toán độc lập", "auditor_report"),
    ("báo cáo của ban tổng giám đốc", "mda"),
    ("báo cáo của ban giám đốc", "mda"),
    ("báo cáo của hội đồng quản trị", "mda"),
]


def slugify_vietnamese(text: str) -> str:
    """Chuyển đổi chuỗi tiếng Việt thành slug thân thiện cho ID."""
    if not text:
        return "section"
    text = text.lower().replace("đ", "d")
    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    slug = re.sub(r"[^\w\s-]", "", without_accents)
    slug = re.sub(r"[-\s]+", "_", slug).strip("_")
    return slug[:40] or "section"


@dataclass
class HeadingMatch:
    """Thực thể kết quả nhận diện tiêu đề."""
    title: str
    level: int                                              # 2: H2, 3: H3, 4: H4, 5: H5
    canonical_code: str | None = None
    roman_code: str | None = None
    item_code: str | None = None
    sub_code: str | None = None
    section_type: str = "notes"


class HierarchyState:
    """Máy trạng thái theo dõi ngữ cảnh cây phân cấp tài liệu BCTC."""
    def __init__(self):
        self.emitted_core_major: bool = False
        self.emitted_notes_major: bool = False
        self.current_major_id: str | None = None
        self.current_major_title: str = ""
        
        self.current_roman: str = ""
        self.current_roman_num: int = 0
        self.current_roman_title: str = ""
        self.current_roman_sec_id: str | None = None
        
        self.current_note_num: int = 0
        self.current_note_title: str = ""
        self.current_note_sec_id: str | None = None
        
        self.current_sub_code: str = ""
        self.current_sub_sec_id: str | None = None


class SectionDetector:
    """Bộ nhận diện tiêu đề và phân đoạn Section phân cấp chuẩn TT 200."""

    def __init__(self, enable_major_sections: bool = False) -> None:
        self.enable_major_sections = enable_major_sections

    def detect_sections(
        self,
        blocks: list[ClassifiedBlock],
        company: str = "DOANH_NGHIEP",
        year: int = 2024,
        toc_ranges: dict[str, tuple[int, int]] | None = None,
    ) -> list[Section]:
        """
        Duyệt qua danh sách ClassifiedBlock và gom nhóm thành cây Section ngữ nghĩa có cấu trúc.
        """
        if not blocks:
            return []

        # Bước tiền xử lý: Tách các block lớn nếu chứa nhiều tiêu đề nội bộ
        refined_blocks = self._preprocess_split_blocks(blocks)

        sections: list[Section] = []
        current_section: Section | None = None
        state = HierarchyState()
        section_idx = 1
        for block in refined_blocks:
            match = self._extract_heading(block, state)

            if match:
                # Nếu kích hoạt major sections (Level 2), tự động chèn Section H2 phân tách đại phân vùng
                if self.enable_major_sections:
                    major_sec = self._maybe_create_major_section(block, match, state, company, year)
                    if major_sec:
                        if current_section and current_section.blocks:
                            sections.append(current_section)
                            current_section = None
                        sections.append(major_sec)

                # Nếu tiêu đề trùng với canonical_code đang active (ví dụ báo cáo LCTT hoặc BCĐKT kéo dài 2 trang tiếp theo)
                if (
                    current_section
                    and match.canonical_code
                    and current_section.canonical_code == match.canonical_code
                ):
                    current_section.blocks.append(block)
                    current_section.page_end = max(current_section.page_end, block.page)
                    block.breadcrumb = current_section.breadcrumb
                    continue

                # Đóng section trước đó nếu đã có
                if current_section and current_section.blocks:
                    sections.append(current_section)

                # Tạo Section mới theo cấp bậc
                slug = slugify_vietnamese(match.title)
                ref_str = f"_{slugify_vietnamese(match.item_code)}" if match.item_code else ""
                sec_id = f"{company.lower()}_{year}_s_{slug}{ref_str}_{section_idx}"
                section_idx += 1

                # Xác định parent_id và breadcrumb theo phân cấp
                parent_id, breadcrumb = self._resolve_hierarchy(match, state)

                # Cập nhật State Machine
                self._update_state(match, sec_id, state)

                current_section = Section(
                    id=sec_id,
                    title=match.title,
                    company=company,
                    year=year,
                    page_start=block.page,
                    page_end=block.page,
                    blocks=[block],
                    section_type=match.section_type,
                    level=match.level,
                    parent_id=parent_id,
                    canonical_code=match.canonical_code,
                    reference_code=match.item_code,
                    breadcrumb=breadcrumb,
                    metadata={"heading_source": block.block_id},
                )
                block.breadcrumb = breadcrumb
                block.heading_level = match.level

            else:
                # Nếu chưa có section nào (đầu tài liệu), tạo section mở đầu
                if current_section is None:
                    sec_id = f"{company.lower()}_{year}_s_thong_tin_chung"
                    current_section = Section(
                        id=sec_id,
                        title="THÔNG TIN CHUNG VÀ MỞ ĐẦU",
                        company=company,
                        year=year,
                        page_start=block.page,
                        page_end=block.page,
                        blocks=[block],
                        section_type="general_info",
                        level=2,
                        breadcrumb="Báo cáo tài chính > Thông tin chung",
                    )
                    state.current_major_id = sec_id
                    state.current_major_title = "THÔNG TIN CHUNG VÀ MỞ ĐẦU"
                    block.breadcrumb = current_section.breadcrumb
                else:
                    current_section.blocks.append(block)
                    current_section.page_end = max(current_section.page_end, block.page)
                    block.breadcrumb = current_section.breadcrumb

        # Lưu section cuối cùng
        if current_section and current_section.blocks:
            sections.append(current_section)

        # Gắn child_metadata cho tất cả sections (phục vụ Parent-Child RAG)
        for sec in sections:
            sec.child_metadata = {
                "chunk_id": sec.id,
                "title": sec.title,
                "level": sec.level,
                "reference_code": sec.reference_code or "",
                "canonical_code": sec.canonical_code or "",
                "breadcrumb": sec.breadcrumb,
                "parent_id": sec.parent_id or "",
                "page_start": sec.page_start,
                "page_end": sec.page_end,
                "block_count": len(sec.blocks),
                "has_table": any(b.is_table for b in sec.blocks),
            }

        logger.info(
            "SectionDetector: Đã phân đoạn thành công %d sections có phân cấp cho BCTC %s_%d.",
            len(sections),
            company,
            year,
        )
        return sections

    def _extract_heading(self, block: ClassifiedBlock, state: HierarchyState) -> HeadingMatch | None:
        """Nhận diện tiêu đề đa chiến lược (Core Statements, Roman, Numbered Note, Sub-items)."""
        # 1. Bảng BCTC chính mà header chứa tên báo cáo
        if block.block.is_table:
            header = block.block.get_header_row().lower()
            for title_kw, sec_type in _PRIMARY_STATEMENTS:
                if title_kw in header:
                    return HeadingMatch(
                        title=title_kw.title(),
                        level=3,
                        section_type=sec_type,
                    )
            return None

        # 2. Văn bản Text: làm sạch rác boilerplate trước khi kiểm tra dòng đầu
        clean_content = strip_boilerplate_lines(block.content)
        lines = [l.strip() for l in clean_content.splitlines() if l.strip()]
        if not lines:
            return None

        first_line = lines[0]
        first_clean = re.sub(r"^[#*>\-\s]+", "", first_line).strip()
        first_low = first_clean.lower()

        # a. Nhận diện Phần La Mã (Level 3 - H3: I, II, III, IV, V, VI, VII, VIII)
        roman_res = self._match_roman_section(first_clean, first_low)
        if roman_res:
            can_code, canon_title, r_num = roman_res
            # Guard: kiểm tra tính đơn điệu (Roman num không được thụt lùi)
            if r_num >= state.current_roman_num:
                roman_str = list(ROMAN_CANONICAL.keys())[r_num - 1]
                return HeadingMatch(
                    title=canon_title,
                    level=3,
                    canonical_code=can_code,
                    roman_code=roman_str,
                    section_type="notes",
                )

        # b. Nhận diện Báo cáo tài chính cốt lõi (Level 3 - H3: Trang 5-12)
        if block.page <= 12 and not any(k in first_low for k in ["thông tin bổ sung", "thuyết minh", "bổ sung"]):
            if (
                re.match(r"^(báo cáo tình hình tài chính|bảng cân đối kế toán)", first_low)
                or (len(first_clean) < 75 and any(kw in first_low for kw in ["báo cáo tình hình tài chính", "bảng cân đối kế toán"]))
            ):
                return HeadingMatch(
                    title=CORE_STATEMENT_CANONICAL["CORE_BALANCE_SHEET"],
                    level=3,
                    canonical_code="CORE_BALANCE_SHEET",
                    section_type="financial_statements",
                )
            if (
                re.match(r"^(báo cáo kết quả hoạt động kinh doanh|báo cáo kết quả kinh doanh)", first_low)
                or (len(first_clean) < 75 and any(kw in first_low for kw in ["báo cáo kết quả hoạt động kinh doanh", "báo cáo kết quả kinh doanh"]))
            ):
                return HeadingMatch(
                    title=CORE_STATEMENT_CANONICAL["CORE_INCOME_STATEMENT"],
                    level=3,
                    canonical_code="CORE_INCOME_STATEMENT",
                    section_type="financial_statements",
                )
            if (
                re.match(r"^(báo cáo lưu chuyển tiền tệ)", first_low)
                or (len(first_clean) < 75 and "báo cáo lưu chuyển tiền tệ" in first_low)
            ):
                return HeadingMatch(
                    title=CORE_STATEMENT_CANONICAL["CORE_CASH_FLOW"],
                    level=3,
                    canonical_code="CORE_CASH_FLOW",
                    section_type="financial_statements",
                )

        # c. Nhận diện Mục số Thuyết minh (Level 4 - H4: "1. Tiền...", "19. Vốn chủ sở hữu...")
        num_res = self._match_numbered_note(first_clean, first_low, state)
        if num_res:
            num, clean_title = num_res
            ref_code = f"{state.current_roman}.{num}" if state.current_roman else f"{num}"
            return HeadingMatch(
                title=f"{num}. {clean_title}",
                level=4,
                item_code=ref_code,
                section_type="numeric_note",
            )

        # d. Nhận diện Tiểu mục chữ Thuyết minh (Level 5 - H5: "(a) Các công ty con...")
        sub_res = self._match_sub_item(first_clean, first_low)
        if sub_res and (state.current_note_num > 0 or state.current_roman_num > 0):
            sub_code, clean_title = sub_res
            ref_code = f"{state.current_roman}.{state.current_note_num}({sub_code})" if state.current_roman and state.current_note_num else f"({sub_code})"
            return HeadingMatch(
                title=f"({sub_code}) {clean_title}",
                level=5,
                sub_code=sub_code,
                item_code=ref_code,
                section_type="sub_item",
            )

        # e. Fallback kiểm tra danh sách _PRIMARY_STATEMENTS cho các văn bản khác (Kiểm toán, HĐQT)
        for title_kw, sec_type in _PRIMARY_STATEMENTS:
            if title_kw == first_low or first_low.startswith(title_kw):
                return HeadingMatch(
                    title=first_clean.upper(),
                    level=3,
                    section_type=sec_type,
                )

        return None

    def _match_roman_section(self, text: str, text_low: str) -> tuple[str, str, int] | None:
        """Nhận diện phần La Mã dựa trên Prefix Regex và Từ khóa Thông tư 200."""
        # 1. Prefix La Mã trực tiếp
        m = re.match(r"^(VIII|VII|VI|IV|V|III|II|I|L|1)[\.:\s\-]+(.*)$", text, re.I)
        if m:
            pref = m.group(1).upper()
            rest = m.group(2).strip().lower()
            if pref in ("L", "1") and ("thông tin doanh nghiệp" in rest or "đặc điểm" in rest):
                return ROMAN_CANONICAL["I"]
            if pref in ROMAN_CANONICAL and len(rest) > 2:
                return ROMAN_CANONICAL[pref]

        # 2. Nhận diện từ khóa không phụ thuộc số La Mã
        if "thông tin doanh nghiệp" in text_low or "đặc điểm hoạt động của doanh nghiệp" in text_low:
            return ROMAN_CANONICAL["I"]
        if "kỳ kế toán" in text_low and "tiền tệ" in text_low:
            return ROMAN_CANONICAL["II"]
        if "chuẩn mực" in text_low and "chế độ kế toán" in text_low:
            return ROMAN_CANONICAL["III"]
        if "chính sách kế toán" in text_low and len(text) < 80:
            return ROMAN_CANONICAL["IV"]
        if ("thông tin bổ sung" in text_low or "thông tin bo sung" in text_low) and (
            "bảng cân đối" in text_low or "trình báo cáo" in text_low or "trình bày" in text_low
        ):
            return ROMAN_CANONICAL["V"]
        if ("thông tin bổ sung" in text_low or "thông tin bo sung" in text_low) and "kết quả" in text_low:
            return ROMAN_CANONICAL["VI"]
        if ("thông tin bổ sung" in text_low or "thông tin bo sung" in text_low) and "lưu chuyển" in text_low:
            return ROMAN_CANONICAL["VII"]
        if "những thông tin khác" in text_low or "thông tin khác" in text_low:
            return ROMAN_CANONICAL["VIII"]

        return None

    def _match_numbered_note(self, text: str, text_low: str, state: HierarchyState) -> tuple[int, str] | None:
        """Nhận diện mục số thuyết minh (ví dụ: '1. Tiền...', '19. Vốn chủ sở hữu...')."""
        m = re.match(r"^(\d{1,2})[\.:\s\-]+([A-ZÀ-Ỹa-zà-ỹ0-9\s,–—\-\/]{3,80})$", text)
        if not m:
            return None

        num = int(m.group(1))
        title = m.group(2).strip()

        # Loại trừ các câu không phải đề mục
        if title.endswith(".") or title.endswith(";") or title.endswith(","):
            return None
        if any(unit in text_low for unit in ["vnd", "triệu đồng", "tỷ đồng", "%"]):
            return None
        if re.search(r"\b(ngày \d+|\d{1,2}/\d{1,2}/\d{4})\b", text_low):
            return None
        if re.search(r"\b\d{1,3}\.\d{3}\b", title):  # chứa số tiền dạng 10.000
            return None

        # Guard tính đơn điệu trong cùng một phần La Mã:
        # Nếu đang ở Phần V và số nhảy từ 15 lên 19 là hợp lý, nhưng từ 19 tụt về 1 (khi chưa sang Phần VI) thì coi là văn bản thường
        if state.current_roman_num == 5 and state.current_note_num >= 15 and num < 5:
            return None

        return num, title

    def _match_sub_item(self, text: str, text_low: str) -> tuple[str, str] | None:
        """Nhận diện tiểu mục chữ (ví dụ: '(a) Các công ty con', '(b) Biến động vốn...')."""
        m = re.match(r"^\(([a-zđ0-9]{1,2})\)[\s\.:\-]+([A-ZÀ-Ỹa-zà-ỹ0-9\s,–—\-\/]{3,80})$", text)
        if not m:
            m = re.match(r"^([a-zđ])\)[\s\.:\-]+([A-ZÀ-Ỹa-zà-ỹ0-9\s,–—\-\/]{3,80})$", text)
        if not m:
            return None

        sub_code = m.group(1).lower()
        title = m.group(2).strip()

        if title.endswith(";") or title.endswith(","):
            return None
        if any(unit in text_low for unit in ["vnd", "triệu đồng"]):
            return None

        return sub_code, title

    def _maybe_create_major_section(
        self,
        block: ClassifiedBlock,
        match: HeadingMatch,
        state: HierarchyState,
        company: str,
        year: int,
    ) -> Section | None:
        """Tự động chèn Section H2 phân tách giữa Phần BCTC Cốt Lõi và Bản Thuyết Minh."""
        if match.canonical_code in CORE_STATEMENT_CANONICAL and not state.emitted_core_major:
            state.emitted_core_major = True
            sec_id = f"{company.lower()}_{year}_s_part_1_core"
            state.current_major_id = sec_id
            state.current_major_title = "PHẦN 1: BÁO CÁO TÀI CHÍNH CỐT LÕI"
            return Section(
                id=sec_id,
                title="PHẦN 1: BÁO CÁO TÀI CHÍNH CỐT LÕI",
                company=company,
                year=year,
                page_start=block.page,
                page_end=block.page,
                blocks=[],
                section_type="major_part",
                level=2,
                breadcrumb="Báo cáo tài chính > PHẦN 1: BÁO CÁO TÀI CHÍNH CỐT LÕI",
            )

        if (match.roman_code or match.section_type in ("notes", "numeric_note")) and not state.emitted_notes_major:
            state.emitted_notes_major = True
            sec_id = f"{company.lower()}_{year}_s_part_2_notes"
            state.current_major_id = sec_id
            state.current_major_title = "PHẦN 2: BẢN THUYẾT MINH BÁO CÁO TÀI CHÍNH"
            return Section(
                id=sec_id,
                title="PHẦN 2: BẢN THUYẾT MINH BÁO CÁO TÀI CHÍNH",
                company=company,
                year=year,
                page_start=block.page,
                page_end=block.page,
                blocks=[],
                section_type="major_part",
                level=2,
                breadcrumb="Báo cáo tài chính > PHẦN 2: BẢN THUYẾT MINH BÁO CÁO TÀI CHÍNH",
            )

        return None

    def _resolve_hierarchy(self, match: HeadingMatch, state: HierarchyState) -> tuple[str | None, str]:
        """Xác định parent_id và breadcrumb phân cấp cho từng Section."""
        major_title = state.current_major_title or "Báo cáo tài chính"
        
        if match.level == 2:
            return None, major_title

        if match.level == 3:
            parent_id = state.current_major_id
            breadcrumb = f"{major_title} > {match.title}"
            return parent_id, breadcrumb

        if match.level == 4:
            parent_id = state.current_roman_sec_id or state.current_major_id
            roman_title = state.current_roman_title or "Thuyết minh BCTC"
            breadcrumb = f"{major_title} > {roman_title} > {match.title}"
            return parent_id, breadcrumb

        if match.level == 5:
            parent_id = state.current_note_sec_id or state.current_roman_sec_id or state.current_major_id
            roman_title = state.current_roman_title or "Thuyết minh BCTC"
            note_title = state.current_note_title or "Mục chi tiết"
            breadcrumb = f"{major_title} > {roman_title} > {note_title} > {match.title}"
            return parent_id, breadcrumb

        return state.current_major_id, f"{major_title} > {match.title}"

    def _update_state(self, match: HeadingMatch, sec_id: str, state: HierarchyState):
        """Cập nhật trạng thái của Hierarchy State Tracker."""
        if match.level == 2:
            state.current_major_id = sec_id
            state.current_major_title = match.title
        elif match.level == 3:
            if match.roman_code:
                state.current_roman = match.roman_code
                state.current_roman_num = ROMAN_CANONICAL.get(match.roman_code, (None, None, 1))[2]
                state.current_roman_title = match.title
                state.current_roman_sec_id = sec_id
                state.current_note_num = 0
                state.current_note_title = ""
                state.current_sub_code = ""
        elif match.level == 4:
            state.current_note_sec_id = sec_id
            state.current_note_title = match.title
            m = re.match(r"^(\d+)", match.title)
            if m:
                state.current_note_num = int(m.group(1))
            state.current_sub_code = ""
        elif match.level == 5:
            state.current_sub_sec_id = sec_id
            state.current_sub_code = match.sub_code or ""

    def _preprocess_split_blocks(self, blocks: list[ClassifiedBlock]) -> list[ClassifiedBlock]:
        """
        Tiền xử lý: Nếu một khối văn bản chứa nhiều tiêu đề phân cấp liên tiếp
        (ví dụ: 'L THÔNG TIN DOANH NGHIỆP' và '1. Hình thức sở hữu vốn'),
        tự động phân tách thành các blocks độc lập để mỗi đề mục sở hữu nội dung riêng.
        """
        output: list[ClassifiedBlock] = []
        for cb in blocks:
            if cb.is_table or not cb.content:
                output.append(cb)
                continue

            cleaned = strip_boilerplate_lines(cb.content)
            lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
            if len(lines) <= 1:
                output.append(cb)
                continue

            # Tìm các vị trí dòng là tiêu đề mới (từ dòng 1 trở đi)
            split_indices = []
            for idx in range(1, len(lines)):
                line = lines[idx]
                # Kiểm tra xem dòng có phải là tiêu đề số hoặc La Mã không
                if (
                    re.match(r"^\d{1,2}\.\s+[A-ZÀ-Ỹ]", line)
                    or re.match(r"^[IVXLCDM]+\.\s+[A-ZÀ-Ỹ]", line)
                    or re.match(r"^\([a-zđ]\)\s+[A-ZÀ-Ỹ]", line)
                ):
                    split_indices.append(idx)

            if not split_indices:
                output.append(cb)
                continue

            # Thực hiện phân tách
            prev_idx = 0
            split_indices.append(len(lines))
            for chunk_i, split_idx in enumerate(split_indices):
                sub_lines = lines[prev_idx:split_idx]
                if sub_lines:
                    sub_content = "\n".join(sub_lines)
                    from src.models import ParsedBlock
                    new_pb = ParsedBlock(
                        block_id=f"{cb.block_id}_sub_{chunk_i}",
                        block_type=cb.block.block_type,
                        page=cb.page,
                        content=sub_content,
                        bbox=cb.block.bbox,
                        source=cb.block.source,
                        metadata=cb.block.metadata.copy() if cb.block.metadata else {},
                    )
                    new_cb = ClassifiedBlock(
                        block=new_pb,
                        block_type=cb.block_type,
                        target=list(cb.target),
                        confidence=cb.confidence,
                        classification_method=cb.classification_method,
                        reasoning=cb.reasoning,
                    )
                    output.append(new_cb)
                prev_idx = split_idx

        return output
