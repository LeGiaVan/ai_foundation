"""
table_utils.py — Tiện ích chuẩn hóa bảng biểu, số liệu tài chính và Markdown.
Xử lý các đặc thù của Báo cáo tài chính Việt Nam:
- Số âm trong ngoặc đơn: `(1,234)` -> `-1234`
- Định dạng số Việt Nam (chấm/phẩy) và quốc tế
- Header đa tầng (merged cells)
- Nhận diện đơn vị tiền tệ (VND, Triệu VND, Tỷ VND, USD)
- Tính toán mật độ số (Numeric Density) để phân loại bảng
"""

import re
from typing import Any

# Regex phát hiện số âm trong ngoặc đơn: vd: (1,234), ( 1.234.567 ), (50.5)
_PAREN_NEGATIVE_RE = re.compile(r"^\s*\(\s*([\d\s.,]+)\s*\)\s*$")

# Regex kiểm tra chuỗi có cấu trúc số (bao gồm cả dấu âm, phẩy, chấm)
_NUMERIC_PATTERN_RE = re.compile(r"^-?\s*[\d]{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?$|^-?\s*\d+(?:[.,]\d+)?$")

# Regex tìm đơn vị tính trong văn bản hoặc header
_CURRENCY_PATTERNS = [
    (re.compile(r"đơn\s*vị\s*tính\s*:\s*tỷ\s*(?:đồng|vnd)", re.IGNORECASE), "TY_VND"),
    (re.compile(r"đơn\s*vị\s*tính\s*:\s*triệu\s*(?:đồng|vnd)", re.IGNORECASE), "TRIEU_VND"),
    (re.compile(r"đơn\s*vị\s*tính\s*:\s*nghìn\s*(?:đồng|vnd)", re.IGNORECASE), "NGHIN_VND"),
    (re.compile(r"đơn\s*vị\s*tính\s*:\s*(?:đồng|vnd)", re.IGNORECASE), "VND"),
    (re.compile(r"đơn\s*vị\s*tính\s*:\s*usd", re.IGNORECASE), "USD"),
    (re.compile(r"\b(?:tỷ\s*đồng|tỷ\s*vnd)\b", re.IGNORECASE), "TY_VND"),
    (re.compile(r"\b(?:triệu\s*đồng|triệu\s*vnd)\b", re.IGNORECASE), "TRIEU_VND"),
    (re.compile(r"\b(?:đồng\s*việt\s*nam|vnd)\b", re.IGNORECASE), "VND"),
]


def clean_cell_value(val: Any) -> str:
    """Làm sạch giá trị một ô bảng: loại bỏ newline, tab, khoảng trắng thừa."""
    if val is None:
        return ""
    text = str(val).strip()
    if not text or text.lower() in ("none", "nan", "null"):
        return ""
    # Chuyển các ký tự gạch ngang unicode thành '-'
    text = text.replace("–", "-").replace("—", "-")
    # Thay thế newlines và khoảng trắng thừa bên trong ô
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def parse_financial_number(text: str) -> tuple[float | None, bool]:
    """
    Phân tích chuỗi thành số thực tài chính.
    Hỗ trợ:
      - Số âm trong ngoặc: `(1,234)` -> `-1234.0`, True
      - Số âm dấu trừ: `-1,234` -> `-1234.0`, True
      - Định dạng VN: `1.234.567,89` -> `1234567.89`
      - Định dạng US/EN: `1,234,567.89` -> `1234567.89`
      - Ký hiệu gạch ngang biểu thị 0: "-" -> 0.0, False

    Returns:
        (parsed_float, is_negative)
    """
    clean_str = clean_cell_value(text)
    if not clean_str:
        return None, False

    # Gạch ngang đơn thuần trong BCTC thường đại diện cho giá trị 0 hoặc không có
    if clean_str in ("-", "—", "–", "‐"):
        return 0.0, False

    is_negative = False

    # Kiểm tra số âm trong ngoặc
    paren_match = _PAREN_NEGATIVE_RE.match(clean_str)
    if paren_match:
        is_negative = True
        raw_num = paren_match.group(1).strip()
    else:
        raw_num = clean_str
        if raw_num.startswith("-"):
            is_negative = True
            raw_num = raw_num[1:].strip()

    # Bỏ dấu khoảng trắng bên trong số (ví dụ '1 000 000')
    raw_num = raw_num.replace(" ", "")

    # Phân biệt định dạng VN (chấm phân cách nghìn, phẩy thập phân)
    # vs EN (phẩy phân cách nghìn, chấm thập phân)
    has_dot = "." in raw_num
    has_comma = "," in raw_num

    try:
        if has_dot and has_comma:
            last_dot = raw_num.rfind(".")
            last_comma = raw_num.rfind(",")
            if last_comma > last_dot:
                # Dạng VN: 1.234.567,89
                std_num = raw_num.replace(".", "").replace(",", ".")
            else:
                # Dạng EN: 1,234,567.89
                std_num = raw_num.replace(",", "")
        elif has_dot:
            parts = raw_num.split(".")
            # Nếu phần sau dấu chấm có đúng 3 chữ số và xuất hiện nhiều lần -> phân cách nghìn
            if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3 and not raw_num.startswith("0.")):
                std_num = raw_num.replace(".", "")
            else:
                std_num = raw_num
        elif has_comma:
            parts = raw_num.split(",")
            # Nếu phần sau dấu phẩy có đúng 3 chữ số và xuất hiện nhiều lần -> phân cách nghìn kiểu EN
            if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3 and not raw_num.startswith("0,")):
                std_num = raw_num.replace(",", "")
            else:
                std_num = raw_num.replace(",", ".")
        else:
            std_num = raw_num

        val = float(std_num)
        if is_negative:
            val = -abs(val)
        return val, is_negative
    except ValueError:
        return None, False


def normalize_cell_for_markdown(text: str) -> str:
    """
    Chuẩn hóa ô bảng khi render Markdown:
    Nếu là số âm ngoặc đơn `(1,234)` thì chuyển thành `-1,234` để mô hình và người đọc dễ hiểu.
    Nếu ô có ký tự pipe `|` thì escape thành `\\|`.
    """
    val, is_neg = parse_financial_number(text)
    cleaned = clean_cell_value(text)
    if is_neg and val is not None and "(" in cleaned:
        # Chuyển (1,234) -> -1,234
        inside = _PAREN_NEGATIVE_RE.match(cleaned)
        if inside:
            cleaned = f"-{inside.group(1).strip()}"

    # Escape markdown pipe
    return cleaned.replace("|", "\\|")


def detect_currency_unit(text: str) -> str | None:
    """Quét văn bản hoặc tiêu đề để tìm đơn vị tiền tệ."""
    if not text:
        return None
    for pattern, unit in _CURRENCY_PATTERNS:
        if pattern.search(text):
            return unit
    return None


def compute_numeric_density(table: list[list[str]]) -> float:
    """
    Tính mật độ số (Numeric Density) trong dữ liệu bảng (bỏ qua dòng header đầu).
    Numeric density cao (> 0.35 - 0.40) là đặc trưng của bảng số liệu BCTC và thuyết minh số.
    """
    if not table or len(table) <= 1:
        return 0.0

    total_cells = 0
    numeric_cells = 0

    # Duyệt qua các dòng data (từ dòng 1 trở đi)
    for row in table[1:]:
        for cell in row:
            cleaned = clean_cell_value(cell)
            if not cleaned:
                continue
            total_cells += 1
            val, _ = parse_financial_number(cleaned)
            if val is not None:
                numeric_cells += 1

    if total_cells == 0:
        return 0.0
    return numeric_cells / total_cells


def merge_multi_level_headers(header_rows: list[list[str]]) -> list[str]:
    """
    Gộp các dòng tiêu đề đa tầng (merged header rows) thành một dòng duy nhất.
    Ví dụ:
      Row 0: ["", "Năm 2024", ""]
      Row 1: ["Chỉ tiêu", "Kỳ này", "Kỳ trước"]
    -> Kết quả: ["Chỉ tiêu", "Năm 2024 - Kỳ này", "Năm 2024 - Kỳ trước"]
    """
    if not header_rows:
        return []
    if len(header_rows) == 1:
        return [clean_cell_value(c) for c in header_rows[0]]

    max_cols = max(len(r) for r in header_rows)
    normalized_rows = []
    for r in header_rows:
        row_padded = [clean_cell_value(c) for c in r]
        while len(row_padded) < max_cols:
            row_padded.append("")
        normalized_rows.append(row_padded)

    # Lan truyền header cha sang các ô rỗng bên phải (forward fill merged header)
    filled_parent = list(normalized_rows[0])
    current_parent = ""
    for idx, cell in enumerate(filled_parent):
        if cell:
            current_parent = cell
        else:
            filled_parent[idx] = current_parent

    combined_headers = []
    for col_idx in range(max_cols):
        col_parts = []
        # Phần tử từ dòng cha
        p = filled_parent[col_idx]
        if p:
            col_parts.append(p)
        # Các dòng con
        for r_idx in range(1, len(normalized_rows)):
            child_val = normalized_rows[r_idx][col_idx]
            if child_val and child_val != p:
                col_parts.append(child_val)

        final_header = " - ".join(col_parts) if col_parts else f"Cột_{col_idx + 1}"
        combined_headers.append(final_header)

    return combined_headers


def format_table_to_markdown(table: list[list[Any]], header_rows_count: int = 1) -> str:
    """
    Chuyển ma trận bảng biểu thành Markdown Table chuẩn GitHub Flavored Markdown (GFM).
    - Tự động gộp header đa tầng nếu `header_rows_count > 1`.
    - Chuẩn hóa số âm ngoặc đơn `(1,234)` -> `-1,234`.
    - Thêm dòng ngăn cách `| --- | --- |`.
    - Loại bỏ các dòng hoàn toàn rỗng.
    """
    if not table or not any(table):
        return ""

    # Làm sạch toàn bộ các ô
    cleaned_table: list[list[str]] = []
    for row in table:
        if not row:
            continue
        cleaned_row = [clean_cell_value(c) for c in row]
        # Bỏ qua nếu cả dòng không có chữ nào
        if any(cleaned_row):
            cleaned_table.append(cleaned_row)

    if not cleaned_table:
        return ""

    num_cols = max(len(r) for r in cleaned_table)

    # Đồng bộ số lượng cột cho tất cả các hàng
    padded_table: list[list[str]] = []
    for r in cleaned_table:
        row_copy = list(r)
        while len(row_copy) < num_cols:
            row_copy.append("")
        padded_table.append(row_copy)

    # Xử lý Header
    if header_rows_count > 1 and len(padded_table) >= header_rows_count:
        header_rows = padded_table[:header_rows_count]
        header = merge_multi_level_headers(header_rows)
        data_rows = padded_table[header_rows_count:]
    else:
        header = [
            normalize_cell_for_markdown(c) or f"Cột_{i+1}"
            for i, c in enumerate(padded_table[0])
        ]
        data_rows = padded_table[1:]

    lines: list[str] = []
    # Dòng Header
    lines.append("| " + " | ".join(header) + " |")
    # Dòng phân cách
    lines.append("| " + " | ".join(["---"] * num_cols) + " |")

    # Các dòng dữ liệu
    for row in data_rows:
        norm_row = [normalize_cell_for_markdown(c) for c in row]
        lines.append("| " + " | ".join(norm_row) + " |")

    return "\n".join(lines)
