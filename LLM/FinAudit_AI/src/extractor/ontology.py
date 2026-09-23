"""
ontology.py — Financial Ontology và Canonical Concept Mapping theo Chuẩn mực Kế toán Việt Nam (Thông tư 200/2014/TT-BTC).
Định nghĩa ánh xạ từ mã số BCTC (100, 110, 270, 300, 440...) và tên chỉ tiêu tiếng Việt sang Canonical Concept.
Hỗ trợ phân định rành mạch 3 báo cáo: Bảng Cân đối kế toán (BS), Kết quả kinh doanh (IS), Lưu chuyển tiền tệ (CF).
"""

import re
from typing import NamedTuple


class ConceptDefinition(NamedTuple):
    concept: str
    code: str
    standard_name: str
    aliases: list[str]
    statement_type: str  # "BALANCE_SHEET" | "INCOME_STATEMENT" | "CASH_FLOW"


# Danh mục các khoản mục tài chính trọng yếu theo Thông tư 200
ONTOLOGY_DEFINITIONS: list[ConceptDefinition] = [
    # ── BẢNG CÂN ĐỐI KẾ TOÁN (TÀI SẢN - CÁC KHOẢN MỤC CHÍNH) ───────────────────
    ConceptDefinition(
        concept="TOTAL_ASSETS",
        code="270",
        standard_name="TỔNG CỘNG TÀI SẢN",
        aliases=[
            r"tổng cộng tài sản",
            r"tong cong tai san",
            r"tổng tài sản",
            r"tong tai san",
            r"^tài sản$",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="CURRENT_ASSETS",
        code="100",
        standard_name="TÀI SẢN NGẮN HẠN",
        aliases=[
            r"tài sản ngắn hạn",
            r"tai san ngan han",
            r"a\s*[-–]\s*tài sản ngắn hạn",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="CASH_AND_EQUIVALENTS",
        code="110",
        standard_name="Tiền và các khoản tương đương tiền",
        aliases=[
            r"tiền và các khoản tương đương tiền",
            r"tiền và tương đương tiền",
            r"tien va tuong duong tien",
            r"i\.\s*tiền và các khoản tương đương tiền",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="SHORT_TERM_INVESTMENTS",
        code="120",
        standard_name="Đầu tư tài chính ngắn hạn",
        aliases=[
            r"đầu tư tài chính ngắn hạn",
            r"dau tu tai chinh ngan han",
            r"các khoản đầu tư tài chính ngắn hạn",
            r"ii\.\s*đầu tư tài chính ngắn hạn",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="SHORT_TERM_RECEIVABLES",
        code="130",
        standard_name="Các khoản phải thu ngắn hạn",
        aliases=[
            r"^(?:iii\.\s*)?(?:các\s+khoản\s+)?phải\s+thu\s+ngắn\s+hạn(?:\s*\(.*\))?$",
            r"các khoản phải thu ngắn hạn",
            r"phải thu ngắn hạn",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="INVENTORIES",
        code="140",
        standard_name="Hàng tồn kho",
        aliases=[
            r"^(?:iv\.\s*)?hàng\s+tồn\s+kho$",
            r"hàng tồn kho",
            r"hang ton kho",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="OTHER_CURRENT_ASSETS",
        code="150",
        standard_name="Tài sản ngắn hạn khác",
        aliases=[
            r"tài sản ngắn hạn khác",
            r"tai san ngan han khac",
            r"v\.\s*tài sản ngắn hạn khác",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="NON_CURRENT_ASSETS",
        code="200",
        standard_name="TÀI SẢN DÀI HẠN",
        aliases=[
            r"tài sản dài hạn",
            r"tai san dai han",
            r"b\s*[-–]\s*tài sản dài hạn",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="LONG_TERM_RECEIVABLES",
        code="210",
        standard_name="Các khoản phải thu dài hạn",
        aliases=[
            r"^(?:i\.\s*)?(?:các\s+khoản\s+)?phải\s+thu\s+dài\s+hạn(?:\s*\(.*\))?$",
            r"các khoản phải thu dài hạn",
            r"phải thu dài hạn",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="FIXED_ASSETS",
        code="220",
        standard_name="Tài sản cố định",
        aliases=[
            r"^(?:ii\.\s*)?tài\s+sản\s+cố\s+định(?:\s*\(.*\))?$",
            r"tài sản cố định",
            r"tai san co dinh",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="INVESTMENT_PROPERTIES",
        code="230",
        standard_name="Bất động sản đầu tư",
        aliases=[
            r"bất động sản đầu tư",
            r"bat dong san dau tu",
            r"iii\.\s*bất động sản đầu tư",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="LONG_TERM_ASSETS_IN_PROGRESS",
        code="240",
        standard_name="Tài sản dở dang dài hạn",
        aliases=[
            r"tài sản dở dang dài hạn",
            r"iv\.\s*tài sản dở dang dài hạn",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="LONG_TERM_INVESTMENTS",
        code="250",
        standard_name="Đầu tư tài chính dài hạn",
        aliases=[
            r"đầu tư tài chính dài hạn",
            r"các khoản đầu tư tài chính dài hạn",
            r"v\.\s*đầu tư tài chính dài hạn",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="OTHER_NON_CURRENT_ASSETS",
        code="260",
        standard_name="Tài sản dài hạn khác",
        aliases=[
            r"tài sản dài hạn khác",
            r"vi\.\s*tài sản dài hạn khác",
        ],
        statement_type="BALANCE_SHEET",
    ),

    # ── BẢNG CÂN ĐỐI KẾ TOÁN (CÁC KHOẢN MỤC CON CHI TIẾT) ─────────────────────
    ConceptDefinition(
        concept="SHORT_TERM_TRADE_RECEIVABLES",
        code="131",
        standard_name="Phải thu khách hàng",
        aliases=[r"phải thu khách hàng", r"phải thu của khách hàng"],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="SHORT_TERM_PREPAYMENTS",
        code="132",
        standard_name="Trả trước cho người bán",
        aliases=[r"trả trước cho người bán"],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="SHORT_TERM_OTHER_RECEIVABLES",
        code="136",
        standard_name="Phải thu ngắn hạn khác",
        aliases=[r"phải thu ngắn hạn khác"],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="SHORT_TERM_BAD_DEBT_PROVISION",
        code="137",
        standard_name="Dự phòng phải thu khó đòi",
        aliases=[r"dự phòng phải thu khó đòi", r"dự phòng phải thu ngắn hạn khó đòi"],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="INVENTORY_GROSS",
        code="141",
        standard_name="Hàng tồn kho (nguyên giá)",
        aliases=[r"hàng tồn kho\s*\(nguyên giá\)"],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="INVENTORY_PROVISION",
        code="149",
        standard_name="Dự phòng giảm giá hàng tồn kho",
        aliases=[r"dự phòng giảm giá hàng tồn kho"],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="LONG_TERM_OTHER_RECEIVABLES",
        code="216",
        standard_name="Phải thu dài hạn khác",
        aliases=[r"phải thu dài hạn khác"],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="TANGIBLE_FIXED_ASSETS",
        code="221",
        standard_name="Tài sản cố định hữu hình",
        aliases=[r"tài sản cố định hữu hình"],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="INTANGIBLE_FIXED_ASSETS",
        code="227",
        standard_name="Tài sản cố định vô hình",
        aliases=[r"tài sản cố định vô hình"],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="CONSTRUCTION_IN_PROGRESS",
        code="242",
        standard_name="Xây dựng cơ bản dở dang",
        aliases=[r"xây dựng cơ bản dở dang"],
        statement_type="BALANCE_SHEET",
    ),

    # ── BẢNG CÂN ĐỐI KẾ TOÁN (NGUỒN VỐN) ─────────────────────────────────────────
    ConceptDefinition(
        concept="TOTAL_RESOURCES",
        code="440",
        standard_name="TỔNG CỘNG NGUỒN VỐN",
        aliases=[
            r"tổng cộng nguồn vốn",
            r"tong cong nguon von",
            r"tổng nguồn vốn",
            r"^nguồn vốn$",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="LIABILITIES",
        code="300",
        standard_name="NỢ PHẢI TRẢ",
        aliases=[
            r"nợ phải trả",
            r"no phai tra",
            r"c\s*[-–]\s*nợ phải trả",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="CURRENT_LIABILITIES",
        code="310",
        standard_name="Nợ ngắn hạn",
        aliases=[
            r"nợ ngắn hạn",
            r"no ngan han",
            r"i\.\s*nợ ngắn hạn",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="NON_CURRENT_LIABILITIES",
        code="330",
        standard_name="Nợ dài hạn",
        aliases=[
            r"nợ dài hạn",
            r"no dai han",
            r"ii\.\s*nợ dài hạn",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="EQUITY",
        code="400",
        standard_name="VỐN CHỦ SỞ HỮU",
        aliases=[
            r"vốn chủ sở hữu",
            r"von chu so huu",
            r"d\s*[-–]\s*vốn chủ sở hữu",
        ],
        statement_type="BALANCE_SHEET",
    ),

    # ── BẢNG CÂN ĐỐI KẾ TOÁN (CHI TIẾT NỢ PHẢI TRẢ & VỐN CSH) ────────────────
    ConceptDefinition(
        concept="SHORT_TERM_TRADE_PAYABLES",
        code="311",
        standard_name="Phải trả người bán ngắn hạn",
        aliases=[
            r"phải trả người bán ngắn hạn",
            r"phải trả cho người bán ngắn hạn",
            r"phải trả người bán",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="SHORT_TERM_ADVANCES_FROM_CUSTOMERS",
        code="312",
        standard_name="Người mua trả tiền trước ngắn hạn",
        aliases=[
            r"người mua trả tiền trước ngắn hạn",
            r"người mua trả tiền trước",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="SHORT_TERM_UNEARNED_REVENUE",
        code="318",
        standard_name="Doanh thu chưa thực hiện ngắn hạn",
        aliases=[
            r"doanh thu chưa thực hiện ngắn hạn",
            r"doanh thu chưa thực hiện",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="SHORT_TERM_OTHER_PAYABLES",
        code="319",
        standard_name="Phải trả ngắn hạn khác",
        aliases=[
            r"phải trả ngắn hạn khác",
            r"phải trả khác ngắn hạn",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="SHORT_TERM_BORROWINGS",
        code="320",
        standard_name="Vay và nợ thuê tài chính ngắn hạn",
        aliases=[
            r"vay và nợ thuê tài chính ngắn hạn",
            r"vay ngắn hạn",
            r"vay và nợ ngắn hạn",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="LONG_TERM_UNEARNED_REVENUE",
        code="336",
        standard_name="Doanh thu chưa thực hiện dài hạn",
        aliases=[
            r"doanh thu chưa thực hiện dài hạn",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="LONG_TERM_OTHER_PAYABLES",
        code="337",
        standard_name="Phải trả dài hạn khác",
        aliases=[
            r"phải trả dài hạn khác",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="LONG_TERM_BORROWINGS",
        code="338",
        standard_name="Vay và nợ thuê tài chính dài hạn",
        aliases=[
            r"vay và nợ thuê tài chính dài hạn",
            r"vay dài hạn",
            r"vay và nợ dài hạn",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="CONTRIBUTED_CAPITAL",
        code="411",
        standard_name="Vốn góp của chủ sở hữu",
        aliases=[
            r"vốn góp của chủ sở hữu",
            r"vốn đầu tư của chủ sở hữu",
            r"vốn cổ phần",
            r"vốn điều lệ",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="SHARE_PREMIUM",
        code="412",
        standard_name="Thặng dư vốn cổ phần",
        aliases=[
            r"thặng dư vốn cổ phần",
            r"thặng dư vốn",
        ],
        statement_type="BALANCE_SHEET",
    ),
    ConceptDefinition(
        concept="RETAINED_EARNINGS",
        code="421",
        standard_name="Lợi nhuận sau thuế chưa phân phối",
        aliases=[
            r"lợi nhuận sau thuế chưa phân phối",
            r"lợi nhuận chưa phân phối",
        ],
        statement_type="BALANCE_SHEET",
    ),

    # ── BÁO CÁO KẾT QUẢ KINH DOANH (KQKD) ──────────────────────────────────────
    ConceptDefinition(
        concept="GROSS_REVENUE",
        code="01",
        standard_name="Doanh thu bán hàng và cung cấp dịch vụ",
        aliases=[
            r"doanh thu bán hàng và cung cấp dịch vụ",
            r"doanh thu bán hàng",
            r"tổng doanh thu",
            r"1\.\s*doanh thu bán hàng",
        ],
        statement_type="INCOME_STATEMENT",
    ),
    ConceptDefinition(
        concept="REVENUE_DEDUCTIONS",
        code="02",
        standard_name="Các khoản giảm trừ doanh thu",
        aliases=[
            r"các khoản giảm trừ doanh thu",
            r"giảm trừ doanh thu",
            r"2\.\s*các khoản giảm trừ",
        ],
        statement_type="INCOME_STATEMENT",
    ),
    ConceptDefinition(
        concept="NET_REVENUE",
        code="10",
        standard_name="Doanh thu thuần về bán hàng và cung cấp dịch vụ",
        aliases=[
            r"doanh thu thuần",
            r"doanh thu thuan",
            r"doanh thu thuần về bán hàng",
            r"3\.\s*doanh thu thuần",
        ],
        statement_type="INCOME_STATEMENT",
    ),
    ConceptDefinition(
        concept="COGS",
        code="11",
        standard_name="Giá vốn hàng bán",
        aliases=[
            r"giá vốn hàng bán",
            r"gia von hang ban",
            r"4\.\s*giá vốn hàng bán",
        ],
        statement_type="INCOME_STATEMENT",
    ),
    ConceptDefinition(
        concept="GROSS_PROFIT",
        code="20",
        standard_name="Lợi nhuận gộp về bán hàng và cung cấp dịch vụ",
        aliases=[
            r"lợi nhuận gộp",
            r"loi nhuan gop",
            r"5\.\s*lợi nhuận gộp",
        ],
        statement_type="INCOME_STATEMENT",
    ),
    ConceptDefinition(
        concept="FINANCIAL_INCOME",
        code="21",
        standard_name="Doanh thu hoạt động tài chính",
        aliases=[
            r"doanh thu hoạt động tài chính",
            r"doanh thu tài chính",
            r"6\.\s*doanh thu hoạt động tài chính",
        ],
        statement_type="INCOME_STATEMENT",
    ),
    ConceptDefinition(
        concept="FINANCIAL_EXPENSES",
        code="22",
        standard_name="Chi phí tài chính",
        aliases=[
            r"chi phí tài chính",
            r"chi phi tai chinh",
            r"7\.\s*chi phí tài chính",
        ],
        statement_type="INCOME_STATEMENT",
    ),
    ConceptDefinition(
        concept="INTEREST_EXPENSE",
        code="23",
        standard_name="Trong đó: Chi phí lãi vay",
        aliases=[
            r"chi phí lãi vay",
            r"trong đó: chi phí lãi vay",
            r"lãi tiền vay",
        ],
        statement_type="INCOME_STATEMENT",
    ),
    ConceptDefinition(
        concept="SELLING_EXPENSES",
        code="25",
        standard_name="Chi phí bán hàng",
        aliases=[
            r"chi phí bán hàng",
            r"chi phi ban hang",
            r"8\.\s*chi phí bán hàng",
        ],
        statement_type="INCOME_STATEMENT",
    ),
    ConceptDefinition(
        concept="ADMIN_EXPENSES",
        code="26",
        standard_name="Chi phí quản lý doanh nghiệp",
        aliases=[
            r"chi phí quản lý doanh nghiệp",
            r"chi phí quản lý",
            r"9\.\s*chi phí quản lý doanh nghiệp",
        ],
        statement_type="INCOME_STATEMENT",
    ),
    ConceptDefinition(
        concept="OPERATING_PROFIT",
        code="30",
        standard_name="Lợi nhuận thuần từ hoạt động kinh doanh",
        aliases=[
            r"lợi nhuận thuần từ hoạt động kinh doanh",
            r"lợi nhuận kinh doanh",
            r"10\.\s*lợi nhuận thuần từ hoạt động kinh doanh",
        ],
        statement_type="INCOME_STATEMENT",
    ),
    ConceptDefinition(
        concept="OTHER_PROFIT",
        code="40",
        standard_name="Kết quả từ hoạt động khác",
        aliases=[
            r"kết quả từ hoạt động khác",
            r"lợi nhuận khác",
        ],
        statement_type="INCOME_STATEMENT",
    ),
    ConceptDefinition(
        concept="PROFIT_BEFORE_TAX",
        code="50",
        standard_name="Tổng lợi nhuận kế toán trước thuế",
        aliases=[
            r"tổng lợi nhuận kế toán trước thuế",
            r"lợi nhuận trước thuế",
            r"15\.\s*tổng lợi nhuận kế toán trước thuế",
        ],
        statement_type="INCOME_STATEMENT",
    ),
    ConceptDefinition(
        concept="NET_PROFIT",
        code="60",
        standard_name="Lợi nhuận sau thuế thu nhập doanh nghiệp",
        aliases=[
            r"lợi nhuận sau thuế",
            r"loi nhuan sau thue",
            r"lợi nhuận sau thuế thu nhập doanh nghiệp",
            r"17\.\s*lợi nhuận sau thuế",
            r"lợi nhuận thuần sau thuế",
        ],
        statement_type="INCOME_STATEMENT",
    ),
    ConceptDefinition(
        concept="CURRENT_TAX_EXPENSE",
        code="51",
        standard_name="Chi phí thuế TNDN hiện hành",
        aliases=[
            r"chi phí thuế thu nhập doanh nghiệp hiện hành",
            r"chi phí thuế tndn hiện hành",
            r"thuế tndn hiện hành",
            r"chi phí thuế hiện hành",
        ],
        statement_type="INCOME_STATEMENT",
    ),
    ConceptDefinition(
        concept="DEFERRED_TAX_EXPENSE",
        code="52",
        standard_name="Chi phí thuế TNDN hoãn lại",
        aliases=[
            r"chi phí thuế thu nhập doanh nghiệp hoãn lại",
            r"chi phí thuế tndn hoãn lại",
            r"thuế tndn hoãn lại",
            r"chi phí thuế hoãn lại",
        ],
        statement_type="INCOME_STATEMENT",
    ),
    ConceptDefinition(
        concept="BASIC_EPS",
        code="70",
        standard_name="Lãi cơ bản trên cổ phiếu",
        aliases=[
            r"lãi cơ bản trên cổ phiếu",
            r"lãi trên một cổ phiếu",
            r"lãi cơ bản trên mỗi cổ phiếu",
            r"eps cơ bản",
        ],
        statement_type="INCOME_STATEMENT",
    ),
    ConceptDefinition(
        concept="DILUTED_EPS",
        code="71",
        standard_name="Lãi suy giảm trên cổ phiếu",
        aliases=[
            r"lãi suy giảm trên cổ phiếu",
            r"lãi suy giảm trên mỗi cổ phiếu",
            r"eps pha loãng",
            r"eps suy giảm",
        ],
        statement_type="INCOME_STATEMENT",
    ),

    # ── BÁO CÁO LƯU CHUYỂN TIỀN TỆ (LCTT) THEO THÔNG TƯ 200 ─────────────────────
    ConceptDefinition(
        concept="CF_OPERATING_PROFIT_BEFORE_WC",
        code="08",
        standard_name="Lợi nhuận từ HĐKD trước thay đổi vốn lưu động",
        aliases=[
            r"lợi nhuận từ hoạt động kinh doanh trước những thay đổi",
            r"trước những thay đổi vốn lưu động",
        ],
        statement_type="CASH_FLOW",
    ),
    ConceptDefinition(
        concept="CF_INVENTORY_CHANGE",
        code="10",
        standard_name="Biến động hàng tồn kho",
        aliases=[
            r"biến động hàng tồn kho",
            r"tăng,?\s*giảm hàng tồn kho",
        ],
        statement_type="CASH_FLOW",
    ),
    ConceptDefinition(
        concept="CF_PAYABLES_CHANGE",
        code="11",
        standard_name="Biến động các khoản phải trả",
        aliases=[
            r"biến động các khoản phải trả",
            r"tăng,?\s*giảm các khoản phải trả",
        ],
        statement_type="CASH_FLOW",
    ),
    ConceptDefinition(
        concept="CF_PREPAID_EXPENSES_CHANGE",
        code="12",
        standard_name="Biến động chi phí trả trước",
        aliases=[
            r"biến động chi phí trả trước",
            r"tăng,?\s*giảm chi phí trả trước",
        ],
        statement_type="CASH_FLOW",
    ),
    ConceptDefinition(
        concept="CF_INTEREST_PAID",
        code="14",
        standard_name="Tiền lãi vay đã trả",
        aliases=[r"tiền lãi vay đã trả"],
        statement_type="CASH_FLOW",
    ),
    ConceptDefinition(
        concept="CF_TAX_PAID",
        code="15",
        standard_name="Thuế TNDN đã nộp",
        aliases=[r"thuế thu nhập doanh nghiệp đã nộp"],
        statement_type="CASH_FLOW",
    ),
    ConceptDefinition(
        concept="CF_NET_OPERATING",
        code="20",
        standard_name="Lưu chuyển tiền thuần từ hoạt động kinh doanh",
        aliases=[r"lưu chuyển tiền thuần từ hoạt động kinh doanh"],
        statement_type="CASH_FLOW",
    ),
    ConceptDefinition(
        concept="CF_CAPEX",
        code="21",
        standard_name="Tiền chi mua sắm tài sản cố định và tài sản dài hạn khác",
        aliases=[
            r"tiền chi mua tài sản cố định",
            r"tiền chi mua sắm,?\s*xây dựng tscđ",
        ],
        statement_type="CASH_FLOW",
    ),
    ConceptDefinition(
        concept="CF_PROCEEDS_DISPOSAL_ASSETS",
        code="22",
        standard_name="Tiền thu từ thanh lý tài sản cố định",
        aliases=[
            r"tiền thu từ thanh lý tài sản cố định",
            r"tiền thu từ thanh lý,?\s*nhượng bán tscđ",
        ],
        statement_type="CASH_FLOW",
    ),
    ConceptDefinition(
        concept="CF_NET_INVESTING",
        code="30",
        standard_name="Lưu chuyển tiền thuần từ hoạt động đầu tư",
        aliases=[r"lưu chuyển tiền thuần từ hoạt động đầu tư"],
        statement_type="CASH_FLOW",
    ),
    ConceptDefinition(
        concept="CF_BORROWINGS",
        code="33",
        standard_name="Tiền thu từ đi vay",
        aliases=[r"tiền thu từ đi vay"],
        statement_type="CASH_FLOW",
    ),
    ConceptDefinition(
        concept="CF_REPAYMENTS",
        code="34",
        standard_name="Tiền chi trả nợ gốc vay",
        aliases=[r"tiền chi trả nợ gốc vay"],
        statement_type="CASH_FLOW",
    ),
    ConceptDefinition(
        concept="CF_DIVIDENDS_PAID",
        code="36",
        standard_name="Tiền chi trả cổ tức",
        aliases=[r"tiền chi trả cổ tức", r"cổ tức, lợi nhuận đã trả"],
        statement_type="CASH_FLOW",
    ),
    ConceptDefinition(
        concept="CF_NET_FINANCING",
        code="40",
        standard_name="Lưu chuyển thuần từ hoạt động tài chính",
        aliases=[
            r"lưu chuyển thuần từ hoạt động tài chính",
            r"lưu chuyển tiền thuần từ hoạt động tài chính",
        ],
        statement_type="CASH_FLOW",
    ),
    ConceptDefinition(
        concept="CF_NET_CHANGE",
        code="50",
        standard_name="Lưu chuyển tiền thuần trong năm",
        aliases=[
            r"lưu chuyển tiền thuần trong năm",
            r"lưu chuyển tiền thuần trong kỳ",
        ],
        statement_type="CASH_FLOW",
    ),
    ConceptDefinition(
        concept="CF_BEGINNING_CASH",
        code="60",
        standard_name="Tiền và các khoản tương đương tiền đầu năm",
        aliases=[
            r"tiền và các khoản tương đương tiền đầu năm",
            r"tiền và tương đương tiền đầu kỳ",
        ],
        statement_type="CASH_FLOW",
    ),
    ConceptDefinition(
        concept="CF_ENDING_CASH",
        code="70",
        standard_name="Tiền và các khoản tương đương tiền cuối năm",
        aliases=[
            r"tiền và các khoản tương đương tiền cuối năm",
            r"tiền và tương đương tiền cuối kỳ",
        ],
        statement_type="CASH_FLOW",
    ),
]

# Từ điển tra cứu nhanh theo concept
CONCEPT_TO_DEF: dict[str, ConceptDefinition] = {d.concept: d for d in ONTOLOGY_DEFINITIONS}
CONCEPT_TO_CODE: dict[str, str] = {d.concept: d.code for d in ONTOLOGY_DEFINITIONS}

# Từ điển tra cứu theo mã số chuẩn (ưu tiên CĐKT và KQKD trước LCTT)
CODE_TO_CONCEPT: dict[str, str] = {}
for d in ONTOLOGY_DEFINITIONS:
    if d.code not in CODE_TO_CONCEPT or d.statement_type in ("BALANCE_SHEET", "INCOME_STATEMENT"):
        CODE_TO_CONCEPT[d.code] = d.concept


def match_concept_from_label_and_code(
    raw_label: str,
    raw_code: str = "",
    statement_type_hint: str | None = None,
) -> tuple[str | None, str]:
    """
    Ánh xạ tên khoản mục và mã số trên báo cáo sang Canonical Concept.

    Ưu tiên:
      1. Tự động suy luận statement_type nếu chưa có hint từ ngữ cảnh nhãn khoản mục.
      2. Khớp theo mã số chuẩn Thông tư 200 có kiểm tra phù hợp loại báo cáo và ngữ nghĩa nhãn.
      3. Khớp theo regex alias tên khoản mục (loại trừ các khoản mục con đè khoản mục cha).

    Returns:
        tuple[concept, standard_code]
    """
    clean_label = raw_label.strip().lower()
    clean_code = re.sub(r"[^\d]", "", str(raw_code).strip())

    # Tự động suy luận loại báo cáo nếu không có hint
    inferred_type = statement_type_hint
    if not inferred_type:
        # Nếu nhãn chứa các từ ngữ đặc thù của Bảng Cân đối kế toán
        # (Đặt trước để tránh nhầm "Doanh thu chưa thực hiện ngắn hạn" sang INCOME_STATEMENT)
        if any(k in clean_label for k in [
            "chưa thực hiện", "ngắn hạn", "dài hạn", "tài sản", "nguồn vốn", "nợ phải trả", "vốn chủ sở hữu"
        ]):
            inferred_type = "BALANCE_SHEET"
        elif any(k in clean_label for k in [
            "lưu chuyển", "tiền thuần", "tiền chi", "tiền thu", "biến động",
            "thanh lý", "cổ tức", "tiền gửi có kì hạn", "tiền gửi có kỳ hạn"
        ]):
            inferred_type = "CASH_FLOW"
        elif any(k in clean_label for k in [
            "doanh thu", "giá vốn", "lợi nhuận gộp", "chi phí bán hàng",
            "chi phí quản lý", "kết quả từ hoạt động khác", "thu nhập khác"
        ]):
            inferred_type = "INCOME_STATEMENT"

    # 1. Khớp theo mã số chuẩn Thông tư 200
    if clean_code:
        # Hỗ trợ cả trường hợp OCR làm rớt số 0 ở đầu (ví dụ: '01' vs '1', '08' vs '8')
        matching_defs = [
            d for d in ONTOLOGY_DEFINITIONS
            if d.code == clean_code or (len(clean_code) <= 2 and d.code.lstrip("0") == clean_code.lstrip("0"))
        ]
        if inferred_type:
            filtered = [d for d in matching_defs if d.statement_type == inferred_type]
            if filtered:
                matching_defs = filtered

        for candidate in matching_defs:
            # Kiểm tra semantic compatibility
            if candidate.statement_type == "INCOME_STATEMENT":
                # Không gán chỉ tiêu KQKD nếu nhãn mang nghĩa LCTT
                if any(k in clean_label for k in ["lưu chuyển", "tiền thuần", "biến động", "tiền chi", "tiền thu"]):
                    continue
            elif candidate.statement_type == "CASH_FLOW":
                # Không gán chỉ tiêu LCTT nếu nhãn mang nghĩa KQKD rõ rệt
                if any(k in clean_label for k in ["giá vốn", "doanh thu thuần"]):
                    continue
            elif candidate.statement_type == "BALANCE_SHEET":
                # Kiểm tra các khoản mục cha không bị con chiếm
                if candidate.concept == "SHORT_TERM_RECEIVABLES" and any(k in clean_label for k in ["khác", "khách hàng", "dự phòng"]):
                    continue
                if candidate.concept == "INVENTORIES" and any(k in clean_label for k in ["dự phòng", "giảm giá"]):
                    continue
                if candidate.concept == "FIXED_ASSETS" and any(k in clean_label for k in ["hữu hình", "vô hình"]):
                    continue
                if candidate.concept == "LIABILITIES" and any(k in clean_label for k in ["ngắn hạn", "dài hạn"]):
                    continue
                if candidate.concept == "CURRENT_LIABILITIES" and any(k in clean_label for k in ["người bán", "trả trước", "vay", "thuê tài chính", "khác", "chưa thực hiện"]):
                    continue
                if candidate.concept == "NON_CURRENT_LIABILITIES" and any(k in clean_label for k in ["vay", "thuê tài chính", "khác", "chưa thực hiện"]):
                    continue
                if candidate.concept == "EQUITY" and any(k in clean_label for k in ["góp", "cổ phần", "chưa phân phối", "thặng dư"]):
                    continue

            return candidate.concept, clean_code

    # 2. Khớp theo regex alias tên khoản mục
    for cdef in ONTOLOGY_DEFINITIONS:
        if inferred_type and cdef.statement_type != inferred_type:
            continue

        # Nếu có mã số rõ ràng mà mã số đó khác với cdef.code (ví dụ mã 136 vs 130),
        # KHÔNG cho phép alias lỏng khớp sai mã
        if clean_code and clean_code != cdef.code and len(clean_code) >= 2:
            continue

        # Kiểm tra loại trừ ngữ nghĩa khoản mục con
        if cdef.concept == "SHORT_TERM_RECEIVABLES" and any(k in clean_label for k in ["khác", "khách hàng", "người bán", "dự phòng"]):
            continue
        if cdef.concept == "INVENTORIES" and any(k in clean_label for k in ["dự phòng", "giảm giá"]):
            continue
        if cdef.concept == "LONG_TERM_RECEIVABLES" and "khác" in clean_label:
            continue
        if cdef.concept == "FIXED_ASSETS" and any(k in clean_label for k in ["hữu hình", "vô hình"]):
            continue
        if cdef.concept == "LONG_TERM_ASSETS_IN_PROGRESS" and "xây dựng cơ bản" in clean_label:
            continue
        if cdef.concept == "LIABILITIES" and any(k in clean_label for k in ["ngắn hạn", "dài hạn"]):
            continue
        if cdef.concept == "CURRENT_LIABILITIES" and any(k in clean_label for k in ["người bán", "trả trước", "vay", "thuê tài chính", "khác", "chưa thực hiện"]):
            continue
        if cdef.concept == "NON_CURRENT_LIABILITIES" and any(k in clean_label for k in ["vay", "thuê tài chính", "khác", "chưa thực hiện"]):
            continue
        if cdef.concept == "EQUITY" and any(k in clean_label for k in ["góp", "cổ phần", "chưa phân phối", "thặng dư"]):
            continue

        for alias in cdef.aliases:
            if re.search(alias, clean_label, flags=re.IGNORECASE):
                return cdef.concept, cdef.code

    return None, clean_code
