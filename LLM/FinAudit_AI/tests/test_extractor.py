"""
test_extractor.py — Unit tests cho FinancialFactExtractor và Financial Ontology mapping.
"""


from src.extractor.fact_extractor import FinancialFactExtractor
from src.extractor.ontology import match_concept_from_label_and_code
from src.models import ClassifiedBlock, ParsedBlock, StorageTarget


def test_ontology_matching():
    # Khớp qua mã số
    concept, code = match_concept_from_label_and_code("TỔNG CỘNG TÀI SẢN", raw_code="270")
    assert concept == "TOTAL_ASSETS"
    assert code == "270"

    # Khớp qua regex label tiếng Việt
    concept, code = match_concept_from_label_and_code("Tiền và các khoản tương đương tiền")
    assert concept == "CASH_AND_EQUIVALENTS"
    assert code == "110"

    # Khớp qua Doanh thu thuần
    concept, code = match_concept_from_label_and_code("Doanh thu thuần về bán hàng và cung cấp dịch vụ", raw_code="10")
    assert concept == "NET_REVENUE"
    assert code == "10"

    # Phân biệt Lưu chuyển tiền tệ (mã 10, 20 LCTT không nhầm với KQKD)
    cf_concept, cf_code = match_concept_from_label_and_code("Biến động hàng tồn kho", raw_code="10")
    assert cf_concept == "CF_INVENTORY_CHANGE"
    assert cf_concept != "NET_REVENUE"

    cf_gp_concept, _ = match_concept_from_label_and_code("Lưu chuyển tiền thuần từ hoạt động kinh doanh", raw_code="20")
    assert cf_gp_concept == "CF_NET_OPERATING"
    assert cf_gp_concept != "GROSS_PROFIT"

    # Phân biệt khoản mục con CĐKT không đè khoản mục mẹ (136 vs 130, 149 vs 140)
    rec_other_concept, rec_other_code = match_concept_from_label_and_code("Phải thu ngắn hạn khác", raw_code="136")
    assert rec_other_concept == "SHORT_TERM_OTHER_RECEIVABLES"
    assert rec_other_concept != "SHORT_TERM_RECEIVABLES"

    inv_prov_concept, _ = match_concept_from_label_and_code("Dự phòng giảm giá hàng tồn kho", raw_code="149")
    assert inv_prov_concept == "INVENTORY_PROVISION"
    assert inv_prov_concept != "INVENTORIES"

    # Kiểm tra nợ vay và vốn chủ sở hữu mới bổ sung (320, 338, 411, 421)
    st_borrow, _ = match_concept_from_label_and_code("Vay và nợ thuê tài chính ngắn hạn", raw_code="320")
    assert st_borrow == "SHORT_TERM_BORROWINGS"

    lt_borrow, _ = match_concept_from_label_and_code("Vay và nợ thuê tài chính dài hạn", raw_code="338")
    assert lt_borrow == "LONG_TERM_BORROWINGS"

    contrib_cap, _ = match_concept_from_label_and_code("Vốn góp của chủ sở hữu", raw_code="411")
    assert contrib_cap == "CONTRIBUTED_CAPITAL"

    ret_earn, _ = match_concept_from_label_and_code("Lợi nhuận sau thuế chưa phân phối", raw_code="421")
    assert ret_earn == "RETAINED_EARNINGS"

    # Kiểm tra Doanh thu gộp (01) và Giảm trừ (02)
    gross_rev, _ = match_concept_from_label_and_code("Doanh thu bán hàng và cung cấp dịch vụ", raw_code="01")
    assert gross_rev == "GROSS_REVENUE"

    rev_deduct, _ = match_concept_from_label_and_code("Các khoản giảm trừ doanh thu", raw_code="02")
    assert rev_deduct == "REVENUE_DEDUCTIONS"

    # Kiểm tra khả năng chịu lỗi OCR rớt số 0 ở đầu (Leading zero tolerance: '1' -> '01', '8' -> '08')
    gross_rev_ocr, _ = match_concept_from_label_and_code("Doanh thu bán hàng và cung cấp dịch vụ", raw_code="1")
    assert gross_rev_ocr == "GROSS_REVENUE"

    cf_wc_ocr, _ = match_concept_from_label_and_code("Lợi nhuận từ hoạt động kinh doanh trước những thay đổi vốn lưu động", raw_code="8")
    assert cf_wc_ocr == "CF_OPERATING_PROFIT_BEFORE_WC"

    # Kiểm tra không nhầm "Doanh thu chưa thực hiện ngắn hạn" sang INCOME_STATEMENT
    unearned_rev, code_318 = match_concept_from_label_and_code("Doanh thu chưa thực hiện ngắn hạn", raw_code="318")
    assert unearned_rev == "SHORT_TERM_UNEARNED_REVENUE"
    assert code_318 == "318"


def test_fact_extractor_from_table_block():
    extractor = FinancialFactExtractor()

    table_content = """
| CHỈ TIÊU | Mã số | Thuyết minh | 31/12/2024 VND | 1/1/2024 VND |
| --- | --- | --- | --- | --- |
| Tài sản ngắn hạn | 100 |  | 30,000,000,000 | 25,000,000,000 |
| Tiền và tương đương tiền | 110 | V.01 | 5,000,000,000 | 4,000,000,000 |
| Hàng tồn kho | 140 | V.04 | 10,000,000,000 | 8,000,000,000 |
"""
    parsed_block = ParsedBlock(
        block_id="p5_b1",
        block_type="table",
        page=5,
        content=table_content.strip(),
        metadata={"unit": "VND"},
    )
    classified_block = ClassifiedBlock(
        block=parsed_block,
        block_type="FINANCIAL_STATEMENT",
        target=[StorageTarget.SQL],
        confidence=0.95,
        classification_method="rule_based",
    )

    facts, report = extractor.extract_from_blocks([classified_block], company="VNM", year=2024)

    assert len(facts) >= 3
    concepts = {f.concept for f in facts}
    assert "CURRENT_ASSETS" in concepts
    assert "CASH_AND_EQUIVALENTS" in concepts
    assert "INVENTORIES" in concepts

    # Kiểm tra giá trị đã làm sạch dấu phẩy
    ca_fact = next(f for f in facts if f.concept == "CURRENT_ASSETS" and f.period_type == "current")
    assert ca_fact.value == 30000000000.0
    assert ca_fact.prov_id.startswith("VNM_2024_p5_")
