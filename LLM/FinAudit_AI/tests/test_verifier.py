"""
test_verifier.py — Unit tests cho AccountingVerifier (Tự kiểm toán số học Anti-GIGO).
Kiểm tra các phương trình kế toán:
  1. Mã 270 == Mã 440 (Tài sản == Nguồn vốn)
  2. Mã 270 == Mã 100 + Mã 200 (Tài sản ngắn hạn + Dài hạn)
  3. Mã 100 == 110 + 120 + 130 + 140 + 150
  4. Mã 440 == Mã 300 + Mã 400 (Nợ phải trả + Vốn CSH)
  5. Mã 20 == Mã 10 - Mã 11 (Lợi nhuận gộp)
"""


from src.models import FinancialFact, VerificationStatus
from src.verifier.accounting_verifier import AccountingVerifier


def test_accounting_verifier_balanced_equations():
    verifier = AccountingVerifier()
    facts = [
        FinancialFact(
            id="f_ta", prov_id="p1", concept="TOTAL_ASSETS", standard_code="270",
            raw_label="Tổng tài sản", value=1000.0, period="2024", period_type="current",
            company="TEST", year=2024,
        ),
        FinancialFact(
            id="f_tr", prov_id="p2", concept="TOTAL_RESOURCES", standard_code="440",
            raw_label="Tổng nguồn vốn", value=1000.0, period="2024", period_type="current",
            company="TEST", year=2024,
        ),
        FinancialFact(
            id="f_ca", prov_id="p3", concept="CURRENT_ASSETS", standard_code="100",
            raw_label="Tài sản ngắn hạn", value=600.0, period="2024", period_type="current",
            company="TEST", year=2024,
        ),
        FinancialFact(
            id="f_nca", prov_id="p4", concept="NON_CURRENT_ASSETS", standard_code="200",
            raw_label="Tài sản dài hạn", value=400.0, period="2024", period_type="current",
            company="TEST", year=2024,
        ),
        FinancialFact(
            id="f_liab", prov_id="p5", concept="LIABILITIES", standard_code="300",
            raw_label="Nợ phải trả", value=300.0, period="2024", period_type="current",
            company="TEST", year=2024,
        ),
        FinancialFact(
            id="f_eq", prov_id="p6", concept="EQUITY", standard_code="400",
            raw_label="Vốn chủ sở hữu", value=700.0, period="2024", period_type="current",
            company="TEST", year=2024,
        ),
        FinancialFact(
            id="f_rev", prov_id="p7", concept="NET_REVENUE", standard_code="10",
            raw_label="Doanh thu thuần", value=2000.0, period="2024", period_type="current",
            company="TEST", year=2024,
        ),
        FinancialFact(
            id="f_cogs", prov_id="p8", concept="COGS", standard_code="11",
            raw_label="Giá vốn hàng bán", value=1400.0, period="2024", period_type="current",
            company="TEST", year=2024,
        ),
        FinancialFact(
            id="f_gp", prov_id="p9", concept="GROSS_PROFIT", standard_code="20",
            raw_label="Lợi nhuận gộp", value=600.0, period="2024", period_type="current",
            company="TEST", year=2024,
        ),
    ]

    report = verifier.verify_facts(facts, company="TEST", year=2024)

    assert report.is_balanced is True
    assert report.total_checks >= 4
    assert len(report.failed_checks) == 0
    assert len(report.discrepancies) == 0
    assert facts[0].verification_status == VerificationStatus.VERIFIED
    assert facts[1].verification_status == VerificationStatus.VERIFIED


def test_accounting_verifier_detects_discrepancy():
    verifier = AccountingVerifier()
    # Giả lập số liệu bị sai lệch (ví dụ: do OCR nhầm số hoặc ảo giác LLM)
    facts = [
        FinancialFact(
            id="f_ta", prov_id="p1", concept="TOTAL_ASSETS", standard_code="270",
            raw_label="Tổng tài sản", value=1000.0, period="2024", period_type="current",
            company="TEST", year=2024,
        ),
        FinancialFact(
            id="f_tr", prov_id="p2", concept="TOTAL_RESOURCES", standard_code="440",
            raw_label="Tổng nguồn vốn", value=950.0,  # Sai lệch 50
            period="2024", period_type="current", company="TEST", year=2024,
        ),
    ]

    report = verifier.verify_facts(facts, company="TEST", year=2024)

    assert report.is_balanced is False
    assert len(report.failed_checks) == 1
    assert len(report.discrepancies) == 1
    assert report.discrepancies[0]["delta"] == 50.0
    assert facts[0].verification_status == VerificationStatus.DISCREPANCY
    assert facts[1].verification_status == VerificationStatus.DISCREPANCY


def test_accounting_verifier_current_assets_sum():
    verifier = AccountingVerifier()
    facts = [
        FinancialFact(
            id="f_ca", prov_id="p1", concept="CURRENT_ASSETS", standard_code="100",
            raw_label="Tài sản ngắn hạn", value=500.0, period="2024", period_type="current",
            company="TEST", year=2024,
        ),
        FinancialFact(
            id="f_cash", prov_id="p2", concept="CASH_AND_EQUIVALENTS", standard_code="110",
            raw_label="Tiền", value=100.0, period="2024", period_type="current",
            company="TEST", year=2024,
        ),
        FinancialFact(
            id="f_inv", prov_id="p3", concept="INVENTORIES", standard_code="140",
            raw_label="Hàng tồn kho", value=200.0, period="2024", period_type="current",
            company="TEST", year=2024,
        ),
        FinancialFact(
            id="f_rec", prov_id="p4", concept="SHORT_TERM_RECEIVABLES", standard_code="130",
            raw_label="Phải thu", value=200.0, period="2024", period_type="current",
            company="TEST", year=2024,
        ),
    ]

    report = verifier.verify_facts(facts, company="TEST", year=2024)

    assert report.is_balanced is True
    assert facts[0].verification_status == VerificationStatus.VERIFIED


def test_accounting_verifier_smart_priority_and_cash_flow_isolation():
    verifier = AccountingVerifier()
    # Giả lập danh sách fact có cả KQKD và LCTT (nơi mã số 10, 11, 20 bị trùng)
    # và khoản mục con CĐKT xuất hiện sau khoản mục cha
    facts = [
        # Bảng Cân đối kế toán
        FinancialFact(
            id="f_ca", prov_id="p5_1", concept="CURRENT_ASSETS", standard_code="100",
            raw_label="TÀI SẢN NGẮN HẠN", value=1000.0, period="2024", period_type="current",
            company="VNM", year=2024,
        ),
        FinancialFact(
            id="f_cash", prov_id="p5_2", concept="CASH_AND_EQUIVALENTS", standard_code="110",
            raw_label="Tiền và tương đương tiền", value=200.0, period="2024", period_type="current",
            company="VNM", year=2024,
        ),
        FinancialFact(
            id="f_st_inv", prov_id="p5_3", concept="SHORT_TERM_INVESTMENTS", standard_code="120",
            raw_label="Đầu tư tài chính ngắn hạn", value=300.0, period="2024", period_type="current",
            company="VNM", year=2024,
        ),
        FinancialFact(
            id="f_rec", prov_id="p5_4", concept="SHORT_TERM_RECEIVABLES", standard_code="130",
            raw_label="Các khoản phải thu ngắn hạn", value=250.0, period="2024", period_type="current",
            company="VNM", year=2024,
        ),
        FinancialFact(
            id="f_inv", prov_id="p5_5", concept="INVENTORIES", standard_code="140",
            raw_label="Hàng tồn kho", value=200.0, period="2024", period_type="current",
            company="VNM", year=2024,
        ),
        FinancialFact(
            id="f_other_ca", prov_id="p5_6", concept="OTHER_CURRENT_ASSETS", standard_code="150",
            raw_label="Tài sản ngắn hạn khác", value=50.0, period="2024", period_type="current",
            company="VNM", year=2024,
        ),
        # Khoản mục con xuất hiện sau (nếu bị gán nhầm hoặc cùng concept)
        FinancialFact(
            id="f_rec_other", prov_id="p5_7", concept="SHORT_TERM_RECEIVABLES", standard_code="136",
            raw_label="Phải thu ngắn hạn khác", value=40.0, period="2024", period_type="current",
            company="VNM", year=2024,
        ),
        # Báo cáo Kết quả kinh doanh (KQKD)
        FinancialFact(
            id="f_rev", prov_id="p10_1", concept="NET_REVENUE", standard_code="10",
            raw_label="3. Doanh thu thuần về bán hàng", value=5000.0, period="2024", period_type="current",
            company="VNM", year=2024,
        ),
        FinancialFact(
            id="f_cogs", prov_id="p10_2", concept="COGS", standard_code="11",
            raw_label="4. Giá vốn hàng bán", value=3000.0, period="2024", period_type="current",
            company="VNM", year=2024,
        ),
        FinancialFact(
            id="f_gp", prov_id="p10_3", concept="GROSS_PROFIT", standard_code="20",
            raw_label="5. Lợi nhuận gộp về bán hàng", value=2000.0, period="2024", period_type="current",
            company="VNM", year=2024,
        ),
        # Báo cáo Lưu chuyển tiền tệ (LCTT) xuất hiện ở các trang sau
        FinancialFact(
            id="f_cf_inv", prov_id="p11_1", concept="NET_REVENUE", standard_code="10",
            raw_label="Biến động hàng tồn kho", value=-100.0, period="2024", period_type="current",
            company="VNM", year=2024,
        ),
        FinancialFact(
            id="f_cf_pay", prov_id="p11_2", concept="COGS", standard_code="11",
            raw_label="Biến động các khoản phải trả", value=-50.0, period="2024", period_type="current",
            company="VNM", year=2024,
        ),
        FinancialFact(
            id="f_cf_net", prov_id="p11_3", concept="GROSS_PROFIT", standard_code="20",
            raw_label="Lưu chuyển tiền thuần từ hoạt động kinh doanh", value=800.0, period="2024", period_type="current",
            company="VNM", year=2024,
        ),
    ]

    report = verifier.verify_facts(facts, company="VNM", year=2024)

    # Cả kiểm tra CỘNG_DỌC_NGẮN_HẠN và CÂN_ĐỐI_LỢI_NHUẬN_GỘP đều phải PASSED
    assert "CỘNG_DỌC_NGẮN_HẠN: TS Ngắn hạn (100) == ∑(5 khoản mục con)" in report.passed_checks
    assert "CÂN_ĐỐI_LỢI_NHUẬN_GỘP: LN Gộp (20) == Doanh thu thuần (10) - Giá vốn (11)" in report.passed_checks
    assert len(report.failed_checks) == 0
    assert report.is_balanced is True

