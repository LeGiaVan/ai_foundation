"""
accounting_verifier.py — Bộ Tự Kiểm Toán Số Học (Self-Auditing Accounting Invariants Verifier).
Chốt chặn cốt lõi loại bỏ rủi ro GIGO (Garbage In, Garbage Out):
  - Áp dụng các phương trình toán học và đẳng thức kế toán đóng kín (Accounting Invariants).
  - Tự động phát hiện số liệu bị hallucinate, nhầm số, rớt dòng hoặc vỡ cấu trúc.
  - Cập nhật trạng thái `verification_status` (VERIFIED vs DISCREPANCY) cho từng Fact.
"""

import logging
import re
from typing import Any

from src.extractor.ontology import CONCEPT_TO_CODE
from src.models import FinancialFact, VerificationReport, VerificationStatus

logger = logging.getLogger(__name__)


class AccountingVerifier:
    """Bộ kiểm tra tính hợp lệ số học và đẳng thức kế toán cho Fact tài chính."""

    def __init__(self, tolerance_ratio: float = 0.0001, absolute_tolerance: float = 2.0) -> None:
        """
        Args:
            tolerance_ratio: Sai số tương đối cho phép do làm tròn (mặc định 0.01%)
            absolute_tolerance: Sai số tuyệt đối cho phép do làm tròn đơn vị lẻ (mặc định 2.0)
        """
        self.tolerance_ratio = tolerance_ratio
        self.absolute_tolerance = absolute_tolerance

    def verify_facts(
        self,
        facts: list[FinancialFact],
        company: str = "DOANH_NGHIEP",
        year: int = 2024,
    ) -> VerificationReport:
        """
        Chạy toàn bộ các bài kiểm tra đẳng thức số học trên danh sách Fact.
        Tự động cập nhật `verification_status` và `verification_detail` trên từng FinancialFact.

        Returns:
            VerificationReport: Báo cáo kết quả kiểm toán số học chi tiết.
        """
        def _get_fact_priority(f: FinancialFact) -> int:
            """
            Tính điểm ưu tiên cho Fact khi có nhiều fact cùng concept:
            - Điểm cao hơn nếu mã số chuẩn khớp đúng với mã chuẩn của concept theo Thông tư 200.
            - Điểm phạt nếu tên khoản mục chứa các từ khóa chỉ khoản mục con hoặc lưu chuyển tiền.
            """
            score = 10
            expected_code = CONCEPT_TO_CODE.get(f.concept, "")
            clean_fact_code = re.sub(r"[^\d]", "", str(f.standard_code))
            if expected_code and clean_fact_code == expected_code:
                score += 50
            elif clean_fact_code and expected_code and clean_fact_code != expected_code:
                score -= 30

            lbl = f.raw_label.lower()
            # Đối với các chỉ tiêu KQKD: phạt nặng nếu nhãn thuộc về LCTT (lưu chuyển tiền thuần, biến động...)
            if f.concept in ("NET_REVENUE", "COGS", "GROSS_PROFIT", "OPERATING_PROFIT", "PROFIT_BEFORE_TAX", "NET_PROFIT"):
                if any(k in lbl for k in ["lưu chuyển", "tiền thuần", "biến động", "thanh lý", "cổ tức", "tiền chi", "tiền thu"]):
                    score -= 100
                if f.concept == "NET_REVENUE" and "doanh thu" in lbl:
                    score += 20
                if f.concept == "COGS" and "giá vốn" in lbl:
                    score += 20
                if f.concept == "GROSS_PROFIT" and "lợi nhuận gộp" in lbl:
                    score += 20

            # Đối với các chỉ tiêu CĐKT: phạt nếu là khoản mục con (khác, dự phòng...)
            if f.concept in ("SHORT_TERM_RECEIVABLES", "INVENTORIES", "FIXED_ASSETS", "LONG_TERM_RECEIVABLES"):
                if any(k in lbl for k in ["khác", "dự phòng", "nguyên giá", "hao mòn"]):
                    score -= 40

            return score

        # Gom nhóm facts theo kỳ hiện tại có chọn lọc thông minh
        current_facts_by_concept: dict[str, FinancialFact] = {}
        for f in facts:
            if f.period_type == "current" or f.period == str(year):
                if f.concept not in current_facts_by_concept:
                    current_facts_by_concept[f.concept] = f
                else:
                    existing = current_facts_by_concept[f.concept]
                    if _get_fact_priority(f) > _get_fact_priority(existing):
                        current_facts_by_concept[f.concept] = f

        passed_checks: list[str] = []
        failed_checks: list[str] = []
        discrepancies: list[dict[str, Any]] = []

        def is_equal(val1: float, val2: float) -> bool:
            diff = abs(val1 - val2)
            if diff <= self.absolute_tolerance:
                return True
            max_val = max(abs(val1), abs(val2))
            if max_val == 0:
                return True
            return (diff / max_val) <= self.tolerance_ratio

        # ── KIỂM TRA 1: CÂN ĐỐI TỔNG TÀI SẢN == TỔNG NGUỒN VỐN (Mã 270 == Mã 440) ────
        if "TOTAL_ASSETS" in current_facts_by_concept and "TOTAL_RESOURCES" in current_facts_by_concept:
            f_assets = current_facts_by_concept["TOTAL_ASSETS"]
            f_res = current_facts_by_concept["TOTAL_RESOURCES"]
            if is_equal(f_assets.value, f_res.value):
                passed_checks.append("CÂN_ĐỐI_TÀI_SẢN_NGUỒN_VỐN: Tổng tài sản (270) == Tổng nguồn vốn (440)")
                f_assets.verification_status = VerificationStatus.VERIFIED
                f_assets.verification_detail = "Khớp 100% với Tổng cộng nguồn vốn"
                f_res.verification_status = VerificationStatus.VERIFIED
                f_res.verification_detail = "Khớp 100% với Tổng cộng tài sản"
            else:
                delta = abs(f_assets.value - f_res.value)
                msg = f"LỆCH_CÂN_ĐỐI: Tổng tài sản ({f_assets.value:,.0f}) != Tổng nguồn vốn ({f_res.value:,.0f}), Chênh lệch: {delta:,.0f}"
                failed_checks.append(msg)
                discrepancies.append({
                    "check": "TOTAL_ASSETS == TOTAL_RESOURCES",
                    "delta": delta,
                    "reported_assets": f_assets.value,
                    "reported_resources": f_res.value,
                })
                f_assets.verification_status = VerificationStatus.DISCREPANCY
                f_assets.verification_detail = msg
                f_res.verification_status = VerificationStatus.DISCREPANCY
                f_res.verification_detail = msg

        # ── KIỂM TRA 2: TỔNG TÀI SẢN = NGẮN HẠN + DÀI HẠN (Mã 270 == 100 + 200) ──────
        if (
            "TOTAL_ASSETS" in current_facts_by_concept
            and "CURRENT_ASSETS" in current_facts_by_concept
            and "NON_CURRENT_ASSETS" in current_facts_by_concept
        ):
            f_total = current_facts_by_concept["TOTAL_ASSETS"]
            f_cur = current_facts_by_concept["CURRENT_ASSETS"]
            f_non_cur = current_facts_by_concept["NON_CURRENT_ASSETS"]
            calc_total = f_cur.value + f_non_cur.value

            if is_equal(f_total.value, calc_total):
                passed_checks.append("CỘNG_TỔNG_TÀI_SẢN: Tài sản (270) == Ngắn hạn (100) + Dài hạn (200)")
                f_cur.verification_status = VerificationStatus.VERIFIED
                f_cur.verification_detail = "Cộng dồn khớp với Tổng tài sản"
                f_non_cur.verification_status = VerificationStatus.VERIFIED
                f_non_cur.verification_detail = "Cộng dồn khớp với Tổng tài sản"
            else:
                delta = abs(f_total.value - calc_total)
                msg = f"LỆCH_TÀI_SẢN: Báo cáo ({f_total.value:,.0f}) != Tính toán ({calc_total:,.0f}), Lệch: {delta:,.0f}"
                failed_checks.append(msg)
                discrepancies.append({
                    "check": "TOTAL_ASSETS == CURRENT + NON_CURRENT",
                    "delta": delta,
                    "reported_total": f_total.value,
                    "calculated_sum": calc_total,
                })
                f_cur.verification_status = VerificationStatus.DISCREPANCY
                f_non_cur.verification_status = VerificationStatus.DISCREPANCY

        # ── KIỂM TRA 3: TÀI SẢN NGẮN HẠN = TỔNG CÁC KHOẢN MỤC CON (100 == ∑110..150) ───
        cur_children_concepts = [
            "CASH_AND_EQUIVALENTS",
            "SHORT_TERM_INVESTMENTS",
            "SHORT_TERM_RECEIVABLES",
            "INVENTORIES",
            "OTHER_CURRENT_ASSETS",
        ]
        available_children = [
            current_facts_by_concept[c] for c in cur_children_concepts if c in current_facts_by_concept
        ]
        if "CURRENT_ASSETS" in current_facts_by_concept and len(available_children) >= 3:
            f_cur = current_facts_by_concept["CURRENT_ASSETS"]
            calc_cur = sum(ch.value for ch in available_children)

            if is_equal(f_cur.value, calc_cur):
                passed_checks.append(f"CỘNG_DỌC_NGẮN_HẠN: TS Ngắn hạn (100) == ∑({len(available_children)} khoản mục con)")
                f_cur.verification_status = VerificationStatus.VERIFIED
                f_cur.verification_detail = f"Khớp tổng {len(available_children)} khoản mục con ngắn hạn"
                for ch in available_children:
                    ch.verification_status = VerificationStatus.VERIFIED
                    ch.verification_detail = "Thuộc phương trình Tài sản ngắn hạn cân đối"
            else:
                delta = abs(f_cur.value - calc_cur)
                # Chỉ cảnh báo nếu có đủ cả 5 mục con hoặc độ lệch lớn
                if len(available_children) == 5 or delta > 0.05 * f_cur.value:
                    failed_checks.append(f"LỆCH_TS_NGẮN_HẠN: Báo cáo ({f_cur.value:,.0f}) != Tổng con ({calc_cur:,.0f}), Lệch: {delta:,.0f}")
                    discrepancies.append({
                        "check": "CURRENT_ASSETS == SUM_CHILDREN",
                        "delta": delta,
                        "reported": f_cur.value,
                        "calculated": calc_cur,
                    })
                    f_cur.verification_status = VerificationStatus.DISCREPANCY
                    f_cur.verification_detail = f"Lệch tổng khoản mục con ngắn hạn: {delta:,.0f}"

        # ── KIỂM TRA 4: NGUỒN VỐN = NỢ PHẢI TRẢ + VỐN CSH (440 == 300 + 400) ─────────
        if (
            "TOTAL_RESOURCES" in current_facts_by_concept
            and "LIABILITIES" in current_facts_by_concept
            and "EQUITY" in current_facts_by_concept
        ):
            f_res = current_facts_by_concept["TOTAL_RESOURCES"]
            f_liab = current_facts_by_concept["LIABILITIES"]
            f_eq = current_facts_by_concept["EQUITY"]
            calc_res = f_liab.value + f_eq.value

            if is_equal(f_res.value, calc_res):
                passed_checks.append("CỘNG_NGUỒN_VỐN: Nguồn vốn (440) == Nợ phải trả (300) + Vốn CSH (400)")
                f_liab.verification_status = VerificationStatus.VERIFIED
                f_liab.verification_detail = "Cộng dồn khớp với Tổng nguồn vốn"
                f_eq.verification_status = VerificationStatus.VERIFIED
                f_eq.verification_detail = "Cộng dồn khớp với Tổng nguồn vốn"
            else:
                delta = abs(f_res.value - calc_res)
                failed_checks.append(f"LỆCH_NGUỒN_VỐN: Báo cáo ({f_res.value:,.0f}) != Tính toán ({calc_res:,.0f}), Lệch: {delta:,.0f}")
                discrepancies.append({
                    "check": "TOTAL_RESOURCES == LIABILITIES + EQUITY",
                    "delta": delta,
                    "reported": f_res.value,
                    "calculated": calc_res,
                })
                f_liab.verification_status = VerificationStatus.DISCREPANCY
                f_eq.verification_status = VerificationStatus.DISCREPANCY

        # ── KIỂM TRA 5: LỢI NHUẬN GỘP = DOANH THU THUẦN - GIÁ VỐN (Mã 20 == 10 - 11) ───
        if (
            "GROSS_PROFIT" in current_facts_by_concept
            and "NET_REVENUE" in current_facts_by_concept
            and "COGS" in current_facts_by_concept
        ):
            f_gp = current_facts_by_concept["GROSS_PROFIT"]
            f_rev = current_facts_by_concept["NET_REVENUE"]
            f_cogs = current_facts_by_concept["COGS"]
            # Lưu ý giá vốn có thể là số âm ngoặc đơn hoặc số dương đã làm sạch
            cogs_val = abs(f_cogs.value)
            calc_gp = f_rev.value - cogs_val

            if is_equal(f_gp.value, calc_gp):
                passed_checks.append("CÂN_ĐỐI_LỢI_NHUẬN_GỘP: LN Gộp (20) == Doanh thu thuần (10) - Giá vốn (11)")
                f_gp.verification_status = VerificationStatus.VERIFIED
                f_gp.verification_detail = "Khớp công thức Doanh thu thuần - Giá vốn"
                f_rev.verification_status = VerificationStatus.VERIFIED
                f_cogs.verification_status = VerificationStatus.VERIFIED
            else:
                delta = abs(f_gp.value - calc_gp)
                failed_checks.append(f"LỆCH_LN_GỘP: Báo cáo ({f_gp.value:,.0f}) != Tính toán ({calc_gp:,.0f}), Lệch: {delta:,.0f}")
                discrepancies.append({
                    "check": "GROSS_PROFIT == NET_REVENUE - COGS",
                    "delta": delta,
                    "reported": f_gp.value,
                    "calculated": calc_gp,
                })

        total_checks = len(passed_checks) + len(failed_checks)
        is_balanced = len(failed_checks) == 0

        summary = (
            f"Kiểm toán số học BCTC {company} ({year}): Đạt {len(passed_checks)}/{total_checks} bài kiểm tra đẳng thức. "
            f"Trạng thái: {'✅ HOÀN TOÀN CÂN ĐỐI (KHÔNG CÓ GIGO)' if is_balanced else '⚠️ PHÁT HIỆN SAI LỆCH CẦN LƯU Ý'}."
        )

        logger.info("AccountingVerifier: %s", summary)

        return VerificationReport(
            company=company,
            year=year,
            is_balanced=is_balanced,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            discrepancies=discrepancies,
            summary=summary,
        )
