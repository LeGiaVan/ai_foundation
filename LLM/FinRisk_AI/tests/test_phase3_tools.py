"""
test_phase3_tools.py — Test suite cho Phase 3 Blueprint: Tools mới & DoA model.

Coverage:
  - calculate_pro_forma_dscr: PMT formula, edge cases, verdict mapping
  - calculate_ltv: ngưỡng theo loại TSBĐ, SAFE/MARGINAL/BREACH
  - search_external_risk_news: fallback mode (không có API key)
  - _calculate_rule_based_score: Pro-forma DSCR + LTV trong scoring
  - _map_to_doa: DoA 4 cấp theo risk_score và loan_amount
  - _generate_default_covenants: logic sinh covenants
  - Backward-compatibility: tests cũ không bị ảnh hưởng

Chạy: pytest tests/test_phase3_tools.py -v
"""

import pytest
from unittest.mock import MagicMock, patch

from src.tools.financial_tools import (
    calculate_pro_forma_dscr,
    calculate_ltv,
    search_external_risk_news,
    FINANCIAL_TOOLS,
)
from src.agents.risk_agent import (
    _calculate_rule_based_score,
    _map_to_doa,
    _generate_default_covenants,
)


# ── Tests: Pro-forma DSCR ─────────────────────────────────────────────────────

class TestProFormaDSCR:
    """
    Pro-forma DSCR là chỉ số nghiệp vụ quan trọng nhất khi thẩm định cho vay.
    DSCR lịch sử có thể đủ nhưng sau khi gánh khoản vay mới có thể sập.
    """

    def test_adequate_verdict_dscr_above_1_25(self):
        """Trường hợp tốt: DSCR dự phóng >= 1.25 → ADEQUATE."""
        result = calculate_pro_forma_dscr.invoke({
            "cfo": 3_000_000_000,             # 3 tỷ CFO/năm
            "existing_debt_service": 500_000_000,  # 500 triệu nợ cũ
            "new_loan_amount": 5_000_000_000, # Xin vay 5 tỷ
            "term_months": 60,                # 5 năm
            "annual_interest_rate": 9.0,      # 9%/năm
        })
        assert result["verdict"] == "ADEQUATE", f"Expected ADEQUATE, got {result}"
        assert result["pro_forma_dscr"] >= 1.25

    def test_insufficient_verdict_when_dscr_drops_below_1(self):
        """
        Kịch bản chết người: DSCR lịch sử đủ (1.3), nhưng vay thêm 50 tỷ
        làm Pro-forma DSCR sập xuống < 1.0.
        """
        result = calculate_pro_forma_dscr.invoke({
            "cfo": 2_000_000_000,                  # 2 tỷ CFO/năm (tương đương DSCR cũ 1.3)
            "existing_debt_service": 1_500_000_000, # 1.5 tỷ nợ cũ → DSCR lịch sử ≈ 1.33
            "new_loan_amount": 50_000_000_000,      # Xin vay thêm 50 tỷ!
            "term_months": 60,
            "annual_interest_rate": 9.5,
        })
        assert result["verdict"] == "INSUFFICIENT", (
            f"Pro-forma DSCR phải INSUFFICIENT khi gánh khoản vay lớn, got: {result}"
        )
        assert result["pro_forma_dscr"] < 1.0

    def test_marginal_verdict_between_1_and_1_25(self):
        """
        DSCR dự phóng trong vùng 1.0 - 1.25 → MARGINAL.

        Tính tay để chọn số liệu đúng:
          r = 9%/12 = 0.75%; n = 120 tháng
          PMT = 3B × 0.0075 × (1.0075)^120 / ((1.0075)^120 - 1) ≈ 37.97M/tháng
          new_annual_pmt ≈ 455.6M/năm
          total_ds = 300M (cũ) + 455.6M = 755.6M
          Pro-forma DSCR = 850M / 755.6M ≈ 1.12 → MARGINAL ✓
        """
        result = calculate_pro_forma_dscr.invoke({
            "cfo": 850_000_000,               # 850 triệu CFO/năm
            "existing_debt_service": 300_000_000,  # 300 triệu nợ cũ
            "new_loan_amount": 3_000_000_000, # Vay thêm 3 tỷ
            "term_months": 120,               # 10 năm
            "annual_interest_rate": 9.0,      # 9%/năm
        })
        assert result["verdict"] == "MARGINAL", (
            f"Với số liệu này Pro-forma DSCR phải MARGINAL (~1.12), got: {result}"
        )
        assert 1.0 <= result["pro_forma_dscr"] < 1.25

    def test_pmt_formula_correctness(self):
        """Xác nhận công thức PMT tính đúng theo tài chính chuẩn."""
        # PMT tính tay: P=1,200,000, r=10%/12=0.8333%, n=12
        # PMT = 1,200,000 × 0.008333 × (1.008333)^12 / ((1.008333)^12 - 1)
        # ≈ 105,479
        result = calculate_pro_forma_dscr.invoke({
            "cfo": 2_000_000,
            "existing_debt_service": 0.01,   # Gần 0, để tách biệt PMT mới
            "new_loan_amount": 1_200_000,
            "term_months": 12,
            "annual_interest_rate": 10.0,
        })
        # new_annual_pmt ≈ 105,479 × 12 ≈ 1,265,748 (xấp xỉ)
        assert result["new_annual_pmt"] is not None
        assert 1_200_000 < result["new_annual_pmt"] < 1_400_000, (
            f"PMT không hợp lý: {result['new_annual_pmt']}"
        )

    def test_negative_cfo_returns_reject(self):
        """CFO âm → doanh nghiệp không tạo ra tiền từ hoạt động → trả về REJECT."""
        result = calculate_pro_forma_dscr.invoke({
            "cfo": -500_000_000,   # CFO âm
            "existing_debt_service": 1_000_000_000,
            "new_loan_amount": 5_000_000_000,
            "term_months": 60,
            "annual_interest_rate": 9.0,
        })
        assert "error" in result or result.get("verdict") == "REJECT"

    def test_zero_interest_rate_returns_error(self):
        """Lãi suất = 0 → trả về lỗi."""
        result = calculate_pro_forma_dscr.invoke({
            "cfo": 2_000_000_000,
            "existing_debt_service": 500_000_000,
            "new_loan_amount": 5_000_000_000,
            "term_months": 60,
            "annual_interest_rate": 0.0,
        })
        assert "error" in result


# ── Tests: LTV ───────────────────────────────────────────────────────────────

class TestLTV:
    """
    Kiểm tra ngưỡng LTV theo từng loại TSBĐ:
    - BĐS: 70%
    - Máy móc: 50%
    - Chứng khoán: 60%
    - Khác: 40%
    """

    def test_real_estate_safe_below_70pct(self):
        result = calculate_ltv.invoke({
            "loan_amount": 6_000_000_000,         # 6 tỷ
            "collateral_value": 10_000_000_000,   # TSBĐ 10 tỷ → LTV 60%
            "collateral_type": "real_estate",
        })
        assert result["status"] == "SAFE"
        assert result["ltv_pct"] == pytest.approx(60.0, abs=0.1)

    def test_real_estate_breach_above_70pct(self):
        result = calculate_ltv.invoke({
            "loan_amount": 8_500_000_000,         # 8.5 tỷ
            "collateral_value": 10_000_000_000,   # TSBĐ 10 tỷ → LTV 85% > 70%
            "collateral_type": "real_estate",
        })
        assert result["status"] == "BREACH"
        assert result["ltv_pct"] > 70.0

    def test_machinery_breach_above_50pct(self):
        result = calculate_ltv.invoke({
            "loan_amount": 6_000_000_000,         # LTV 60% > 50% máy móc
            "collateral_value": 10_000_000_000,
            "collateral_type": "machinery",
        })
        assert result["status"] == "BREACH"
        assert result["max_allowed_ltv_pct"] == 50.0

    def test_securities_safe_at_55pct(self):
        result = calculate_ltv.invoke({
            "loan_amount": 5_500_000_000,         # LTV 55% < 60% chứng khoán
            "collateral_value": 10_000_000_000,
            "collateral_type": "securities",
        })
        assert result["status"] == "SAFE"

    def test_other_collateral_breach_above_40pct(self):
        result = calculate_ltv.invoke({
            "loan_amount": 5_000_000_000,         # LTV 50% > 40% tài sản khác
            "collateral_value": 10_000_000_000,
            "collateral_type": "other",
        })
        assert result["status"] == "BREACH"

    def test_max_safe_loan_amount_is_correct(self):
        """max_safe_loan_amount = collateral_value × max_ltv_ceiling."""
        result = calculate_ltv.invoke({
            "loan_amount": 9_000_000_000,
            "collateral_value": 10_000_000_000,
            "collateral_type": "real_estate",   # ceiling 70%
        })
        assert result["max_safe_loan_amount"] == pytest.approx(7_000_000_000, rel=0.01)

    def test_zero_collateral_returns_error(self):
        result = calculate_ltv.invoke({
            "loan_amount": 1_000_000,
            "collateral_value": 0,
            "collateral_type": "real_estate",
        })
        assert "error" in result
        assert result["status"] == "ERROR"

    def test_marginal_zone_just_above_ceiling(self):
        """LTV vượt nhẹ (trong 15% buffer) → MARGINAL, không phải BREACH."""
        # Ceiling BĐS = 70%; 70% × 1.15 = 80.5%; thử LTV = 75%
        result = calculate_ltv.invoke({
            "loan_amount": 7_500_000_000,     # 75%
            "collateral_value": 10_000_000_000,
            "collateral_type": "real_estate",
        })
        assert result["status"] == "MARGINAL"


# ── Tests: External News Search ──────────────────────────────────────────────

class TestExternalNewsSearch:
    def test_fallback_mode_when_no_api_key(self):
        """Không có Tavily key → trả về NO_API_KEY với disclaimer rõ ràng."""
        with patch("src.config.get_settings") as mock_settings:
            mock_settings.return_value.tavily_api_key = ""
            result = search_external_risk_news.invoke({
                "company_name": "Công ty TNHH ABC",
                "tax_code": "0123456789",
            })
        assert result["source"] == "NO_API_KEY"
        assert result["found_risks"] == []
        assert "CẢNH BÁO" in result["disclaimer"]

    def test_result_structure_always_consistent(self):
        """Dù có hay không có API key, cấu trúc dict trả về phải nhất quán."""
        with patch("src.config.get_settings") as mock_settings:
            mock_settings.return_value.tavily_api_key = ""
            result = search_external_risk_news.invoke({
                "company_name": "Test Corp",
            })
        required_keys = ["source", "company_name", "found_risks", "risk_keywords_hit", "raw_results"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"


# ── Tests: Financial Tools Registry ─────────────────────────────────────────

class TestFinancialToolsRegistry:
    def test_financial_tools_has_7_tools(self):
        """FINANCIAL_TOOLS phải có đúng 7 tools sau khi thêm Phase 3."""
        assert len(FINANCIAL_TOOLS) == 7, (
            f"Expected 7 tools, got {len(FINANCIAL_TOOLS)}: "
            f"{[t.name for t in FINANCIAL_TOOLS]}"
        )

    def test_new_tools_are_registered(self):
        """Xác nhận 3 tools mới đã được đăng ký."""
        tool_names = {t.name for t in FINANCIAL_TOOLS}
        assert "calculate_pro_forma_dscr" in tool_names
        assert "calculate_ltv" in tool_names
        assert "search_external_risk_news" in tool_names


# ── Tests: Rule-based Scoring với Pro-forma DSCR + LTV ──────────────────────

class TestRuleBasedScoringPhase3:
    def test_pro_forma_dscr_insufficient_adds_45_points(self):
        metrics = {
            "pro_forma_dscr": 0.75,   # < 1.0 → +45
            "dscr": 1.3,              # Lịch sử tốt, không trigger
            "altman_z_score": 3.5,    # SAFE
            "debt_to_equity": 0.8,    # LOW
            "quick_ratio": 1.2,       # ADEQUATE
        }
        score, rules = _calculate_rule_based_score(metrics)
        assert score >= 45, f"Pro-forma DSCR < 1.0 phải cộng ít nhất 45 điểm, got {score}"
        assert any("Pro-forma DSCR" in r for r in rules)

    def test_ltv_breach_adds_25_points(self):
        metrics = {
            "ltv_status": "BREACH",
            "altman_z_score": 3.5,
            "dscr": 1.5,
            "debt_to_equity": 0.8,
            "quick_ratio": 1.2,
        }
        score, rules = _calculate_rule_based_score(metrics)
        assert score >= 25, f"LTV BREACH phải cộng ít nhất 25 điểm, got {score}"
        assert any("LTV" in r for r in rules)

    def test_all_bad_metrics_capped_at_100(self):
        metrics = {
            "pro_forma_dscr": 0.5,    # +45
            "dscr": 0.7,              # +40
            "altman_z_score": 1.2,    # +30
            "ltv_status": "BREACH",   # +25
            "debt_to_equity": 4.0,    # +20
            "quick_ratio": 0.5,       # +10
        }
        score, rules = _calculate_rule_based_score(metrics)
        assert score == 100.0, f"Score phải được cap tại 100, got {score}"

    def test_pro_forma_marginal_adds_20_points(self):
        """Pro-forma DSCR trong vùng 1.0 - 1.25 → +20 điểm."""
        metrics = {
            "pro_forma_dscr": 1.10,   # marginal → +20
            "dscr": 1.5,
            "altman_z_score": 3.5,
            "debt_to_equity": 0.8,
            "quick_ratio": 1.2,
        }
        score, rules = _calculate_rule_based_score(metrics)
        assert score == 20.0, f"Pro-forma DSCR marginal phải cho 20 điểm, got {score}"


# ── Tests: DoA Mapping ───────────────────────────────────────────────────────

class TestDoAMapping:
    def test_low_score_gives_fast_track(self):
        recommendation, _, requires_hitl = _map_to_doa(25.0, 5_000_000_000, [])
        assert recommendation == "FAST_TRACK_REVIEW"
        assert not requires_hitl

    def test_score_40_to_69_gives_standard_audit(self):
        recommendation, reason, requires_hitl = _map_to_doa(55.0, 5_000_000_000, [])
        assert recommendation == "STANDARD_AUDIT"
        assert requires_hitl
        assert len(reason) > 0

    def test_score_70_plus_gives_decline(self):
        recommendation, reason, requires_hitl = _map_to_doa(75.0, None, ["Bad DSCR"])
        assert recommendation == "DECLINE_RECOMMENDED"
        assert requires_hitl

    def test_large_loan_amount_escalates_to_credit_committee(self):
        """Khoản vay >= 10 tỷ VND → nâng lên CREDIT_COMMITTEE dù score 40-69."""
        recommendation, reason, requires_hitl = _map_to_doa(
            50.0, 15_000_000_000, []   # 15 tỷ > ngưỡng 10 tỷ
        )
        assert recommendation == "CREDIT_COMMITTEE"
        assert requires_hitl

    def test_large_loan_low_score_still_committee(self):
        """Khoản vay rất lớn dù score thấp vẫn cần CREDIT_COMMITTEE."""
        recommendation, reason, requires_hitl = _map_to_doa(
            20.0, 12_000_000_000, []   # Score thấp nhưng khoản vay 12 tỷ
        )
        assert recommendation == "CREDIT_COMMITTEE"
        assert requires_hitl


# ── Tests: Credit Covenants ──────────────────────────────────────────────────

class TestCreditCovenants:
    def test_high_de_generates_de_covenant(self):
        metrics = {"debt_to_equity": 2.8, "dscr": 1.2, "quick_ratio": 0.9}
        covenants = _generate_default_covenants(metrics, "STANDARD_AUDIT")
        assert any("D/E" in c or "nợ" in c.lower() for c in covenants), (
            f"D/E cao phải sinh covenant về đòn bẩy: {covenants}"
        )

    def test_low_quick_ratio_generates_liquidity_covenant(self):
        metrics = {"debt_to_equity": 1.0, "dscr": 1.3, "quick_ratio": 0.6}
        covenants = _generate_default_covenants(metrics, "FAST_TRACK_REVIEW")
        assert any("Quick Ratio" in c or "thanh khoản" in c.lower() for c in covenants)

    def test_standard_audit_generates_cash_flow_covenant(self):
        """STANDARD_AUDIT và CREDIT_COMMITTEE phải sinh covenant về tài khoản ngân hàng."""
        metrics = {"debt_to_equity": 1.5, "dscr": 1.1, "quick_ratio": 0.9}
        covenants = _generate_default_covenants(metrics, "STANDARD_AUDIT")
        assert any("doanh thu" in c.lower() or "70%" in c for c in covenants)

    def test_max_6_covenants(self):
        """Không sinh quá 6 covenants."""
        metrics = {
            "debt_to_equity": 2.8,
            "dscr": 0.9,
            "quick_ratio": 0.6,
            "pro_forma_dscr": 0.8,
        }
        covenants = _generate_default_covenants(metrics, "CREDIT_COMMITTEE")
        assert len(covenants) <= 6
