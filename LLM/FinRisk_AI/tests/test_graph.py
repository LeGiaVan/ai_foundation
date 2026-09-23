"""
test_graph.py — Test suite cho Phase 1: LangGraph Multi-Agent Core.

Triết lý test:
  - Unit test từng Node độc lập (không cần chạy cả graph)
  - Integration test toàn bộ pipeline với dữ liệu tài chính mẫu thực tế
  - HITL test: Kiểm tra interrupt và resume đúng cách

Chạy: pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch

from src.tools.financial_tools import (
    calculate_altman_z_score,
    calculate_dscr,
    calculate_debt_to_equity,
    calculate_quick_ratio,
)
from src.agents.supervisor import supervisor_node, route_after_supervisor
from src.agents.risk_agent import _calculate_rule_based_score


# ── Fixtures: Dữ liệu tài chính mẫu ─────────────────────────────────────────

@pytest.fixture
def healthy_company_state():
    """Doanh nghiệp khỏe mạnh — Altman Z > 3, DSCR > 1.5, D/E < 1.0."""
    return {
        "session_id": "test-healthy-001",
        "user_id": "analyst_test",
        "question": "Phân tích rủi ro tín dụng",
        "company_name": "Vinamilk JSC",
        "document_paths": [],
        "next_agent": "",
        "iteration": 0,
        "retrieved_context": [],
        "raw_financials": {
            "working_capital": 5_000_000_000,
            "total_assets": 50_000_000_000,
            "retained_earnings": 20_000_000_000,
            "ebit": 8_000_000_000,
            "market_cap": 40_000_000_000,
            "total_liabilities": 15_000_000_000,
            "revenue": 60_000_000_000,
        },
        "financial_metrics": {},
        "risk_assessment": {},
        "requires_human_approval": False,
        "hitl_reason": "",
        "human_decision": None,
        "human_comment": None,
        "final_report": "",
        "messages": [],
        "loan_application": None,
        "external_news_context": [],
    }


@pytest.fixture
def distressed_company_state():
    """Doanh nghiệp rủi ro cao — Z-Score DISTRESS, DSCR < 1.0."""
    return {
        "session_id": "test-distress-001",
        "user_id": "analyst_test",
        "question": "Phân tích rủi ro tín dụng",
        "company_name": "RiskyCompany Corp",
        "document_paths": [],
        "next_agent": "",
        "iteration": 0,
        "retrieved_context": [],
        "raw_financials": {},
        "financial_metrics": {
            "altman_z_score": 1.2,    # DISTRESS zone
            "dscr": 0.85,             # Không đủ khả năng trả nợ
            "debt_to_equity": 4.5,    # Đòn bẩy rất cao
            "quick_ratio": 0.6,       # Thanh khoản yếu
            "gross_margin": None,
            "net_cash_from_operations": None,
        },
        "risk_assessment": {},
        "requires_human_approval": False,
        "hitl_reason": "",
        "human_decision": None,
        "human_comment": None,
        "final_report": "",
        "messages": [],
        "loan_application": None,
        "external_news_context": [],
    }


# ── Unit Tests: Financial Tools ──────────────────────────────────────────────

class TestAltmanZScore:
    def test_safe_zone(self):
        result = calculate_altman_z_score.invoke({
            "working_capital": 5_000_000,
            "total_assets": 20_000_000,
            "retained_earnings": 8_000_000,
            "ebit": 3_000_000,
            "market_cap": 15_000_000,
            "total_liabilities": 6_000_000,
            "revenue": 25_000_000,
        })
        assert result["zone"] == "SAFE"
        assert result["z_score"] > 2.99

    def test_distress_zone(self):
        result = calculate_altman_z_score.invoke({
            "working_capital": -1_000_000,    # Vốn lưu động âm
            "total_assets": 10_000_000,
            "retained_earnings": -500_000,    # Lỗ lũy kế
            "ebit": 100_000,                  # EBIT rất thấp
            "market_cap": 2_000_000,
            "total_liabilities": 9_000_000,   # Nợ gần bằng tài sản
            "revenue": 5_000_000,
        })
        assert result["zone"] == "DISTRESS"
        assert result["z_score"] < 1.81

    def test_zero_assets_returns_error(self):
        result = calculate_altman_z_score.invoke({
            "working_capital": 100,
            "total_assets": 0,    # Edge case: tổng tài sản = 0
            "retained_earnings": 100,
            "ebit": 100,
            "market_cap": 100,
            "total_liabilities": 100,
            "revenue": 100,
        })
        assert "error" in result


class TestDSCR:
    def test_adequate_dscr(self):
        result = calculate_dscr.invoke({
            "net_operating_income": 1_500_000,
            "annual_debt_service": 1_000_000,  # DSCR = 1.5
        })
        assert result["dscr"] == pytest.approx(1.5, abs=0.01)
        assert result["verdict"] == "ADEQUATE"

    def test_insufficient_dscr(self):
        result = calculate_dscr.invoke({
            "net_operating_income": 800_000,
            "annual_debt_service": 1_000_000,  # DSCR = 0.8 < 1.0
        })
        assert result["verdict"] == "INSUFFICIENT"


class TestDebtToEquity:
    def test_negative_equity(self):
        result = calculate_debt_to_equity.invoke({
            "total_debt": 5_000_000,
            "total_equity": -100_000,    # Vốn âm → mất hết vốn
        })
        assert result["risk_level"] == "CRITICAL"
        assert result["debt_to_equity"] is None


# ── Unit Tests: Rule-based Risk Scoring ─────────────────────────────────────

class TestRuleBasedScoring:
    def test_high_risk_metrics_give_high_score(self):
        metrics = {
            "altman_z_score": 1.2,   # DISTRESS → +30
            "dscr": 0.7,             # < 1.0 → +40
            "debt_to_equity": 4.0,   # > 3.0 → +20
            "quick_ratio": 0.5,      # < 0.8 → +10
        }
        score, rules = _calculate_rule_based_score(metrics)
        assert score >= 70, "Tất cả chỉ số xấu phải cho score >= 70"
        assert len(rules) == 4, "Phải trigger 4 rules"

    def test_healthy_metrics_give_low_score(self):
        metrics = {
            "altman_z_score": 3.5,   # SAFE
            "dscr": 1.8,             # ADEQUATE
            "debt_to_equity": 0.5,   # LOW
            "quick_ratio": 1.5,      # ADEQUATE
        }
        score, rules = _calculate_rule_based_score(metrics)
        assert score == 0.0, "Tất cả chỉ số tốt phải cho score = 0"
        assert len(rules) == 0


# ── Unit Tests: Supervisor Routing ──────────────────────────────────────────

class TestSupervisorRouting:
    def test_routes_to_math_when_no_metrics(self, healthy_company_state):
        mock_llm = MagicMock()
        result = supervisor_node(healthy_company_state, llm=mock_llm)
        assert result["next_agent"] == "math_agent"

    def test_routes_to_risk_when_metrics_exist(self, healthy_company_state):
        healthy_company_state["financial_metrics"] = {
            "altman_z_score": 3.5, "dscr": 1.5,
            "debt_to_equity": None, "quick_ratio": None,
            "gross_margin": None, "net_cash_from_operations": None,
        }
        mock_llm = MagicMock()
        result = supervisor_node(healthy_company_state, llm=mock_llm)
        assert result["next_agent"] == "risk_agent"

    def test_routes_to_end_when_complete(self, healthy_company_state):
        healthy_company_state["financial_metrics"] = {
            "altman_z_score": 3.5, "dscr": 1.5,
            "debt_to_equity": 0.5, "quick_ratio": 1.2,
            "gross_margin": None, "net_cash_from_operations": None,
        }
        healthy_company_state["risk_assessment"] = {
            "risk_score": 10.0, "risk_level": "LOW",
            "recommendation": "FAST_TRACK_REVIEW", "red_flags": [],
            "five_c_summary": {}, "credit_covenants": [],
        }
        mock_llm = MagicMock()
        result = supervisor_node(healthy_company_state, llm=mock_llm)
        assert result["next_agent"] == "END"

    def test_max_iterations_forces_end(self, healthy_company_state):
        healthy_company_state["iteration"] = 999
        mock_llm = MagicMock()
        result = supervisor_node(healthy_company_state, llm=mock_llm)
        assert result["next_agent"] == "END"
