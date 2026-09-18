"""
risk_agent.py — Node Risk Agent: Phán quyết rủi ro tín dụng tổng thể.

Đây là node "não phán xét" cuối cùng của pipeline:
  1. Đọc financial_metrics từ Math Agent
  2. Đọc retrieved_context từ RAG Agent (phần thuyết minh BCTC, kiện tụng, v.v.)
  3. Áp dụng Khung 5C → Tính risk_score (0–100) và classification
  4. Nếu risk_score vượt ngưỡng → đặt requires_human_approval = True

Scoring model (có thể thay bằng ML model trong tương lai):
  - DSCR < 1.0           → +40 điểm rủi ro (nghiêm trọng nhất)
  - Z-Score DISTRESS      → +30 điểm rủi ro
  - D/E > 3.0             → +20 điểm rủi ro
  - Quick Ratio < 0.8     → +10 điểm rủi ro
  - LLM tìm thêm red flags từ context → +variable điểm

Scale-up hint:
  - Thay rule-based scoring bằng ML model (scikit-learn / XGBoost) khi có dữ liệu lịch sử.
  - Phase 3: Wrap trong @observe(name="risk_assessment") để track trên Langfuse.
"""

import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

from src.agents.state import FinRiskState, RiskAssessment
from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

RISK_AGENT_SYSTEM_PROMPT = """
Bạn là Giám định viên Rủi ro Tín dụng cấp cao, áp dụng Khung 5C (Capacity, Capital, Conditions, Character, Collateral).

Dựa trên các chỉ số tài chính và thông tin định tính từ BCTC được cung cấp, hãy:
1. Đánh giá từng chữ C theo thang: "TỐT" / "TRUNG BÌNH" / "YẾU" / "NGUY HIỂM"
2. Liệt kê các Red Flags phát hiện được (nếu có)
3. Đưa ra khuyến nghị: "APPROVE" (Phê duyệt) / "REVIEW" (Cần xem xét thêm) / "REJECT" (Từ chối)

Trả về JSON nghiêm ngặt theo định dạng:
{
  "five_c_summary": {
    "Capacity": "...",
    "Capital": "...",
    "Conditions": "...",
    "Character": "Cần Human Review - không thể tự đánh giá",
    "Collateral": "..."
  },
  "red_flags": ["...", "..."],
  "qualitative_risk_addition": <số từ 0-20, điểm rủi ro bổ sung từ phân tích định tính>
}
""".strip()


def _calculate_rule_based_score(metrics: dict) -> tuple[float, list[str]]:
    """
    Tính risk_score bằng rule-based model dựa trên ngưỡng nghiệp vụ.
    Trả về (score, list_of_triggered_rules).

    Note: Score 0 = Rủi ro thấp nhất, 100 = Rủi ro cao nhất.
    """
    score = 0.0
    triggered = []

    dscr = metrics.get("dscr")
    if dscr is not None:
        if dscr < settings.dscr_minimum:            # < 1.0
            score += 40
            triggered.append(f"DSCR={dscr:.2f} < {settings.dscr_minimum} (Không đủ khả năng trả nợ)")
        elif dscr < 1.25:
            score += 15
            triggered.append(f"DSCR={dscr:.2f} — Biên an toàn mỏng")

    z = metrics.get("altman_z_score")
    if z is not None:
        if z < settings.zscore_distress_threshold:  # < 1.81
            score += 30
            triggered.append(f"Z-Score={z:.2f} — Vùng nguy hiểm (phá sản)")
        elif z < 2.99:
            score += 10
            triggered.append(f"Z-Score={z:.2f} — Vùng xám (Grey Zone)")

    de = metrics.get("debt_to_equity")
    if de is not None:
        if de > settings.debt_equity_max:           # > 3.0
            score += 20
            triggered.append(f"D/E={de:.2f} > {settings.debt_equity_max} (Đòn bẩy quá cao)")
        elif de > 2.0:
            score += 8
            triggered.append(f"D/E={de:.2f} — Đòn bẩy cao")

    qr = metrics.get("quick_ratio")
    if qr is not None:
        if qr < settings.quick_ratio_minimum:       # < 0.8
            score += 10
            triggered.append(f"Quick Ratio={qr:.2f} — Thanh khoản yếu")

    return min(score, 100.0), triggered


def risk_agent_node(state: FinRiskState, llm: BaseChatModel) -> dict:
    """
    Node Risk Agent — tổng hợp phán quyết tín dụng theo Khung 5C.
    """
    logger.info("Risk Agent invoked for session=%s", state.get("session_id"))

    metrics = state.get("financial_metrics") or {}
    context = "\n\n".join(state.get("retrieved_context", []))

    # Bước 1: Rule-based score từ chỉ số định lượng
    rule_score, triggered_rules = _calculate_rule_based_score(metrics)

    # Bước 2: LLM bổ sung phân tích định tính (qualitative) từ context
    llm_response = llm.invoke([
        SystemMessage(content=RISK_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Chỉ số tài chính đã tính:
{metrics}

Thông tin từ Báo cáo Tài chính:
{context or "Chưa có context (Phase 1 — sẽ bổ sung ở Phase 2)"}

Câu hỏi: {state.get("question", "")}
"""),
    ])

    # Parse JSON từ LLM (có fallback nếu LLM trả về format sai)
    import json, re
    llm_text = llm_response.content
    qualitative_addition = 0
    five_c_summary = {}
    llm_red_flags = []

    try:
        # Tìm JSON block trong response
        json_match = re.search(r'\{.*\}', llm_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            five_c_summary = parsed.get("five_c_summary", {})
            llm_red_flags = parsed.get("red_flags", [])
            qualitative_addition = min(float(parsed.get("qualitative_risk_addition", 0)), 20)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("LLM returned non-JSON for risk assessment: %s", e)

    # Bước 3: Tính score cuối + phân loại
    final_score = min(rule_score + qualitative_addition, 100.0)
    all_red_flags = triggered_rules + llm_red_flags

    if final_score >= settings.risk_score_reject_threshold:    # >= 70
        risk_level = "CRITICAL"
        recommendation = "REJECT"
        requires_hitl = True
        hitl_reason = f"Risk Score={final_score:.0f} vượt ngưỡng từ chối {settings.risk_score_reject_threshold}."
    elif final_score >= settings.risk_score_review_threshold:  # >= 40
        risk_level = "HIGH"
        recommendation = "REVIEW"
        requires_hitl = True
        hitl_reason = f"Risk Score={final_score:.0f} yêu cầu Giám đốc Tín dụng xem xét."
    elif final_score >= 20:
        risk_level = "MEDIUM"
        recommendation = "APPROVE"
        requires_hitl = False
        hitl_reason = ""
    else:
        risk_level = "LOW"
        recommendation = "APPROVE"
        requires_hitl = False
        hitl_reason = ""

    logger.info(
        "Risk assessment complete: score=%.1f level=%s recommendation=%s hitl=%s",
        final_score, risk_level, recommendation, requires_hitl,
    )

    risk_assessment: RiskAssessment = {
        "risk_score": round(final_score, 2),
        "risk_level": risk_level,
        "recommendation": recommendation,
        "red_flags": all_red_flags,
        "five_c_summary": five_c_summary,
    }

    return {
        "risk_assessment": risk_assessment,
        "requires_human_approval": requires_hitl,
        "hitl_reason": hitl_reason,
    }
