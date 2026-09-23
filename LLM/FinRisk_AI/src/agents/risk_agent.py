"""
risk_agent.py — Node Risk Agent: Phán quyết rủi ro tín dụng tổng thể.

Đây là node "não phán xét" cuối cùng của pipeline:
  1. Đọc financial_metrics từ Math Agent
  2. Đọc retrieved_context từ RAG Agent (phần thuyết minh BCTC, kiện tụng, v.v.)
  3. Đọc external_news_context từ Math Agent (Tavily Search)
  4. Áp dụng Khung 5C → Tính risk_score (0–100) và classification
  5. Áp dụng Ma trận phân quyền (DoA) → recommendation
  6. Sinh credit_covenants (giao ước tín dụng) tự động
  7. Nếu risk_score vượt ngưỡng → đặt requires_human_approval = True

Scoring model (có thể thay bằng ML model trong tương lai):
  - Pro-forma DSCR < 1.0    → +45 điểm rủi ro (quan trọng nhất — dòng tiền dự phóng)
  - DSCR lịch sử < 1.0      → +40 điểm rủi ro
  - Z-Score DISTRESS         → +30 điểm rủi ro
  - LTV BREACH               → +25 điểm rủi ro
  - D/E > 3.0                → +20 điểm rủi ro
  - Quick Ratio < 0.8        → +10 điểm rủi ro
  - LLM tìm thêm red flags từ context → +variable điểm

Ma trận phân quyền (DoA) — chuẩn nghiệp vụ ngân hàng thương mại:
  - "FAST_TRACK_REVIEW"   : Risk score < 40  → Chuyên viên 1-click duyệt
  - "STANDARD_AUDIT"      : Risk score 40-69 → HITL: Trưởng phòng phúc tra
  - "CREDIT_COMMITTEE"    : Hồ sơ phức tạp / khoản vay lớn → HITL: Hội đồng
  - "DECLINE_RECOMMENDED" : Risk score >= 70  → Kiến nghị từ chối

Scale-up hint:
  - Thay rule-based scoring bằng ML model (scikit-learn / XGBoost) khi có dữ liệu lịch sử.
  - Phase 3: Wrap trong @observe(name="risk_assessment") để track trên Langfuse.
"""

import json
import logging
import re
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
3. Đề xuất Credit Covenants (tối đa 5 giao ước ràng buộc) phù hợp với hồ sơ rủi ro
4. Ước tính điểm rủi ro định tính bổ sung (0–20) từ thông tin phi tài chính

Trả về JSON nghiêm ngặt theo định dạng:
{
  "five_c_summary": {
    "Capacity": "...",
    "Capital": "...",
    "Conditions": "...",
    "Character": "Cần Human Review - không thể tự đánh giá / [kết quả nếu có external_news]",
    "Collateral": "..."
  },
  "red_flags": ["...", "..."],
  "credit_covenants": [
    "Cam kết duy trì DSCR >= 1.20 mỗi năm, báo cáo kiểm toán quý",
    "Doanh thu phải đổ tối thiểu 70% qua tài khoản ngân hàng cho vay",
    "..."
  ],
  "qualitative_risk_addition": <số từ 0-20>
}
""".strip()


def _calculate_rule_based_score(metrics: dict) -> tuple[float, list[str]]:
    """
    Tính risk_score bằng rule-based model dựa trên ngưỡng nghiệp vụ.
    Trả về (score, list_of_triggered_rules).

    Note: Score 0 = Rủi ro thấp nhất, 100 = Rủi ro cao nhất.
    Thứ tự ưu tiên: Pro-forma DSCR > DSCR lịch sử > Z-Score > LTV > D/E > Quick Ratio.
    """
    score = 0.0
    triggered = []

    # ── Pro-forma DSCR (Quan trọng nhất — phản ánh gánh nặng khoản vay MỚI) ──
    pro_dscr = metrics.get("pro_forma_dscr")
    if pro_dscr is not None:
        if pro_dscr < 1.0:
            score += 45
            triggered.append(
                f"Pro-forma DSCR={pro_dscr:.2f} < 1.0 — Dòng tiền dự phóng KHÔNG đủ trả nợ sau giải ngân"
            )
        elif pro_dscr < 1.25:
            score += 20
            triggered.append(
                f"Pro-forma DSCR={pro_dscr:.2f} — Biên an toàn mỏng sau khi gánh khoản vay mới"
            )

    # ── DSCR lịch sử ──────────────────────────────────────────────────────────
    dscr = metrics.get("dscr")
    if dscr is not None:
        if dscr < settings.dscr_minimum:            # < 1.0
            score += 40
            triggered.append(f"DSCR lịch sử={dscr:.2f} < {settings.dscr_minimum} (Không đủ khả năng trả nợ hiện tại)")
        elif dscr < 1.25:
            score += 15
            triggered.append(f"DSCR lịch sử={dscr:.2f} — Biên an toàn mỏng")

    # ── Altman Z-Score ────────────────────────────────────────────────────────
    z = metrics.get("altman_z_score")
    if z is not None:
        if z < settings.zscore_distress_threshold:  # < 1.81
            score += 30
            triggered.append(f"Z-Score={z:.2f} — Vùng nguy hiểm (nguy cơ phá sản)")
        elif z < 2.99:
            score += 10
            triggered.append(f"Z-Score={z:.2f} — Vùng xám (Grey Zone, cần theo dõi)")

    # ── LTV (Loan-to-Value) ───────────────────────────────────────────────────
    ltv_status = metrics.get("ltv_status")
    if ltv_status == "BREACH":
        score += 25
        triggered.append(f"LTV vượt ngưỡng an toàn cho loại TSBĐ — rủi ro phát mãi thu hồi nợ")
    elif ltv_status == "MARGINAL":
        score += 10
        triggered.append(f"LTV ở mức cận ngưỡng — cần bổ sung TSBĐ")

    # ── D/E Ratio ─────────────────────────────────────────────────────────────
    de = metrics.get("debt_to_equity")
    if de is not None:
        if de > settings.debt_equity_max:           # > 3.0
            score += 20
            triggered.append(f"D/E={de:.2f} > {settings.debt_equity_max} (Đòn bẩy quá cao)")
        elif de > 2.0:
            score += 8
            triggered.append(f"D/E={de:.2f} — Đòn bẩy cao, cần theo dõi")

    # ── Quick Ratio ───────────────────────────────────────────────────────────
    qr = metrics.get("quick_ratio")
    if qr is not None:
        if qr < settings.quick_ratio_minimum:       # < 0.8
            score += 10
            triggered.append(f"Quick Ratio={qr:.2f} — Thanh khoản ngắn hạn yếu")

    return min(score, 100.0), triggered


def _map_to_doa(
    final_score: float,
    loan_amount: float | None,
    all_red_flags: list[str],
) -> tuple[str, str, bool]:
    """
    Ánh xạ risk_score → Ma trận phân quyền (Delegation of Authority - DoA).

    Returns:
        (recommendation, hitl_reason, requires_hitl)
    """
    # Ngưỡng khoản vay lớn (10 tỷ VND) → nâng lên CREDIT_COMMITTEE
    LARGE_LOAN_THRESHOLD = 10_000_000_000

    if final_score >= settings.risk_score_reject_threshold:  # >= 70
        return (
            "DECLINE_RECOMMENDED",
            f"Risk Score={final_score:.0f} >= {settings.risk_score_reject_threshold}. "
            f"Kiến nghị từ chối. Red Flags: {', '.join(all_red_flags[:3])}.",
            True,
        )

    if final_score >= settings.risk_score_review_threshold:  # >= 40
        if loan_amount and loan_amount >= LARGE_LOAN_THRESHOLD:
            return (
                "CREDIT_COMMITTEE",
                f"Risk Score={final_score:.0f} và khoản vay lớn ({loan_amount:,.0f} VND). "
                "Trình Hội đồng Tín dụng họp dual-signer.",
                True,
            )
        return (
            "STANDARD_AUDIT",
            f"Risk Score={final_score:.0f} trong ngưỡng 40-69. "
            "Chuyển Trưởng phòng Tín dụng và Thẩm định độc lập phúc tra.",
            True,
        )

    # Risk score < 40 — nhưng nếu khoản vay rất lớn vẫn cần Committee
    if loan_amount and loan_amount >= LARGE_LOAN_THRESHOLD:
        return (
            "CREDIT_COMMITTEE",
            f"Khoản vay lớn ({loan_amount:,.0f} VND) dù Risk Score={final_score:.0f} thấp. "
            "Quy trình phê duyệt đặc biệt theo chính sách ngân hàng.",
            True,
        )

    return (
        "FAST_TRACK_REVIEW",
        "",   # Không cần giải thích HITL vì không trigger HITL
        False,
    )


def _generate_default_covenants(metrics: dict, recommendation: str) -> list[str]:
    """
    Sinh credit covenants mặc định dựa trên hồ sơ rủi ro.
    Sẽ được bổ sung/thay thế bởi covenants từ LLM.
    """
    covenants = []

    dscr = metrics.get("dscr") or metrics.get("pro_forma_dscr")
    if dscr is not None and dscr < 1.5:
        covenants.append(
            "Duy trì DSCR >= 1.20 được kiểm tra hàng năm qua BCTC kiểm toán; "
            "vi phạm → ngân hàng có quyền thu hồi nợ trước hạn."
        )

    de = metrics.get("debt_to_equity")
    if de is not None and de > 1.5:
        covenant_de = max(round(de * 0.85, 1), 2.0)
        covenants.append(
            f"D/E không vượt {covenant_de}x tại bất kỳ thời điểm kiểm tra nào. "
            "Không phát hành thêm nợ mới khi chưa có sự chấp thuận bằng văn bản của ngân hàng."
        )

    qr = metrics.get("quick_ratio")
    if qr is not None and qr < 1.0:
        covenants.append(
            "Quick Ratio duy trì >= 0.80 tại thời điểm báo cáo quý. "
            "Cung cấp báo cáo dòng tiền 13 tuần (rolling cash flow) khi được yêu cầu."
        )

    if recommendation in ("STANDARD_AUDIT", "CREDIT_COMMITTEE", "DECLINE_RECOMMENDED"):
        covenants.append(
            "Tối thiểu 70% doanh thu phải được chuyển về tài khoản mở tại ngân hàng cho vay."
        )
        covenants.append(
            "Cung cấp BCTC quản trị (management accounts) định kỳ mỗi quý trong vòng 30 ngày sau khi kết thúc quý."
        )

    return covenants


def risk_agent_node(state: FinRiskState, llm: BaseChatModel) -> dict:
    """
    Node Risk Agent — tổng hợp phán quyết tín dụng theo Khung 5C + Chuẩn DoA.
    """
    logger.info("Risk Agent invoked for session=%s", state.get("session_id"))

    metrics = state.get("financial_metrics") or {}
    context = "\n\n".join(state.get("retrieved_context", []))
    external_news = "\n\n".join(state.get("external_news_context", []))
    loan_app = state.get("loan_application") or {}
    loan_amount = loan_app.get("loan_amount")

    # Bước 1: Rule-based score từ chỉ số định lượng
    rule_score, triggered_rules = _calculate_rule_based_score(metrics)

    # Bước 2: LLM bổ sung phân tích định tính (qualitative) từ context
    human_msg = f"""
Chỉ số tài chính đã tính:
{metrics}

Thông tin hồ sơ vay vốn:
{loan_app or "Chưa có thông tin hồ sơ vay"}

Thông tin từ Báo cáo Tài chính:
{context or "Chưa có context (Phase 1 — sẽ bổ sung ở Phase 2)"}

Thông tin từ nguồn bên ngoài (tin tức, pháp lý):
{external_news or "Không có dữ liệu ngoài (cần Tavily API hoặc tra cứu thủ công CIC)"}

Câu hỏi: {state.get("question", "")}
"""

    llm_response = llm.invoke([
        SystemMessage(content=RISK_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=human_msg.strip()),
    ])

    # Parse JSON từ LLM (có fallback nếu LLM trả về format sai)
    llm_text = llm_response.content
    qualitative_addition = 0
    five_c_summary = {}
    llm_red_flags = []
    llm_covenants = []

    try:
        # Tìm JSON block trong response
        json_match = re.search(r'\{.*\}', llm_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            five_c_summary = parsed.get("five_c_summary", {})
            llm_red_flags = parsed.get("red_flags", [])
            llm_covenants = parsed.get("credit_covenants", [])
            qualitative_addition = min(float(parsed.get("qualitative_risk_addition", 0)), 20)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("LLM returned non-JSON for risk assessment: %s", e)

    # Bước 3: Tính score cuối + phân loại
    final_score = min(rule_score + qualitative_addition, 100.0)
    all_red_flags = triggered_rules + llm_red_flags

    # Bước 4: Ánh xạ sang DoA
    recommendation, hitl_reason, requires_hitl = _map_to_doa(
        final_score, loan_amount, all_red_flags
    )

    # Bước 5: Phân loại mức rủi ro
    if final_score >= 70:
        risk_level = "CRITICAL"
    elif final_score >= 40:
        risk_level = "HIGH"
    elif final_score >= 20:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Bước 6: Tổng hợp Credit Covenants
    # LLM covenants ưu tiên; bổ sung default covenants nếu LLM thiếu
    default_covenants = _generate_default_covenants(metrics, recommendation)
    # Hợp nhất: LLM trước, default bổ sung những gì chưa có (tránh trùng lặp)
    final_covenants = list(llm_covenants)
    for cov in default_covenants:
        if not any(cov[:30] in existing for existing in final_covenants):
            final_covenants.append(cov)
    final_covenants = final_covenants[:6]  # Giới hạn tối đa 6 covenants

    logger.info(
        "Risk assessment complete: score=%.1f level=%s recommendation=%s hitl=%s covenants=%d",
        final_score, risk_level, recommendation, requires_hitl, len(final_covenants),
    )

    risk_assessment: RiskAssessment = {
        "risk_score": round(final_score, 2),
        "risk_level": risk_level,
        "recommendation": recommendation,
        "red_flags": all_red_flags,
        "five_c_summary": five_c_summary,
        "credit_covenants": final_covenants,
    }

    return {
        "risk_assessment": risk_assessment,
        "requires_human_approval": requires_hitl,
        "hitl_reason": hitl_reason,
    }
