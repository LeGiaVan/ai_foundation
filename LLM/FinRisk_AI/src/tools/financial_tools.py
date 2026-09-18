"""
financial_tools.py — Python Tools cho Math Agent.

LÝ DO QUAN TRỌNG: LLM KHÔNG ĐƯỢC TỰ TÍNH TOÁN SỐ HỌC!
Altman Z-Score có 5 biến số và các hệ số thập phân — LLM rất dễ "ảo giác" sai
khi tính toán phức tạp. Do đó, tất cả phép tính được thực hiện bởi Python
100% chính xác, LLM chỉ làm nhiệm vụ trích xuất số liệu và diễn giải kết quả.

Mỗi tool là một @tool function được LangChain nhận dạng và gắn vào Agent.

Scale-up hint:
  - Thêm tool mới bằng cách tạo @tool function mới và đăng ký vào FINANCIAL_TOOLS.
  - Phase 2: Thêm tool `search_financial_context(query)` gọi Qdrant.
  - Phase 4: Thêm validation nghiêm ngặt hơn (Pydantic schemas cho args).
"""

import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ── Altman Z-Score ──────────────────────────────────────────────────────────

@tool
def calculate_altman_z_score(
    working_capital: float,
    total_assets: float,
    retained_earnings: float,
    ebit: float,
    market_cap: float,
    total_liabilities: float,
    revenue: float,
) -> dict:
    """
    Tính Altman Z-Score để dự báo nguy cơ phá sản trong 2 năm tới.
    Áp dụng cho công ty sản xuất có niêm yết (Mô hình Z gốc - Altman 1968).

    Args:
        working_capital: Vốn lưu động thuần = Tài sản ngắn hạn - Nợ ngắn hạn
        total_assets: Tổng tài sản
        retained_earnings: Lợi nhuận giữ lại tích lũy
        ebit: Lợi nhuận trước lãi vay và thuế (EBIT)
        market_cap: Vốn hóa thị trường (giá cổ phiếu × số CP lưu hành)
        total_liabilities: Tổng nợ phải trả
        revenue: Doanh thu thuần

    Returns:
        dict với z_score, zone, và interpretation
    """
    if total_assets <= 0:
        return {"error": "total_assets phải > 0", "z_score": None}

    # Tính 5 biến số theo công thức Altman
    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = market_cap / total_liabilities if total_liabilities > 0 else 0
    x5 = revenue / total_assets

    # Hệ số gốc của Altman (1968)
    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    # Phân loại theo ngưỡng nghiệp vụ
    if z > 2.99:
        zone = "SAFE"
        interpretation = f"An toàn (Z={z:.2f}). Xác suất phá sản thấp trong 2 năm tới."
    elif z >= 1.81:
        zone = "GREY"
        interpretation = f"Vùng cảnh báo (Z={z:.2f}). Cần theo dõi chặt chẽ."
    else:
        zone = "DISTRESS"
        interpretation = f"NGUY HIỂM (Z={z:.2f}). Xác suất phá sản cao trong 2 năm tới. Kích hoạt Human Review!"

    logger.info("Z-Score computed: z=%.2f zone=%s company_total_assets=%.0f", z, zone, total_assets)
    return {
        "z_score": round(z, 4),
        "zone": zone,
        "interpretation": interpretation,
        "breakdown": {"X1": round(x1, 4), "X2": round(x2, 4), "X3": round(x3, 4), "X4": round(x4, 4), "X5": round(x5, 4)},
    }


# ── DSCR ────────────────────────────────────────────────────────────────────

@tool
def calculate_dscr(
    net_operating_income: float,
    annual_debt_service: float,
) -> dict:
    """
    Tính DSCR (Debt Service Coverage Ratio) — Hệ số Khả năng Trả Nợ.
    Đây là chỉ số QUAN TRỌNG NHẤT trong 5C (Capacity).

    Args:
        net_operating_income: Thu nhập hoạt động thuần (EBITDA hoặc EBIT + Khấu hao)
        annual_debt_service: Tổng trả nợ trong năm = Gốc + Lãi phải trả

    Returns:
        dict với dscr, verdict, và interpretation
    """
    if annual_debt_service <= 0:
        return {"error": "annual_debt_service phải > 0", "dscr": None}

    dscr = net_operating_income / annual_debt_service

    if dscr >= 1.25:
        verdict = "ADEQUATE"
        interpretation = f"Đạt chuẩn (DSCR={dscr:.2f}). Doanh nghiệp tạo ra đủ tiền để trả nợ, còn dư an toàn."
    elif dscr >= 1.0:
        verdict = "MARGINAL"
        interpretation = f"Biên an toàn mỏng (DSCR={dscr:.2f}). Đủ trả nợ nhưng không có đệm dự phòng."
    else:
        verdict = "INSUFFICIENT"
        interpretation = f"KHÔNG ĐỦ (DSCR={dscr:.2f}). Dòng tiền KHÔNG đủ trả nợ. Cần từ chối hoặc yêu cầu tài sản đảm bảo bổ sung!"

    return {
        "dscr": round(dscr, 4),
        "verdict": verdict,
        "interpretation": interpretation,
    }


# ── Debt-to-Equity ──────────────────────────────────────────────────────────

@tool
def calculate_debt_to_equity(
    total_debt: float,
    total_equity: float,
) -> dict:
    """
    Tính D/E (Debt-to-Equity Ratio) — Hệ số Đòn bẩy Tài chính.
    Thuộc nhóm 5C: Capital.

    Args:
        total_debt: Tổng nợ phải trả (Nợ ngắn hạn + Nợ dài hạn)
        total_equity: Tổng vốn chủ sở hữu

    Returns:
        dict với ratio, risk_level, và interpretation
    """
    if total_equity <= 0:
        return {
            "debt_to_equity": None,
            "risk_level": "CRITICAL",
            "interpretation": "Vốn chủ sở hữu âm — doanh nghiệp kỹ thuật mà nói đã mất hết vốn."
        }

    ratio = total_debt / total_equity

    # Ngưỡng mặc định cho ngành sản xuất/dịch vụ phi tài chính
    if ratio <= 1.0:
        risk_level = "LOW"
        interpretation = f"Thấp (D/E={ratio:.2f}). Cơ cấu vốn lành mạnh, vốn tự có là chủ yếu."
    elif ratio <= 2.0:
        risk_level = "MEDIUM"
        interpretation = f"Trung bình (D/E={ratio:.2f}). Đòn bẩy vừa phải, cần theo dõi lãi suất."
    elif ratio <= 3.0:
        risk_level = "HIGH"
        interpretation = f"Cao (D/E={ratio:.2f}). Gánh nặng lãi vay đáng kể."
    else:
        risk_level = "CRITICAL"
        interpretation = f"RẤT CAO (D/E={ratio:.2f}). Rủi ro tài chính nghiêm trọng, kích hoạt Human Review!"

    return {
        "debt_to_equity": round(ratio, 4),
        "risk_level": risk_level,
        "interpretation": interpretation,
    }


# ── Quick Ratio ─────────────────────────────────────────────────────────────

@tool
def calculate_quick_ratio(
    cash_and_equivalents: float,
    short_term_investments: float,
    accounts_receivable: float,
    current_liabilities: float,
) -> dict:
    """
    Tính Quick Ratio (Acid-Test Ratio) — Thanh khoản Tức thì.
    Loại trừ Hàng tồn kho vì không thể bán ngay.

    Args:
        cash_and_equivalents: Tiền và tương đương tiền
        short_term_investments: Đầu tư tài chính ngắn hạn
        accounts_receivable: Khoản phải thu ngắn hạn
        current_liabilities: Tổng nợ ngắn hạn

    Returns:
        dict với ratio, verdict
    """
    if current_liabilities <= 0:
        return {"error": "current_liabilities phải > 0", "quick_ratio": None}

    liquid_assets = cash_and_equivalents + short_term_investments + accounts_receivable
    ratio = liquid_assets / current_liabilities

    if ratio >= 1.0:
        verdict = "ADEQUATE"
        interpretation = f"Đạt chuẩn (Quick Ratio={ratio:.2f}). Tài sản thanh khoản đủ trả nợ ngắn hạn."
    elif ratio >= 0.8:
        verdict = "MARGINAL"
        interpretation = f"Biên mỏng (Quick Ratio={ratio:.2f}). Cần theo dõi chu kỳ thu hồi công nợ."
    else:
        verdict = "WEAK"
        interpretation = f"YẾU (Quick Ratio={ratio:.2f}). Rủi ro thanh khoản ngắn hạn."

    return {
        "quick_ratio": round(ratio, 4),
        "verdict": verdict,
        "interpretation": interpretation,
    }


# ── Registry: danh sách tool đăng ký với Agent ──────────────────────────────

FINANCIAL_TOOLS = [
    calculate_altman_z_score,
    calculate_dscr,
    calculate_debt_to_equity,
    calculate_quick_ratio,
]
"""
Danh sách tools được bind vào Math Agent (llm.bind_tools(FINANCIAL_TOOLS)).
Scale-up: Thêm tool mới vào list này là đủ, không cần sửa Agent.
"""
