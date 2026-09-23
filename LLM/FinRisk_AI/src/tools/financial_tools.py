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
  - Phase 3: Pro-forma DSCR, LTV, External News Search đã được bổ sung.
  - Phase 4: Thêm validation nghiêm ngặt hơn (Pydantic schemas cho args).
"""

import logging
import math
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


# ── DSCR (Lịch sử) ──────────────────────────────────────────────────────────

@tool
def calculate_dscr(
    net_operating_income: float,
    annual_debt_service: float,
) -> dict:
    """
    Tính DSCR lịch sử (Debt Service Coverage Ratio) — Hệ số Khả năng Trả Nợ.
    Đây là chỉ số QUAN TRỌNG NHẤT trong 5C (Capacity).
    Dùng số liệu BCTC quá khứ — KHÔNG phản ánh gánh nặng khoản vay MỚI.
    Để tính DSCR dự phóng, dùng calculate_pro_forma_dscr().

    Args:
        net_operating_income: Thu nhập hoạt động thuần (EBITDA hoặc EBIT + Khấu hao)
        annual_debt_service: Tổng trả nợ trong năm = Gốc + Lãi phải trả (nợ HIỆN TẠI)

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


# ── Pro-forma DSCR ───────────────────────────────────────────────────────────

@tool
def calculate_pro_forma_dscr(
    cfo: float,
    existing_debt_service: float,
    new_loan_amount: float,
    term_months: int,
    annual_interest_rate: float,
) -> dict:
    """
    Tính Pro-forma DSCR — DSCR DỰ PHÓNG sau khi gánh thêm khoản vay mới.

    ĐÂY LÀ CHỈ SỐ QUAN TRỌNG NHẤT CHO QUYẾT ĐỊNH CHO VAY:
    DSCR lịch sử có thể đủ (ví dụ 1.3), nhưng sau khi gánh khoản vay mới,
    tổng nghĩa vụ nợ tăng → Pro-forma DSCR có thể sập xuống < 1.0 (nguy cơ vỡ nợ)!

    Công thức niên kim (PMT - Payment):
        r_monthly = annual_interest_rate / 100 / 12
        PMT = P × r × (1+r)^n / ((1+r)^n - 1)
        Pro-forma DSCR = CFO / (existing_debt_service + PMT × 12)

    Args:
        cfo: Dòng tiền thuần từ hoạt động kinh doanh (Cash Flow from Operations)
             Lấy từ Báo cáo Lưu chuyển Tiền tệ — KHÔNG dùng EBIT hay lợi nhuận kế toán.
        existing_debt_service: Tổng nghĩa vụ trả nợ hiện tại mỗi năm (gốc + lãi nợ cũ)
        new_loan_amount: Số tiền khoản vay mới đề nghị (VND hoặc đơn vị bất kỳ nhất quán)
        term_months: Thời hạn vay (số tháng, vd: 60 = 5 năm)
        annual_interest_rate: Lãi suất hàng năm (%, vd: 9.5 = 9.5%/năm)

    Returns:
        dict với pro_forma_dscr, new_annual_pmt, total_debt_service, verdict, interpretation
    """
    if cfo <= 0:
        return {
            "error": "CFO phải > 0 để đủ điều kiện vay",
            "pro_forma_dscr": None,
            "verdict": "REJECT",
            "interpretation": f"Dòng tiền hoạt động kinh doanh âm (CFO={cfo:,.0f}). Doanh nghiệp không tạo ra tiền từ hoạt động cốt lõi.",
        }

    if term_months <= 0:
        return {"error": "term_months phải > 0", "pro_forma_dscr": None}

    if annual_interest_rate <= 0:
        return {"error": "annual_interest_rate phải > 0", "pro_forma_dscr": None}

    if new_loan_amount <= 0:
        return {"error": "new_loan_amount phải > 0", "pro_forma_dscr": None}

    # Tính niên kim tháng (PMT) theo công thức tài chính chuẩn
    r = annual_interest_rate / 100 / 12          # Lãi suất tháng
    n = term_months
    if r == 0:
        monthly_pmt = new_loan_amount / n
    else:
        monthly_pmt = new_loan_amount * r * (1 + r) ** n / ((1 + r) ** n - 1)

    new_annual_pmt = monthly_pmt * 12            # Quy đổi sang trả nợ năm
    total_debt_service = existing_debt_service + new_annual_pmt
    pro_forma_dscr = cfo / total_debt_service

    # Phân loại theo ngưỡng nghiệp vụ
    if pro_forma_dscr >= 1.25:
        verdict = "ADEQUATE"
        interpretation = (
            f"Pro-forma DSCR={pro_forma_dscr:.2f} đạt chuẩn. "
            f"Sau khi gánh khoản vay mới {new_loan_amount:,.0f}, doanh nghiệp vẫn còn đủ dòng tiền trả nợ an toàn."
        )
    elif pro_forma_dscr >= 1.0:
        verdict = "MARGINAL"
        interpretation = (
            f"Pro-forma DSCR={pro_forma_dscr:.2f} — biên mỏng. "
            f"Khoản trả nợ mới {new_annual_pmt:,.0f}/năm đẩy dòng tiền gần đến giới hạn. "
            "Cần yêu cầu bổ sung tài sản bảo đảm hoặc giảm hạn mức vay."
        )
    else:
        verdict = "INSUFFICIENT"
        interpretation = (
            f"NGUY HIỂM: Pro-forma DSCR={pro_forma_dscr:.2f} < 1.0. "
            f"Với khoản trả nợ mới {new_annual_pmt:,.0f}/năm, tổng nghĩa vụ nợ ({total_debt_service:,.0f}) "
            f"VƯỢT QUÁ dòng tiền thuần ({cfo:,.0f}). Nguy cơ vỡ nợ ngay sau khi giải ngân!"
        )

    logger.info(
        "Pro-forma DSCR computed: %.2f (existing_ds=%.0f, new_pmt_annual=%.0f, cfo=%.0f)",
        pro_forma_dscr, existing_debt_service, new_annual_pmt, cfo,
    )

    return {
        "pro_forma_dscr": round(pro_forma_dscr, 4),
        "new_annual_pmt": round(new_annual_pmt, 2),
        "total_debt_service": round(total_debt_service, 2),
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


# ── LTV (Loan-to-Value) ──────────────────────────────────────────────────────

@tool
def calculate_ltv(
    loan_amount: float,
    collateral_value: float,
    collateral_type: str,
) -> dict:
    """
    Tính LTV (Loan-to-Value Ratio) — Tỷ lệ Vay trên Giá trị Tài sản Bảo đảm.
    Áp dụng ngưỡng theo chuẩn Basel II / Thông tư NHNN.

    Args:
        loan_amount: Số tiền vay đề nghị (VND)
        collateral_value: Giá trị định giá tài sản bảo đảm (VND)
                          — do thẩm định viên độc lập cung cấp, KHÔNG tự khai
        collateral_type: Loại TSBĐ:
                         "real_estate"  → Bất động sản (nhà, đất)
                         "machinery"    → Máy móc, thiết bị, nhà xưởng
                         "securities"   → Cổ phiếu, trái phiếu niêm yết
                         "other"        → Tài sản khác (phương tiện, hàng hóa,...)

    Returns:
        dict với ltv_ratio, ltv_pct, max_allowed_ltv, status, interpretation
    """
    # Map ngưỡng an toàn theo loại TSBĐ
    LTV_CEILINGS = {
        "real_estate": 0.70,    # Thông tư 39/2016/TT-NHNN: BĐS max 70%
        "machinery":   0.50,    # Máy móc khấu hao nhanh: max 50%
        "securities":  0.60,    # Cổ phiếu niêm yết thanh khoản cao: max 60%
        "other":       0.40,    # Tài sản khác thanh khoản thấp: max 40%
    }

    if collateral_value <= 0:
        return {
            "error": "collateral_value phải > 0",
            "ltv_ratio": None,
            "status": "ERROR",
        }

    if loan_amount <= 0:
        return {
            "error": "loan_amount phải > 0",
            "ltv_ratio": None,
            "status": "ERROR",
        }

    collateral_type_normalized = collateral_type.lower().strip()
    max_ltv = LTV_CEILINGS.get(collateral_type_normalized, LTV_CEILINGS["other"])
    ltv_ratio = loan_amount / collateral_value
    ltv_pct = round(ltv_ratio * 100, 2)
    max_ltv_pct = round(max_ltv * 100, 1)

    if ltv_ratio <= max_ltv:
        status = "SAFE"
        interpretation = (
            f"LTV={ltv_pct:.1f}% nằm trong ngưỡng an toàn ({max_ltv_pct}% với TSBĐ loại '{collateral_type}'). "
            f"Tài sản bảo đảm đủ bảo vệ ngân hàng trong trường hợp phát mãi."
        )
    elif ltv_ratio <= max_ltv * 1.15:   # Quá ngưỡng nhưng trong 15% buffer
        status = "MARGINAL"
        interpretation = (
            f"LTV={ltv_pct:.1f}% vượt nhẹ ngưỡng {max_ltv_pct}% (với TSBĐ '{collateral_type}'). "
            f"Cần yêu cầu bổ sung TSBĐ hoặc giảm hạn mức vay xuống tối đa "
            f"{collateral_value * max_ltv:,.0f}."
        )
    else:
        status = "BREACH"
        interpretation = (
            f"LTV={ltv_pct:.1f}% VƯỢT NGƯỠNG AN TOÀN {max_ltv_pct}% với TSBĐ loại '{collateral_type}'. "
            f"Hạn mức vay tối đa được phép: {collateral_value * max_ltv:,.0f}. "
            "Không đủ điều kiện cho vay theo quy định hiện hành!"
        )

    logger.info(
        "LTV computed: %.1f%% (max=%.0f%%) status=%s type=%s",
        ltv_pct, max_ltv_pct, status, collateral_type,
    )

    return {
        "ltv_ratio": round(ltv_ratio, 4),
        "ltv_pct": ltv_pct,
        "max_allowed_ltv_pct": max_ltv_pct,
        "max_safe_loan_amount": round(collateral_value * max_ltv, 2),
        "status": status,
        "interpretation": interpretation,
    }


# ── External Risk News Search ────────────────────────────────────────────────

@tool
def search_external_risk_news(
    company_name: str,
    tax_code: str = "",
) -> dict:
    """
    Tìm kiếm thông tin rủi ro pháp lý và uy tín doanh nghiệp từ nguồn bên ngoài.
    Đây là thành phần 5C: Character (Đặc tính / Uy tín người vay).

    Các từ khóa được quét:
      - Kiện tụng, tranh chấp thương mại
      - Nợ thuế, trốn thuế, vi phạm thuế
      - Đình chỉ hoạt động, thu hồi giấy phép
      - Vỡ nợ, trái phiếu vỡ hạn, CIC nhóm xấu
      - Sai phạm của ban lãnh đạo, bị khởi tố

    Args:
        company_name: Tên đầy đủ của doanh nghiệp cần tra cứu
        tax_code: Mã số thuế (nếu có — giúp tăng độ chính xác)

    Returns:
        dict với found_risks (list), risk_keywords_hit (list), source, raw_results (list)
    """
    try:
        from src.config import get_settings
        settings = get_settings()
        tavily_key = settings.tavily_api_key
    except Exception:
        tavily_key = ""

    RISK_KEYWORDS = [
        "kiện tụng", "khởi tố", "tranh chấp",
        "nợ thuế", "trốn thuế", "vi phạm",
        "đình chỉ", "thu hồi giấy phép", "phá sản",
        "vỡ nợ", "trái phiếu vỡ hạn", "nợ xấu",
        "CIC nhóm 3", "CIC nhóm 4", "CIC nhóm 5",
        "bị bắt", "hầu tòa", "bị điều tra",
    ]

    if not tavily_key:
        # Fallback: không có Tavily API → trả về disclaimer rõ ràng
        logger.warning(
            "Tavily API key not configured. External news search unavailable for '%s'.",
            company_name,
        )
        return {
            "source": "NO_API_KEY",
            "company_name": company_name,
            "tax_code": tax_code,
            "found_risks": [],
            "risk_keywords_hit": [],
            "raw_results": [],
            "disclaimer": (
                "CẢNH BÁO: Không có Tavily API key. Không thể tra cứu thông tin bên ngoài. "
                "Chuyên viên tín dụng PHẢI tra cứu thủ công: CIC, thuế, tòa án, báo chí."
            ),
        }

    # Sử dụng Tavily Search API
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=tavily_key)

        query = f'"{company_name}"'
        if tax_code:
            query += f' OR "{tax_code}"'
        query += " (kiện tụng OR nợ thuế OR phá sản OR vi phạm OR đình chỉ OR khởi tố)"

        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
        )

        results = response.get("results", [])
        raw_snippets = [r.get("content", "") for r in results]

        # Quét keywords rủi ro trong kết quả
        all_text = " ".join(raw_snippets).lower()
        hit_keywords = [kw for kw in RISK_KEYWORDS if kw.lower() in all_text]
        found_risks = [
            f"[Web] {kw.upper()} phát hiện trong kết quả tìm kiếm cho '{company_name}'"
            for kw in hit_keywords
        ]

        logger.info(
            "External search complete for '%s': %d results, %d risk keywords hit",
            company_name, len(results), len(hit_keywords),
        )

        return {
            "source": "TAVILY_API",
            "company_name": company_name,
            "tax_code": tax_code,
            "found_risks": found_risks,
            "risk_keywords_hit": hit_keywords,
            "raw_results": raw_snippets[:3],   # Chỉ trả 3 snippet đầu để tránh overflow context
        }

    except ImportError:
        return {
            "source": "TAVILY_NOT_INSTALLED",
            "company_name": company_name,
            "found_risks": [],
            "risk_keywords_hit": [],
            "raw_results": [],
            "disclaimer": "Chưa cài tavily-python. Chạy: pip install tavily-python",
        }
    except Exception as e:
        logger.error("Tavily search failed for '%s': %s", company_name, e)
        return {
            "source": "TAVILY_ERROR",
            "company_name": company_name,
            "found_risks": [],
            "risk_keywords_hit": [],
            "raw_results": [],
            "error": str(e),
        }


# ── Registry: danh sách tool đăng ký với Agent ──────────────────────────────

FINANCIAL_TOOLS = [
    calculate_altman_z_score,
    calculate_dscr,
    calculate_pro_forma_dscr,
    calculate_debt_to_equity,
    calculate_quick_ratio,
    calculate_ltv,
    search_external_risk_news,
]
"""
Danh sách tools được bind vào Math Agent (llm.bind_tools(FINANCIAL_TOOLS)).
Scale-up: Thêm tool mới vào list này là đủ, không cần sửa Agent.

Tools hiện có (7):
  1. calculate_altman_z_score  — Dự báo phá sản (5C: Capacity)
  2. calculate_dscr             — Khả năng trả nợ lịch sử (5C: Capacity)
  3. calculate_pro_forma_dscr   — DSCR dự phóng sau khoản vay mới (5C: Capacity) ← QUAN TRỌNG NHẤT
  4. calculate_debt_to_equity   — Đòn bẩy tài chính (5C: Capital)
  5. calculate_quick_ratio      — Thanh khoản tức thì (5C: Capital)
  6. calculate_ltv              — Tỷ lệ vay/TSBĐ (5C: Collateral)
  7. search_external_risk_news  — Tra cứu uy tín/rủi ro pháp lý (5C: Character)
"""
