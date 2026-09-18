"""
main.py — Entry point để chạy thử pipeline FinRisk AI (Phase 1 — CLI mode).

Đây là script chạy tay để test pipeline mà KHÔNG cần API hay Docker.
Phase 4: Entry point sẽ là uvicorn fastapi_app:app (xem src/api/main.py).

Chạy: python main.py
"""

import logging
import uuid
from dotenv import load_dotenv

# Load .env trước mọi import khác
load_dotenv()

from src.agents.graph import get_graph

# Cấu hình logging JSON-friendly (Phase 4 sẽ dùng pythonjsonlogger)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("finriskai.main")


def run_demo():
    """
    Chạy demo pipeline với dữ liệu tài chính mẫu của một công ty có rủi ro cao.
    Mục đích: Kiểm tra toàn bộ luồng, bao gồm HITL interrupt.
    """
    graph = get_graph()
    session_id = str(uuid.uuid4())

    # ── Dữ liệu đầu vào: Công ty rủi ro cao ──────────────────────────────
    # Số liệu mô phỏng từ BCTC (Phase 2 sẽ tự động trích xuất từ PDF)
    initial_state = {
        "session_id": session_id,
        "user_id": "analyst_demo",
        "question": "Đánh giá toàn diện rủi ro tín dụng và khả năng phê duyệt khoản vay 50 tỷ VND.",
        "company_name": "Công ty TNHH Xây Dựng ABC",
        "document_paths": [],
        "next_agent": "",
        "iteration": 0,
        "retrieved_context": [],
        "raw_financials": {
            # Các số liệu từ BCTC (đơn vị: VND)
            "working_capital": -2_000_000_000,      # Vốn lưu động thuần ÂM
            "total_assets": 80_000_000_000,
            "retained_earnings": -500_000_000,       # Lỗ lũy kế
            "ebit": 1_200_000_000,
            "market_cap": 15_000_000_000,
            "total_liabilities": 70_000_000_000,    # Nợ rất cao
            "revenue": 40_000_000_000,
            # DSCR inputs
            "net_operating_income": 1_500_000_000,
            "annual_debt_service": 2_000_000_000,   # DSCR = 0.75 < 1.0 → Nguy hiểm!
            # D/E inputs
            "total_debt": 70_000_000_000,
            "total_equity": 10_000_000_000,          # D/E = 7.0 → Cực kỳ cao
            # Quick Ratio inputs
            "cash_and_equivalents": 500_000_000,
            "short_term_investments": 0,
            "accounts_receivable": 3_000_000_000,
            "current_liabilities": 8_000_000_000,   # Quick Ratio = 0.44 → Rất yếu
        },
        "financial_metrics": {},
        "risk_assessment": {},
        "requires_human_approval": False,
        "hitl_reason": "",
        "human_decision": None,
        "human_comment": None,
        "final_report": "",
        "messages": [],
    }

    config = {"configurable": {"thread_id": session_id}}

    print("\n" + "="*60)
    print("🏦 FINRISK AI — PHASE 1 DEMO")
    print("="*60)
    print(f"📋 Công ty: {initial_state['company_name']}")
    print(f"❓ Yêu cầu: {initial_state['question']}")
    print("="*60 + "\n")

    # Chạy pipeline (có thể bị interrupt bởi HITL)
    try:
        for event in graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_output in event.items():
                if node_name == "__interrupt__":
                    # HITL triggered!
                    interrupt_data = node_output[0].value
                    print("\n" + "🛑 "*15)
                    print("HUMAN-IN-THE-LOOP REQUIRED")
                    print(f"Lý do: {interrupt_data.get('reason')}")
                    print(f"Risk Score: {interrupt_data.get('risk_score')}")
                    print(f"Risk Level: {interrupt_data.get('risk_level')}")
                    print(f"Red Flags: {interrupt_data.get('red_flags')}")
                    print("🛑 "*15 + "\n")

                    # Mô phỏng Giám đốc Tín dụng nhập quyết định
                    decision = input("Nhập quyết định (APPROVED/REJECTED): ").strip().upper()
                    comment = input("Ghi chú (Enter để bỏ qua): ").strip()

                    # Resume pipeline với quyết định của Human
                    human_response = {"decision": decision, "comment": comment}
                    for event2 in graph.stream(
                        {"human_decision": decision, "human_comment": comment},
                        config=config,
                        stream_mode="updates",
                    ):
                        for n2, o2 in event2.items():
                            print(f"  ✅ [{n2}] resumed")
                else:
                    print(f"  ▶ [{node_name}] completed")

        # Lấy state cuối cùng
        final_state = graph.get_state(config).values
        print("\n" + "="*60)
        print("📄 BÁO CÁO THẨM ĐỊNH CUỐI CÙNG")
        print("="*60)
        print(final_state.get("final_report", "Không có báo cáo."))
        print("\n")
        print(f"Risk Score: {final_state.get('risk_assessment', {}).get('risk_score')}/100")
        print(f"Recommendation: {final_state.get('risk_assessment', {}).get('recommendation')}")

    except Exception as e:
        logger.error("Pipeline error: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    run_demo()
