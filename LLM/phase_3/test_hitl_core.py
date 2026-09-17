import sys
import uuid

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import HumanMessage, AIMessage
from src.capstone.agent_graph import research_graph

def print_separator(title: str):
    print("\n" + "=" * 60)
    print(f"🧪 {title}")
    print("=" * 60)

def test_hitl_approve_flow():
    """
    Test Case 1: Kiểm tra luồng Phê duyệt (Approve)
    Kỳ vọng: Đồ thị dừng lại trước web_search_worker -> Người dùng duyệt -> Đồ thị chạy tiếp đến END.
    """
    print_separator("TEST CASE 1: LUỒNG PHÊ DUYỆT (APPROVE)")
    
    thread_id = f"test_thread_approve_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    
    inputs = {
        "messages": [HumanMessage(content="Tìm kiếm trên web về xu hướng công nghệ AI Agent mới nhất 2026?")],
        "user_id": "test_user_hitl",
        "agents_used": [],
        "sources": []
    }
    
    print(f"👉 Bước 1: Gửi câu hỏi vào đồ thị (Thread ID: {thread_id})...")
    research_graph.invoke(inputs, config=config)
    
    print("\n👉 Bước 2: Kiểm tra trạng thái đồ thị sau lần chạy 1...")
    state = research_graph.get_state(config)
    print(f"   - Node tiếp theo dự kiến (state.next): {state.next}")
    
    # 🛑 ASSERTION QUAN TRỌNG: Đồ thị PHẢI dừng lại trước web_search_worker
    assert state.next == ("web_search_worker",), (
        f"❌ LỖI: Đồ thị không dừng lại trước web_search_worker! state.next hiện tại là: {state.next}"
    )
    print("   ✅ BẮT ĐƯỢC ĐIỂM DỪNG (HITL) CHUẨN XÁC!")
    print("      Đồ thị đã bị đóng băng an toàn, chưa hề chạy hàm web_search.")
    
    print("\n👉 Bước 3: Giả lập người dùng bấm nút [ĐỒNG Ý] (Approve)...")
    print("      Tiếp tục gọi research_graph.invoke(None, config=config)...")
    research_graph.invoke(None, config=config)
    
    print("\n👉 Bước 4: Kiểm tra trạng thái đồ thị sau khi Approve...")
    final_state = research_graph.get_state(config)
    print(f"   - Trạng thái hoàn thành (state.next): {final_state.next}")
    assert final_state.next == (), "❌ LỖI: Đồ thị phải kết thúc (state.next rỗng)!"
    
    final_res = final_state.values.get("final_result")
    assert final_res is not None, "❌ LỖI: final_result không được rỗng!"
    print(f"   - Độ tự tin: {final_res.get('confidence')}")
    print(f"   - Các Agent đã tham gia: {final_res.get('agents_used')}")
    print("\n🎉 TEST CASE 1: ĐẠT CHUẨN XUẤT SẮC ✅")

def test_hitl_reject_flow():
    """
    Test Case 2: Kiểm tra luồng Từ chối (Reject)
    Kỳ vọng: Đồ thị dừng lại trước web_search_worker -> Người dùng từ chối -> Đồ thị không gọi tool mà tổng hợp luôn.
    """
    print_separator("TEST CASE 2: LUỒNG TỪ CHỐI (REJECT)")
    
    thread_id = f"test_thread_reject_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    
    inputs = {
        "messages": [HumanMessage(content="Tìm kiếm tin tức giá cổ phiếu mới nhất hôm nay trên web?")],
        "user_id": "test_user_hitl",
        "agents_used": [],
        "sources": []
    }
    
    print(f"👉 Bước 1: Gửi câu hỏi vào đồ thị (Thread ID: {thread_id})...")
    research_graph.invoke(inputs, config=config)
    
    print("\n👉 Bước 2: Kiểm tra điểm dừng HITL...")
    state = research_graph.get_state(config)
    assert state.next == ("web_search_worker",), f"Kỳ vọng dừng trước web_search_worker nhưng nhận {state.next}"
    print(f"   ✅ Đồ thị đã tạm dừng tại: {state.next}")
    
    print("\n👉 Bước 3: Giả lập người dùng bấm nút [TỪ CHỐI] (Reject)...")
    print("      Inject thông báo từ chối vào State bằng research_graph.update_state()...")
    agents = list(state.values.get("agents_used", []))
    agents.append("Web Search (Bị từ chối bởi User)")
    
    research_graph.update_state(
        config,
        {
            "messages": [AIMessage(content="Người dùng đã từ chối cấp quyền gọi Web Search. Hãy tổng hợp câu trả lời dựa trên những gì đã có.")],
            "agents_used": agents
        }
    )
    
    print("👉 Bước 4: Chạy tiếp đồ thị sau khi từ chối...")
    research_graph.invoke(None, config=config)
    
    final_state = research_graph.get_state(config)
    assert final_state.next == (), "❌ LỖI: Đồ thị phải kết thúc!"
    
    final_res = final_state.values.get("final_result")
    assert final_res is not None
    print(f"   - Các Agent đã tham gia: {final_res.get('agents_used')}")
    print(f"   - Câu trả lời khi bị từ chối:\n{final_res.get('answer')[:180]}...")
    
    print("\n🎉 TEST CASE 2: ĐẠT CHUẨN XUẤT SẮC ✅")

if __name__ == "__main__":
    print("\n🚀 BẮT ĐẦU CHẠY KIỂM THỬ HUMAN-IN-THE-LOOP (HITL) TRÊN LANGGRAPH 🚀")
    test_hitl_approve_flow()
    test_hitl_reject_flow()
    print("\n" + "=" * 60)
    print("🌟 TẤT CẢ TEST CASES HITL ĐỀU VƯỢT QUA 100%! 🌟")
    print("=" * 60)
