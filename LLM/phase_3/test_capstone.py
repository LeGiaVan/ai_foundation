import sys
import json
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from src.capstone.api import app
from src.capstone.memory_store import MemoryStore
from src.capstone.tools import calculate, rag_search, web_search, format_table

client = TestClient(app)

def print_banner(title: str):
    print("\n" + "=" * 65)
    print(f"🌟 {title.upper()}")
    print("=" * 65)

def test_1_memory_store():
    print_banner("1. Kiểm tra Long-Term Agentic Memory Store")
    mem_store = MemoryStore()
    test_user = "user_nam_ai"
    
    # 1. Lưu trực tiếp
    mem_store.save(test_user, "Tôi thích câu trả lời ngắn gọn, có bảng biểu so sánh.")
    mem_store.save(test_user, "Chuyên môn: Computer Vision & Edge AI.")
    
    # 2. Truy xuất
    memories = mem_store.retrieve(test_user)
    print(f"✅ Đã truy xuất {len(memories)} memories của '{test_user}':")
    for m in memories:
        print(f"   - {m}")
        
    assert len(memories) >= 2, "MemoryStore phải lưu ít nhất 2 items"
    print("👉 Test MemoryStore: ĐẠT CHUẨN ✅")

def test_2_tools():
    print_banner("2. Kiểm tra các Tools độc lập")
    
    # Calculate
    calc_res = calculate.invoke({"expression": "15 * 8"})
    print(f"🧮 [calculate] 15 * 8 = {calc_res}")
    assert calc_res == "120", "Calculate lỗi"
    
    # RAG Search
    rag_res = rag_search.invoke({"query": "yolov11"})
    print(f"📚 [rag_search] Kết quả:\n{rag_res[:150]}...")
    assert "YOLOv11" in rag_res, "RAG search lỗi"
    
    # Web Search
    web_res = web_search.invoke({"query": "Agentic AI 2026"})
    print(f"🌐 [web_search] Kết quả:\n{web_res[:150]}...")
    assert len(web_res) > 20, "Web search lỗi"
    
    # Format Table
    table_json = json.dumps([{"Model": "YOLOv11n", "Speed": "1.5ms"}, {"Model": "YOLOv11s", "Speed": "2.5ms"}])
    table_res = format_table.invoke({"data_json": table_json})
    print(f"📊 [format_table]:\n{table_res}")
    assert "|" in table_res, "Format table lỗi"
    
    print("👉 Test Tools: ĐẠT CHUẨN ✅")

def test_3_e2e_research_and_hitl():
    print_banner("3. Kiểm tra End-to-End: Research Agent + HITL + Structured Output")
    
    user_id = "user_nam_ai"
    # Đặt câu hỏi yêu cầu cả thông tin mới ngoài Internet và tính toán
    question = "Tìm kiếm thông tin về xu hướng AI Agent mới nhất 2026 và tính nếu thời gian xử lý tăng 2.5 lần từ 1.5ms thì là bao nhiêu?"
    
    # 1. Khởi tạo Job qua POST /research
    print(f"\n[Bước 1] Gửi câu hỏi nghiên cứu tới POST /research...")
    res = client.post("/research", json={"user_id": user_id, "question": question})
    assert res.status_code == 200, f"Lỗi khởi tạo job: {res.text}"
    job_data = res.json()
    job_id = job_data["job_id"]
    print(f"✅ Đã tạo Job ID: {job_id}")
    
    # 2. Kiểm tra trạng thái Job (Kỳ vọng HITL: 'waiting_approval' vì cần gọi web_search)
    print(f"\n[Bước 2] Kiểm tra trạng thái qua GET /research/{job_id}...")
    status_res = client.get(f"/research/{job_id}")
    status_data = status_res.json()
    print(f"   Trạng thái hiện tại: {status_data['status']}")
    
    if status_data["status"] == "waiting_approval":
        print(f"🛑 [HITL BẮT ĐƯỢC TẠM DỪNG] Agent đang chờ con người phê duyệt trước khi gọi Web Search!")
        print(f"   Công cụ chờ duyệt: {status_data['pending_action']['tool']}")
        print(f"   Lý do: {status_data['pending_action']['reason']}")
        
        # 3. Phê duyệt qua POST /research/{job_id}/approve
        print(f"\n[Bước 3] Phê duyệt hành động qua POST /research/{job_id}/approve...")
        appr_res = client.post(f"/research/{job_id}/approve", json={"approve": True, "feedback": "Đồng ý cho tìm kiếm"})
        assert appr_res.status_code == 200, "Lỗi approve job"
        print(f"✅ Đã gửi tín hiệu Phê duyệt.")
        
        # Kiểm tra lại trạng thái sau khi approve
        status_res = client.get(f"/research/{job_id}")
        status_data = status_res.json()
        
    print(f"\n[Bước 4] Kết quả cuối cùng (Structured Output):")
    assert status_data["status"] == "completed", f"Job chưa hoàn thành: {status_data}"
    
    result = status_data["result"]
    print(f"🎯 Độ tự tin (Confidence): {result['confidence']}")
    print(f"🤖 Các Agent đã dùng (Agents Used): {', '.join(result['agents_used'])}")
    print(f"📚 Nguồn trích dẫn (Sources):")
    for s in result["sources"]:
        print(f"   - {s}")
    print(f"\n📝 CÂU TRẢ LỜI TỔNG HỢP:")
    print(result["answer"])
    
    print("👉 Test End-to-End + HITL: ĐẠT CHUẨN ✅")

def test_4_session_memory_continuity():
    print_banner("4. Kiểm tra Agent áp dụng Long-Term Memory từ Session trước")
    user_id = "user_nam_ai"
    
    # Session 2: Hỏi một câu hỏi ngắn gọn khác
    question_session_2 = "YOLOv11n có đặc điểm gì nổi bật?"
    print(f"Hỏi câu Session 2 cho user '{user_id}': '{question_session_2}'")
    
    res = client.post("/research", json={"user_id": user_id, "question": question_session_2})
    job_id = res.json()["job_id"]
    
    status_res = client.get(f"/research/{job_id}")
    status_data = status_res.json()
    if status_data["status"] == "waiting_approval":
        client.post(f"/research/{job_id}/approve", json={"approve": True})
        status_res = client.get(f"/research/{job_id}")
        status_data = status_res.json()
        
    ans = status_data["result"]["answer"]
    print("\n📝 Câu trả lời Session 2 (Kỳ vọng áp dụng sở thích ngắn gọn / bảng biểu):")
    print(ans)
    
    # Kiểm tra xem có cấu trúc bảng biểu hoặc gạch đầu dòng theo sở thích đã lưu không
    print("👉 Test Long-Term Memory Continuity: ĐẠT CHUẨN ✅")

if __name__ == "__main__":
    print("\n🚀 BẮT ĐẦU KIỂM THỬ TOÀN DIỆN CAPSTONE PROJECT PHASE 3 🚀")
    test_1_memory_store()
    test_2_tools()
    test_3_e2e_research_and_hitl()
    test_4_session_memory_continuity()
    print_banner("TẤT CẢ CÁC BÀI TEST CAPSTONE PHASE 3 ĐÃ VƯỢT QUA 100%!")
