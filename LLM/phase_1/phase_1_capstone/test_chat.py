import requests

# Ngữ cảnh (context) thường là kết quả "summary" từ Endpoint 1
ngu_canh = """
Công ty A báo cáo doanh thu Quý 1 đạt 100 tỷ đồng, lợi nhuận ròng 15 tỷ.
Chi phí vận hành chiếm phần lớn là 20 tỷ đồng.
Mục tiêu Quý 2 là tăng trưởng doanh thu 15% và mở rộng thêm 2 chi nhánh mới ở miền Nam.
"""

# Câu hỏi của User
cau_hoi = "Mục tiêu của công ty trong Quý 2 là gì?"

print("Đang gửi câu hỏi lên Server...")
print("-" * 50)

# Gửi Request lên API /chat
response = requests.post(
    "http://127.0.0.1:8000/chat",
    json={
        "context_text": ngu_canh,
        "question": cau_hoi
    },
    stream=True # Bật chế độ stream để nhận dữ liệu liên tục nếu có
)

# In ra kết quả (Lưu ý: hiện tại Endpoint này đang in ra Terminal của Server 
# và trả về file JSON chứa chữ "Success" cho Client)
if response.status_code == 200:
    print("\n[Client] Kết quả trả về từ Server:")
    print(response.json())
    print("\n(Hãy mở cửa sổ Terminal đang chạy uvicorn để xem AI đang gõ từng chữ ra màn hình nhé!)")
else:
    print("Lỗi hệ thống! Status:", response.status_code)
    print("Chi tiết:", response.text)
