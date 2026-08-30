# 📖 Tổng hợp kiến thức Ngày 4: Gọi LLM Cơ Bản (Bản thay thế: GROQ API - Miễn phí)

Do tài khoản Anthropic đã hết tiền, chúng ta sẽ linh hoạt chuyển sang dùng **Groq** (nhà cung cấp các model Llama hoàn toàn miễn phí và siêu tốc độ).
Kiến thức lý thuyết về System Prompt, User Prompt, Token hoàn toàn GIỐNG NHAU. Chỉ khác ở một chút "cú pháp" (Syntax) của thư viện.

---

## 1. Cấu trúc cốt lõi của Groq API (Chuẩn OpenAI)

Groq sử dụng cấu trúc API giống y hệt OpenAI. Để nhờ AI trả lời, bạn gửi lên:

- **`model`**: Tên phiên bản AI (Ví dụ: `llama3-8b-8192` hoặc `llama3-70b-8192`).
- **`max_tokens`**: Giới hạn độ dài câu trả lời.
- **`messages`**: Trái ngược với Anthropic (tách riêng system ra ngoài), trong Groq/OpenAI, **System Prompt** được nhét chung vào danh sách `messages` luôn, nhưng đánh dấu `"role": "system"`.

> **Sự khác biệt cốt lõi:**
> - `System Prompt` (role: system) là **Bối cảnh** (Sống cố định suốt cuộc hội thoại).
> - `User Prompt` (role: user) là **Câu lệnh cụ thể** (Thay đổi liên tục theo mỗi lần người dùng gõ).

---

## 2. Cài đặt thư viện cần thiết

Mở terminal mới (hoặc tắt uvicorn bằng Ctrl+C) và chạy lệnh cài đặt thư viện của Groq:
```bash
pip install groq pydantic-settings python-dotenv
```

---

## 3. Viết thử 1 đoạn script gọi API thuần (Chưa cần FastAPI)

Sửa lại file `.env` của bạn, thay API key bằng API key của Groq (lấy miễn phí tại console.groq.com):
```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
```

Code Python gọi Groq (Sửa lại nội dung file `test.py`):
```python
from groq import Groq
import os
from dotenv import load_dotenv

# Tải biến từ file .env
load_dotenv()

# Tự động lấy API key từ biến môi trường GROQ_API_KEY
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama3-8b-8192", # Dùng model Llama3 cực nhẹ và nhanh
    max_tokens=500,
    messages=[
        # System prompt nằm ngay trong mảng messages
        {"role": "system", "content": "Bạn là một tên cướp biển hung dữ. Cuối mỗi câu phải có chữ 'Arrr'."},
        # User prompt
        {"role": "user", "content": "Thủ đô của Việt Nam là gì?"}
    ]
)

# Cú pháp trích xuất kết quả trả về của Groq hơi dài hơn Anthropic một chút
print(response.choices[0].message.content)
```

---

## 4. Tích hợp Groq vào FastAPI (Kết hợp kiến thức Ngày 1, 2, 3)

Bây giờ ráp nối lại với **Request Body (Pydantic) + Dependency Injection (Depends) + Groq API**.

```python
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from groq import Groq

# 1. Đọc Settings từ file .env 
class Settings(BaseSettings):
    groq_api_key: str  # Pydantic sẽ tự tìm GROQ_API_KEY trong .env
    class Config:
        env_file = ".env"

def get_settings():
    return Settings()

# 2. Định nghĩa Request Body
class ChatRequest(BaseModel):
    message: str

app = FastAPI()

# 3. Tạo Endpoint
@app.post("/chat")
def chat_with_llama(request: ChatRequest, settings: Settings = Depends(get_settings)):
    # Bước 1: Khởi tạo Client
    client = Groq(api_key=settings.groq_api_key)
    
    # Bước 2: Gọi Groq API
    response = client.chat.completions.create(
        model="groq/compound-mini",
        max_tokens=500,
        messages=[
            {"role": "system", "content": "Bạn là một trợ lý thông minh, lịch sự và trả lời ngắn gọn."},
            {"role": "user", "content": request.message}
        ]
    )
    
    # Bước 3: Trích xuất nội dung chữ và trả về
    return {"reply": response.choices[0].message.content}
```
