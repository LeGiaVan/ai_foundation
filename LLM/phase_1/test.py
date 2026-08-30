from fastapi import FastAPI, Depends
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from groq import Groq

# 1. Đọc Settings từ file .env 
class Settings(BaseSettings):
    groq_api_key: str  # Pydantic sẽ tự tìm GROQ_API_KEY trong .env
    class Config:
        env_file = ".env"
        extra = "ignore"  # Bỏ qua các biến bị thừa trong .env (vd: ANTHROPIC_API_KEY)

def get_settings():
    return Settings()

# 2. Định nghĩa Request Body
class ChatRequest(BaseModel):
    message: str

from fastapi.responses import StreamingResponse

app = FastAPI()

def generate_stream(user_message: str):
    """Generator: Nhận chunk từ Groq, lập tức yield ra cho Client"""
    settings = get_settings()
    client = Groq(api_key=settings.groq_api_key)
    
    # Groq dùng cú pháp OpenAI: chat.completions.create(stream=True)
    # KHÔNG PHẢI cú pháp Anthropic: messages.stream()
    stream = client.chat.completions.create(
        model="groq/compound-mini",  # Model của Groq (không phải Claude)
        max_tokens=1024,
        stream=True,  # Bật streaming
        messages=[
            {"role": "system", "content": "Bạn là trợ lý thông minh, trả lời bằng tiếng Việt."},
            {"role": "user", "content": user_message}
        ]
    )
    
    import time
    for chunk in stream:
        # Mỗi chunk chứa một mảnh nhỏ của câu trả lời
        content = chunk.choices[0].delta.content
        if content is not None:
            yield content  # Gửi từng mảnh nhỏ ra ngoài Client NGAY LẬP TỨC
            time.sleep(0.05)  # 🐌 CỐ TÌNH LÀM CHẬM LẠI ĐỂ MẮT NGƯỜI NHÌN THẤY HIỆU ỨNG GÕ CHỮ

@app.post("/chat/stream")
async def chat_stream(message: str):
    return StreamingResponse(
        generate_stream(message),
        media_type="text/plain"
    )

# # 3. Tạo Endpoint
# @app.post("/chat")
# def chat_with_llama(request: ChatRequest, settings: Settings = Depends(get_settings)):
#     # Bước 1: Khởi tạo Client
#     client = Groq(api_key=settings.groq_api_key)
    
#     # Bước 2: Gọi Groq API
#     response = client.chat.completions.create(
#         model="groq/compound-mini",
#         max_tokens=500,
#         messages=[
#             {"role": "system", "content": "Bạn là một trợ lý thông minh, lịch sự và trả lời ngắn gọn."},
#             {"role": "user", "content": request.message}
#         ]
#     )
    
#     # Bước 3: Trích xuất nội dung chữ và trả về
#     return {"reply": response.choices[0].message.content}