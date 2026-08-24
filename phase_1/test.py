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