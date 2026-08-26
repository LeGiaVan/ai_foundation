# from fastapi import FastAPI
# from groq import Groq
# import os
# from dotenv import load_dotenv
# import time


# load_dotenv()

# API_KEY = os.getenv("GROQ_API_KEY")
# client = Groq(api_key=API_KEY)
# app = FastAPI(title="Capstone API")


# @app.get("/health")
# def read_root():
#     return {"status": "ok", "message": "Server đã khởi động!"}

# @app.post("/chat")
# def chat(message: str):
#     response = client.chat.completions.create(
#         model="groq/compound-mini",
#         messages=[
#             {"role": "user", "content": message}
#         ],
#         temperature=0.7,
#         max_tokens=100,
#         # stream=True,
#     )
#     return {"reply": response.choices[0].message.content}

# def chat(message: str):
#     streaming = client.chat.completions.create(
#         model="groq/compound-mini",
#         messages=[
#             {"role": "user", "content": message}
#         ],
#         temperature=0.7,
#         max_tokens=100,
#         stream=True,
#     )

#     import time
#     for chunk in streaming:
#         # Mỗi chunk chứa một mảnh nhỏ của câu trả lời
#         content = chunk.choices[0].delta.content
#         if content is not None:
#             yield content  # Gửi từng mảnh nhỏ ra ngoài Client NGAY LẬP TỨC
#             time.sleep(0.05)  # 🐌 CỐ TÌNH LÀM CHẬM LẠI ĐỂ MẮT NGƯỜI NHÌN THẤY HIỆU ỨNG GÕ CHỮ

# from fastapi.responses import StreamingResponse
# @app.post("/chat/stream")
# async def chat_stream(message: str):
#     return StreamingResponse(
#         chat(message),
#         media_type="text/plain"
#     )

# curl -N -X POST "http://localhost:8000/chat/stream?message="

# /extract	POST	Nhận đoạn văn bản tự do, trả về dữ liệu có cấu trúc (Pydantic model tự chọn chủ đề, vd: trích thông tin liên hệ, trích ý chính...)

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
from groq import Groq
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os
# 1. Load Environment Variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 2. Setup FastAPI app
app = FastAPI(title="Capstone Extraction API")

# 3. Define Pydantic Model (The "Structure")
# Ta sẽ làm ví dụ: Trích xuất CV xin việc thành JSON
class ContactInfo(BaseModel):
    """Thông tin liên hệ trích ra được"""
    name: str = Field(..., description="Họ tên đầy đủ")
    phone: Optional[str] = Field(None, description="Số điện thoại dạng +84...")
    email: Optional[str] = Field(None, description="Email")
    linkedin: Optional[str] = Field(None, description="URL LinkedIn")

class CVParseResult(BaseModel):
    """Kết quả trích xuất toàn bộ CV"""
    objective: Optional[str] = Field(None, description="Mục tiêu nghề nghiệp (Objective/Summary)")
    contact: ContactInfo
    skills: list[str] = Field(default_factory=list, description="Danh sách kỹ năng (Skills)")

# 4. Create the Endpoint
class ExtractRequest(BaseModel):
    text_input: str

@app.post("/extract")
def extract_data(request: ExtractRequest):
    """
    Nhận text thô, yêu cầu LLM bóc tách thông tin vào JSON
    """
    text_input = request.text_input
    if not GROQ_API_KEY:
        return {"error": "Missing GROQ_API_KEY in .env"}

    client = Groq(api_key=GROQ_API_KEY)

    try:
        # Tạo Prompt yêu cầu LLM trả về JSON
        system_prompt = f"""
        Bạn là một công cụ bóc tách thông tin từ CV. Hãy trích xuất thông tin dưới đây 
        vào định dạng JSON chính xác theo schema được cung cấp.
        Yêu cầu:
        - Nếu không tìm thấy thông tin nào, trả về null cho trường đó.
        - Phone phải theo format +84xxxx...
        - Email phải hợp lệ.
        - Skills là một danh sách các từ khóa.
        
        Dữ liệu input:
        {text_input}
        """

        response = client.chat.completions.create(
            model="groq/compound-mini",
            messages=[
                # System prompt yêu cầu format JSON
                {
                    "role": "system",
                    "content": f"Trích xuất thông tin vào JSON theo schema sau: {CVParseResult.schema_json()}"
                },
                # User prompt chứa dữ liệu cần xử lý
                {
                    "role": "user",
                    "content": f"Hãy phân tích đoạn văn sau: \n\n{text_input}"
                }
            ],
            temperature=0.1, # Set thấp để giảm tính sáng tạo, tăng tính chính xác
            max_tokens=500
        )
        
        # Trích xuất kết quả text từ LLM
        raw_output = response.choices[0].message.content
        
        # Xử lý chuỗi JSON thô để tránh lỗi (Clean up)
        # LLM đôi khi bao JSON trong ```json ... ```
        clean_json = raw_output.strip()
        if clean_json.startswith("```json"):
            clean_json = clean_json[7:-3].strip()
        elif clean_json.startswith("```"):
            clean_json = clean_json[3:-3].strip()
            
        # Parse JSON string thành Pydantic Object
        result = CVParseResult.parse_raw(clean_json)
        
        # Trả về dạng dict để API dễ xử lý
        return result.dict()

    except Exception as e:
        return {"error": str(e)}

# 5. Thêm Test Input example vào phần docstring cho dễ test
# Ví dụ: Input một đoạn CV mẫu
