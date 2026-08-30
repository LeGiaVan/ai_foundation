from asyncio import streams
from pydantic import BaseModel
import json
from groq import Groq
import os
from schemas import DocumentSummary
from typing import Type
from text_utils import chunk_text
from prompts import MAP_PROMPT_TEMPLATE, REDUCE_PROMPT_TEMPLATE, QA_SYSTEM_PROMPT
import asyncio
from dotenv import load_dotenv
import time
from abc import ABC, abstractmethod
# Tải các biến môi trường từ file .env (ví dụ: GROQ_API_KEY)
load_dotenv()

# Tạo lớp trừu tượng (Abstract Base Class)
class BaseLLMClient(ABC):
    async def generate_structured(self, prompt: str, schema: Type[BaseModel]) -> BaseModel:
        pass
    
    @abstractmethod
    async def generate_text(self, prompt: str) -> str:
        pass
    
    @abstractmethod
    async def stream_chat(self, system_prompt: str, user_prompt: str):
        pass

class GroqClient(BaseLLMClient):
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    async def generate_structured(self, prompt: str, schema: Type[BaseModel]) -> BaseModel:
        # Sử dụng chat.completions.create chuẩn của Groq/OpenAI
        # Ép model trả về chuẩn JSON thông qua tham số response_format
        response = self.client.chat.completions.create(
            model="groq/compound-mini",  # Model của Groq (không phải Claude)
            messages=[
                {
                    "role": "system", 
                    "content": f"You are a helpful assistant that outputs JSON. The JSON schema is: {json.dumps(schema.model_json_schema())}"
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0 # Bắt buộc = 0 để trả về chính xác cấu trúc
        )
        
        # Lấy kết quả chuỗi JSON
        json_string = response.choices[0].message.content
        
        # Chuyển đổi chuỗi JSON thô thành Object Python (DocumentSummary) 
        # và trả về (return) cho các hàm khác sử dụng.
        return schema.model_validate_json(json_string)
        
    async def generate_text(self, prompt: str) -> str:
        # Hàm sinh văn bản thô thông thường (chuyên dùng cho bước Map)
        response = self.client.chat.completions.create(
            model="groq/compound-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2 # Tùy chỉnh nhẹ độ sáng tạo
        )
        return response.choices[0].message.content
    
    async def stream_chat(self, system_prompt: str, user_prompt: str):
        with self.client.chat.completions.create(
        model="groq/compound-mini",  # Model của Groq (không phải Claude)
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            stream = True
        ) as stream:
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token is not None:
                    print(token, end="", flush=True) 
                    time.sleep(0.05)  # 🐌 CỐ TÌNH LÀM CHẬM LẠI ĐỂ MẮT NGƯỜI NHÌN THẤY HIỆU ỨNG GÕ CHỮ


class DocumentProcessor:
    def __init__(self, llm_client: GroqClient):
        self.llm_client = llm_client

    async def summarize_long_document(self, text: str) -> DocumentSummary:
        """
        Logic gọi chunk_text, thực hiện Map-Reduce và cuối cùng gọi generate_structured.
        """
        # Truyền cứng "gpt-3.5-turbo" vì thư viện tiktoken chỉ hỗ trợ model của OpenAI
        chunks = chunk_text(text, "gpt-3.5-turbo")
        
        # BƯỚC MAP: Chỉ cần tóm tắt thô (không ép JSON) để tối ưu hóa tài nguyên
        map_prompts = [MAP_PROMPT_TEMPLATE.format(text=chunk) for chunk in chunks]
        map_tasks = [self.llm_client.generate_text(prompt) for prompt in map_prompts]
        map_results = await asyncio.gather(*map_tasks) # Kết quả là List các string
        
        # BƯỚC REDUCE: Gom mảng string đó lại và ép ra Object DocumentSummary chuẩn chỉ 1 lần
        reduce_prompt = REDUCE_PROMPT_TEMPLATE.format(text="\n\n---\n\n".join(map_results))
        return await self.llm_client.generate_structured(reduce_prompt, DocumentSummary)

    async def answer_question_stream(self, text: str, question: str):
        """
        Logic xây dựng ngữ cảnh RAG, gọi stream_chat để lấy luồng dữ liệu trả về.
        """
        system_prompt = QA_SYSTEM_PROMPT.format(context_text=text)
        await self.llm_client.stream_chat(
            system_prompt=system_prompt,
            user_prompt=question
        )