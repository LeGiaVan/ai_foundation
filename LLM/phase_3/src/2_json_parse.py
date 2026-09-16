import os
import json
import sys
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import (
    PydanticOutputParser, 
    JsonOutputParser
)
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

# Đảm bảo in tiếng Việt mượt mà trên console Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# JSON mẫu đóng vai trò khuôn đúc (blueprint)
sample_structure = {
    "answer": "Nội dung câu trả lời ngắn gọn, súc tích",
    "confidence_score": 0.95,
    "key_takeaways": [
        "Ý chính thứ nhất",
        "Ý chính thứ hai"
    ],
    "sources": [
        "https://example.com"
    ]
}

print("="*50)
print("BÀI TẬP 1: JsonOutputParser với JSON Mẫu")
json_parser = JsonOutputParser()

json_prompt = PromptTemplate(
    template="""Bạn là trợ lý AI thông minh. Hãy trả lời câu hỏi sau của người dùng.

[CÂU HỎI]:
{query}

[ĐỊNH DẠNG ĐẦU RA]:
Đầu ra BẮT BUỘC là một JSON Object tuân theo cấu trúc mẫu sau (chỉ thay đổi giá trị, giữ nguyên các keys):
{sample_json}

{format_instructions}""",

    input_variables=["query"],
    # biến bắt buộc là query
    # biến partial_variables không bắt buộc phải có, nó sẽ được inject vào prompt sau input_variables và nó là cố định, không thể thay đổi khi invoke
    partial_variables={
        "format_instructions": json_parser.get_format_instructions(),
        # json_parser.get_format_instructions() là hàm sẵn của Langchain, tự động xuất ra Prompt, không cần phải setup thủ công.
        "sample_json": json.dumps(sample_structure, ensure_ascii=False, indent=2)
    },
)

json_chain = json_prompt | model | json_parser
res_json = json_chain.invoke({"query": "LangChain được viết bằng ngôn ngữ lập trình nào?"})
print(res_json)
print("Kết quả JSON:")
print(json.dumps(res_json, indent=2, ensure_ascii=False))
print("\nKiểu dữ liệu nhận được:", type(res_json))
print("- Câu trả lời:", res_json.get("answer"))
print("- Độ tự tin:", res_json.get("confidence_score"))
print("- Ý chính:", res_json.get("key_takeaways"))
print("- Nguồn:", res_json.get("sources"))

"""
Pipeline của code:
prompt -> llm -> json_parse

prompt:
res_json = json_chain.invoke({"query": "LangChain được viết bằng ngôn ngữ lập trình nào?"})

query được inject vào đầu chain, là prompt
prompt lúc này: 
"""