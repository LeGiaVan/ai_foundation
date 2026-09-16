'''
BÀI HỌC: TOOL CALLING & AGENTS VỚI LANGCHAIN
1. Chuyển hàm sang dùng @tool decorator (tự động sinh JSON Schema).
2. Dùng .bind_tools() và LCEL để xây dựng chain cơ bản: prompt | model.bind_tools().
3. Tích hợp RAG Qdrant vào một tool riêng biệt (rag_search).
4. Khởi tạo Agent tự động điều phối và gọi tools nhiều bước (ReAct Agent).
'''

import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import warnings
warnings.filterwarnings("ignore")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.agents import create_agent

# Đảm bảo in tiếng Việt mượt mà trên console Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# === Khởi tạo Qdrant cho Tool rag_search (với Fallback an toàn) ===
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "rag_docs"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

retriever = None
try:
    qdrant_client = QdrantClient(url=QDRANT_URL, timeout=1.0)
    vector_store = QdrantVectorStore(
        client=qdrant_client, 
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
        vector_name="dense",
        content_payload_key="text",      
        metadata_payload_key="metadata"  
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    print("✅ Đã kết nối thành công tới Qdrant Vector Store.")
except Exception as e:
    print("⚠️ Lưu ý: Qdrant local (cổng 6333) chưa bật. Tool rag_search sẽ dùng dữ liệu giả lập dự phòng.")

# === 1️⃣ Định nghĩa các tools bằng @tool ===

@tool
def calculate(expression: str) -> str:
    """Tính toán biểu thức toán học (ví dụ: '2 * 3.5 + 10'). Input là chuỗi biểu thức."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Lỗi tính toán: {str(e)}"

@tool
def get_weather(city: str) -> str:
    """Lấy thông tin thời tiết hiện tại tại một thành phố (Hà Nội, TP.HCM, Đà Nẵng,...)."""
    print(f"\n[Hệ thống] Đang lấy thời tiết tại {city}...")
    city_lower = city.lower()
    if "hanoi" in city_lower or "hà nội" in city_lower:
        return "Thời tiết tại Hà Nội: Nắng, nhiệt độ 32°C, độ ẩm 70%"
    elif "ho chi minh" in city_lower or "hcm" in city_lower:
        return "Thời tiết tại TP.HCM: Mưa rào, nhiệt độ 28°C, độ ẩm 85%"
    elif "da nang" in city_lower or "đà nẵng" in city_lower:
        return "Thời tiết tại Đà Nẵng: Có mây, nhiệt độ 30°C, độ ẩm 75%"
    return f"Không tìm thấy dữ liệu thời tiết cho {city}"

@tool
def rag_search(question: str) -> str:
    """Tìm kiếm thông tin trong cơ sở dữ liệu nội bộ (tài liệu kỹ thuật, YOLOv11n, Groq LLM, dự án công ty)."""
    print(f"\n[Hệ thống] Đang truy vấn RAG Database cho: '{question}'...")
    if retriever is None:
        return "Tài liệu kỹ thuật nội bộ: Quá trình Tích hợp AI (kết hợp YOLOv11n phát hiện vật thể và Groq LLM xử lý ngôn ngữ) mất trung bình 3.5 giây cho các tác vụ phức tạp."
    try:
        docs = retriever.invoke(question)
        if not docs:
            return "Không tìm thấy thông tin trong tài liệu."
        parts = []
        for d in docs:
            title = d.metadata.get("title") or d.metadata.get("source") or "unknown"
            parts.append(f"Source: {title}\n{d.page_content}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Lỗi truy vấn Qdrant ({e}). Dữ liệu dự phòng: Thời gian xử lý Tích hợp AI là 3.5 giây."

tools = [calculate, get_weather, rag_search]
print("✅ Tools đã sẵn sàng. Kiểm tra Schema tự sinh của calculate:")
print(f"- Tên tool: {calculate.name}")
print(f"- Mô tả: {calculate.description}")
print(f"- Tham số: {calculate.args}")

# === 2️⃣ Dùng .bind_tools() và LCEL (Clean Code) ===
print("\n" + "="*50)
print("BÀI TẬP 2: Xây chain với .bind_tools() và ChatPromptTemplate")
model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Bạn là trợ lý AI thông minh. Hãy trả lời ngắn gọn, chính xác bằng tiếng Việt."),
    ("human", "{question}")
])

chain = prompt | model.bind_tools(tools)
ai_msg = chain.invoke({
    "question": "Thời tiết Hà Nội hôm nay thế nào? và tính chính xác 984712 * 48213 bằng bao nhiêu?"
})

print(type(ai_msg))
for call in ai_msg.tool_calls:
    print(f"-> Tool: {call['name']} | Tham số: {call['args']}")

# # === 3️⃣ Chạy tự động khép kín với Agent (ReAct) ===
# print("\n" + "="*50)
# print("BÀI TẬP 3 & 4: Chạy tự động đa bước với Agent")

# agent = create_agent(
#     model=model,
#     tools=tools,
#     system_prompt="Bạn là trợ lý thông minh. Hãy dùng các công cụ được cung cấp để trả lời câu hỏi của người dùng. Luôn trả lời bằng tiếng Việt."
# )

# query_both = "Nếu nhân đôi thời gian xử lý các tác vụ phức tạp của việc Tích hợp AI (đặc biệt là YOLOv11n và Groq LLM) thì mất bao lâu?"
# print(f"\nUser: {query_both}")

# inputs = {"messages": [{"role": "user", "content": query_both}]}
# final_result = agent.invoke(inputs)

# print(f"\n🤖 CÂU TRẢ LỜI CUỐI CÙNG:\n{final_result['messages'][-1].content}")
