'''
**Bài tập:**
1. Chuyển 2 tools từ Ngày 6 sang dùng `@tool` decorator → in ra `.schema` để xem JSON tự generate.
2. Dùng `.bind_tools()` và LCEL để xây chain: `prompt | llm_with_tools`.
3. Thêm tool `rag_search(question: str) -> str` — wrap Qdrant RAG retrieval từ Phase 2 thành 1 tool. Giờ LLM có thể tự quyết định khi nào cần search document.
4. Test: hỏi câu cần RAG → LLM gọi `rag_search`. Hỏi câu toán → gọi `calculate`. Hỏi câu cả hai → gọi cả hai.
'''

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.agents import create_agent

# === Setup API Key ===
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# === Khởi tạo Qdrant cho Tool rag_search ===
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "rag_docs"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
qdrant_client = QdrantClient(url=QDRANT_URL)
vector_store = QdrantVectorStore(
    client=qdrant_client, 
    collection_name=QDRANT_COLLECTION,
    embedding=embeddings,
    vector_name="dense",
    content_payload_key="text",      
    metadata_payload_key="metadata"  
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# === 1️⃣ Định nghĩa các tools ===

@tool
def calculate(expression: str) -> str:
    """Tính toán một biểu thức toán học. Input là string, output là string của kết quả."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

@tool
def get_weather(city: str) -> str:
    """Lấy thời tiết hiện tại ở một thành phố."""
    print(f"\n[Hệ thống] Đang lấy thời tiết tại {city}...")
    if "hanoi" in city.lower():
        return f"Thời tiết tại Hà Nội: Nắng, nhiệt độ 32°C, độ ẩm 70%"
    elif "ho chi minh" in city.lower() or "hcmc" in city.lower():
        return f"Thời tiết tại TP.HCM: Mưa rào, nhiệt độ 28°C, độ ẩm 85%"
    elif "da nang" in city.lower():
        return f"Thời tiết tại Đà Nẵng: Có mây, nhiệt độ 30°C, độ ẩm 75%"
    else:
        return f"Không tìm thấy dữ liệu thời tiết cho {city}"

@tool
def rag_search(question: str) -> str:
    """Tìm kiếm thông tin trong cơ sở dữ liệu nội bộ (tài liệu công ty, sản phẩm, v.v.). Dùng tool này nếu người dùng hỏi về kiến thức, thông tin."""
    print(f"\n[Hệ thống] Đang truy vấn RAG Database cho câu hỏi: '{question}'...")
    docs = retriever.invoke(question)
    if not docs:
        return "Không tìm thấy thông tin trong tài liệu."
    
    parts = []
    for d in docs:
        title = d.metadata.get("title") or d.metadata.get("source") or "unknown"
        parts.append(f"Source: {title}\n{d.page_content}")
    return "\n\n".join(parts)

print("✅ Tools đã được chuyển sang @tool và sẵn sàng sử dụng.")

# === 2️⃣ Dùng .bind_tools() và LCEL ===

# Khởi tạo LLM
model = ChatGroq(model="openai/gpt-oss-120b", temperature=0) # Model này hỗ trợ tool calling

# Binding tools vào LLM
llm_with_tools = model.bind_tools([calculate, get_weather, rag_search])
print("✅ Đã bind 3 tools vào LLM.")

# Tạo Prompt (LCEL style)
prompt_template = """
Bạn là một trợ lý thông minh.

Câu hỏi của người dùng: {question}

Hãy quyết định xem có cần dùng tools để trả lời không.
- Nếu cần tính toán, dùng tool 'calculate'.
- Nếu cần thông tin thời tiết, dùng tool 'get_weather'.
- Nếu cần thông tin về kiến thức, tài liệu, sản phẩm, dùng tool 'rag_search'.
- Nếu không cần, trả lời trực tiếp.

Trả lời ngắn gọn và chính xác bằng thông tin từ tool.
"""

prompt = PromptTemplate.from_template(prompt_template)

# Xây dựng chain LCEL (LangChain Expression Language)
chain = prompt | llm_with_tools

print("✅ Đã xây dựng xong chain với LCEL.")

# === 4️⃣ Xây dựng Agent để TỰ ĐỘNG chạy tool và lấy câu trả lời cuối ===

print("\n" + "="*50)
print("BÀI TẬP 4: Chạy tự động với Agent")
print("="*50)

# Khai báo danh sách tools
tools = [calculate, get_weather, rag_search]

# Khởi tạo Agent (LangChain >= 1.0)
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="Bạn là một trợ lý thông minh. Hãy dùng các công cụ được cung cấp để trả lời câu hỏi của người dùng. Luôn trả lời bằng tiếng Việt."
)

# Test câu hỏi cực khó cần cả RAG và Toán
query_both = "Nếu nhân đôi thời gian xử lý các tác vụ phức tạp của việc Tích hợp AI (đặc biệt là YOLOv11n và Groq LLM) thì mất bao lâu?"
print(f"\nUser: {query_both}")

# Chạy! agent sẽ tự làm hết
inputs = {"messages": [{"role": "user", "content": query_both}]}
final_result = agent.invoke(inputs)

print(f"\n🤖 CÂU TRẢ LỜI CUỐI CÙNG: {final_result['messages'][-1].content}")
