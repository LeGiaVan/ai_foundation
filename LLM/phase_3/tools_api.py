import os
import json
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage

# --- Imports cho Qdrant (RAG) ---
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

# Thiết lập API keys
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Khởi tạo model - Tăng timeout lên hẳn 300 giây (5 phút) để cứu code không bị crash
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_retries=1, timeout=300)

app = FastAPI(title="LangChain Tools API - Week 8")

# --- KHỞI TẠO RAG ---
try:
    qdrant_client = QdrantClient("http://localhost:6333")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = QdrantVectorStore(
        client=qdrant_client, 
        collection_name="rag_docs",
        embedding=embeddings,
        vector_name="dense",
        content_payload_key="text",
        metadata_payload_key="metadata"
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    print("✅ Đã kết nối thành công tới Qdrant localhost:6333")
except Exception as e:
    retriever = None
    print(f"⚠️ Lỗi kết nối Qdrant: {e}. Sẽ chạy ở chế độ fallback.")

# --- ĐỊNH NGHĨA 4 TOOLS ---

@tool
def calculate(expression: str) -> str:
    """Tính toán biểu thức toán học. Tham số truyền vào là một chuỗi biểu thức (ví dụ: '50 + 75')."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

@tool
def get_current_datetime() -> str:
    """Lấy ngày và giờ hệ thống hiện tại."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def format_table(data_json: str) -> str:
    """
    Format dữ liệu dạng JSON string chứa array of objects thành bảng Markdown.
    Ví dụ: '[{"Tên": "A", "Tuổi": 20}]'
    """
    try:
        data = json.loads(data_json)
        if not data or not isinstance(data, list):
            return "Error: Data must be a non-empty JSON array of objects."
        
        headers = list(data[0].keys())
        header_row = "| " + " | ".join(headers) + " |"
        sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        
        rows = []
        for item in data:
            row = "| " + " | ".join(str(item.get(h, "")) for h in headers) + " |"
            rows.append(row)
            
        return "\n".join([header_row, sep_row] + rows)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def rag_search(query: str) -> str:
    """Tìm kiếm thông tin, kiến thức từ cơ sở dữ liệu nội bộ."""
    if retriever:
        try:
            docs = retriever.invoke(query)
            if not docs:
                return "Không tìm thấy thông tin nào."
            return "\n\n".join(doc.page_content for doc in docs)
        except Exception as e:
            return f"Lỗi RAG: {e}"
    return "Lỗi RAG: Máy chủ Qdrant chưa được kết nối."

# Gộp tools
tools = [calculate, get_current_datetime, format_table, rag_search]

# Khởi tạo Agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "Bạn là một trợ lý thông minh và phân tích dữ liệu tốt. "
        "Bạn có các công cụ (tools) để tìm kiếm dữ liệu, tính toán, định dạng bảng và xem giờ. "
        "Hãy luôn trả lời bằng tiếng Việt và trình bày thật đẹp mắt."
    )
)

# --- FASTAPI ENDPOINTS ---

class AskRequest(BaseModel):
    question: str

class ToolStep(BaseModel):
    tool_name: str
    tool_input: dict
    tool_output: str

class AskResponse(BaseModel):
    final_answer: str
    chain_of_thought: list[ToolStep]

@app.post("/ask", response_model=AskResponse)
async def ask_agent(req: AskRequest):
    try:
        # Gọi Agent
        inputs = {"messages": [{"role": "user", "content": req.question}]}
        result = agent.invoke(inputs)
        
        # Parse Messages để lấy Chain of Thought
        messages = result["messages"]
        final_answer = messages[-1].content
        
        chain_of_thought = []
        tool_calls_map = {}
        
        for msg in messages:
            # Nếu là lệnh gọi tool từ LLM
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    tool_calls_map[tc["id"]] = {
                        "name": tc["name"],
                        "args": tc["args"]
                    }
                    
            # Nếu là kết quả trả về từ tool
            elif isinstance(msg, ToolMessage):
                tc_info = tool_calls_map.get(msg.tool_call_id, {})
                chain_of_thought.append(ToolStep(
                    tool_name=msg.name,
                    tool_input=tc_info.get("args", {}),
                    tool_output=msg.content
                ))
                
        return AskResponse(
            final_answer=final_answer,
            chain_of_thought=chain_of_thought
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

