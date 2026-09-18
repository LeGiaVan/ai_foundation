import os
import sys
import json
import re
from typing import Annotated, TypedDict, Optional, Literal, List
from dotenv import load_dotenv

# Hỗ trợ import linh hoạt dù chạy từ thư mục nào
current_dir = os.path.dirname(os.path.abspath(__file__))
phase_3_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if phase_3_dir not in sys.path:
    sys.path.insert(0, phase_3_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

try:
    from src.capstone.schemas import ResearchResult
    from src.capstone.tools import rag_search, web_search, calculate, format_table, get_current_datetime
    from src.capstone.memory_store import MemoryStore
except ImportError:
    from schemas import ResearchResult
    from tools import rag_search, web_search, calculate, format_table, get_current_datetime
    from memory_store import MemoryStore

load_dotenv()

# Khởi tạo LLM chính và LLM sinh Structured Output
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_retries=2, timeout=60)
structured_llm = llm.with_structured_output(ResearchResult)

# Khởi tạo MemoryStore
memory_store = MemoryStore()

# === 1. ĐỊNH NGHĨA GRAPH STATE ===
class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    user_memories: list[str]
    agents_used: list[str]
    sources: list[str]
    pending_query: Optional[str]
    final_result: Optional[dict]

# === 2. ĐỊNH NGHĨA CÁC NODES ===

def supervisor_node(state: ResearchState):
    """
    Supervisor phân tích yêu cầu, nạp Long-Term Memory, và quyết định worker tiếp theo.
    """
    user_id = state.get("user_id", "user_default")
    user_memories = memory_store.retrieve(user_id)
    
    memory_context = "\n".join(f"- {m}" for m in user_memories) if user_memories else "Chưa có thông tin ghi nhớ."
    
    system_prompt = (
        "Bạn là Supervisor của hệ thống Multi-Agent Research Assistant.\n"
        f"Thông tin cá nhân & sở thích của người dùng ({user_id}):\n{memory_context}\n\n"
        "Nhiệm vụ của bạn là xem xét tiến trình nghiên cứu và ra quyết định bước tiếp theo:\n"
        "- Nếu câu hỏi cần thông tin kỹ thuật nội bộ (YOLOv11, Fintech...) mà chưa tìm: gọi 'RAG'.\n"
        "- Nếu cần thông tin cập nhật Internet, tin tức mới hoặc RAG không có: gọi 'WEB'.\n"
        "- Nếu cần tính toán số liệu toán học: gọi 'CALC'.\n"
        "- Nếu đã thu thập đủ thông tin để trả lời hoàn chỉnh: gọi 'FINISH'.\n\n"
        "Hãy suy nghĩ ngắn gọn và trả về cú pháp chính xác ở dòng cuối cùng:\n"
        "DECISION: [RAG|WEB|CALC|FINISH]"
    )
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    
    # Trích xuất quyết định
    content = response.content
    decision = "FINISH"
    match = re.search(r"DECISION:\s*(RAG|WEB|CALC|FINISH)", content, re.IGNORECASE)
    # Bỏ hết các từ không liên quan: 
    # - DECISION: là từ khóa
    # - \s* là khoảng trắng
    # - (RAG|WEB|CALC|FINISH) là các lựa chọn (chỉ 1 trong số đó)
    # - re.IGNORECASE là không phân biệt chữ hoa chữ thường
    if match:
        decision = match.group(1).upper()
        
    return {
        "messages": [response],
        "user_memories": user_memories
    }

def rag_worker_node(state: ResearchState):
    """Worker chuyên tra cứu tài liệu nội bộ (Qdrant)."""
    last_human_msg = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    
    print(f">> [RAG Worker 📚] Đang tra cứu tài liệu cho: '{last_human_msg}'")
    rag_output = rag_search.invoke({"query": last_human_msg})
    
    agents = list(state.get("agents_used", []))
    if "RAG Agent" not in agents:
        agents.append("RAG Agent")
        
    sources = list(state.get("sources", []))
    sources.append("Tài liệu nội bộ Qdrant Knowledge Base")
    
    return {
        "messages": [AIMessage(content=f"📚 [Kết quả tra cứu RAG]:\n{rag_output}")],
        "agents_used": agents,
        "sources": sources
    }

def web_search_worker_node(state: ResearchState):
    """
    Worker tìm kiếm Internet (Được bảo vệ bởi HITL interrupt).
    """
    last_human_msg = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    print(f">> [Web Search Worker 🌐] Đang tìm kiếm Internet: '{last_human_msg}'")
    
    web_output = web_search.invoke({"query": last_human_msg})
    
    agents = list(state.get("agents_used", []))
    if "Web Search Agent" not in agents:
        agents.append("Web Search Agent")
        
    sources = list(state.get("sources", []))
    sources.append("DuckDuckGo Web Search")
    
    return {
        "messages": [AIMessage(content=f"🌐 [Kết quả tìm kiếm Web]:\n{web_output}")],
        "agents_used": agents,
        "sources": sources
    }

def calculator_worker_node(state: ResearchState):
    """Worker tính toán toán học an toàn."""
    last_human_msg = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    print(f">> [Calculator Worker 🧮] Đang phân tích biểu thức tính toán...")
    
    # Cho LLM trích xuất biểu thức toán học
    extract_prompt = f"Trích xuất biểu thức toán học cần tính từ câu: '{last_human_msg}'. Chỉ trả về biểu thức (ví dụ: '1.5 * 2' hoặc '100 * 4')."
    expr_res = llm.invoke(extract_prompt)
    clean_expr = expr_res.content.strip().replace("`", "").replace("'", "").replace('"', "")
    calc_output = calculate.invoke({"expression": clean_expr})
    
    agents = list(state.get("agents_used", []))
    if "Calculator Agent" not in agents:
        agents.append("Calculator Agent")
        
    return {
        "messages": [AIMessage(content=f"🧮 [Kết quả phép tính ({clean_expr})]: {calc_output}")],
        "agents_used": agents
    }

from langchain_core.output_parsers import PydanticOutputParser
result_parser = PydanticOutputParser(pydantic_object=ResearchResult)

def synthesizer_node(state: ResearchState):
    """
    Tổng hợp toàn bộ thông tin từ các workers, định dạng theo Structured Output
    và tuân thủ sở thích trong Long-Term Memory.
    """
    user_id = state.get("user_id", "user_default")
    user_memories = state.get("user_memories", [])
    memory_notes = "\n".join(f"- {m}" for m in user_memories) if user_memories else "Không có"
    
    collected_texts = []
    for m in state["messages"]:
        if isinstance(m, AIMessage) and any(tag in m.content for tag in ["[Kết quả", "[RAG", "[Web", "[Calculator"]):
            collected_texts.append(m.content)
            
    data_str = "\n\n".join(collected_texts) if collected_texts else "Dữ liệu trực tiếp từ ngữ cảnh."
    last_human_msg = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Bạn là chuyên gia tổng hợp nghiên cứu AI. Dưới đây là các dữ liệu do các Worker thu thập được.\n\n"
         f"RÀNG BUỘC PHONG CÁCH CỦA NGƯỜI DÙNG ({user_id}):\n{memory_notes}\n"
         "BẮT BUỘC tuân theo phong cách và sở thích trên (ví dụ: ngắn gọn, lập bảng, hoặc chi tiết)!\n\n"
         "Bạn BẮT BUỘC phải trả về một JSON object hợp lệ theo schema sau:\n{format_instructions}"),
        ("human", "Dữ liệu thu thập được:\n{data}\n\nHãy trả lời câu hỏi: '{question}'")
    ])
    
    try:
        chain = prompt | llm | result_parser
        result: ResearchResult = chain.invoke({
            "data": data_str, 
            "question": last_human_msg,
            "format_instructions": result_parser.get_format_instructions()
        })
    except Exception as e:
        print(f"[Synthesizer fallback]: {e}")
        raw_res = (prompt | llm).invoke({
            "data": data_str, 
            "question": last_human_msg,
            "format_instructions": result_parser.get_format_instructions()
        })
        text = raw_res.content
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            clean_json = match.group()
            data = json.loads(clean_json)
            result = ResearchResult(**data)
        else:
            result = ResearchResult(
                answer=text,
                sources=state.get("sources", []),
                agents_used=state.get("agents_used", ["Supervisor"]),
                confidence=0.9
            )
    
    # Đảm bảo danh sách sources và agents_used đầy đủ
    existing_sources = list(set(state.get("sources", []) + result.sources))
    existing_agents = list(set(state.get("agents_used", []) + result.agents_used))
    if not existing_agents:
        existing_agents = ["Supervisor Agent"]
        
    result.sources = existing_sources
    result.agents_used = existing_agents
    
    # Trích xuất và cập nhật Long-Term Memory từ câu hỏi của user
    if last_human_msg:
        memory_store.extract_and_save(user_id, last_human_msg)
        
    return {
        "final_result": result.model_dump()
    }

# === 3. ĐIỀU HƯỚNG CONDITIONAL EDGES ===

def route_supervisor(state: ResearchState) -> Literal["rag_worker", "web_search_worker", "calculator_worker", "synthesizer"]:
    last_msg = state["messages"][-1]
    content = getattr(last_msg, "content", "")
    
    agents = state.get("agents_used", [])
    
    # Kiểm tra nếu đã chạy RAG rồi mà chưa đủ thì cho Web Search
    if "DECISION: RAG" in content.upper() and "RAG Agent" not in agents:
        return "rag_worker"
    elif "DECISION: WEB" in content.upper() and "Web Search Agent" not in agents:
        return "web_search_worker"
    elif "DECISION: CALC" in content.upper() and "Calculator Agent" not in agents:
        return "calculator_worker"
    elif "DECISION: FINISH" in content.upper() or len(agents) >= 2:
        return "synthesizer"
    else:
        # Mặc định kết thúc nếu không còn tool nào cần gọi
        return "synthesizer"

# === 4. XÂY DỰNG STATEGRAPH & HITL INTERRUPT ===

workflow = StateGraph(ResearchState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("rag_worker", rag_worker_node)
workflow.add_node("web_search_worker", web_search_worker_node)
workflow.add_node("calculator_worker", calculator_worker_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.add_edge(START, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "rag_worker": "rag_worker",
        "web_search_worker": "web_search_worker",
        "calculator_worker": "calculator_worker",
        "synthesizer": "synthesizer"
    }
)

# Các worker sau khi chạy xong quay lại supervisor để đánh giá tiếp
workflow.add_edge("rag_worker", "supervisor")
workflow.add_edge("web_search_worker", "supervisor")
workflow.add_edge("calculator_worker", "supervisor")
workflow.add_edge("synthesizer", END)

# BỘ NHỚ CHECKPOINTER CHO HITL
checkpointer = MemorySaver()

# ⛔ HUMAN-IN-THE-LOOP: Tạm dừng trước khi Web Search Worker thực thi (đắt tiền / Internet)
research_graph = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["web_search_worker"]
)

# === XUẤT SƠ ĐỒ MERMAID ===
if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        
    print("\n" + "=" * 55)
    print("📊 MÃ MERMAID CỦA CAPSTONE RESEARCH AGENT GRAPH:")
    print("=" * 55)
    
    # 1. Lấy mã Mermaid dạng chuỗi
    mermaid_code = research_graph.get_graph().draw_mermaid()
    print(mermaid_code)
    
    # 2. Tự động lưu ra file graph_diagram.md
    diagram_path = os.path.join(current_dir, "graph_diagram.md")
    with open(diagram_path, "w", encoding="utf-8") as f:
        f.write(f"# Sơ đồ Kiến trúc LangGraph Capstone\n\n```mermaid\n{mermaid_code}\n```\n")
        
    print(f"\n✅ Đã lưu sơ đồ vào file: {diagram_path}")

