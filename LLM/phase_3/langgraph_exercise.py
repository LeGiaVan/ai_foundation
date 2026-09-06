import os
import json
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, trim_messages
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# === Setup ===
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_retries=1, timeout=15)

# === 1. Định nghĩa Tools ===
@tool
def calculate(expression: str) -> str:
    """Tính toán một biểu thức toán học. Input là string, output là kết quả."""
    print(f"[Hệ thống] Đang tính toán: {expression}")
    return str(eval(expression))

@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Chuyển đổi tiền tệ (Tỷ giá giả lập)."""
    print(f"[Hệ thống] Đang chuyển {amount} {from_currency} sang {to_currency}...")
    rates = {"USD": 1, "VND": 25000, "EUR": 0.9}
    if from_currency not in rates or to_currency not in rates:
        raise ValueError(f"Không hỗ trợ loại tiền tệ này: {from_currency} hoặc {to_currency}")
    
    usd_amount = amount / rates[from_currency]
    final_amount = usd_amount * rates[to_currency]
    return f"{final_amount:,.2f} {to_currency}"

tools = [calculate, convert_currency]
llm_with_tools = llm.bind_tools(tools)

# === 2. Khái niệm cốt lõi của LangGraph ===

# Định nghĩa State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

"""
TypedDict: Kiểu dữ liệu Dict nhưng có ràng buộc tên Schema, ...
Annotated: Đánh dấu thêm thông tin cho biến đó
add_messages: Hàm cộng dồn các message lại với nhau, tránh ghi đè giữa các Node
"""

# Cấu hình Sliding Window: giữ tối đa 6 tin nhắn (không tính SystemMessage)
# Đảm bảo không cắt ngang cụm ToolCall (include_system=True)
trimmer = trim_messages(
    max_tokens=6,  # Ở đây ta dùng số lượng tin nhắn thay vì token để dễ kiểm chứng
    strategy="last",
    token_counter=len, # đếm theo số lượng item trong list
    include_system=True,
    allow_partial=False,
    start_on="human"
)

# Định nghĩa Nodes
def call_llm(state: AgentState):
    print(">> [Node: agent] LLM đang suy nghĩ...")
    
    # Trim tin nhắn trước khi đưa vào LLM
    trimmed_messages = trimmer.invoke(state["messages"])
    # In ra để xem trimmer hoạt động thế nào
    print(f"   [Context Window] Gửi {len(trimmed_messages)} tin nhắn tới LLM (trên tổng số {len(state['messages'])}).")
    
    response = llm_with_tools.invoke(trimmed_messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        print(f">> [Edge: Conditional] Phải dùng tools! Gọi {len(last_message.tool_calls)} tool(s).")
        return "tools"
    print(">> [Edge: Conditional] Xong! Trả kết quả.")
    return END

# Xây Graph
graph = StateGraph(AgentState)
graph.add_node("agent", call_llm)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

# Thêm checkpointer để có bộ nhớ
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# === 3. Thực thi Vòng Lặp Chat ===

if __name__ == "__main__":
    # Lưu biểu đồ Mermaid
    try:
        mermaid_code = app.get_graph().draw_mermaid()
        with open("langgraph_agent.md", "w", encoding="utf-8") as f:
            f.write(f"```mermaid\n{mermaid_code}\n```")
        print("✅ Đã cập nhật sơ đồ Graph vào file: langgraph_agent.md")
    except Exception as e:
        print(f"⚠️ Không thể vẽ biểu đồ Mermaid: {e}")

    print("\n" + "="*50)
    print("🚀 CHATBOT KHỞI ĐỘNG (Dùng /quit để thoát)")
    print("="*50)
    
    # Cấu hình Thread ID (Session ID)
    config = {"configurable": {"thread_id": "user_123"}}
    
    # Gửi tin nhắn hệ thống đầu tiên vào bộ nhớ
    system_msg = SystemMessage(content="Bạn là trợ lý ảo thân thiện. Hãy trả lời ngắn gọn và dùng tiếng Việt.")
    app.update_state(config, {"messages": [system_msg]})
    
    while True:
        try:
            user_input = input("\n👤 Bạn: ")
            if user_input.lower() in ["/quit", "quit", "exit"]:
                print("Tạm biệt!")
                break
                
            if not user_input.strip():
                continue
                
            inputs = {"messages": [HumanMessage(content=user_input)]}
            
            # In ra stream
            for output in app.stream(inputs, config=config):
                for key, value in output.items():
                    if "messages" in value:
                        last_msg = value["messages"][-1]
                        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                            pass # Đã có dòng log ở node
                        elif isinstance(last_msg, ToolMessage):
                            print(f"   ✅ Tool trả về: {last_msg.content}")
                        elif isinstance(last_msg, AIMessage) and last_msg.content:
                            print(f"\n🤖 Agent: {last_msg.content}")
        except EOFError:
            break
        except Exception as e:
            print(f"Lỗi: {e}")
