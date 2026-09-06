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
    print(f"[Tool] Đang tính toán: {expression}")
    return str(eval(expression))

@tool
def transfer_money(amount: float, recipient: str) -> str:
    """Chuyển tiền cho người khác."""
    print(f"[Tool 💸] ĐANG THỰC HIỆN GIAO DỊCH CHUYỂN {amount:,.0f} CHO {recipient.upper()}...")
    return f"Giao dịch chuyển {amount:,.0f} cho {recipient} đã thành công tốt đẹp."

tools = [calculate, transfer_money]
llm_with_tools = llm.bind_tools(tools)

# === 2. Khái niệm cốt lõi của LangGraph ===
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

trimmer = trim_messages(
    max_tokens=6, 
    strategy="last",
    token_counter=len,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

def call_llm(state: AgentState):
    print(">> [Node: agent] LLM đang suy nghĩ...")
    trimmed_messages = trimmer.invoke(state["messages"])
    response = llm_with_tools.invoke(trimmed_messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        print(f">> [Edge: Conditional] Bắt được lệnh gọi Tool.")
        return "tools"
    print(">> [Edge: Conditional] Xong! Trả kết quả.")
    return END

graph = StateGraph(AgentState)
graph.add_node("agent", call_llm)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

memory = MemorySaver()
# THÊM INTERRUPT TẠI ĐÂY
app = graph.compile(checkpointer=memory, interrupt_before=["tools"])

# === 3. Thực thi Vòng Lặp Chat (HITL) ===

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 CHATBOT BẢO MẬT (Human-in-the-Loop) - Dùng /quit để thoát")
    print("="*50)
    
    config = {"configurable": {"thread_id": "secure_session_02"}}
    
    system_msg = SystemMessage(content="Bạn là trợ lý ngân hàng ảo. Bạn có thể làm toán và chuyển tiền. Hãy dùng tiếng Việt.")
    app.update_state(config, {"messages": [system_msg]})
    
    while True:
        try:
            # Kiểm tra xem graph có đang bị tạm dừng không
            current_state = app.get_state(config)
            
            # Nếu KHÔNG dừng ở tools, thì hỏi user câu mới
            if not current_state.next or current_state.next == ("agent",):
                user_input = input("\n👤 Khách hàng: ")
                if user_input.lower() in ["/quit", "quit", "exit"]:
                    print("Tạm biệt!")
                    break
                if not user_input.strip():
                    continue
                inputs = {"messages": [HumanMessage(content=user_input)]}
            elif current_state.next == ("tools",):
                # Nếu ĐANG bị dừng lại trước node tools
                inputs = None # Resume không cần input mới
                print("\n⛔ HỆ THỐNG ĐÃ TẠM DỪNG!")
                
                # Trích xuất thông tin tool_call để hỏi user
                last_msg = current_state.values["messages"][-1]
                tool_calls = last_msg.tool_calls
                
                for tc in tool_calls:
                    print(f"⚠️ AI muốn gọi hàm: {tc['name']}({tc['args']})")
                
                decision = ""
                while decision.lower() not in ['y', 'n']:
                    decision = input("❓ Bạn có cho phép thực hiện không? (y/n): ").strip()
                
                if decision.lower() == 'y':
                    print("✅ Lệnh đã được phê duyệt. Tiếp tục chạy...")
                else:
                    print("❌ Lệnh bị TỪ CHỐI. Hệ thống tự động hủy mà không cần gọi lại LLM (Tiết kiệm API)...")
                    
                    messages_to_inject = []
                    for tc in tool_calls:
                        messages_to_inject.append(ToolMessage(
                            tool_call_id=tc["id"],
                            name=tc["name"],
                            content="User rejected the tool call."
                        ))
                    
                    # Tiêm sẵn câu trả lời của AI vào bộ nhớ
                    apology = "Xin lỗi, tôi đã hủy yêu cầu thực hiện hành động này để đảm bảo an toàn."
                    messages_to_inject.append(AIMessage(content=apology))
                    
                    # Đẩy vào bộ nhớ, giả vờ như Node "agent" vừa chạy xong và nhả ra câu xin lỗi
                    app.update_state(config, {"messages": messages_to_inject}, as_node="agent")
                    
                    print(f"\n🤖 Agent: {apology}")
                    
                    # Gọi luồng tiếp tục (nó sẽ đi từ Agent -> END mà không tốn API)
                    inputs = None
            
            # In ra luồng xử lý
            for output in app.stream(inputs, config=config):
                for key, value in output.items():
                    if "messages" in value:
                        msg = value["messages"][-1]
                        if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
                            print(f"\n🤖 Agent: {msg.content}")
                            
        except EOFError:
            break
        except Exception as e:
            print(f"Lỗi: {e}")
            break
