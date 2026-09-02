import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

# === Setup ===
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Khởi tạo model hỗ trợ Tool Calling (Thêm timeout để không bị treo vĩnh viễn khi Server sập)
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_retries=1, timeout=15)

# === 1️⃣ Định nghĩa Tools ===

@tool
def calculate(expression: str) -> str:
    """Tính toán một biểu thức toán học. Input là string, output là kết quả."""
    print(f"[Hệ thống] Đang tính toán: {expression}")
    return str(eval(expression))

@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Chuyển đổi tiền tệ (Tỷ giá giả lập)."""
    print(f"[Hệ thống] Đang chuyển {amount} {from_currency} sang {to_currency}...")
    # Tỷ giá giả lập
    rates = {"USD": 1, "VND": 25000, "EUR": 0.9}
    if from_currency not in rates or to_currency not in rates:
        raise ValueError(f"Không hỗ trợ loại tiền tệ này: {from_currency} hoặc {to_currency}")
    
    # Đổi sang USD rồi đổi sang đích
    usd_amount = amount / rates[from_currency]
    final_amount = usd_amount * rates[to_currency]
    return f"{final_amount:,.2f} {to_currency}"

@tool
def get_weather(city: str) -> str:
    """Lấy thời tiết. Cố tình cài cắm lỗi để test Error Handling."""
    print(f"[Hệ thống] Đang gọi API thời tiết cho {city}...")
    try:
        if city.lower() == "error_city":
            # Cố tình quăng lỗi để xem LLM xử lý ra sao
            raise TimeoutError("Connection to Weather API timed out!")
        return "32°C, có nắng"
    except Exception as e:
        return f"Tool execution failed: {str(e)}"

# Gom tools và đưa cho LLM
tools = [calculate, convert_currency, get_weather]

# === 2️⃣ Hàm thực thi tự động với Agent ===
from langchain.agents import create_agent

# Khởi tạo Agent
agent = create_agent(
    model=llm, 
    tools=tools, 
    system_prompt="Bạn là trợ lý thông minh. Hãy trả lời bằng tiếng Việt."
)

print("="*50)
print("TEST 1: Tính toán và Đổi tiền tệ")
print("="*50)
inputs1 = {"messages": [{"role": "user", "content": "Tính 50 + 75, và chuyển 100 USD sang VND giúp tôi."}]}
res1 = agent.invoke(inputs1)
print(f"\n🤖 CÂU TRẢ LỜI CUỐI: {res1['messages'][-1].content}")

print("\n" + "="*50)
print("TEST 2: Lỗi Tool (Error Handling)")
print("="*50)
inputs2 = {"messages": [{"role": "user", "content": "Thời tiết ở hà nội hiện tại thế nào?"}]}
res2 = agent.invoke(inputs2)
print(f"\n🤖 CÂU TRẢ LỜI CUỐI: {res2['messages'][-1].content}")
