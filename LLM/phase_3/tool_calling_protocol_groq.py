import os
import json
import asyncio
from groq import AsyncGroq
from dotenv import load_dotenv

# 1. Định nghĩa tool thực tế (Mock) - Dùng async để không block server
async def get_weather(city: str) -> str:
    """Hàm giả lập lấy thời tiết."""
    print(f"\n[Hệ thống] Đang chạy hàm get_weather({city})...")
    await asyncio.sleep(1)  # Giả lập call API thời tiết bên ngoài mất 1s
    weather_data = {
        "hanoi": "32°C, có mây",
        "ho chi minh": "35°C, nắng gắt",
        "london": "15°C, mưa phùn"
    }
    return weather_data.get(city.lower(), "25°C, trời quang đãng")

async def get_time(timezone: str) -> str:
    """Hàm giả lập lấy thời gian."""
    print(f"\n[Hệ thống] Đang chạy hàm get_time({timezone})...")
    await asyncio.sleep(1)  # Giả lập delay
    time_data = {
        "new york": "10:00 AM",
        "hanoi": "21:00 PM",
        "london": "15:00 PM"
    }
    return time_data.get(timezone.lower(), "12:00 PM")

# Dictionary map tên tool với hàm Python tương ứng
AVAILABLE_TOOLS = {
    "get_weather": get_weather,
    "get_time": get_time
}

async def execute_tool(tool_name: str, tool_kwargs: dict):
    if tool_name in AVAILABLE_TOOLS:
        func = AVAILABLE_TOOLS[tool_name]
        return await func(**tool_kwargs)  # Hàm thực tế giờ là async
    return "Error: Tool not found"

# 2. Định nghĩa Schema để LLM hiểu
tool_schema = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Lấy thông tin thời tiết hiện tại của 1 thành phố",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Tên thành phố bằng tiếng Anh (VD: Hanoi, London)"
                }
            },
            "required": ["city"]
        }
    }
}

tool_schema_2 = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Lấy thời gian hiện tại của một thành phố hoặc múi giờ",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Tên thành phố bằng tiếng Anh (VD: New York, Hanoi)"
                }
            },
            "required": ["timezone"]
        }
    }
}

async def main():
    load_dotenv() # Đọc file .env vào os.environ
    
    # 3. Khởi tạo Async Client (Quan trọng cho server FastAPI/Starlette)
    client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
    
    # 4. Bắt đầu hội thoại
    messages = [{"role": "user", "content": "Thời tiết ở Hanoi hiện tại như thế nào? Và bây giờ là mấy giờ ở New York?"}]
    print("User:", messages[0]["content"])
    
    # --- VÒNG LẶP 1: Gửi câu hỏi và Tool cho LLM (Non-blocking) ---
    response = await client.chat.completions.create(
        model="openai/gpt-oss-120b", 
        messages=messages,
        tools=[tool_schema, tool_schema_2],
        tool_choice="auto"
    )
    
    message = response.choices[0].message
    messages.append(message)  
    
    # Sử dụng VÒNG LẶP WHILE để xử lý trường hợp LLM gọi tuần tự nhiều tool
    while message.tool_calls:
        print(f"\n[LLM] Yêu cầu gọi {len(message.tool_calls)} tools cùng lúc.")
        
        # Chạy TẤT CẢ các tools SONG SONG bằng asyncio.gather
        tasks = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_input = json.loads(tool_call.function.arguments)
            tasks.append(execute_tool(tool_name, tool_input))
            
        print("[Hệ thống] Đang thực thi tools song song (Parallel Tool Execution)...")
        # results sẽ chứa kết quả của các tool trả về cùng lúc
        results = await asyncio.gather(*tasks) 
        
        # --- Gửi kết quả lại cho LLM ---
        for tool_call, tool_result in zip(message.tool_calls, results):
            print(f"[Hệ thống] Kết quả tool {tool_call.function.name}: {tool_result}")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": str(tool_result)
            })
            
        # Gọi LLM lần nữa để xem nó có muốn gọi tool tiếp hay là trả lời
        response = await client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=[tool_schema, tool_schema_2]
        )
        message = response.choices[0].message
        messages.append(message)

    print("\nLLM Final Answer:", message.content)

if __name__ == "__main__":
    asyncio.run(main())
