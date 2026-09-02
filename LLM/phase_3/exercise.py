import os
import json
import asyncio
from groq import AsyncGroq
from dotenv import load_dotenv

'''
**Bài tập:**
1. Viết tool `calculate(expression: str) -> float` — dùng Python `eval()` an toàn để tính biểu thức toán học.
2. Viết JSON schema chuẩn OpenAI/Groq cho tool đó. Gửi cho Groq API với câu hỏi "Tính 15% của 350 cộng thêm 42".
3. Implement vòng lặp tool execution: detect `message.tool_calls` → gọi function → gửi kết quả lại (với `role: "tool"`, `tool_call_id`) → lấy câu trả lời cuối.
'''

# 1. Hàm calculate
async def calculate(expression: str) -> float:
    print(f"\n[Hệ thống] Đang tính toán: {expression}")
    # Hàm eval() trong thực tế có rủi ro bảo mật, nhưng dùng cho bài tập này thì OK
    return float(eval(expression))

# 2. Schema chuẩn OpenAI/Groq
tool_schema = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Tính toán một biểu thức toán học (mathematical expression)",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Biểu thức toán học để tính, ví dụ: '0.15 * 350 + 42'"
                }
            },
            "required": ["expression"]
        }
    }
}

AVAILABLE_TOOLS = {
    "calculate": calculate
}

async def execute_tool(tool_name: str, tool_kwargs: dict):
    if tool_name in AVAILABLE_TOOLS:
        return await AVAILABLE_TOOLS[tool_name](**tool_kwargs)
    return "Error: Tool not found"

async def main():
    load_dotenv()
    client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

    messages = [
        {"role": "user", "content": "Tính 15% của 350 cộng thêm 42"}
    ]
    print("User:", messages[0]["content"])

    # --- Vòng lặp 1 ---
    response = await client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        tools=[tool_schema],
        tool_choice="auto"
    )

    message = response.choices[0].message
    messages.append(message)  

    # Kiểm tra LLM có muốn gọi tool không (Dùng while loop)
    while message.tool_calls:
        print(f"\n[LLM] Yêu cầu gọi {len(message.tool_calls)} tools cùng lúc.")
        
        tasks = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            # parse JSON input từ LLM (như {"expression": "0.15 * 350 + 42"})
            tool_input = json.loads(tool_call.function.arguments)
            tasks.append(execute_tool(tool_name, tool_input))
            
        # Thực thi song song (dù ở đây có 1 tool)
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
            
        # Gọi LLM lần nữa (Bắt buộc phải pass tools vào đây để không bị lỗi 400)
        response = await client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=[tool_schema]
        )
        message = response.choices[0].message
        messages.append(message)

    print("\nLLM Final Answer:", message.content)

if __name__ == "__main__":
    asyncio.run(main())