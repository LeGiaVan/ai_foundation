# LangChain Message Architecture & Chain of Thought (CoT) Cheatsheet

Tài liệu chi tiết chuyên sâu về **Kiến trúc tin nhắn chuẩn hóa**, **Cơ chế bóc tách Chain of Thought**, **Kỹ thuật giải phẫu Object**, và **Bộ công cụ Typing** trong LangChain & LangGraph.

---

## 1. Kiến trúc tin nhắn chuẩn hóa (Unified Message Architecture)

Bất kể bạn dùng nhà cung cấp nào (OpenAI, Groq, Anthropic Claude, Google Gemini, Ollama...), LangChain đều chuẩn hóa toàn bộ vòng lặp giao tiếp về **4 loại Message cơ bản** kế thừa từ `BaseMessage`.

```
                       BaseMessage
               (content, type, id, name)
                            │
        ┌──────────────┬────┴─────────┬──────────────┐
        │              │              │              │
  SystemMessage  HumanMessage     AIMessage     ToolMessage
  (role: system) (role: user)  (role: assistant)(role: tool)
```

### Bảng so sánh Bộ tứ quyền lực (Big 4 Messages)

| Message Class | Role tương đương | Ai tạo ra? | Mục đích / Ngữ cảnh | Thuộc tính quan trọng nhất |
| :--- | :--- | :--- | :--- | :--- |
| **`SystemMessage`** | `"system"` | Lập trình viên / Hệ thống | Đặt luật, phân vai (`"Bạn là trợ lý..."`), cấu hình tone giọng và các ràng buộc. | `.content` |
| **`HumanMessage`** | `"user"` | Người dùng | Lời nhắc (prompt), câu hỏi hoặc input đầu vào từ người dùng. | `.content` |
| **`AIMessage`** | `"assistant"` | Mô hình LLM | Câu trả lời, suy nghĩ (`Thought`), hoặc yêu cầu gọi công cụ (`tool_calls`). | `.content`<br>`.tool_calls`<br>`.additional_kwargs` |
| **`ToolMessage`** | `"tool"` | Hàm Python / Tool | Kết quả trả về sau khi thực thi code Python / Database / API (`Observation`). | `.content`<br>`.tool_call_id`<br>`.name` |

> [!TIP]
> **Tính hoán đổi tuyệt đối (Vendor Agnostic):**
> Nhờ 4 component chuẩn hóa này, bạn có thể đổi từ `ChatGroq` sang `ChatOpenAI`, `ChatAnthropic` hay mô hình Local `ChatOllama` chỉ bằng **1 dòng khai báo khởi tạo**. Toàn bộ logic vòng lặp, bóc tách dữ liệu và lưu database ở phía sau được **giữ nguyên 100%**.

---

## 2. Giải phẫu chi tiết đối tượng `AIMessage`

Mỗi khi LLM phản hồi, LangChain gói toàn bộ thông tin vào một đối tượng `AIMessage` gồm **4 ngăn chứa dữ liệu cốt lõi**:

```python
AIMessage(
    # [1] Văn bản trả lời chính cho người dùng (hoặc rỗng nếu LLM gọi tool)
    content="Hôm nay là ngày 17 tháng 9 năm 2026.",

    # [2] Danh sách các hàm LLM muốn gọi (chuẩn hóa dạng list[dict])
    tool_calls=[
        {
            "id": "call_abc123",
            "name": "get_current_datetime",
            "args": {}
        }
    ],

    # [3] Thông tin kỹ thuật (Tokens, thời gian, model...)
    response_metadata={
        "token_usage": {"prompt_tokens": 50, "completion_tokens": 20},
        "model_name": "openai/gpt-oss-120b",
        "finish_reason": "tool_calls"
    },

    # [4] Ngăn chứa "hàng riêng" của từng nhà cung cấp
    additional_kwargs={
        # Groq / DeepSeek-R1 / Qwen Reasoning lưu chuỗi suy luận ở đây:
        "reasoning_content": "User asks for date. Need to call get_time tool."
    }
)
```

---

## 3. Bản chất Chain of Thought & Cơ chế trích xuất "Thought"

### Tại sao có hiện tượng `msg.content` bị rỗng khi gọi Tool?
- Ở các dòng **Reasoning Models** (như `openai/gpt-oss-120b` trên Groq, DeepSeek-R1, OpenAI o1/o3):
  - Token suy luận (Thinking/Reasoning tokens) được tách riêng thành luồng dữ liệu độc lập.
  - Khi AI quyết định gọi Tool, `content` thường bằng `""` (rỗng), còn chuỗi suy luận nội tại được đặt trong `additional_kwargs["reasoning_content"]`.
- Ở các dòng **LLM truyền thống** (GPT-4o, Claude 3.5, Llama 3):
  - Nếu được nhắc suy nghĩ trước khi hành động qua prompt, AI sẽ ghi trực tiếp suy nghĩ vào `msg.content`.

### Code trích xuất Thought chuẩn phòng thủ (Two-Tier Extraction)
Đoạn code sau tự động bắt trọn suy nghĩ dù bạn dùng bất kỳ dòng model nào:

```python
thought = None
# Tầng 1: Bắt suy nghĩ từ Reasoning Engine (Groq / DeepSeek / OSS Reasoning)
if hasattr(msg, "additional_kwargs") and msg.additional_kwargs.get("reasoning_content"):
    thought = msg.additional_kwargs["reasoning_content"].strip()
# Tầng 2: Bắt suy nghĩ từ Text thông thường (Prompt-based CoT)
elif isinstance(msg.content, str) and msg.content.strip():
    thought = msg.content.strip()
```

---

## 4. Kỹ thuật "Kính hiển vi" soi Object trong Python

Khi nhận một đối tượng mới từ thư viện mà không rõ bên trong gồm những gì, hãy dùng 4 kỹ thuật thực chiến sau:

```python
import json

# 1. Xem dưới dạng JSON/Dict đầy đủ (Hiệu quả nhất với Pydantic / LangChain Objects)
print(json.dumps(msg.model_dump(), indent=2, ensure_ascii=False))

# 2. Kiểm tra kiểu Class chính xác
print("Kiểu dữ liệu:", type(msg))  # <class 'langchain_core.messages.ai.AIMessage'>

# 3. Liệt kê toàn bộ thuộc tính và phương thức có trong object
print("Các thuộc tính:", dir(msg))

# 4. In từ điển thuộc tính nội bộ
print("Thuộc tính nội bộ:", msg.__dict__)
```

### Nguyên tắc lập trình phòng thủ (Defensive Programming)
Không bao giờ truy cập trực tiếp bằng dấu chấm hoặc ngoặc vuông cứng nếu không chắc chắn trường đó luôn tồn tại:

```python
# ❌ Nguy hiểm (dễ văng AttributeError hoặc KeyError làm sập server):
calls = msg.tool_calls
thought = msg.additional_kwargs["reasoning_content"]

# ✅ An toàn tuyệt đối (không bao giờ crash, trả về None nếu thiếu):
calls = getattr(msg, "tool_calls", None)
thought = msg.additional_kwargs.get("reasoning_content") if hasattr(msg, "additional_kwargs") else None
```

---

## 5. Bộ 3 công cụ gõ kiểu (Typing Toolkit) trong LangGraph

LangGraph sử dụng 3 khái niệm typing cốt lõi của Python để quản lý trạng thái (`State`) và điều hướng luồng chạy (`Edges`):

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages

# 1. TypedDict: Khuôn mẫu State dạng Dictionary nhẹ, gợi ý tên key cho IDE & LLM
class AgentState(TypedDict):
    # 2. Annotated + add_messages: State mặc định ghi đè, reducer add_messages giúp NỐI THÊM (Append)
    messages: Annotated[list, add_messages]
    user_id: str
    retry_count: int

# 3. Literal: Ràng buộc giá trị trả về chính xác để Router Edge rẽ nhánh chuẩn
def router_node(state: AgentState) -> Literal["tools", "human_feedback", "__end__"]:
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "tools"
    return "__end__"
```

| Kiểu Type | Bản chất là gì? | Giải quyết bài toán gì trong Agent? |
| :--- | :--- | :--- |
| **`TypedDict`** | Dictionary thuần có gợi ý key | Nhẹ hơn `BaseModel` của Pydantic, không tốn tài nguyên validate runtime, định hình cấu trúc State rõ ràng. |
| **`Annotated`** | Kiểu dữ liệu + Hàm xử lý kèm theo (`reducer`) | Mặc định State của LangGraph bị **ghi đè** giá trị mới. `Annotated[list, add_messages]` bảo vệ lịch sử hội thoại bằng cách **nối thêm (append)** tin nhắn mới vào danh sách. |
| **`Literal`** | Tập hợp các giá trị hằng số cố định | Ràng buộc hàm Router chỉ được trả về đúng các tên node đã thiết kế, ngăn chặn hoàn toàn lỗi gõ sai tên nhánh. |

---

## 6. Mẫu chuẩn Production: FastAPI + Agent với Chain of Thought

Code mẫu hoàn chỉnh cho một API dịch vụ AI Agent trả về đầy đủ chuỗi suy luận **Thought $\rightarrow$ Action $\rightarrow$ Observation $\rightarrow$ Final Answer**:

```python
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage

app = FastAPI(title="Agent CoT API")

# 1. Khởi tạo LLM & Agent
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "Bạn là trợ lý thông minh. Trước khi gọi bất kỳ công cụ nào, "
        "hãy luôn suy nghĩ ngắn gọn (Thought) bằng tiếng Việt giải thích lý do."
    )
)

# 2. Pydantic Schemas
class AskRequest(BaseModel):
    question: str

class ToolStep(BaseModel):
    thought: Optional[str] = None  # 🧠 Suy nghĩ / lý do AI quyết định gọi tool
    tool_name: str                 # ⚡ Tên tool được gọi
    tool_input: dict               # 📥 Tham số truyền vào
    tool_output: str               # 👁️ Kết quả trả về từ tool

class AskResponse(BaseModel):
    final_answer: str
    chain_of_thought: list[ToolStep]

# 3. Endpoint xử lý và trích xuất CoT
@app.post("/ask", response_model=AskResponse)
async def ask_agent(req: AskRequest):
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": req.question}]})
        messages = result["messages"]
        final_answer = messages[-1].content
        
        chain_of_thought = []
        tool_calls_map = {}
        pending_thought = None
        
        for msg in messages:
            if isinstance(msg, AIMessage):
                # 🧠 Trích xuất suy nghĩ (Hỗ trợ cả Reasoning Tokens và Text Thông thường)
                thought = None
                if hasattr(msg, "additional_kwargs") and msg.additional_kwargs.get("reasoning_content"):
                    thought = msg.additional_kwargs["reasoning_content"].strip()
                elif isinstance(msg.content, str) and msg.content.strip():
                    thought = msg.content.strip()
                
                # ⚡ Lưu thông tin tool call kèm suy nghĩ
                if getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        tool_calls_map[tc["id"]] = {
                            "name": tc["name"],
                            "args": tc["args"],
                            "thought": thought or pending_thought
                        }
                    pending_thought = None
                else:
                    pending_thought = thought
                    
            elif isinstance(msg, ToolMessage):
                # 👁️ Gom thành 1 bước hoàn chỉnh (Thought -> Tool -> Input -> Output)
                tc_info = tool_calls_map.get(msg.tool_call_id, {})
                chain_of_thought.append(ToolStep(
                    thought=tc_info.get("thought"),
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
```
