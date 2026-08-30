# Giai đoạn 3 — Tuần 7-9: LangChain, Function Calling & AI Agent

## Mục tiêu tổng thể

Sau giai đoạn này, bạn có thể tự xây **1 hệ thống AI Agent hoàn chỉnh**: kết nối LLM với các công cụ bên ngoài (search, database, API), tự lên kế hoạch đa bước (multi-step reasoning), và điều phối workflow phức tạp bằng LangGraph — vượt xa RAG đơn thuần.

**Sản phẩm đầu ra (capstone project):** 1 "Research Agent" — người dùng đặt câu hỏi phức tạp, agent tự chia nhỏ thành sub-tasks, dùng nhiều tools (web search, RAG, calculator...), tổng hợp kết quả và trả lời có nguồn trích dẫn.

## Tiền đề (đã học ở Giai đoạn 2)

FastAPI, Qdrant vector search, RAG pipeline hoàn chỉnh, embedding model, reranking, async/await. Giai đoạn này sẽ dùng lại RAG pipeline từ Phase 2 như một "tool" trong agent.

---

## Bản đồ kiến thức tổng quan — Từ RAG đến Agent

```
Phase 2 (RAG):           Phase 3 (Agent):
User → Query            User → Complex Task
         ↓                        ↓
    Vector Search         Agent (LLM + ReAct Loop)
         ↓                  ↓         ↓         ↓
    Top-k Chunks         Tool 1    Tool 2    Tool 3
         ↓               (RAG)  (Search)  (API...)
    LLM Answer                    ↓
                          Tổng hợp → Answer
```

LangChain là framework kết nối các mảnh ghép lại. LangGraph là engine điều phối luồng logic phức tạp.

---

## TUẦN 7 — LangChain & LCEL

### Ngày 1: LangChain cơ bản & LCEL (LangChain Expression Language)

**Mục tiêu:** Hiểu triết lý của LangChain, nắm cú pháp LCEL pipe operator `|`, tự xây chain đầu tiên.

**Khái niệm cần nắm:**

- **LangChain** — framework mã nguồn mở giúp kết nối LLM với các thành phần khác (prompt template, output parser, retriever, tool...). Thay vì viết boilerplate lặp đi lặp lại, LangChain cung cấp các abstraction chuẩn.

- **LCEL (LangChain Expression Language)** — cú pháp dùng toán tử `|` để nối các thành phần thành pipeline. Mỗi thành phần là 1 `Runnable` (có thể là prompt, model, parser, retriever...). Điểm mạnh: hỗ trợ streaming, async, batching tự động.

```python
# Ví dụ LCEL:
chain = prompt_template | llm | output_parser
result = chain.invoke({"question": "..."})
# Dữ liệu chảy: prompt_template → llm → output_parser
```

- **Runnable Interface** — interface chuẩn mà tất cả thành phần trong LangChain đều implement:
  - `.invoke(input)` — chạy đồng bộ
  - `.stream(input)` — streaming từng token
  - `.batch([input1, input2])` — chạy song song nhiều input
  - `.ainvoke(input)` — async version

- **PromptTemplate** — template với biến động, ví dụ: `"Trả lời câu hỏi: {question}"`. Tách biệt nội dung prompt khỏi code Python.

- **ChatPromptTemplate** — tương tự nhưng cho chat model, hỗ trợ system/human/ai messages.

**Tài liệu đọc:**
- LangChain LCEL Quickstart: https://python.langchain.com/docs/concepts/lcel/
- LangChain Expression Language: https://python.langchain.com/docs/how_to/lcel_cheatsheet/

**Cài đặt:**
```bash
pip install langchain langchain-anthropic langchain-openai python-dotenv
```

**Bài tập:**
1. Tạo `PromptTemplate` với 2 biến `{topic}` và `{language}`, gọi `.format()` để kiểm tra output.
2. Xây chain đơn giản: `ChatPromptTemplate | ChatAnthropic | StrOutputParser` → hỏi LLM 1 câu bất kỳ.
3. Thêm `.stream()` vào chain → in từng token ra console khi LLM đang trả lời.
4. Dùng `.batch()` để gửi 3 câu hỏi khác nhau cùng lúc → so sánh với việc gọi `.invoke()` 3 lần tuần tự về tốc độ.
5. Dùng `RunnablePassthrough` để tạo chain nhận dict input, pass thêm metadata qua pipeline.

**Tiêu chí hoàn thành:** Giải thích được "LCEL khác gì so với gọi API LLM thủ công?" và tự xây 1 chain từ PromptTemplate → LLM → Parser hoạt động được.

---

### Ngày 2: Output Parser & Structured Output

**Mục tiêu:** Ép LLM trả về dữ liệu có cấu trúc (JSON, Pydantic model) một cách đáng tin cậy.

**Khái niệm cần nắm:**

- **Output Parser** — thành phần cuối chain, chuyển đổi raw text output của LLM thành Python object. Các loại phổ biến:
  - `StrOutputParser` — đơn giản, trả về string thuần
  - `JsonOutputParser` — parse JSON, raise error nếu LLM trả về JSON sai format
  - `PydanticOutputParser` — parse và validate theo Pydantic model, mạnh nhất

- **`.with_structured_output()`** — cách hiện đại nhất (2024+). Thay vì dùng parser riêng, gọi trực tiếp trên model. LangChain tự chọn phương pháp tốt nhất (tool calling hoặc JSON mode) tùy theo model.

```python
from pydantic import BaseModel, Field

class BookReview(BaseModel):
    title: str = Field(description="Tên sách")
    rating: int = Field(description="Điểm từ 1-10", ge=1, le=10)
    summary: str = Field(description="Tóm tắt đánh giá trong 2 câu")

model = ChatAnthropic(model="claude-3-5-haiku-20241022")
structured_model = model.with_structured_output(BookReview)
result = structured_model.invoke("Đánh giá cuốn 'Atomic Habits'")
# result là BookReview object, type-safe, validated
print(result.rating)  # int, không phải string
```

- **Retry logic** — LLM đôi khi trả về JSON sai. `OutputFixingParser` tự động gửi lại output lỗi cho LLM để sửa.

**Tài liệu đọc:**
- LangChain Structured Output: https://python.langchain.com/docs/how_to/structured_output/

**Bài tập:**
1. Dùng `JsonOutputParser` để ép LLM trả về JSON có 3 trường: `answer`, `confidence` (0-1), `sources` (list).
2. Tạo Pydantic model `ProductInfo` (name, price, category, in_stock: bool). Dùng `PydanticOutputParser` để extract thông tin sản phẩm từ 1 đoạn text mô tả.
3. Dùng `.with_structured_output()` với cùng Pydantic model → so sánh code với cách dùng `PydanticOutputParser`.
4. Thử cố tình tạo prompt mơ hồ để LLM trả sai JSON → implement retry logic thủ công.

**Tiêu chí hoàn thành:** Tự tin tạo Pydantic model cho bất kỳ output nào và ép LLM trả về đúng format.

---

### Ngày 3: LangChain Retriever & RAG Chain

**Mục tiêu:** Tích hợp Qdrant từ Phase 2 vào LangChain Retriever interface, xây RAG chain chuẩn LCEL.

**Khái niệm cần nắm:**

- **Retriever Interface** — abstraction chuẩn của LangChain cho tất cả vector store. Method chính: `.get_relevant_documents(query)` hoặc `.invoke(query)` → trả về `list[Document]`. Lợi ích: swap backend (Qdrant, Pinecone, FAISS...) mà không đổi code phía trên.

- **LangChain Document** — object chuẩn gồm `page_content: str` và `metadata: dict`. Toàn bộ pipeline LangChain dùng object này.

- **RAG Chain chuẩn với LCEL:**

```python
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("Câu hỏi của user?")
```

- **`RunnableParallel`** — chạy nhiều runnable song song và merge kết quả. Ví dụ: vừa retrieve context vừa pass question cùng lúc.

- **VectorStoreRetriever** — Qdrant có thể wrap thành LangChain retriever qua `QdrantVectorStore.as_retriever()`.

**Tài liệu đọc:**
- LangChain RAG Quickstart: https://python.langchain.com/docs/tutorials/rag/

**Bài tập:**
1. Wrap Qdrant collection từ Phase 2 thành `QdrantVectorStore`, gọi `.as_retriever(search_kwargs={"k": 5})`.
2. Xây full RAG chain theo cú pháp LCEL như trên → so sánh với code thủ công từ Phase 2.
3. Thêm source trích dẫn: chain trả về cả `answer` và `sources` (list tên file/page từ metadata).
4. Bật streaming cho RAG chain → người dùng thấy token xuất hiện dần thay vì chờ toàn bộ câu trả lời.

**Tiêu chí hoàn thành:** RAG chain LCEL chạy được, có streaming, có trích dẫn nguồn.

---

### Ngày 4: Conversation Memory — Chatbot nhớ lịch sử hội thoại

**Mục tiêu:** Cho chatbot khả năng nhớ các lượt hội thoại trước, hỏi lại câu hỏi liên quan đến câu trả lời trước mà không cần lặp lại context.

**Khái niệm cần nắm:**

- **Vấn đề:** LLM là stateless — mỗi lần gọi API là 1 request mới, hoàn toàn không nhớ lần trước. Memory là cách bạn tự quản lý lịch sử và nhét vào prompt.

- **Các loại Memory:**

| Loại | Cách hoạt động | Khi nào dùng |
|---|---|---|
| **Buffer Memory** | Lưu toàn bộ messages, nhét hết vào prompt | Chat ngắn, đơn giản |
| **Window Memory** | Chỉ giữ N lượt gần nhất | Chat dài, muốn tiết kiệm token |
| **Summary Memory** | Dùng LLM tóm tắt hội thoại cũ | Chat rất dài, cần ngữ cảnh xa |
| **Vector Memory** | Lưu memory vào vector DB, retrieve theo semantic | Agent cần nhớ "sự kiện" cụ thể |

- **`ChatMessageHistory`** — class chuẩn của LangChain lưu messages. Backend có thể là in-memory, Redis, Postgres...

- **`RunnableWithMessageHistory`** — wrapper tự động inject history vào chain và lưu response mới vào history.

```python
from langchain_core.runnables.history import RunnableWithMessageHistory

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,  # hàm lấy history theo session_id
    input_messages_key="question",
    history_messages_key="chat_history",
)

# Mỗi session_id là 1 cuộc hội thoại riêng biệt
result = chain_with_memory.invoke(
    {"question": "Hôm nay tôi mệt quá"},
    config={"configurable": {"session_id": "user_123"}}
)
```

**Tài liệu đọc:**
- LangChain Memory: https://python.langchain.com/docs/how_to/message_history/

**Bài tập:**
1. Xây chatbot cơ bản với `InMemoryChatMessageHistory` — gõ 5 lượt, hỏi câu liên quan đến lượt trước → confirm bot nhớ.
2. Thêm Window: chỉ giữ 4 messages gần nhất → test với hội thoại 10 lượt.
3. Tích hợp memory vào RAG chain từ Ngày 3: "Câu hỏi tiếp theo" phải hiểu "nó" trỏ đến đối tượng nào từ câu hỏi trước.
4. Implement `get_session_history` nhận `session_id` → mỗi user có history riêng, không trộn lẫn.

**Tiêu chí hoàn thành:** RAG chatbot nhớ được ít nhất 3 lượt hội thoại trước, câu hỏi follow-up không cần lặp lại context.

---

### Ngày 5: LangSmith — Debug & Tracing Pipeline

**Mục tiêu:** Dùng LangSmith để visualize và debug chain LCEL, hiểu từng bước xử lý bên trong.

**Khái niệm cần nắm:**

- **LangSmith** — platform observability của LangChain. Mỗi lần chain chạy, LangSmith ghi lại toàn bộ: input/output của từng bước, latency, số token dùng, cost. Không thể debug LangChain hiệu quả nếu không có LangSmith.

- **Trace** — 1 lần chạy chain đầy đủ từ input đến output, có cấu trúc cây (tree) thể hiện các bước con lồng nhau.

- **Tại sao cần LangSmith:**
  - Nhìn thấy **prompt thực tế** được gửi cho LLM (sau khi template đã được fill) — rất quan trọng để debug hallucination
  - Xem chính xác **chunks nào được retrieve** — phát hiện retrieval quality issue
  - Đo **latency từng bước** — tìm bottleneck
  - So sánh output giữa các lần chạy khác nhau

- **Setup:**
```bash
pip install langsmith
```
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__..."
os.environ["LANGCHAIN_PROJECT"] = "phase3-practice"
# Sau đó chạy chain bình thường — tự động được trace
```

**Tài liệu đọc:**
- LangSmith Quickstart: https://docs.smith.langchain.com/

**Bài tập:**
1. Tạo account LangSmith (free tier đủ dùng), lấy API key.
2. Bật tracing cho RAG chain từ Ngày 3-4 → chạy 5 queries → xem trace trên dashboard.
3. Tìm trong trace: prompt thực tế gửi cho LLM là gì, chunks nào được retrieve, token count là bao nhiêu.
4. Cố tình tạo 1 query trả lời sai → dùng LangSmith trace để diagnose: lỗi ở bước retrieve hay bước generate?

**Tiêu chí hoàn thành:** Khi chain cho kết quả sai, bạn có thể dùng LangSmith để chỉ ra nguyên nhân trong vòng 2 phút.

---

### Checkpoint cuối Tuần 7

Ghép lại thành 1 file `rag_chatbot.py`:
- LangChain RAG chain với LCEL
- Conversation memory (Window = 6 messages)
- Structured output cho câu trả lời (có field `answer`, `confidence`, `sources`)
- Tracing với LangSmith
- FastAPI endpoint `POST /chat` nhận `{session_id, question}` → trả về structured response

---

## TUẦN 8 — Function Calling & Tool Use

### Ngày 6: Function Calling / Tool Use — Nguyên lý cốt lõi

**Mục tiêu:** Hiểu cơ chế Function Calling từ góc độ protocol-level, viết tool đầu tiên không dùng framework.

**Khái niệm cần nắm:**

- **Function Calling là gì?** — Tính năng cho phép LLM không tự trả lời ngay mà thay vào đó ra quyết định "tôi cần gọi function này với arguments này". LLM trả về structured JSON thay vì text, code phía bạn nhận JSON đó, thực thi function thật, rồi gửi kết quả lại cho LLM → LLM tổng hợp câu trả lời cuối.

- **Luồng hoạt động (cực kỳ quan trọng!):**

```
User: "Thời tiết Hà Nội hôm nay thế nào?"
           ↓
     [LLM nhận câu hỏi + danh sách tools có sẵn]
           ↓
LLM output: {"tool": "get_weather", "args": {"city": "Hanoi"}}
           ↓  (LLM KHÔNG tự chạy function, chỉ ra quyết định!)
     [Code của bạn thực thi get_weather("Hanoi")]
           ↓  (gọi API thời tiết thật)
     kết quả: {"temp": 32, "condition": "cloudy"}
           ↓
     [Gửi kết quả này lại cho LLM]
           ↓
LLM output: "Thời tiết Hà Nội hôm nay 32°C và có mây..."
```

- **JSON Schema cho tool** — mỗi tool phải có schema mô tả cho LLM biết tool đó làm gì, nhận tham số gì:
```python
tool_schema = {
    "name": "get_weather",
    "description": "Lấy thông tin thời tiết hiện tại của 1 thành phố",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "Tên thành phố bằng tiếng Anh"
            }
        },
        "required": ["city"]
    }
}
```

- **Cách Anthropic implement** (Claude API trực tiếp, không qua LangChain):
```python
import anthropic

client = anthropic.Anthropic()
tools = [tool_schema]

response = client.messages.create(
    model="claude-3-5-haiku-20241022",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "Thời tiết HN?"}]
)

# Kiểm tra LLM có muốn gọi tool không
if response.stop_reason == "tool_use":
    tool_use = next(b for b in response.content if b.type == "tool_use")
    tool_name = tool_use.name       # "get_weather"
    tool_input = tool_use.input    # {"city": "Hanoi"}
    tool_result = execute_tool(tool_name, tool_input)  # code của bạn
    # Gửi kết quả lại...
```

**Tài liệu đọc (BẮT BUỘC):**
- Anthropic Tool Use Guide: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
- Đọc kỹ phần "How tool use works" và "Implementing tool use"

**Bài tập:**
1. Viết tool `calculate(expression: str) -> float` — dùng Python `eval()` an toàn để tính biểu thức toán học.
2. Viết JSON schema cho tool đó. Gửi cho Claude API với câu hỏi "Tính 15% của 350 cộng thêm 42".
3. Implement vòng lặp tool execution: detect `stop_reason == "tool_use"` → gọi function → gửi kết quả lại → lấy câu trả lời cuối.
4. Thêm tool thứ hai `get_current_time(timezone: str) -> str`. Test câu hỏi cần dùng cả 2 tools.

**Tiêu chí hoàn thành:** Giải thích được "LLM có tự chạy function không?" và implement được vòng lặp tool calling từ đầu không dùng framework.

---

### Ngày 7: LangChain Tools — Tích hợp Tools vào Chain

**Mục tiêu:** Chuyển tool thủ công sang LangChain `@tool` decorator, bind tools vào LLM, xây tool-calling chain.

**Khái niệm cần nắm:**

- **`@tool` decorator** — LangChain tự động tạo JSON schema từ Python function signature và docstring:
```python
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Tìm kiếm thông tin trên internet. Dùng khi cần thông tin hiện tại hoặc không có trong training data.
    
    Args:
        query: Câu truy vấn tìm kiếm bằng tiếng Anh
    """
    # implement thật hoặc mock
    return f"Kết quả tìm kiếm cho: {query}"
```

- **`.bind_tools()`** — attach danh sách tools vào LLM model:
```python
llm_with_tools = llm.bind_tools([search_web, calculate, get_weather])
```

- **`ToolNode`** — LangChain component tự động execute tất cả tool_calls trong AIMessage:
```python
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode

tool_node = ToolNode([search_web, calculate])
```

- **`AgentExecutor`** — executor cổ điển của LangChain (đang dần bị thay bởi LangGraph). Vẫn quan trọng để hiểu concept.

**Tài liệu đọc:**
- LangChain Tool Calling: https://python.langchain.com/docs/how_to/tool_calling/
- LangChain Custom Tools: https://python.langchain.com/docs/how_to/custom_tools/

**Bài tập:**
1. Chuyển 2 tools từ Ngày 6 sang dùng `@tool` decorator → in ra `.schema` để xem JSON tự generate.
2. Dùng `.bind_tools()` và LCEL để xây chain: `prompt | llm_with_tools`.
3. Thêm tool `rag_search(question: str) -> str` — wrap Qdrant RAG retrieval từ Phase 2 thành 1 tool. Giờ LLM có thể tự quyết định khi nào cần search document.
4. Test: hỏi câu cần RAG → LLM gọi `rag_search`. Hỏi câu toán → gọi `calculate`. Hỏi câu cả hai → gọi cả hai.

**Tiêu chí hoàn thành:** Tools tích hợp mượt mà, LLM tự chọn đúng tool cho từng loại câu hỏi.

---

### Ngày 8: Parallel Tool Calls & Error Handling

**Mục tiêu:** Xử lý trường hợp LLM gọi nhiều tools cùng lúc, handle tool errors gracefully.

**Khái niệm cần nắm:**

- **Parallel Tool Calls** — LLM hiện đại (Claude 3+, GPT-4) có thể ra quyết định gọi nhiều tools trong 1 lần respond. VD: câu hỏi "So sánh thời tiết Hà Nội và Hồ Chí Minh" → LLM call `get_weather("Hanoi")` VÀ `get_weather("Ho Chi Minh")` song song trong 1 response.

```python
# AIMessage.tool_calls có thể là list với nhiều phần tử
for tool_call in ai_message.tool_calls:
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    # execute mỗi cái...
```

- **Tool Errors — LLM cần biết tool thất bại:**
```python
from langchain_core.messages import ToolMessage

# Khi tool raise exception, ĐỪNG crash — gửi error message lại cho LLM
try:
    result = execute_tool(tool_call)
    tool_msg = ToolMessage(content=str(result), tool_call_id=tool_call["id"])
except Exception as e:
    tool_msg = ToolMessage(
        content=f"Tool execution failed: {str(e)}",
        tool_call_id=tool_call["id"],
        status="error"
    )
# LLM nhận được error → có thể thử cách khác hoặc báo user
```

- **Tool Result Formatting** — LLM đọc kết quả tool từ text. Format tốt (có cấu trúc, ngắn gọn) → LLM hiểu đúng hơn.

**Bài tập:**
1. Thêm tool `convert_currency(amount: float, from_currency: str, to_currency: str) -> float`. Test câu hỏi cần `calculate` + `convert_currency` song song.
2. Cố tình làm tool bị lỗi (network timeout giả) → implement error handling đúng cách → LLM vẫn phải trả lời gracefully.
3. Log toàn bộ tool_calls và results — visualize trên LangSmith.

**Tiêu chí hoàn thành:** Tool pipeline không crash khi tool lỗi, LLM xử lý partial failure được.

---

### Checkpoint cuối Tuần 8

Xây `tools_api.py` — FastAPI server với endpoint `POST /ask`:
- Nhận câu hỏi tự do
- LLM có 4 tools: `rag_search`, `calculate`, `get_current_datetime`, `format_table`
- Hỗ trợ parallel tool calls
- Trả về answer + danh sách tools đã được gọi (với input/output) để frontend hiển thị "chain of thought"

---

## TUẦN 9 — LangGraph & AI Agent

### Ngày 9: LangGraph — Stateful Agent từ đầu

**Mục tiêu:** Hiểu LangGraph khác AgentExecutor thế nào, tự xây agent graph đầu tiên với node/edge/state.

**Khái niệm cần nắm:**

- **Vì sao cần LangGraph?** AgentExecutor của LangChain chạy loop đơn giản: LLM -> Tool -> LLM -> Tool... Không có cách để: phân nhánh logic phức tạp, có nhiều agents phối hợp, implement human-in-the-loop, hoặc quay lại bước trước khi cần. LangGraph giải quyết tất cả bằng cách mô hình hóa agent như 1 **State Machine (máy trạng thái)**.

- **3 khái niệm cốt lõi LangGraph:**
  - **State** — TypedDict chứa toàn bộ thông tin của workflow tại 1 thời điểm. Mỗi node đọc/ghi State.
  - **Node** — 1 Python function nhận State -> trả về dict update State (không cần return toàn bộ state).
  - **Edge** — kết nối giữa các nodes. Có 2 loại: edge cố định (A -> B luôn luôn) và conditional edge (A -> B hoặc A -> C tùy điều kiện).

`python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# 1. Định nghĩa State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # add_messages = reducer: append, không replace

# 2. Định nghĩa Nodes
def call_llm(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"  # -> chạy tool
    return END           # -> kết thúc

# 3. Xây Graph
graph = StateGraph(AgentState)
graph.add_node("agent", call_llm)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")  # sau tools -> quay lại agent
app = graph.compile()
`

- **dd_messages reducer** — khi State update messages, không replace list cũ mà append vào. Đây là pattern quan trọng để giữ conversation history trong State.

**Tài liệu đọc (BẮT BUỘC):**
- LangGraph Quickstart: https://langchain-ai.github.io/langgraph/tutorials/introduction/

**Cài đặt:**
`ash
pip install langgraph
`

**Bài tập:**
1. Xây ReAct agent cơ bản như code trên với 2 tools từ Tuần 8.
2. Vẽ graph bằng pp.get_graph().draw_mermaid() -> hiểu cấu trúc node/edge.
3. Dùng pp.invoke() với câu hỏi cần nhiều tool calls -> print từng bước state thay đổi.
4. Dùng pp.stream() để thấy state update sau mỗi node.

**Tiêu chí hoàn thành:** Giải thích được "LangGraph khác AgentExecutor ở điểm gì?" và tự xây được graph 3 nodes.

---

### Ngày 10: Human-in-the-Loop & Checkpointing

**Mục tiêu:** Cho phép con người xem xét và phê duyệt hành động của agent trước khi thực thi, persist state qua sessions.

**Khái niệm cần nắm:**

- **Human-in-the-Loop (HITL)** — agent dừng lại và chờ người dùng xác nhận trước khi gọi tool nguy hiểm (xóa file, thanh toán, gửi email...). Đây là tính năng quan trọng cho production agent.

- **Interrupt** — cơ chế LangGraph để dừng graph giữa chừng, trả control cho user:
`python
from langgraph.checkpoint.memory import MemorySaver

# Checkpointer lưu state -> cho phép resume
checkpointer = MemorySaver()
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["tools"]  # dừng TRƯỚC khi chạy tools
)

config = {"configurable": {"thread_id": "session-1"}}

# Lần 1: chạy đến interrupt
result = app.invoke({"messages": [HumanMessage("Xóa file test.txt")]}, config)
# Agent đề xuất gọi tool delete_file -> dừng lại

# User xem xét và approve
print("Agent muốn:", result["messages"][-1].tool_calls)
user_input = input("Cho phép? (y/n): ")

if user_input == "y":
    # Resume từ điểm dừng
    result = app.invoke(None, config)  # None = tiếp tục từ checkpoint
`

- **Checkpointer & Thread** — mỗi 	hread_id là 1 conversation riêng biệt. State được persist -> agent có thể resume sau khi bị ngắt (crash, restart...).

- **interrupt_after** vs **interrupt_before** — dừng sau hoặc trước khi node chạy xong.

**Tài liệu đọc:**
- LangGraph Human-in-the-loop: https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/

**Bài tập:**
1. Thêm MemorySaver và interrupt_before=["tools"] vào agent từ Ngày 9.
2. Test: hỏi câu cần tool -> agent dừng -> bạn in ra đề xuất tool call -> approve/reject.
3. Implement "reject" flow: khi user từ chối, gửi message "User từ chối hành động này" vào state -> LLM thử cách khác.
4. Test persistence: chạy graph đến interrupt -> "tắt" (không resume) -> khởi động lại, dùng cùng 	hread_id -> graph nhớ được điểm dừng.

**Tiêu chí hoàn thành:** Agent production-safe: không tự ý thực hiện hành động nhạy cảm, state persist qua sessions.

---

### Ngày 11: Multi-Agent System — Subgraph & Agent Orchestration

**Mục tiêu:** Xây hệ thống nhiều agent chuyên biệt, có 1 orchestrator điều phối.

**Khái niệm cần nắm:**

- **Vì sao Multi-Agent?** — 1 agent đơn với 20 tools sẽ chọn tool kém hơn agent chuyên biệt với 3 tools. Các agent nhỏ, chuyên sâu phối hợp tốt hơn 1 agent "biết tuốt".

- **Pattern: Supervisor + Workers:**
`
User Request
     |
[Supervisor Agent]     -> LLM đọc request, quyết định giao việc cho ai
   /        |        \
[Research  [RAG    [Code
 Agent]    Agent]  Agent]
  (web)   (docs)  (python)
     \        |        /
[Supervisor tổng hợp kết quả -> trả lời User]
`

- **Subgraph** — 1 LangGraph graph có thể là node của graph khác. Mỗi worker agent là 1 subgraph độc lập.

- **Handoff** — cơ chế 1 agent chuyển việc sang agent khác, kèm theo context cần thiết:
`python
# Supervisor node quyết định
def supervisor_node(state):
    decision = supervisor_llm.invoke(state["messages"])
    # Decision: "research", "rag", "code", hay "FINISH"
    return {"next": decision}

graph.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {"research": "research_agent", "rag": "rag_agent", "FINISH": END}
)
`

**Tài liệu đọc:**
- LangGraph Multi-Agent: https://langchain-ai.github.io/langgraph/concepts/multi_agent/

**Bài tập:**
1. Tạo ResearchAgent (tools: web search, summarize) và RAGAgent (tools: rag_search từ Phase 2).
2. Tạo SupervisorAgent — nhận task, dùng LLM quyết định giao cho Research hay RAG agent.
3. Implement handoff: mỗi agent hoàn thành việc -> report kết quả về cho Supervisor.
4. Test câu hỏi phức tạp cần cả 2 agents -> Supervisor tổng hợp.

**Tiêu chí hoàn thành:** Supervisor chọn đúng agent >80% các trường hợp test.

---

### Ngày 12: Long-Term Memory & Agent Personalization

**Mục tiêu:** Agent nhớ thông tin về user qua nhiều session (preferences, history), cá nhân hóa câu trả lời.

**Khái niệm cần nắm:**

- **Short-term Memory** (đã học Ngày 4) — trong 1 conversation, bị xóa khi session kết thúc.
- **Long-term Memory** — persist qua nhiều sessions. Lưu vào external store (database, vector store).

- **Các loại Long-term Memory:**
  - **Semantic Memory** — facts về user: "User thích Python, ghét Java, đang học AI"
  - **Episodic Memory** — các sự kiện cụ thể: "Ngày 20/8 user hỏi về Transformer, có vẻ chưa hiểu rõ"
  - **Procedural Memory** — cách làm việc với user: "User thích giải thích bằng ví dụ code"

- **Pattern chuẩn:** Lưu memory vào vector store ở mỗi đầu conversation, retrieve memories liên quan -> inject vào system prompt.

`python
# Cuối mỗi conversation, extract và lưu memories
def save_memories(conversation: list[Message], user_id: str):
    memory_extractor_prompt = "Từ hội thoại này, extract thông tin quan trọng về user..."
    memories = llm.invoke(memory_extractor_prompt + str(conversation))
    vector_store.add_texts([memories], metadata={"user_id": user_id})

# Đầu conversation mới, load memories
def load_memories(query: str, user_id: str) -> str:
    relevant_memories = vector_store.similarity_search(
        query, filter={"user_id": user_id}, k=3
    )
    return "\n".join([m.page_content for m in relevant_memories])
`

**Tài liệu đọc:**
- DeepLearning.AI Long-Term Agentic Memory: https://www.deeplearning.ai/courses/long-term-agentic-memory-with-langgraph

**Bài tập:**
1. Implement MemoryStore class: methods save(user_id, memory_text) và etrieve(user_id, query) -> list[str].
2. Tích hợp vào agent: đầu conversation retrieve memories -> thêm vào system prompt -> cuối conversation save memories mới.
3. Test: 2 sessions riêng biệt, session 2 agent nhớ preferences từ session 1.
4. Implement memory consolidation: nếu có >10 memories cho 1 user, dùng LLM tóm tắt bớt.

**Tiêu chí hoàn thành:** Agent nhớ ít nhất 3 điều về user từ session trước và áp dụng vào câu trả lời session sau.

---

## 🏆 Capstone Project Phase 3: Research Agent

**Mô tả:** Xây 1 "Research Assistant Agent" hoàn chỉnh — người dùng đặt câu hỏi phức tạp, agent tự nghiên cứu và trả lời với nguồn trích dẫn rõ ràng.

**Kiến trúc:**

`
FastAPI
   |
LangGraph Supervisor
   |           |           |
RAG Agent   Web Search   Calculator
(Phase 2    Agent        Agent
 Qdrant)    (Tavily)     (python eval)
   |
Long-term Memory (Qdrant)
   |
Human-in-the-loop (interrupt before action)
   |
Structured Output (answer + sources + confidence)
`

**Yêu cầu kỹ thuật:**
1. **LangGraph**: Supervisor + >=2 worker agents (RAG + Web Search)
2. **Tools**: >=4 tools tổng cộng, bao gồm RAG search từ Phase 2
3. **Memory**: Long-term memory lưu user preferences
4. **HITL**: Interrupt trước khi gọi tool search web (đắt tiền) -> user approve
5. **Structured Output**: Response có nswer, sources: list, gents_used: list, confidence: float
6. **Observability**: Full LangSmith tracing
7. **API**: FastAPI với endpoints:
   - POST /research -> gửi question, nhận job_id
   - GET /research/{job_id} -> poll kết quả
   - POST /research/{job_id}/approve -> approve HITL interrupt

**Tiêu chí hoàn thành:**
- [ ] Agent trả lời đúng câu hỏi factual cần tra cứu
- [ ] HITL hoạt động: dừng đúng lúc, resume sau approve
- [ ] Long-term memory: lần 2 hỏi cùng chủ đề, agent nhớ context từ lần 1
- [ ] LangSmith trace hiển thị rõ từng bước agent suy nghĩ
- [ ] Code coverage: unit test cho từng tool

---

## ✅ Checklist hoàn thành Phase 3

**Tuần 7 — LangChain:**
- [ ] Xây LCEL chain từ PromptTemplate -> LLM -> Parser
- [ ] RAG chain với LangChain Retriever, có streaming
- [ ] Chatbot với Conversation Memory (Window)
- [ ] LangSmith trace hiển thị từng bước

**Tuần 8 — Function Calling:**
- [ ] Implement tool calling thủ công (không dùng framework) -> hiểu protocol
- [ ] Tối thiểu 4 tools với @tool decorator
- [ ] Parallel tool calls hoạt động
- [ ] Error handling: tool fail -> LLM vẫn trả lời được

**Tuần 9 — LangGraph:**
- [ ] ReAct agent với StateGraph, node/edge/conditional
- [ ] Human-in-the-loop với Checkpoint + interrupt
- [ ] Multi-agent: Supervisor + 2 workers hoạt động
- [ ] Long-term memory persist qua >=2 sessions

**Câu hỏi phải trả lời được khi kết thúc Phase 3:**
- "LangGraph giải quyết vấn đề gì mà AgentExecutor không làm được?"
- "Khi nào nên dùng 1 agent với nhiều tools vs nhiều agents chuyên biệt?"
- "Human-in-the-loop implement ở tầng nào của stack?"
- "Long-term memory khác short-term memory ở điểm gì trong kiến trúc?"
