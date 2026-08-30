# LCEL (LangChain Expression Language) Cheat Sheet

LCEL là cú pháp cốt lõi của LangChain giúp khai báo và kết nối các thành phần (Runnables) thành một pipeline (chuỗi - chain) một cách tường minh, dễ đọc và hỗ trợ sẵn các tính năng ưu việt như streaming, async, và batching.

---

## 1. Cú pháp `|` (Pipe Operator)

Giống như toán tử pipe trong Unix, `|` trong LCEL dùng để lấy **output của thành phần bên trái** làm **input cho thành phần bên phải**.

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate.from_template("Kể một câu chuyện ngắn về {topic}.")
llm = ChatOpenAI(model="gpt-3.5-turbo")
parser = StrOutputParser()

# Khai báo LCEL Chain
chain = prompt | llm | parser

# Chạy chain
result = chain.invoke({"topic": "AI Agent"})
print(result)
```

**Luồng dữ liệu (Data flow):**
1. `{"topic": "AI Agent"}` truyền vào `prompt` ➡️ Trả về string/prompt value: `"Kể một câu chuyện ngắn về AI Agent."`
2. Truyền string đó vào `llm` ➡️ Trả về object `AIMessage(...)`
3. Truyền `AIMessage` vào `parser` ➡️ Trả về string thuần chứa câu trả lời.

---

## 2. Các phương thức của `Runnable` Interface

Mọi thành phần trong LangChain (Prompt, Model, Parser, Retriever...) đều là 1 `Runnable`. Do đó, bất kỳ chain nào được tạo từ `|` cũng là 1 `Runnable` và hỗ trợ các hàm sau:

### Chạy đồng bộ (Sync)
- **`invoke(input)`**: Chạy chain và chờ kết quả cuối cùng.
  ```python
  chain.invoke({"topic": "mèo"})
  ```
- **`batch([input1, input2])`**: Chạy song song nhiều input cùng lúc (nhanh hơn nhiều so với chạy vòng lặp for từng cái một).
  ```python
  chain.batch([{"topic": "mèo"}, {"topic": "chó"}])
  ```
- **`stream(input)`**: Trả về một generator yield từng token khi LLM đang sinh văn bản. Rất hữu ích làm tính năng gõ chữ (typing effect) trên UI.
  ```python
  for chunk in chain.stream({"topic": "mèo"}):
      print(chunk, end="", flush=True)
  ```

### Chạy bất đồng bộ (Async) - Hữu ích cho FastAPI
- **`ainvoke(input)`**, **`abatch(inputs)`**, **`astream(input)`**: Thêm tiền tố `a` (async).
  ```python
  result = await chain.ainvoke({"topic": "mèo"})
  
  async for chunk in chain.astream({"topic": "mèo"}):
      print(chunk, end="", flush=True)
  ```

---

## 3. Quản lý Prompt (Prompt Templates)

LangChain tách biệt code Python và nội dung text prompt thông qua Prompt Templates.

### `PromptTemplate` (Cho LLM thường)
Dùng cho mô hình text-in/text-out cơ bản.
```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["language", "topic"],
    template="Viết một bài thơ bằng tiếng {language} về chủ đề {topic}."
)
# Hoặc dùng shorthand:
prompt = PromptTemplate.from_template("Viết một bài thơ bằng tiếng {language} về chủ đề {topic}.")
```

### `ChatPromptTemplate` (Cho Chat Models)
Định nghĩa sẵn các vai trò (System, Human, AI).
```python
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Bạn là một trợ lý ảo tên là {assistant_name}. Trả lời ngắn gọn."),
    ("human", "Giải thích {concept} là gì?")
])
```

---

## 4. `RunnablePassthrough` & `RunnableParallel`

Khi xây dựng chain, đôi khi bạn cần truyền dữ liệu đầu vào đi xuyên qua pipeline mà không bị thay đổi, hoặc phân nhánh chạy song song.

### `RunnablePassthrough`
Giữ nguyên input và pass nó xuống bước tiếp theo. Thường dùng trong RAG để gom nhóm context và question.

```python
from langchain_core.runnables import RunnablePassthrough

# Ví dụ RAG Chain:
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

rag_chain.invoke("LangChain là gì?")
```
*Giải thích: Output của dict khởi đầu sẽ có `context` lấy từ hàm search của `retriever`, còn `question` chính là input gốc ("LangChain là gì?") được truyền thẳng qua nhờ `RunnablePassthrough()`.*

### `RunnableParallel` (Khai báo tự động qua dict)
Chạy song song nhiều hàm/runnable cùng lúc.
```python
from langchain_core.runnables import RunnableParallel

map_chain = RunnableParallel(
    joke=prompt_joke | llm | parser,
    poem=prompt_poem | llm | parser
)
# Cả 2 chain (joke và poem) sẽ chạy đồng thời
res = map_chain.invoke({"topic": "Lập trình viên"})
print(res["joke"])
print(res["poem"])
```

---

## 5. LCEL khác gì gọi API thủ công?

| Gọi API Thủ Công (OpenAI SDK, `requests`) | Dùng LangChain LCEL |
|---|---|
| Phải tự format chuỗi string rườm rà. | `PromptTemplate` tự lo việc thay thế biến. |
| Nếu đổi từ OpenAI sang Groq/Anthropic, phải viết lại toàn bộ code gọi hàm. | Đổi class (`ChatOpenAI` -> `ChatGroq`), pipeline `\|` không cần sửa. |
| Muốn streaming hoặc async phải tự code logic khá phức tạp, dùng queue/generator. | Tích hợp sẵn hàm `.stream()`, `.ainvoke()` dùng chung mọi chỗ. |
| Phải tự parse chuỗi JSON trả về. | Có các OutputParser ép kiểu trực tiếp. |
| Cực kỳ khó quản lý khi làm hệ thống nhiều bước (Agent, RAG, API call). | Ghép nối đơn giản bằng `\|`, dễ debug bằng LangSmith. |
