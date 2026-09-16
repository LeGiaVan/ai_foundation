# LangChain Cheatsheet

## 1️⃣ Các hàm & cú pháp thường dùng

| Chức năng | Lớp / Hàm | Mẫu cú pháp | Ghi chú |
|-----------|-----------|--------------|--------|
| **LLM** | `ChatOpenAI`, `ChatAnthropic`, `ChatGroq` | `llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)` | Thay `model` tùy nhà cung cấp. |
| **Prompt** | `PromptTemplate` | `prompt = PromptTemplate.from_template("Trả lời: {question}")` | `{var}` → biến. |
| **LLMChain** | `LLMChain` | `chain = LLMChain(llm=llm, prompt=prompt); chain.run(question="LangChain là gì?")` | Kết hợp LLM + Prompt. |
| **RAG (RetrievalQA)** | `RetrievalQA` | `qa = RetrievalQA.from_chain_type(llm, retriever=vstore.as_retriever(k=4)); qa.invoke({"query": "RAG là gì?"})` | Tự động tạo chuỗi hỏi‑đáp. |
| **Vector Store** | `FAISS`, `Qdrant`, `Chroma` | `vstore = FAISS.from_documents(docs, embeddings); retriever = vstore.as_retriever(search_kwargs={"k": 5})` | `as_retriever` trả về `BaseRetriever`. |
| **Memory** | `ConversationBufferMemory` | `memory = ConversationBufferMemory(k=6)` | Lưu 6 tin nhắn cuối. |
| **Tool / Agent** | `Tool`, `initialize_agent` | `tool = Tool(name="calc", func=lambda x,y:x+y, description="Cộng 2 số"); agent = initialize_agent([tool], llm, agent="zero-shot-react-description")` | LLM gọi hàm Python. |
| **LangGraph** | `StateGraph` | `graph = StateGraph(router); graph.add_node("agent", agent); graph.set_entry_point("agent"); app = graph.compile()` | Workflow có trạng thái. |
| **Streaming** | `StreamingStdOutCallbackHandler` | `handler = StreamingStdOutCallbackHandler(); llm = ChatOpenAI(streaming=True, callbacks=[handler])` | Kết quả trả về từng token. |
| **LangSmith** | `traceable` decorator | `@traceable()\ndef my_chain(...): ...` | Theo dõi chi phí, latency. |
| **Evaluation** | `ChatEvaluationChain` | `eval_chain = ChatEvaluationChain.from_llm(llm); eval_chain.evaluate({"question": "...", "answer": "...", "ground_truth": "..."})` | Đánh giá chất lượng. |
| **Document Loader** | `TextLoader`, `PDFMinerLoader`, `CSVLoader` | `loader = TextLoader("data.txt"); docs = loader.load_and_split()` | Hỗ trợ nhiều định dạng. |
| **Text Splitter** | `RecursiveCharacterTextSplitter` | `splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50); chunks = splitter.split_documents(docs)` | Tách tài liệu cho vector store. |

## 2️⃣ Hướng mở rộng & ví dụ nhanh

### 🔹 1. Agent + Tools (Gọi hàm / Công cụ)
Tạo bot có khả năng dùng Tools để tính toán, gọi API hoặc truy vấn dữ liệu.

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# 1. Định nghĩa tool với decorator @tool
@tool
def add(a: int, b: int) -> int:
    """Cộng 2 số nguyên a và b."""
    return a + b

tools = [add]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2. Khởi tạo ReAct Agent (chuẩn hiện đại với LangGraph prebuilt)
agent = create_react_agent(llm, tools)
res = agent.invoke({"messages": [("user", "Tính 12 + 7 bằng bao nhiêu?")]})
print(res["messages"][-1].content)
```

---

### 🔹 2. RAG với Multi-Vector Retriever
Dùng nhiều vector / embedding (tách riêng summary và full document) để tăng độ chính xác tìm kiếm.

```python
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_openai import OpenAIEmbeddings

# Vectorstore lưu tóm tắt (summary vectors), docstore lưu nội dung gốc đầy đủ
byte_store = InMemoryByteStore()
vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings())

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=byte_store,
    id_key="doc_id"
)
```

---

### 🔹 3. LangGraph Workflow (Đồ thị có State)
Xây dựng pipeline dạng đồ thị trạng thái: Retrieve → Rerank → Generate.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class RAGState(TypedDict):
    query: str
    docs: list[str]
    answer: str

def retrieve(state: RAGState):
    return {"docs": ["Tài liệu 1", "Tài liệu 2"]}

def generate(state: RAGState):
    return {"answer": f"Đã trả lời câu hỏi: '{state['query']}'"}

# Khởi tạo Graph
workflow = StateGraph(RAGState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# Định nghĩa luồng chạy
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
output = app.invoke({"query": "LangGraph hoạt động như thế nào?"})
print(output["answer"])
```

---

### 🔹 4. Streaming Token (Thời gian thực)
Truyền kết quả từng token về UI / WebSocket ngay khi LLM đang sinh văn bản.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
prompt = ChatPromptTemplate.from_template("Giải thích ngắn gọn: {topic}")
chain = prompt | llm

# Stream trực tiếp từng chunk token
for chunk in chain.stream({"topic": "Tại sao cần Streaming?"}):
    print(chunk.content, end="", flush=True)
```

---

### 🔹 5. LangSmith Tracing (Giám sát & Debug)
Ghi lại chi tiết execution trace, latency, token usage và cost lên dashboard LangSmith.

```python
import os
from langsmith import traceable

# Cấu hình biến môi trường
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_pt_your_api_key_here"

@traceable(name="My RAG Pipeline")
def rag_pipeline(query: str):
    docs = retriever.invoke(query)
    response = chain.invoke({"query": query, "context": docs})
    return response
```

---

### 🔹 6. Evaluation (Đánh giá chất lượng mô hình)
Chạy bộ test dataset đánh giá đầu ra (độ chính xác, ground truth) bằng LLM-as-a-judge.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

eval_prompt = ChatPromptTemplate.from_template("""
Bạn là giám khảo chấm điểm. So sánh câu trả lời của AI với đáp án chuẩn:
- Câu hỏi: {question}
- AI trả lời: {answer}
- Đáp án chuẩn: {ground_truth}

Hãy chấm điểm từ 1 đến 10 và giải thích ngắn gọn:
""")

eval_chain = eval_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0)

score = eval_chain.invoke({
    "question": "RAG viết tắt của gì?",
    "answer": "Retrieval-Augmented Generation",
    "ground_truth": "Retrieval-Augmented Generation"
})
print(score.content)
```

---

💡 **Tips nhanh**
- Dùng `from_xxx import *` chỉ khi cần, tránh import thừa.
- Đặt `k` (số tài liệu trả về) từ 4‑8 cho RAG, tùy vào độ dài tài liệu.
- Khi dùng `Memory`, bật `return_messages=True` để duy trì ngữ cảnh đầy đủ.
- `langchain-community` chứa các loader, vectorstore và utils bổ sung.

> Nếu cần bổ sung phần nào hoặc muốn ví dụ chi tiết hơn, cứ nhắn nhé!
