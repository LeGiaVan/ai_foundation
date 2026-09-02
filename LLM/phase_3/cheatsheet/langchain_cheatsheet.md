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

| Hướng | Mô tả | Mini‑example |
|------|------|--------------|
| **Agent + Tools** | Tạo bot có thể thực hiện tính toán, truy vấn DB, v.v. | ```python\nfrom langchain.agents import initialize_agent, Tool\n\ndef calc(a,b): return a+b\n\ntool = Tool(name="calc", func=calc, description="Cộng 2 số")\nagent = initialize_agent([tool], llm, agent="zero-shot-react-description")\nres = agent.run("Cộng 12 và 7")\nprint(res)\n``` |
| **RAG với multi‑vector** | Dùng nhiều embedding (text + metadata) để nâng độ chính xác. | ```python\nfrom langchain_community.vectorstores import Qdrant\nemb_text = OpenAIEmbeddings(); emb_meta = CohereEmbeddings()\nvec = Qdrant.from_documents(docs, emb_text, collection_name="texts")\nmeta_vec = Qdrant.from_documents(meta_docs, emb_meta, collection_name="meta")\nretriever = MultiVectorRetriever(vectorstore=vec, metadata_store=meta_vec)\nqa = RetrievalQA.from_chain_type(llm, retriever=retriever)\n``` |
| **LangGraph workflow** | Xây dựng pipeline: Retrieve → Rerank → Generate → Post‑process. | ```python\nfrom langgraph.graph import StateGraph\ndef retrieve(state): ...\ndef rerank(state): ...\ndef generate(state): ...\nworkflow = StateGraph(lambda s:s)\nworkflow.add_node("retrieve", retrieve)\nworkflow.add_node("rerank", rerank)\nworkflow.add_node("gen", generate)\nworkflow.add_edge("retrieve","rerank")\nworkflow.add_edge("rerank","gen")\napp = workflow.compile()\napp.invoke({"query":"Giải thích RAG"})\n``` |
| **Streaming + UI** | Kết hợp `streaming=True` và WebSocket để hiển thị câu trả lời từng token. | ```python\nllm = ChatOpenAI(streaming=True, callbacks=[WebSocketCallback()])\nchain = LLMChain(llm=llm, prompt=prompt)\n# client receives chunks via WS\n``` |
| **LangSmith tracing** | Ghi lại toàn bộ pipeline, xem cost & latency trên dashboard. | ```python\nfrom langsmith import traceable\n@traceable()\ndef rag_pipeline(query):\n    docs = retriever.get_relevant_documents(query)\n    answer = llm.invoke(... )\n    return answer\n``` |
| **Evaluation loop** | Tự động tạo dataset đánh giá, chạy `ChatEvaluationChain`, thu thập score. | ```python\nevals = []\nfor q,a,g in test_set:\n    score = eval_chain.evaluate({"question":q,"answer":a,"ground_truth":g})\n    evals.append(score)\nprint(sum(evals)/len(evals))\n``` |

---

💡 **Tips nhanh**
- Dùng `from_xxx import *` chỉ khi cần, tránh import thừa.
- Đặt `k` (số tài liệu trả về) từ 4‑8 cho RAG, tùy vào độ dài tài liệu.
- Khi dùng `Memory`, bật `return_messages=True` để duy trì ngữ cảnh đầy đủ.
- `langchain-community` chứa các loader, vectorstore và utils bổ sung.

> Nếu cần bổ sung phần nào hoặc muốn ví dụ chi tiết hơn, cứ nhắn nhé!
