import os
from operator import itemgetter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load env vars
load_dotenv()

# SỬA LỖI 1: Cú pháp os.getenv đúng (Tham số 1 là tên biến, tham số 2 là giá trị mặc định)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_docs")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in environment")

# Initialise Qdrant vector store and retriever
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
qdrant_client = QdrantClient(url=QDRANT_URL)
vector_store = QdrantVectorStore(
    client=qdrant_client, 
    collection_name=QDRANT_COLLECTION,
    embedding=embeddings,
    vector_name="dense",
    content_payload_key="text",      # Bổ sung dòng này để LangChain biết chỗ lấy chữ
    metadata_payload_key="metadata"  # Bổ sung dòng này để lấy siêu dữ liệu
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Prompt template – Ép LLM trả về JSON thuần túy
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the user's question using the provided context.\n\nYou MUST return a valid JSON object with EXACTLY three fields: \"answer\" (string), \"confidence\" (float between 0 and 1), and \"sources\" (list of source strings). Do not include any markdown formatting like ```json."),
    MessagesPlaceholder(variable_name="history"), # BỔ SUNG DÒNG NÀY ĐỂ CHÈN LỊCH SỬ
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# Structured output parser
parser = JsonOutputParser()

# Helper to format retrieved documents into a single string
def format_docs(docs: list[Document]) -> str:
    parts = []
    for d in docs:
        title = d.metadata.get("title") or d.metadata.get("source") or "unknown"
        parts.append(f"Source: {title}\n{d.page_content}")
    return "\n\n".join(parts)

# Build the LCEL chain
# SỬA LỖI 2 & 3: Dùng itemgetter để lấy đúng chuỗi "question" từ dict; Đổi tên model Groq hợp lệ; Thêm bind response_format
rag_chain = (
    {
        "context": itemgetter("question") | retriever | RunnableLambda(format_docs),
        "question": itemgetter("question"),
        "history": itemgetter("history") # BỔ SUNG DÒNG NÀY
    }
    | prompt
    | ChatGroq(model="groq/compound-mini", api_key=GROQ_API_KEY, temperature=0).bind(response_format={"type": "json_object"})
    | parser
)

# In‑memory store for per‑session memory objects
_memory_store: dict[str, ConversationBufferWindowMemory] = {}

def get_memory(session_id: str) -> ConversationBufferWindowMemory:
    if session_id not in _memory_store:
        _memory_store[session_id] = ConversationBufferWindowMemory(k=6, return_messages=True)
    return _memory_store[session_id]

import uuid
SERVER_SESSION_ID = str(uuid.uuid4())

app = FastAPI()

class ChatRequest(BaseModel):
    question: str = Field(..., description="User's question")

class ChatResponse(BaseModel):
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources: list[str]

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    memory = get_memory(SERVER_SESSION_ID)
    
    # 1. Lấy lịch sử chat từ bộ nhớ
    chat_history = memory.load_memory_variables({})["history"]
    
    try:
        # 2. Truyền thêm history vào invoke
        result = rag_chain.invoke({
            "question": req.question,
            "history": chat_history
        })
    except Exception as e:
        print(f"Error details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    if not isinstance(result, dict) or not all(k in result for k in ("answer", "confidence", "sources")):
        raise HTTPException(status_code=500, detail="LLM did not return the expected JSON structure")

    # 3. Lưu ngữ cảnh mới vào bộ nhớ
    memory.save_context({"input": req.question}, {"output": result["answer"]})

    return ChatResponse(**result)