import os
# Lưu ý: Cần đảm bảo bạn đã cài đặt các thư viện:
# pip install langchain-qdrant langchain-huggingface

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# API Key của bạn
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_langchain_api_key_here" # Dán key của bạn vào đây
os.environ["LANGCHAIN_PROJECT"] = "Hoc_LangChain"                  # Đặt tên project tùy ý
print("="*50)
print("BƯỚC 1: Kết nối Qdrant và tạo Retriever")
# 1. Định nghĩa embedding model y hệt như lúc bạn index ở Phase 2
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Kết nối tới Qdrant local
client = QdrantClient(host="localhost", port=6333)

# 3. Wrap collection "my_docs" thành LangChain Vector Store
vector_store = QdrantVectorStore(
    client=client,
    collection_name="rag_docs",
    embedding=embeddings,
    vector_name="dense", # Trỏ đúng vào named vector 'dense' của collection
    content_payload_key="text" # Sửa lỗi "black box": Báo cho Langchain biết text nằm ở key "text" (vì mặc định nó tìm key "page_content")
)

# 4. Biến thành Retriever, lấy top 5 chunks
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
print("=> Đã tạo xong retriever!\n")


print("="*50)
print("BƯỚC 2 & 3: Xây RAG Chain bằng LCEL (hỗ trợ sources)")
model = ChatGroq(model="groq/compound-mini", temperature=0)

# Hàm helper gom text từ documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Cú pháp LCEL để vừa lấy context vừa lấy docs gốc làm trích dẫn (sources)
setup_and_retrieval = RunnableParallel({
    "context": retriever | format_docs, # Lấy text ghép lại để đưa vào LLM
    "docs": retriever,                  # Giữ nguyên object Document để trích xuất metadata
    "question": RunnablePassthrough()   # Nhận trực tiếp câu hỏi đầu vào
})

prompt = PromptTemplate.from_template(
    "Dựa vào các thông tin trong ngoặc kép dưới đây, hãy trả lời câu hỏi.\n\n"
    "Thông tin (Context):\n\"\"\"\n{context}\n\"\"\"\n\n"
    "Câu hỏi: {question}\n\n"
    "Trả lời:"
)

# Hàm assign sẽ nhận dict từ bước 1, chạy qua LLM và gán kết quả vào key 'answer'
rag_chain = setup_and_retrieval.assign(
    answer = prompt | model | StrOutputParser()
)
print("=> Đã build RAG chain LCEL thành công!\n")


print("="*50)
print("BƯỚC 4: Bật Streaming và lấy kết quả trích dẫn")
question = "OCR được dùng ở đâu trong dự án này?"

print(f"Câu hỏi: {question}")
print("Trả lời: ", end="")

# Chạy chain ở chế độ stream
sources = []
for chunk in rag_chain.stream(question):
    # Vì rag_chain trả về 1 dict, hàm stream sẽ sinh ra các phần nhỏ cập nhật
    
    # Nếu chunk chứa nội dung câu trả lời (answer) đang được sinh ra, ta in ra màn hình
    if "answer" in chunk:
        print(chunk["answer"], end="", flush=True)
        
    # Nếu chunk chứa tài liệu retrieved (docs), ta trích xuất lấy thông tin source file
    if "docs" in chunk:
        # Lấy metadata 'source' (loại bỏ trùng lặp bằng set)
        sources = list(set([doc.metadata.get("source", "Không rõ nguồn") for doc in chunk["docs"]]))

print("\n\n" + "-"*30)
print("NGUỒN TRÍCH DẪN (Sources):")
for idx, s in enumerate(sources, 1):
    print(f"{idx}. {s}")
