# 📚 Document Q&A API (Phase 2 Capstone)

Đây là Capstone Project của Giai đoạn 2 trong lộ trình học AI Foundation. Dự án này là một hệ thống **RAG (Retrieval-Augmented Generation) backend** hoàn chỉnh, cho phép tải lên các tài liệu thật (PDF, DOCX) và đặt câu hỏi dựa trên nội dung của chúng.

## 🚀 Tính năng chính

- **Xử lý tài liệu:** Trích xuất text từ file PDF và DOCX.
- **Chunking thông minh:** Sử dụng `RecursiveCharacterTextSplitter` chia nhỏ văn bản, giữ nguyên ngữ cảnh qua cấu hình overlap.
- **Vector Search:** Lưu trữ và tìm kiếm vector siêu tốc bằng **Qdrant**.
- **Metadata Filtering:** Tìm kiếm thông tin chính xác theo từng tài liệu được chỉ định (`doc_id`).
- **Hybrid Search:** Kết hợp Dense Vector Search (semantic) và Sparse Vector Search (BM25 keyword search) để tối ưu kết quả (Optional).
- **RAG Generation:** Kết hợp context lấy từ tài liệu với LLM (Groq/Anthropic) để trả lời chính xác, hạn chế hallucination. Có trả về trích dẫn nguồn (sources).
- **Streaming Response:** Hỗ trợ trả lời streaming từng token để nâng cao trải nghiệm người dùng (UX).

## 🛠️ Công nghệ sử dụng

- **Web Framework:** FastAPI, Pydantic
- **Vector Database:** Qdrant (Docker)
- **Document Parsing:** PyMuPDF (`fitz`), `python-docx`
- **Embedding:** `sentence-transformers` / `fastembed`
- **LLM API:** Groq hoặc Anthropic (quản lý qua `.env`)

---

## 🌐 API Endpoints

Hệ thống cung cấp các endpoint sau (có thể xem chi tiết trên giao diện Swagger UI tại `/docs` sau khi chạy server):

| Endpoint | Method | Mô tả chức năng |
|---|---|---|
| `/health` | `GET` | Kiểm tra trạng thái server, trả về version và embedding model đang cấu hình. |
| `/documents/upload` | `POST` | Upload file PDF/DOCX. Hệ thống tự parse text, chunk, embed và lưu vào Qdrant. Trả về `doc_id`. |
| `/documents` | `GET` | Liệt kê danh sách các tài liệu đã được upload và index (gồm tên file, ID, thời gian tạo). |
| `/documents/{doc_id}` | `DELETE` | Xóa một tài liệu cụ thể và dọn dẹp toàn bộ vector liên quan khỏi Qdrant. |
| `/ask` | `POST` | Đặt câu hỏi. Cần truyền `question`. Có thể truyền `doc_id` (optional) để filter tài liệu cụ thể. Trả về `answer` kèm `sources`. |
| `/ask/stream` | `POST` | Tương tự `/ask` nhưng trả về dạng Server-Sent Events (SSE) streaming. |

---

## ⚙️ Cài đặt & Chạy dự án

### 1. Cài đặt thư viện
Hãy đảm bảo bạn đã tạo môi trường ảo (virtualenv) và cài đặt các thư viện cần thiết:
```bash
pip install fastapi uvicorn python-dotenv pydantic groq
pip install pymupdf python-docx langchain-text-splitters sentence-transformers qdrant-client fastembed
```

### 2. Cấu hình môi trường (.env)
Tạo file `.env` ở thư mục gốc của project (cùng cấp với `main.py`) và điền các thông tin:
```env
LLM_API_KEY="sk-..."
QDRANT_HOST="localhost"
QDRANT_PORT=6333
CHUNK_SIZE=500
CHUNK_OVERLAP=50
EMBEDDING_MODEL="all-MiniLM-L6-v2"
```

### 3. Chạy Qdrant Vector Database
Bạn cần chạy Qdrant bằng Docker ở background:
```bash
docker run -d -p 6333:6333 qdrant/qdrant
```
*Truy cập bảng điều khiển của Qdrant tại: http://localhost:6333/dashboard*

### 4. Chạy API Server
```bash
uvicorn main:app --reload
```

Sau khi chạy xong, truy cập vào **http://127.0.0.1:8000/docs** để test trực tiếp các API.

---

## 📂 Cấu trúc thư mục

```text
phase_2_capstone/
├── main.py              # Entry point FastAPI, định nghĩa routes
├── config.py            # Quản lý settings từ file .env
├── models.py            # Pydantic schemas (Request/Response)
├── document_parser.py   # Logic đọc file PDF/DOCX (PyMuPDF, docx)
├── chunker.py           # Logic cắt text (RecursiveCharacterTextSplitter)
├── embedder.py          # Wrapper cho SentenceTransformer
├── vector_store.py      # Giao tiếp với Qdrant (upsert, search, delete)
├── rag_pipeline.py      # Nối kết retrieval context và gọi LLM API
├── readme.MD            # Tài liệu dự án (bạn đang đọc)
└── .env                 # (Không commit) Các biến môi trường
```
