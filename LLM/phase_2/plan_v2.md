Viewed system_design_mindset.md:1-80

**Kế hoạch Phase 2 – RAG Chuyên Sâu (với tư duy Software Architect)**  

---  

## 1. Mục tiêu tổng thể  
- **Xây dựng hệ thống RAG production‑grade**:  
  1. **Parsing → Chunking → Embedding → Vector store** (indexing, chỉ chạy một lần).  
  2. **Query → Retrieve → (optional) Rerank → LLM** (truy vấn, thực thi cho mỗi request).  
- **Kiến trúc**: Clean Architecture + Design Patterns (Strategy, Adapter, Repository, Factory) → **Loose‑coupling**, **testable**, **scalable**.  

## 2. Bản đồ kiến thức – RAG Pipeline (Architect View)

```
[Tài liệu thô: PDF/DOCX/TXT]
        ↓
[1. Document Parsing — trích xuất text thuần]
        ↓
[2. Chunking — chia nhỏ thành đoạn vừa phải]
        ↓
[3. Embedding — chuyển mỗi chunk thành vector số]
        ↓
[4. Lưu vào Vector Database (Qdrant)]
        ↓  ← Đây là quá trình INDEXING (làm 1 lần)

[Câu hỏi của người dùng]
        ↓
[5. Embed câu hỏi → query vector]
        ↓
[6. Vector Search — tìm top‑k chunks gần nhất]
        ↓
[7. Reranking (tùy chọn) — sắp xếp lại kết quả]
        ↓
[8. Đưa chunks vào context của LLM → sinh câu trả lời]
        ↓
[Câu trả lời có nguồn gốc từ tài liệu thật]
        ↓  ← Đây là quá trình QUERYING (mỗi câu hỏi làm lại)
```

---  

## 3. Tuần 4 – Document Parsing + Chunking  

| Ngày | Chủ đề | Kiến trúc / Pattern | Bài tập chính |
|------|-------|---------------------|---------------|
| **1** | **Document Parsing** | **Interface Segregation (ISP) + Dependency Inversion** – định nghĩa `BaseDocumentParser` (ABC). Implement `PyMuPDFParser` & `DocxParser`. | 1. `interfaces.py` với `BaseDocumentParser`. <br>2. Factory `get_parser(ext)` trả về đúng parser. |
| **2‑3** | **Chunking** | **Strategy Pattern** – `BaseChunker` → `FixedSizeChunker`, `RecursiveChunker`, `SemanticChunker`. Cấu hình qua `.env`. | 1. Cài `langchain-text-splitters`. <br>2. Viết `fixed_size_chunk` thủ công. <br>3. So sánh với `RecursiveCharacterTextSplitter`. <br>4. Thử `chunk_size/overlap` và đo trade‑off. |
| **3** | **Semantic Chunking** | **Trade**Kế hoạch Phase 2 – RAG Chuyên Sâu (với tư duy Software Architect)**  

---  

## 1. Mục tiêu tổng thể  
- **Xây dựng hệ thống RAG production‑grade**:  
  1. **Parsing → Chunking → Embedding → Vector store** (indexing, chỉ chạy một lần).  
  2. **Query → Retrieve → (optional) Rerank → LLM** (truy vấn, thực thi cho mỗi request).  
- **Kiến trúc**: Clean Architecture + Design Patterns (Strategy, Adapter, Repository, Factory) → **Loose‑coupling**, **testable**, **scalable**.  

## 2. Bản đồ kiến thức – RAG Pipeline (Architect View)

```
[Tài liệu thô: PDF/DOCX/TXT]
        ↓
[1. Document Parsing — trích xuất text thuần]
        ↓
[2. Chunking — chia nhỏ thành đoạn vừa phải]
        ↓
[3. Embedding — chuyển mỗi chunk thành vector số]
        ↓
[4. Lưu vào Vector Database (Qdrant)]
        ↓  ← Đây là quá trình INDEXING (làm 1 lần)

[Câu hỏi của người dùng]
        ↓
[5. Embed câu hỏi → query vector]
        ↓
[6. Vector Search — tìm top‑k chunks gần nhất]
        ↓
[7. Reranking (tùy chọn) — sắp xếp lại kết quả]
        ↓
[8. Đưa chunks vào context của LLM → sinh câu trả lời]
        ↓
[Câu trả lời có nguồn gốc từ tài liệu thật]
        ↓  ← Đây là quá trình QUERYING (mỗi câu hỏi làm lại)
```

---  

## 3. Tuần 4 – Document Parsing + Chunking  

| Ngày | Chủ đề | Kiến trúc / Pattern | Bài tập chính |
|------|-------|---------------------|---------------|
| **1** | **Document Parsing** | **Interface Segregation (ISP) + Dependency Inversion** – định nghĩa `BaseDocumentParser` (ABC). Implement `PyMuPDFParser` & `DocxParser`. | 1. `interfaces.py` với `BaseDocumentParser`. <br>2. Factory `get_parser(ext)` trả về đúng parser. |
| **2‑3** | **Chunking** | **Strategy Pattern** – `BaseChunker` → `FixedSizeChunker`, `RecursiveChunker`, `SemanticChunker`. Cấu hình qua `.env`. | 1. Cài `langchain-text-splitters`. <br>2. Viết `fixed_size_chunk` thủ công. <br>3. So sánh với `RecursiveCharacterTextSplitter`. <br>4. Thử `chunk_size/overlap` và đo trade‑off. |
| **3** | **Semantic Chunking** | **Trade‑off analysis** – khi nên dùng Semantic (độ chính xác cao, tốc độ chậm). | 1. Cài `langchain-community` + `sentence‑transformers`. <br>2. So sánh thời gian và chất lượng output. |

## 4. Tuần 5 – Embedding & Vector DB  

| Ngày | Chủ đề | Kiến trúc / Pattern | Bài tập |
|------|-------|---------------------|--------|
| **4** | **Embedding Model** | **Adapter Pattern** – `BaseEmbedder` → `LocalSentenceEmbedder`, `OpenAIEmbedder`, `GeminiEmbedder`. | 1. Embed 5 câu, tính cosine similarity. |
| **5** | **Qdrant (Vector DB)** | **Repository Pattern** – `BaseVectorStore` → `QdrantVectorStore`. | 1. Docker `qdrant/qdrant`. <br>2. Upsert vectors + payload. <br>3. Thực hiện một truy vấn top‑k. |

## 5. Tuần 6 – FastAPI Integration (Non‑blocking, Resilient)  

| Ngày | Chủ đề | Kiến trúc / Pattern | Bài tập |
|------|-------|---------------------|--------|
| **6** | **Retrieval Quality** | Đánh giá **Precision@k**, **False‑Positive/Negative**. | Thay đổi `chunk_size`, đo Precision@3. |
| **7** | **FastAPI RAG Endpoint** | **Rate‑limit + Semaphore** (giới hạn đồng thời LLM calls). <br>**Error handling** – `gather(return_exceptions=True)`. <br>**Dependency Injection** – `Depends()` để inject `VectorStore`, `Embedder`. | Tạo `POST /ask` trả về `answer` + `sources`. |
| **8** | **Upload File** | **Asynchronous Background Processing** – `BackgroundTasks` (hoặc Celery). Trả về `202 Accepted`. | `POST /upload` nhận `UploadFile`, xử lý trong background. |
| **9‑10** | **Hybrid Search + Reranking** | **Trade‑off analysis** – latency vs accuracy. | 1. Cài `fastembed` cho BM25. <br>2. Implement Reranker (`CrossEncoder`). <br>3. Đo latency, lập bảng so sánh. |

## 6. Tuần 7 – Advanced RAG & Capstone  

| Ngày | Chủ đề | Kiến trúc / Pattern | Bài tập |
|------|-------|---------------------|--------|
| **11** | **Metadata Filtering** | **Payload Index** trên Qdrant (filter by `source_file`, `doc_id`). | Nâng cấp indexing pipeline để lưu `source_file`, `page`, `chunk_index`. |
| **12** | **RAGAS Evaluation** | **Data‑driven improvement** – 4 chiều: Faithfulness, Answer Relevancy, Context Precision, Context Recall. | Cài `ragasas`, tạo test‑set 10 câu hỏi, chạy evaluation, đề xuất cải tiến. |
| **13‑15** | **Capstone – Production‑grade Document Q&A API** | **Clean Architecture** (Controller‑Service‑Repository). <br>**Resilience** (Semaphore, Retry). <br>**Background Jobs** (upload → indexing). <br>**Streaming** (`/ask/stream`). | Xây **capstone/** với: `main.py`, `config.py`, `models.py`, `document_parser.py`, `chunker.py`, `embedder.py`, `vector_store.py`, `rag_pipeline.py`. |

### Điểm nhấn kiến trúc Clean Architecture (capstone)

```
capstone/
├─ api/            # FastAPI routers (Controllers)
├─ core/           # Config, ABC interfaces, Exceptions
├─ services/       # Business logic (RAG pipeline, workflow)
├─ infrastructure/ # QdrantVectorStore, GroqLLMClient, PyMuPDFParser
└─ schemas/        # Pydantic request/response models
```

- **Controllers** chỉ nhận request, inject dependencies qua `Depends()`.  
- **Services** thực hiện chuỗi: retrieve → (optional) rerank → LLM.  
- **Infrastructure** chịu trách nhiệm giao tiếp với bên ngoài (Qdrant, LLM, file parsers).  

## 7. Định nghĩa “Done” (DOD) – Kiểm tra trước khi lên Phase 3  

- [ ] **Parsing**: PDF/DOCX → text (đúng nội dung).  
- [ ] **Chunking**: Hiểu *fixed‑size*, *recursive*, *semantic*; áp dụng `chunk_overlap`.  
- [ ] **Embedding**: Giải thích dense vs sparse, cosine similarity.  
- [ ] **Vector DB**: Qdrant chạy, upsert, search, payload index.  
- [ ] **FastAPI**: Endpoints `/upload`, `/ask`, `/ask/stream` hoạt động, non‑blocking, rate‑limited.  
- [ ] **Hybrid + Rerank**: Có thể bật/tắt, đo latency.  
- [ ] **Metadata filtering**: `doc_id` filter hoạt động.  
- [ ] **RAGAS**: Có báo cáo 4 chiều, đưa ra cải tiến.  
- [ ] **Clean Architecture**: Thư mục chuẩn, các lớp phụ thuộc đúng.  

---  

## 8. Tham khảo nhanh (từ `system_design_mindset.md`)  

- **Non‑Blocking & Async** – luôn `await` các I/O, không dùng sync lib trong `async def`.  
- **Rate‑limit & Semaphore** – bảo vệ API bên thứ 3, tránh lỗi 429.  
- **Stateless** – lưu trạng thái (chat history, uploads) trong DB/Cache, không trong RAM.  
- **Design for Failure** – retry + graceful degradation + `gather(return_exceptions=True)`.  
- **UX** – streaming, progress bar, WebSocket để báo tiến độ dài.  

---  

### Hành động tiếp theo  

1. **Bắt đầu Ngày 1**: Tạo `interfaces.py` với `BaseDocumentParser` và các parser mẫu.  
2. **Cài đặt môi trường** (Docker, pip) theo phần “Thư viện sẽ dùng trong Giai đoạn 2”.  

Nếu bạn muốn **bắt đầu ngay** với một file cụ thể (ví dụ `interfaces.py`), hoặc cần **chi tiết code mẫu** cho một phần (Chunker, Embedder, v.v.), hãy cho tôi biết để tôi tạo nhanh các template. 

---  

*Bạn có muốn tôi tạo template cho `BaseDocumentParser` và các lớp parser ngay bây giờ không?*