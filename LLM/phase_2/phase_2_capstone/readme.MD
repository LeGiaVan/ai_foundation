
### Ngày 13-15: Capstone Project — "Document Q&A API"

Đây là bài kiểm tra thực sự của Giai đoạn 2.

**Yêu cầu chức năng:**

| Endpoint | Method | Mô tả |
|---|---|---|
| `GET /health` | GET | Server status, trả về version và embedding model đang dùng |
| `POST /documents/upload` | POST | Upload PDF/DOCX, index vào Qdrant, trả về `doc_id` và số chunks |
| `GET /documents` | GET | Liệt kê tất cả tài liệu đã index (tên file, số chunks, thời gian upload) |
| `DELETE /documents/{doc_id}` | DELETE | Xóa tài liệu và tất cả vectors liên quan khỏi Qdrant |
| `POST /ask` | POST | Nhận `question` + optional `doc_id` filter, trả về `answer` + `sources` |
| `POST /ask/stream` | POST | Giống `/ask` nhưng streaming response |

**Yêu cầu kỹ thuật:**
- Toàn bộ request/response dùng **Pydantic model** với description rõ ràng trên Swagger
- API key đọc từ **`.env`**, không hardcode
- Qdrant client được inject qua **`Depends()`**
- Chunking dùng `RecursiveCharacterTextSplitter` với `chunk_size` và `overlap` cấu hình được từ `.env`
- **Metadata filtering** theo `doc_id` trong endpoint `/ask`
- Nếu LLM không tìm thấy câu trả lời trong context, phải nói rõ **"Không có trong tài liệu"** thay vì hallucinate
- `sources` trong response có: `text` (snippet), `doc_name`, `score`, `page_number` nếu có
- **Hybrid search** (dense + sparse) trong endpoint `/ask`
- Xử lý lỗi rõ ràng: file không hỗ trợ, Qdrant không kết nối được, LLM API lỗi → HTTP error code phù hợp

**Không bắt buộc nhưng nên thử:**
- Thêm reranking vào pipeline `/ask`
- Cache embedding của query phổ biến
- Endpoint `POST /ask/evaluate` nhận expected_answer để tính faithfulness score tự động

**Cấu trúc thư mục gợi ý:**
```
phase_2/
├── capstone/
│   ├── main.py              # FastAPI app, router
│   ├── config.py            # Settings từ .env (BaseSettings)
│   ├── models.py            # Pydantic models
│   ├── document_parser.py   # extract_text_from_pdf/docx
│   ├── chunker.py           # chunking logic
│   ├── embedder.py          # embedding model wrapper
│   ├── vector_store.py      # Qdrant operations (upsert, search, delete)
│   ├── rag_pipeline.py      # RAG logic (retrieve + LLM)
│   └── .env
└── plan.md
```

---
