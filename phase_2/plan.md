# Giai đoạn 2 — Tuần 4-6: RAG Chuyên Sâu

## Mục tiêu tổng thể

Sau giai đoạn này, bạn có thể tự xây **1 hệ thống RAG (Retrieval-Augmented Generation) hoàn chỉnh từ đầu đến cuối**: đọc tài liệu PDF/DOCX, chia nhỏ thành chunks thông minh, chuyển thành vector, lưu vào vector database (Qdrant), và tích hợp vào FastAPI để LLM có thể trả lời câu hỏi dựa trên tài liệu thật — chứ không phải chỉ dựa vào kiến thức đã học sẵn.

**Sản phẩm đầu ra (capstone project):** 1 "Document Q&A API" — người dùng upload tài liệu, rồi đặt câu hỏi, API trả lời dựa trên nội dung tài liệu đó. Chi tiết ở cuối tài liệu.

## Tiền đề (đã học ở Giai đoạn 1)

FastAPI (endpoint, Pydantic, Depends, StreamingResponse), Anthropic/Groq API (messages, system prompt, structured output, streaming), prompt engineering cơ bản (.env, type hint, async/await). Nếu quên, ôn lại nhanh trước — đặc biệt phần Pydantic model và Depends() sẽ dùng lại xuyên suốt giai đoạn này.

---

## Bản đồ kiến thức tổng quan — RAG Pipeline

Trước khi bắt đầu từng ngày, hãy đọc và hiểu sơ đồ này. Mỗi bước tương ứng với 1-2 ngày học:

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
[6. Vector Search — tìm top-k chunks gần nhất]
        ↓
[7. Reranking (tùy chọn) — sắp xếp lại kết quả]
        ↓
[8. Đưa chunks vào context của LLM → sinh câu trả lời]
        ↓
[Câu trả lời có nguồn gốc từ tài liệu thật]
        ↓  ← Đây là quá trình QUERYING (mỗi câu hỏi làm lại)
```

---

## TUẦN 4 — Document Parsing + Chunking

### Ngày 1: Document Parsing — đọc tài liệu từ file thật

**Mục tiêu:** Trích xuất được text thuần túy từ file PDF và DOCX bằng Python.

**Khái niệm cần nắm:**
- **Document parsing** — quá trình đọc file nhị phân (PDF, DOCX...) và chuyển thành text thuần (plain text) mà Python có thể xử lý tiếp. Không thể gửi thẳng file PDF vào LLM — phải extract text ra trước.
- **PyMuPDF (fitz)** — thư viện phổ biến nhất để đọc PDF, nhanh hơn pdfplumber, hỗ trợ đọc theo từng page (trang)
- **python-docx** — thư viện đọc file `.docx` (Word), đọc theo từng `paragraph`
- Lưu ý thực tế: PDF scan (ảnh chụp) cần OCR riêng — nằm ngoài phạm vi bài này. Chỉ xử lý PDF có text thật (text-based PDF).

**Cài thư viện:**
```bash
pip install pymupdf python-docx
```

**Tài liệu đọc:**
- PyMuPDF docs: https://pymupdf.readthedocs.io/en/latest/tutorial.html (đọc phần "Opening a Document" và "Accessing Pages")
- python-docx: https://python-docx.readthedocs.io/en/latest/user/text.html (đọc phần "Accessing Text")

**Bài tập:**
1. Tìm hoặc tạo 1 file PDF thật (vd: export CV, hoặc tải PDF bất kỳ từ internet)
2. Viết hàm `extract_text_from_pdf(file_path: str) -> str` dùng `pymupdf`, lặp qua từng page, ghép text lại, trả về 1 chuỗi
3. Tìm hoặc tạo 1 file `.docx`, viết hàm `extract_text_from_docx(file_path: str) -> str` dùng `python-docx`, lặp qua `doc.paragraphs`, bỏ qua đoạn rỗng
4. Viết hàm `extract_text(file_path: str) -> str` gộp 2 hàm trên — tự detect loại file qua phần mở rộng (`.pdf` hay `.docx`), raise `ValueError` nếu không hỗ trợ
5. In ra vài trăm ký tự đầu của text trích xuất được — tự kiểm tra có đọc đúng không

**Tiêu chí hoàn thành:** Hàm `extract_text()` chạy được với cả PDF và DOCX, text trích xuất đọc được và đúng nội dung.

---

### Ngày 2: Chunking — Fixed-size và Recursive Character Splitting

**Mục tiêu:** Chia văn bản thành các đoạn nhỏ có kích thước kiểm soát được, có hiểu về `chunk_size` và `chunk_overlap`.

**Khái niệm cần nắm:**
- **Chunking** — tại sao cần chia nhỏ? Embedding model có giới hạn token đầu vào (thường 512-8192 tokens). Nếu đoạn văn quá dài, bị cắt bớt → mất thông tin → vector kém chất lượng → kết quả tìm kiếm sai. Nếu quá ngắn → thiếu ngữ cảnh, vector không đủ ý nghĩa.

- **Fixed-size chunking** — cách đơn giản nhất: cắt đều theo số ký tự (hoặc token) cố định. Nhanh, dễ cài, nhưng có thể cắt giữa câu/đoạn → mất ngữ nghĩa.

- **Chunk overlap** — giải pháp cho fixed-size: các chunk kề nhau được phép "chồng lên nhau" một phần. Đảm bảo câu bị cắt ở cuối chunk 1 vẫn xuất hiện đầy đủ ở chunk 2 → không mất ngữ cảnh liền mạch.
  ```
  chunk_size=100, chunk_overlap=20
  Chunk 1: ký tự 0-99
  Chunk 2: ký tự 80-179  ← overlap 20 ký tự với chunk 1
  Chunk 3: ký tự 160-259
  ```

- **Recursive Character Text Splitting** — thông minh hơn fixed-size: thử tách theo thứ tự ưu tiên `["\n\n", "\n", ".", " "]`. Trước tiên tách theo đoạn văn (`\n\n`), nếu đoạn nào vẫn còn dài quá thì tách theo dòng (`\n`), tiếp tục tách theo câu (`.`), cuối cùng mới tách theo từ (` `). Kết quả: chunk thường kết thúc ở ranh giới tự nhiên (cuối câu, cuối đoạn) thay vì cắt giữa từ.

**Tài liệu đọc (BẮT BUỘC đọc bài này):**
- Pinecone — "Chunking Strategies for LLM Applications": https://www.pinecone.io/learn/chunking-strategies/
  - Đọc các mục: "What is chunking?", "Why do we need chunking?", "Fixed size chunking", "Recursive character text splitting", "Figuring out the best chunking strategy"

**Bài tập:**
1. Cài LangChain text splitter: `pip install langchain-text-splitters`
2. Viết hàm `fixed_size_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]` **từ đầu, không dùng thư viện** — dùng vòng lặp và string slicing
3. So sánh output của hàm tự viết với `RecursiveCharacterTextSplitter` từ LangChain trên cùng 1 văn bản
4. Thử nghiệm: lấy text PDF từ Ngày 1, chunk với `chunk_size=200` và `chunk_overlap=20` → đếm số chunks tạo ra. Tăng lên `chunk_size=1000` → đếm lại → cảm nhận trade-off
5. Viết hàm `count_tokens_rough(text: str) -> int` ước lượng token bằng công thức đơn giản: `len(text.split()) * 1.3`

**Tiêu chí hoàn thành:** Hiểu và giải thích được bằng lời: "Tại sao overlap giúp ích? Khi nào dùng Recursive thay vì Fixed-size?"

---

### Ngày 3: Semantic Chunking + Chiến lược chọn chunk size

**Mục tiêu:** Hiểu cách chunking nâng cao dựa trên ngữ nghĩa (semantic similarity), biết cách chọn chiến lược chunking phù hợp với từng loại tài liệu.

**Khái niệm cần nắm:**
- **Semantic chunking** — thay vì cắt theo ký tự/câu cố định, cắt tại chỗ mà *ý nghĩa thay đổi*. Cách làm: embed từng câu riêng, tính cosine similarity giữa các câu liên tiếp, phát hiện "điểm gãy" (breakpoint) khi similarity giảm đột ngột → đó là ranh giới tự nhiên của chunk.
  - Ưu điểm: mỗi chunk là 1 ý hoàn chỉnh, chất lượng vector tốt hơn
  - Nhược điểm: chậm hơn nhiều (phải embed từng câu), kích thước chunk không đều

- **Khi nào dùng cái gì?**

| Tình huống | Chiến lược phù hợp |
|---|---|
| Tài liệu kỹ thuật, FAQ, có cấu trúc rõ | Recursive (cắt theo heading/đoạn) |
| Sách, báo cáo, văn xuôi liên tục | Recursive + overlap 10-20% |
| Tài liệu cần độ chính xác cao (pháp lý, y tế) | Semantic chunking |
| Prototype nhanh | Fixed-size đơn giản |

- **Quy tắc vàng khi chọn chunk size:** "Câu trả lời cho query thường nằm trong bao nhiêu token?" → chunk size ~ bằng độ dài đó. Thực tế phổ biến: 256-512 tokens cho semantic search, 512-1024 tokens cho Q&A.

**Tài liệu đọc:**
- Tiếp tục bài Pinecone — phần "Semantic Chunking" và "Figuring out the best chunking strategy for your application"
- LangChain docs — SemanticChunker: https://python.langchain.com/docs/how_to/semantic-chunker/

**Bài tập:**
1. Cài: `pip install langchain-community sentence-transformers`
2. Dùng `SemanticChunker` từ LangChain với embedding model `all-MiniLM-L6-v2`
3. So sánh output của `SemanticChunker` vs `RecursiveCharacterTextSplitter` trên cùng 1 đoạn văn có nhiều chủ đề khác nhau
4. Thiết kế 1 bảng so sánh cá nhân: với tài liệu bạn sẽ dùng trong capstone, chiến lược nào phù hợp? Viết ra lý do.

**Tiêu chí hoàn thành:** Có thể quyết định chiến lược chunking phù hợp với 1 loại tài liệu bất kỳ và giải thích lý do mà không cần tra cứu.

---

### Ngày 4: Embedding Model — Từ text thành vector số

**Mục tiêu:** Hiểu embedding là gì, tự tay embed 1 đoạn text, so sánh cosine similarity giữa các văn bản.

**Khái niệm cần nắm:**
- **Embedding** — quá trình chuyển đổi đoạn text thành 1 mảng số thực (vector) có chiều cố định (thường 384, 768, 1536 chiều...). Văn bản có ý nghĩa gần nhau → vector gần nhau trong không gian nhiều chiều.

- **Dense vector** — vector dày đặc: hầu hết các chiều đều có giá trị khác 0. Trái ngược với sparse vector (BM25) — phần lớn là số 0. Dense embedding tốt cho semantic search (tìm theo ý nghĩa), sparse tốt cho keyword search (tìm chính xác từ khóa).

- **Cosine similarity** — cách đo "khoảng cách" giữa 2 vector trong không gian n chiều. Giá trị từ -1 đến 1: 1 = giống nhau hoàn toàn, 0 = không liên quan, -1 = đối lập.
  ```python
  from numpy import dot
  from numpy.linalg import norm
  cosine_sim = dot(a, b) / (norm(a) * norm(b))
  ```

- **Lựa chọn embedding model:**

| Model | Chiều | Ưu điểm | Nhược điểm |
|---|---|---|---|
| `all-MiniLM-L6-v2` (sentence-transformers) | 384 | Miễn phí, nhanh, chạy local | Chỉ tiếng Anh tốt |
| `text-embedding-3-small` (OpenAI) | 1536 | Đa ngôn ngữ, chất lượng cao | Tốn phí |
| `nomic-embed-text` (via Ollama) | 768 | Miễn phí, đa ngôn ngữ | Cần cài thêm |

**Bài tập:**
1. Cài và dùng `sentence-transformers` (chạy local, miễn phí):
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer("all-MiniLM-L6-v2")
   vector = model.encode("Hello world")
   print(vector.shape)  # (384,)
   ```
2. Embed 5 câu: 3 câu về cùng chủ đề (vd: bóng đá), 2 câu về chủ đề khác (vd: nấu ăn). Tính cosine similarity giữa tất cả cặp → confirm: câu cùng chủ đề similarity cao hơn
3. Embed 1 câu hỏi ("Làm thế nào để nấu phở?") và 3 đoạn text. Sắp xếp theo cosine similarity → đây chính là cơ chế retrieval của RAG
4. Đo thời gian embed 100 chunks đã tạo ở Ngày 2 → cảm nhận tốc độ

**Tiêu chí hoàn thành:** Giải thích được "Dense vector là gì, cosine similarity đo cái gì, vì sao câu hỏi và câu trả lời không cần giống nhau từng chữ mà vẫn được tìm thấy?"

---

### Ngày 5: Qdrant — Vector Database nền tảng

**Mục tiêu:** Chạy Qdrant local bằng Docker, tạo collection, upsert vector, thực hiện vector search đầu tiên.

**Khái niệm cần nắm:**
- **Vector Database** — database được thiết kế đặc biệt để lưu trữ và tìm kiếm vector nhanh chóng trong không gian nhiều chiều. Database thông thường (SQL) không làm được điều này hiệu quả.

- **Qdrant** — vector database mã nguồn mở, viết bằng Rust (nhanh), có Python client, hỗ trợ local và cloud. Các khái niệm cơ bản:
  - **Collection** — tương đương "table" trong SQL. Mỗi collection chứa các vector có cùng kích thước (dimension) và cùng phương pháp đo khoảng cách (distance metric)
  - **Point** — 1 bản ghi trong collection, gồm: `id` (unique), `vector` (mảng số), `payload` (dict JSON chứa metadata: text gốc, tên file, số trang...)
  - **Payload** — metadata gắn kèm theo vector. Ví dụ: `{"text": "...", "source": "cv.pdf", "page": 3, "chunk_id": 15}`. Khi tìm kiếm, payload giúp bạn lấy lại text gốc
  - **HNSW (Hierarchical Navigable Small World)** — thuật toán index mặc định của Qdrant. Cho phép tìm kiếm xấp xỉ (approximate nearest neighbor) cực kỳ nhanh. Đây là lý do vector search nhanh hơn brute-force hàng trăm lần.
  - **Top-k retrieval** — truy vấn để lấy k kết quả gần nhất (vd: top_k=3 → lấy 3 chunks liên quan nhất)

**Cài đặt:**
```bash
# Chạy Qdrant bằng Docker (cần cài Docker trước)
docker run -p 6333:6333 qdrant/qdrant

# Cài Python client
pip install qdrant-client
```

**Tài liệu đọc (BẮT BUỘC):**
- Qdrant Quickstart: https://qdrant.tech/documentation/quickstart/
- Qdrant Overview: https://qdrant.tech/documentation/overview/

**Bài tập:**
1. Chạy Qdrant local bằng Docker, mở `http://localhost:6333/dashboard` → confirm dashboard hiển thị
2. Tạo collection `"test_collection"` với `vector_size=384` và `distance=Distance.COSINE`
3. Tạo 10 câu text mẫu, embed tất cả, upsert vào collection với payload chứa text gốc
4. Tạo 1 câu query, embed, gọi `collection.search()` với `limit=3` → in ra text và score của từng kết quả
5. Dùng Qdrant Dashboard để xem collection và các points vừa tạo

**Tiêu chí hoàn thành:** Tự tay làm được: text → embed → upsert vào Qdrant → query → lấy về top-k. Giải thích được `payload` dùng để làm gì.

---

### Checkpoint cuối tuần 4 (mini project)

Ghép lại thành 1 script `indexing_pipeline.py`:
- Đọc 1 file PDF/DOCX thật
- Chunk bằng `RecursiveCharacterTextSplitter` (chunk_size=500, overlap=50)
- Embed tất cả chunks bằng `all-MiniLM-L6-v2`
- Upsert vào Qdrant collection `"my_docs"` với payload: `{"text": ..., "source": ..., "chunk_index": ...}`
- Viết hàm `search(query: str, top_k: int = 3) -> list[dict]` — embed query, search Qdrant, trả về list payload của top-k chunks

---

## TUẦN 5 — Vector Search + RAG Integration

### Ngày 6: Retrieval Quality — Đánh giá chất lượng tìm kiếm

**Mục tiêu:** Biết cách kiểm tra pipeline retrieval có đang trả về đúng kết quả không, và phát hiện khi nào nó thất bại.

**Khái niệm cần nắm:**
- **Retrieval quality** — bước retrieval là "nút cổ chai" của toàn bộ RAG. Nếu tìm sai chunk, LLM có context sai → trả lời sai dù LLM rất thông minh. Garbage in, garbage out.

- **Các kiểu thất bại retrieval thường gặp:**
  1. **False positive** — tìm về chunk không liên quan (có từ giống nhau nhưng ý khác)
  2. **False negative** — bỏ sót chunk liên quan (query và document dùng từ đồng nghĩa khác nhau)
  3. **Chunk quá lớn** — vector "trung bình" của nhiều chủ đề → không đại diện tốt cho bất kỳ chủ đề nào
  4. **Chunk quá nhỏ** — thiếu ngữ cảnh, vector không đủ ý nghĩa

- **Cách test đơn giản (manual evaluation):**
  1. Tạo 5-10 câu hỏi mà bạn biết câu trả lời nằm ở đoạn nào
  2. Chạy retrieval cho từng câu hỏi, xem top-3 kết quả
  3. Tính **Precision@k** = số chunk relevant trong top-k / k

**Bài tập:**
1. Dùng tài liệu đã index từ Ngày 5, tạo 10 câu hỏi test
2. Chạy retrieval, so sánh top-3 trả về với kết quả mong đợi, tính Precision@3
3. Tìm ít nhất 2 ví dụ thất bại — phân tích nguyên nhân
4. Thử thay đổi `chunk_size` (200 vs 500 vs 1000) → chạy lại đánh giá → kết luận chunk_size nào cho retrieval tốt nhất

**Tiêu chí hoàn thành:** Tự đánh giá được chất lượng retrieval một cách có số liệu, biết khi nào cần điều chỉnh chunking strategy.

---

### Ngày 7: Tích hợp RAG vào FastAPI — Pipeline hoàn chỉnh

**Mục tiêu:** Xây endpoint FastAPI nhận câu hỏi, tự động retrieve context từ Qdrant, ghép vào prompt, gọi LLM, trả về câu trả lời có nguồn gốc.

**Khái niệm cần nắm:**
- **RAG Prompt Template** — cấu trúc chuẩn:
  ```
  System: Bạn là trợ lý Q&A. Chỉ trả lời dựa trên CONTEXT được cung cấp.
          Nếu không tìm thấy thông tin trong context, nói rõ "Không có trong tài liệu".
  
  User: CONTEXT:
        [Chunk 1]: ...
        [Chunk 2]: ...
        CÂU HỎI: {user_question}
  ```

- **Tại sao phải nói "chỉ trả lời dựa trên context"?** Không có câu lệnh này, LLM sẽ trộn lẫn kiến thức học sẵn với tài liệu → trả lời "có vẻ đúng" nhưng thực ra là hallucination từ training data.

- **Luồng dữ liệu trong 1 request:**
  ```
  POST /ask {question: "..."}
    → embed(question) → query vector
    → Qdrant.search(query_vector, top_k=3) → 3 chunks + payload
    → format_prompt(question, chunks) → prompt string
    → LLM.chat(prompt) → answer
    → return {answer, sources: [source_1, source_2, source_3]}
  ```

**Bài tập:**
1. Tạo file `rag_api.py` với 2 endpoint:
   - `POST /index` — nhận text_content, chunk, embed, upsert vào Qdrant
   - `POST /ask` — nhận `question: str`, thực hiện retrieve + LLM, trả về `answer` và `sources`
2. Pydantic model cho response:
   ```python
   class Source(BaseModel):
       text: str
       source_file: str
       chunk_index: int
       score: float

   class AskResponse(BaseModel):
       answer: str
       sources: list[Source]
   ```
3. Dùng `Depends()` để inject Qdrant client
4. Test bằng Swagger UI: index 1 đoạn văn, đặt câu hỏi, kiểm tra sources có đúng không

**Tiêu chí hoàn thành:** Endpoint `/ask` trả về câu trả lời kèm `sources` — người dùng có thể verify câu trả lời từ đâu.

---

### Ngày 8: Upload File — Nhận PDF/DOCX qua API

**Mục tiêu:** Nhận file upload từ client, xử lý on-the-fly bằng `BytesIO`, không cần ghi ra disk.

**Khái niệm cần nắm:**
- **FastAPI File Upload** — dùng `UploadFile`:
  ```python
  from fastapi import UploadFile, File

  @app.post("/upload")
  async def upload(file: UploadFile = File(...)):
      content = await file.read()  # bytes
  ```
- **`BytesIO`** — class của Python cho phép đọc/ghi bytes như thể là 1 file thật, không cần ghi ra disk. PyMuPDF và python-docx đều hỗ trợ đọc từ `BytesIO`.

**Bài tập:**
1. Nâng cấp endpoint `POST /index` để nhận `UploadFile` thay vì text string
2. Detect loại file qua `file.content_type` hoặc `file.filename` extension → gọi đúng parser
3. Xử lý lỗi: nếu file không phải PDF/DOCX → return `400 Bad Request` với message rõ ràng
4. Test bằng `curl`:
   ```bash
   curl -X POST "http://localhost:8000/index" -F "file=@path/to/your/document.pdf"
   ```
5. (Bonus) Lưu file vào thư mục `uploads/` sau khi xử lý, dùng filename + timestamp để tránh trùng tên

**Tiêu chí hoàn thành:** Upload được file PDF/DOCX qua API, tự động index nội dung, không crash khi gửi file sai định dạng.

---

### Ngày 9: Hybrid Search — Dense + Sparse (BM25)

**Mục tiêu:** Hiểu vì sao chỉ dùng dense vector đôi khi không đủ, và cách kết hợp BM25 với semantic search.

**Khái niệm cần nắm:**
- **Sparse vector (BM25)** — thuật toán tìm kiếm từ khóa cổ điển. Rất tốt khi query chứa từ khóa chính xác, tên riêng, số liệu, mã code. Yếu khi query mơ hồ hoặc dùng từ đồng nghĩa.

- **Điểm mù của dense-only:**
  - Query "lỗi 404 NOT FOUND" → dense có thể tìm về "lỗi HTTP" (đúng ngữ nghĩa) nhưng bỏ sót chunk có mã số "404" chính xác
  - Query chứa tên riêng → dense không giỏi phân biệt

- **Hybrid search** — kết hợp cả 2: cùng 1 query, chạy cả dense search lẫn sparse search (BM25), rồi dùng thuật toán **Reciprocal Rank Fusion (RRF)** để hợp nhất 2 danh sách kết quả thành 1 ranking cuối.

**Cài đặt:**
```bash
pip install fastembed
```

**Tài liệu đọc:**
- Qdrant Hybrid Search: https://qdrant.tech/documentation/tutorials/hybrid-search-fastembed/

**Bài tập:**
1. Index lại collection mới với cả dense và sparse vector (dùng `fastembed`)
2. Thực hiện hybrid search query, so sánh kết quả với dense-only search trên cùng câu hỏi
3. Tìm ví dụ cụ thể: câu query nào hybrid cho kết quả tốt hơn dense-only? Tại sao?
4. Thảo luận: khi nào nên dùng hybrid? Khi nào dense-only là đủ?

**Tiêu chí hoàn thành:** Chạy được hybrid search, so sánh được kết quả của 2 phương pháp, biết khi nào hybrid đáng đầu tư thêm độ phức tạp.

---

### Ngày 10: Reranking — Sắp xếp lại kết quả sau khi retrieve

**Mục tiêu:** Hiểu vì sao retrieval top-k đôi khi không đủ tốt, và cách dùng reranker để cải thiện độ chính xác.

**Khái niệm cần nắm:**
- **Vấn đề của top-k retrieval:** Embedding model được huấn luyện để tìm kiếm nhanh (approximate nearest neighbor). Kết quả top-3 không nhất thiết là 3 chunk *thực sự* liên quan nhất.

- **Reranking** — bước thứ 2 sau retrieval. Lấy top-k (vd: top-10) từ vector search, dùng 1 model riêng (cross-encoder) để đánh giá lại mức độ liên quan thực sự của từng chunk với query. Cross-encoder xem xét cả query và chunk cùng lúc — chính xác hơn nhưng chậm hơn bi-encoder. Cuối cùng chỉ lấy top-3 sau khi rerank.

- **Pattern phổ biến:**
  ```
  Vector DB → top-20 candidates (nhanh, xấp xỉ)
               ↓
  Reranker  → top-3 sau rerank (chậm hơn, nhưng chính xác hơn nhiều)
               ↓
  LLM với 3 chunks chất lượng cao
  ```

**Bài tập:**
1. Cài và thử reranker local:
   ```python
   from sentence_transformers import CrossEncoder
   reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
   scores = reranker.predict([(query, chunk) for chunk in candidates])
   ```
2. Pipeline đầy đủ: retrieve top-10 từ Qdrant → rerank → lấy top-3 → đưa vào LLM
3. So sánh: top-3 trước rerank vs top-3 sau rerank — chunk nào "đổi hạng" sau rerank?
4. Đo thời gian: thêm bước rerank mất bao nhiêu millisecond? Trade-off đó có đáng không?

**Tiêu chí hoàn thành:** Tích hợp được reranking vào pipeline RAG, đo được sự cải thiện với ít nhất 3 ví dụ cụ thể.

---

### Checkpoint cuối tuần 5

Nâng cấp `rag_api.py` thành version 2:
- Upload file PDF/DOCX qua `UploadFile`
- Hybrid search (dense + sparse)
- Reranking trước khi đưa vào LLM
- Response trả về `answer` + `sources` đầy đủ với score sau rerank

---

## TUẦN 6 — Advanced RAG + Capstone Project

### Ngày 11: Metadata Filtering + Payload Index

**Mục tiêu:** Lọc kết quả tìm kiếm theo metadata để tăng độ chính xác khi có nhiều tài liệu.

**Khái niệm cần nắm:**
- **Payload Index** — ngoài vector index (HNSW), Qdrant còn cho phép tạo index trên các field trong payload để filter nhanh mà không cần scan toàn bộ collection.

- **Metadata filtering** — giới hạn không gian tìm kiếm vector bằng điều kiện metadata:
  ```python
  client.search(
      collection_name="docs",
      query_vector=query_vector,
      query_filter=Filter(
          must=[
              FieldCondition(key="source_file", match=MatchValue(value="cv.pdf")),
              FieldCondition(key="page", range=Range(gte=1, lte=5))
          ]
      ),
      limit=3
  )
  ```

**Bài tập:**
1. Nâng cấp indexing pipeline: lưu thêm vào payload: `source_file`, `page_number`, `chunk_index`, `indexed_at` (timestamp)
2. Tạo Qdrant payload index cho field `source_file`
3. Nâng cấp endpoint `POST /ask` để nhận thêm optional param `source_file: str | None`
4. Test: index 2 file khác nhau, hỏi 1 câu rồi filter theo từng file → kết quả khác nhau như thế nào?

**Tiêu chí hoàn thành:** Metadata filtering hoạt động đúng — query chỉ trả về chunks từ file được chỉ định.

---

### Ngày 12: Đánh giá RAG — RAGAS

**Mục tiêu:** Đánh giá chất lượng toàn bộ hệ thống RAG có số liệu cụ thể.

**Khái niệm cần nắm:**
- **4 chiều đánh giá RAG (từ RAGAS framework):**
  1. **Faithfulness** — câu trả lời của LLM có trung thực với context được retrieve không? (không hallucinate)
  2. **Answer Relevancy** — câu trả lời có thực sự trả lời đúng câu hỏi không?
  3. **Context Precision** — context retrieve có chứa đủ thông tin cần thiết không?
  4. **Context Recall** — hệ thống có bỏ sót thông tin quan trọng nào từ tài liệu không?

**Tài liệu đọc:**
- DeepLearning.AI — Building and Evaluating Advanced RAG: https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/

**Bài tập:**
1. Cài: `pip install ragas`
2. Tạo test set 10 câu hỏi từ tài liệu, kèm expected answer và ground truth context
3. Chạy RAGAS evaluation: tính faithfulness score và answer relevancy score
4. Xác định điểm yếu nhất của hệ thống → đề xuất 1 cải tiến cụ thể

**Tiêu chí hoàn thành:** Có bảng số liệu evaluation 4 chiều, xác định được điểm yếu và biết cách cải thiện.

---

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

## Definition of Done — Tự kiểm tra trước khi qua Giai đoạn 3

Với mỗi mục, tự hỏi: "Tôi có thể giải thích + viết code minh hoạ ngay mà không cần tra cứu không?"

- [ ] Trích xuất được text từ PDF và DOCX, biết giới hạn (không xử lý được PDF scan)
- [ ] Giải thích được tại sao chunking cần thiết và `chunk_overlap` giải quyết vấn đề gì
- [ ] So sánh được Fixed-size vs Recursive vs Semantic chunking — biết khi nào dùng cái nào
- [ ] Giải thích được embedding là gì, cosine similarity đo cái gì, dense vs sparse vector khác nhau ở đâu
- [ ] Cài và chạy được Qdrant local, tạo collection, upsert, query top-k với payload
- [ ] Giải thích được HNSW là gì và tại sao nó nhanh hơn brute-force
- [ ] Xây được endpoint RAG đầy đủ: upload → chunk → embed → upsert → query → LLM → trả kết quả kèm sources
- [ ] Thực hiện được hybrid search (dense + BM25), biết khi nào hybrid tốt hơn dense-only
- [ ] Hiểu reranking giải quyết vấn đề gì, tích hợp được cross-encoder vào pipeline
- [ ] Dùng được metadata filtering trong Qdrant query
- [ ] Đánh giá được chất lượng RAG bằng ít nhất 2 chiều có số liệu (faithfulness, answer relevancy)

---

## Tổng hợp tài liệu tham khảo

| Nguồn | Dùng cho |
|---|---|
| [Pinecone — Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/) | Ngày 2-3 (nền tảng chunking, BẮT BUỘC) |
| [Qdrant Quickstart](https://qdrant.tech/documentation/quickstart/) | Ngày 5 (setup Qdrant, BẮT BUỘC) |
| [Qdrant Overview](https://qdrant.tech/documentation/overview/) | Ngày 5 (khái niệm: collection, payload, HNSW) |
| [Qdrant Hybrid Search](https://qdrant.tech/documentation/tutorials/hybrid-search-fastembed/) | Ngày 9 |
| [LangChain SemanticChunker](https://python.langchain.com/docs/how_to/semantic-chunker/) | Ngày 3 |
| [DeepLearning.AI — Building & Evaluating Advanced RAG](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) | Ngày 12 (RAGAS evaluation) |
| [sentence-transformers docs](https://www.sbert.net/) | Ngày 4 (embedding) và Ngày 10 (reranking) |
| [PyMuPDF docs](https://pymupdf.readthedocs.io/en/latest/tutorial.html) | Ngày 1 (PDF parsing) |

**Lưu ý khi tra cứu:**
- Qdrant Python client API thay đổi nhiều giữa các version — luôn kiểm tra version bằng `pip show qdrant-client` và đọc docs đúng version
- Sentence-transformers model names có thể thay đổi — tra trên HuggingFace Hub để tìm model mới nhất
- LangChain thay đổi rất nhanh — nếu gặp lỗi import, tra cứu docs bản mới nhất

---

## Thư viện sẽ dùng trong Giai đoạn 2

```bash
# Document parsing
pip install pymupdf python-docx

# Chunking
pip install langchain-text-splitters

# Embedding
pip install sentence-transformers

# Vector DB
pip install qdrant-client fastembed

# Evaluation
pip install ragas

# Đã có từ Phase 1
pip install fastapi uvicorn python-dotenv pydantic groq
```
