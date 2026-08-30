# Dự Án Capstone Phase 1: Hệ thống Smart Document RAG & Analyzer API

Dự án này là một Backend API xây dựng trên **FastAPI**, đóng vai trò như một trợ lý ảo có khả năng xử lý tài liệu dài, tóm tắt thông tin và trả lời câu hỏi dựa trên tài liệu (RAG). 

Dự án này được thiết kế để kết nối **tất cả** các keyword bạn đã học thành một luồng hoàn chỉnh và thực tế.

---

## 1. Tính Năng Chính
1. **Endpoint Tóm Tắt (Summarize):** Nhận vào một văn bản rất dài, chia nhỏ văn bản và sử dụng kỹ thuật Map-Reduce để tóm tắt, sau đó trả về kết quả dưới dạng cấu trúc JSON rõ ràng.
2. **Endpoint Hỏi Đáp (Q&A Stream):** Người dùng đặt câu hỏi về tài liệu. Hệ thống trích xuất ngữ cảnh liên quan (RAG), suy luận từng bước (Chain-Of-Thought) và trả về câu trả lời dạng chữ chảy từ từ (Streaming Response).

---

## 2. Cách Ứng Dụng Các Kiến Thức

### 🎯 2.1. FastAPI & asyncio
- **FastAPI:** Xây dựng web server với các route `@app.post("/upload")`, `@app.post("/summarize")` và `@app.post("/chat")`.
- **Dependency Injection:** Sử dụng `Depends()` để tiêm (inject) các class xử lý LLM hoặc xác thực (auth) vào các route.
- **asyncio:** Sử dụng `async def` và `await` cho toàn bộ các network calls (như gọi API của LLM, đọc file) để đảm bảo server không bị block (chặn luồng) khi xử lý nhiều người dùng.

### 🎯 2.2. OOP (Lập Trình Hướng Đối Tượng)
Xây dựng một kiến trúc dễ mở rộng (ví dụ bạn muốn dùng OpenAI, hoặc chuyển sang Gemini, Anthropic):
- Tạo một Base class `BaseLLMClient` chứa các method trừu tượng như `generate()`, `stream()`.
- Tạo các class con như `OpenAIClient(BaseLLMClient)` hoặc `GeminiClient(BaseLLMClient)` kế thừa và override các methods sử dụng `super().__init__` để khởi tạo.
- Tạo class `DocumentProcessor` chuyên quản lý việc xử lý văn bản.

### 🎯 2.3. Pydantic & JSON Schema
- **DataModel & Field:** Định nghĩa các class `DocumentRequest`, `QueryRequest`, `SummaryResponse` để validate dữ liệu đầu vào và đầu ra của FastAPI.
- **Structured Output:** Dùng Pydantic biến đổi thành JSON Schema để ép LLM trả về chính xác format bạn cần (ví dụ trả về mảng các `KeyPoint` thay vì text thô).

### 🎯 2.4. LLM & Prompt Engineering
- **Tokenization & Context Window:** Trước khi gửi cho LLM, dùng thư viện (như `tiktoken`) để đếm số lượng token. Đảm bảo tổng số token không vượt quá Context Window của model.
- **Chunking:** Nếu text quá dài, cắt nhỏ text ra thành các "Chunk" (đoạn) có độ dài token phù hợp.
- **RAG (Retrieval-Augmented Generation):** Lưu các Chunk. Khi người dùng hỏi, viết code tìm kiếm Chunk có chứa thông tin liên quan ghép vào Prompt làm ngữ cảnh.
- **Map - Reduce:** Khi cần tóm tắt toàn bộ file lớn, cho LLM tóm tắt từng Chunk trước (Map), sau đó gộp các bản tóm tắt lại và cho LLM tóm tắt lần cuối (Reduce).
- **Streaming Response:** Trả về kết quả của LLM dưới dạng stream (từng token một) bằng `StreamingResponse` của FastAPI để tăng trải nghiệm người dùng, thay vì bắt họ chờ 10s mới thấy kết quả.
- **Few-Shot & Chain-Of-Thought (CoT):** Trong Prompt thiết kế cho Q&A, cung cấp 1-2 ví dụ (Few-Shot) và yêu cầu LLM "hãy suy nghĩ từng bước trước khi đưa ra kết quả cuối cùng" (CoT) để tăng độ chính xác.
- **Prompt Caching:** Nếu model hỗ trợ, bật tính năng cache đối với System Prompt (vốn chứa toàn bộ hướng dẫn, Few-Shot và Rules) để tiết kiệm chi phí/thời gian khi user gọi API nhiều lần.

---

## 3. Đề Xuất Cấu Trúc Thư Mục
```text
phase_1_capstone/
├── main.py                  # Điểm bắt đầu của FastAPI (khởi tạo app, routes)
├── schemas.py               # Chứa các Pydantic Models (Request, Response)
├── llm_service.py           # Chứa các class OOP xử lý logic tương tác LLM
├── text_utils.py            # Chứa các hàm Tokenization, Chunking
├── prompts.py               # Chứa các chuỗi Prompt (Few-shot, CoT templates)
└── requirements.txt         # fastapi, uvicorn, pydantic, tiktoken, openai...
```

## 4. Các Bước Bắt Đầu Chi Tiết (Roadmap)

Vì bạn đang tự học, dưới đây là gợi ý cụ thể về **tên class** và **tên hàm** cho từng file để bạn dễ hình dung kiến trúc code:

### Bước 1: Khởi tạo Project & Cài đặt (Terminal)
- Tạo và kích hoạt môi trường ảo (venv).
- Cài đặt các thư viện trong `requirements.txt` bằng `pip install -r requirements.txt`.

### Bước 2: `schemas.py` (Pydantic Models)
Định nghĩa các cấu trúc dữ liệu giao tiếp giữa Client (người dùng) -> FastAPI -> LLM.
- `class DocumentSummary(BaseModel):` chứa các field `title`, `summary`, `key_points`, `keywords` (dùng để ép LLM trả về Structured JSON Output).
- `class QueryRequest(BaseModel):` chứa `context_text` (tài liệu) và `question` (câu hỏi).
- `class SummarizeRequest(BaseModel):` chứa `text` (nội dung cần tóm tắt).

### Bước 3: `text_utils.py` (Xử lý chuỗi)
Chứa các hàm (functions) phục vụ việc tính toán Token và cắt nhỏ văn bản (Chunking).
- `def count_tokens(text: str) -> int:` Hàm đếm số lượng token của đoạn text.
- `def chunk_text(text: str, chunk_size: int = 2000) -> list[str]:` Hàm chia văn bản dài thành một List các đoạn văn ngắn hơn (mỗi đoạn không quá số token quy định).

### Bước 4: `prompts.py` (Prompt Engineering)
Tách toàn bộ chuỗi Prompt dài ra file riêng để code sạch sẽ.
- `MAP_PROMPT_TEMPLATE = """..."""`: Mẫu Prompt yêu cầu tóm tắt một đoạn nhỏ (Map).
- `REDUCE_PROMPT_TEMPLATE = """..."""`: Mẫu Prompt yêu cầu tổng hợp các đoạn tóm tắt thành 1 bản tóm tắt cuối (Reduce).
- `QA_SYSTEM_PROMPT = """..."""`: System Prompt hướng dẫn LLM cách trả lời câu hỏi dựa trên văn bản, kèm theo các ví dụ (Few-shot) và yêu cầu suy luận (Chain-Of-Thought).

### Bước 5: `llm_service.py` (OOP & Gọi LLM)
Quản lý việc gọi API (ví dụ Groq) và các thuật toán phức tạp như Map-Reduce.
- **Class `BaseLLMClient`:**
  - `async def generate_structured(self, prompt: str, schema: type[BaseModel]) -> BaseModel:` (Hàm ảo / Interface)
  - `async def stream_chat(self, system_prompt: str, user_prompt: str):` (Hàm ảo / Interface)
- **Class `GroqClient(BaseLLMClient)`:** Kế thừa BaseLLMClient và viết code thực tế gọi thư viện `groq`.
- **Class `DocumentProcessor`:**
  - `def __init__(self, llm_client: BaseLLMClient):` Áp dụng Dependency Injection để truyền Client vào.
  - `async def summarize_long_document(self, text: str) -> DocumentSummary:` Logic gọi `chunk_text`, thực hiện Map-Reduce và cuối cùng gọi `generate_structured`.
  - `async def answer_question_stream(self, text: str, question: str):` Logic xây dựng ngữ cảnh RAG, gọi `stream_chat` để lấy luồng dữ liệu trả về.

### Bước 6: `main.py` (FastAPI Endpoints)
Kết nối mọi thứ thành 1 API Server chạy được bằng `uvicorn`.
- Khởi tạo: `app = FastAPI(title="Capstone API")`
- Khởi tạo Dependencies: `llm_client = GroqClient(...)` và `doc_processor = DocumentProcessor(llm_client)`
- **`@app.post("/summarize", response_model=DocumentSummary):`** Định nghĩa API nhận request, gọi hàm `doc_processor.summarize_long_document()`.
- **`@app.post("/chat"):`** Định nghĩa API hỏi đáp, sử dụng `StreamingResponse` của FastAPI để trả luồng dữ liệu liên tục từ hàm `doc_processor.answer_question_stream()`.
