# 🔭 CẨM NANG TOÀN DIỆN: LANGFUSE OBSERVABILITY CHO LLM & RAG
## Giám sát Thực thi (Tracing), Quản trị Chi phí (Cost), Prompt Versioning & Đánh giá (Evaluation)

---

## 1. MENTAL MODEL: TẠI SAO APM TRUYỀN THỐNG BẤT LỰC TRƯỚC LLM?

Trong các ứng dụng web thông thường, bạn dùng Datadog, Prometheus hoặc New Relic để đo: `CPU`, `RAM`, `RPS`, `HTTP Status Code 200/500`.

Tuy nhiên, trong các hệ thống GenAI / RAG / Multi-Agent:
* **HTTP 200 nhưng kết quả sai hoàn toàn:** LLM trả về mã 200 OK nhưng nội dung lại bịa đặt 100% (Ảo giác / Hallucination).
* **Chi phí biến thiên phi tuyến:** Không tính bằng băng thông mạng, mà tính bằng **Tokens** (Input, Output, Cached Tokens, Reasoning Tokens) nhân với đơn giá USD của từng model.
* **Thời gian trễ (Latency) nghẽn ở bước sinh:** Cần theo dõi **TTFT (Time to First Token)** và tốc độ Streaming (Tokens/sec).
* **Độ phức tạp đa tầng (Non-deterministic Execution Tree):** Một câu hỏi có thể kích hoạt 1 chuỗi LangGraph gồm 3 Agents, 5 Tool calls, 2 lần Vector Search. APM truyền thống chỉ thấy một request HTTP dài 10 giây mà không biết thời gian đang nghẽn ở bước nào.

```
┌──────────────────────────────────────┬──────────────────────────────────────┐
│  TRADITIONAL APM (Datadog / NewRelic) │     LLM OBSERVABILITY (Langfuse)     │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ • Đơn vị đo: Request/Response HTTP.  │ • Đơn vị đo: Traces, Spans, LLM Gens│
│ • Theo dõi lỗi: Exception 500, Crash │ • Theo dõi lỗi: Hallucination, Bad   │
│                                      │   Retrieval, Tool Failures, Toxic.   │
│ • Chi phí: Hạ tầng máy chủ (Server). │ • Chi phí: Từng cent USD cho Token   │
│                                      │   của OpenAI, Claude, DeepSeek...    │
│ • Đánh giá: Uptime 99.9%, Latency.   │ • Đánh giá: Faithfulness, Relevancy, │
│                                      │   User Feedback (👍/👎), Accuracy.   │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

---

## 2. MÔ HÌNH DỮ LIỆU PHÂN CẤP CỐT LÕI (LANGFUSE DATA MODEL)

Langfuse tổ chức dữ liệu giám sát theo một **Cây thực thi phân cấp (Execution Tree)**:

```
[TRACE: Toàn bộ phiên xử lý câu hỏi của User]
  │
  ├── [SPAN: Router Phân loại câu hỏi (50ms)]
  │
  ├── [SPAN: Truy xuất Dữ liệu RAG (350ms)]
  │     ├── [SPAN: Vector Search Qdrant (15ms)]
  │     └── [SPAN: Cross-Encoder Rerank (80ms)]
  │
  ├── [GENERATION: LLM Viết câu trả lời (GPT-4o - 1.2s)]
  │     └── Metrics: 450 prompt tokens, 120 completion tokens -> $0.0042
  │
  ├── [EVENT: Phát hiện cảnh báo rủi ro / Filter trigger]
  │
  └── [SCORE: Điểm đánh giá chất lượng]
        ├── User Feedback: 👍 (+1)
        └── Ragas Faithfulness: 0.95
```

### Chi tiết 5 thực thể dữ liệu của Langfuse:

| Thực thể | Bản chất | Dữ liệu ghi nhận | Ví dụ thực tế |
| :--- | :--- | :--- | :--- |
| **Trace** | Bao trọn 1 request hoàn chỉnh từ User gửi đến khi trả lời xong. | `user_id`, `session_id`, `tags`, `metadata`, tổng thời gian, tổng chi phí USD. | Người dùng hỏi: *"Tính Z-Score cho BCTC Vinamilk 2023"*. |
| **Span** | Đại diện cho một bước xử lý logic trung gian (không gọi LLM). | Input, Output, Thời gian bắt đầu, Kết thúc, Error stack trace. | Bước gọi Tool Python tính toán, bước truy vấn Vector DB, bước nén Context. |
| **Generation** | Đại diện cho **một lời gọi LLM cụ thể**. | Tên Model, Prompt, Output, Token usage (Prompt / Completion), Cost USD, Temperature. | Gọi `ChatOpenAI(model="gpt-4o")` để tổng hợp văn bản. |
| **Event** | Dấu mốc thời gian (Timestamp) ghi nhận một sự kiện tức thời. | Name, Metadata tại thời điểm diễn ra. | Người dùng bấm nút hủy, hệ thống kích hoạt fallback sang Web Search. |
| **Score** | Điểm số đánh giá chất lượng của Trace hoặc Observation. | `name`, `value` (số hoặc boolean), `comment`. | Người dùng vote 👍, Ragas chấm Faithfulness = 0.92, G-Eval = 4/5 sao. |

---

## 3. KIẾN TRÚC TRIỂN KHAI: MANAGED CLOUD VS. SELF-HOSTED (DOCKER COMPOSE)

### 3.1. Langfuse Cloud (SaaS)
* Phù hợp cho Startup, POC, ứng dụng không bị rào cản pháp lý dữ liệu ngân hàng.
* Đăng ký tài khoản tại `cloud.langfuse.com` $\rightarrow$ Tạo Project $\rightarrow$ Lấy `PUBLIC_KEY`, `SECRET_KEY`, `HOST`.

### 3.2. Langfuse Self-Hosted v3 (On-Premise cho Ngân hàng / Doanh nghiệp)
* **Bắt buộc** với các dự án Ngân hàng, Bảo hiểm, Y tế nơi dữ liệu tài chính không được phép bay ra internet.
* Kiến trúc Langfuse v3 kết hợp **PostgreSQL** (lưu Metadata/Users) và **ClickHouse** (CSDL Columnar chuyên dụng để query hàng triệu traces với tốc độ mili-giây).

#### File `docker-compose.yml` chuẩn Production:
```yaml
version: "3.9"

services:
  langfuse-server:
    image: ghcr.io/langfuse/langfuse:3
    restart: always
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgresql://langfuse:langfuse_secure_pwd@postgres:5432/langfuse
      - CLICKHOUSE_URL=http://clickhouse:8123
      - CLICKHOUSE_USER=default
      - CLICKHOUSE_PASSWORD=clickhouse_secure_pwd
      - NEXTAUTH_URL=http://localhost:3000
      - NEXTAUTH_SECRET=your_super_secret_hex_key_minimum_32_chars
      - SALT=your_random_salt_string_at_least_16_chars
      - TELEMETRY_ENABLED=false
    depends_on:
      postgres:
        condition: service_healthy
      clickhouse:
        condition: service_healthy

  postgres:
    image: postgres:16-alpine
    restart: always
    environment:
      - POSTGRES_USER=langfuse
      - POSTGRES_PASSWORD=langfuse_secure_pwd
      - POSTGRES_DB=langfuse
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U langfuse"]
      interval: 5s
      timeout: 5s
      retries: 5

  clickhouse:
    image: clickhouse/clickhouse-server:24.3-alpine
    restart: always
    environment:
      - CLICKHOUSE_USER=default
      - CLICKHOUSE_PASSWORD=clickhouse_secure_pwd
    volumes:
      - chdata:/var/lib/clickhouse
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:8123/ping"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
  chdata:
```

---

## 4. 3 PHƯƠNG PHÁP TÍCH HỢP CODE CHUẨN DOANH NGHIỆP

### Cấu hình biến môi trường (`.env`):
```bash
LANGFUSE_PUBLIC_KEY="pk-lf-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
LANGFUSE_SECRET_KEY="sk-lf-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
LANGFUSE_HOST="https://cloud.langfuse.com" # Hoặc "http://localhost:3000" nếu self-host
```

---

### PHƯƠNG PHÁP 1: Decorator `@observe()` (Pythonic, Khuyến nghị #1 cho Backend/FastAPI)

Decorator `@observe()` tự động liên kết các hàm lồng nhau thành cây Trace/Span mà không cần truyền biến context thủ công.

```python
import os
from langfuse.decorators import observe, langfuse_context
from openai import OpenAI

client = OpenAI()

# 1. Bước RAG Search (Span)
@observe(name="retrieve_financial_docs")
def retrieve_docs(query: str):
    # Cập nhật metadata cho bước tìm kiếm
    langfuse_context.update_current_observation(
        input={"query": query},
        metadata={"top_k": 3, "retriever_type": "Hybrid-Qdrant"}
    )
    # Giả lập kết quả lấy từ Vector DB
    retrieved_chunks = ["BCTC 2023: Doanh thu thuần đạt 60.374 tỷ VNĐ.", "Lợi nhuận gộp đạt 24.500 tỷ."]
    
    langfuse_context.update_current_observation(output={"chunks_found": len(retrieved_chunks)})
    return retrieved_chunks

# 2. Bước LLM Generation
@observe(as_type="generation", name="generate_financial_answer")
def call_llm(query: str, context: list):
    prompt = f"Ngữ cảnh: {context}\nCâu hỏi: {query}"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    answer = response.choices[0].message.content
    
    # Cập nhật thông số token và chi phí
    langfuse_context.update_current_observation(
        model="gpt-4o-mini",
        input=prompt,
        output=answer,
        usage={
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
            "total": response.usage.total_tokens
        }
    )
    return answer

# 3. Hàm gốc bọc toàn bộ Trace (Root Trace)
@observe(name="financial_rag_pipeline")
def handle_user_request(user_id: str, session_id: str, question: str):
    # Gắn User ID và Session ID vào Trace
    langfuse_context.update_current_trace(
        user_id=user_id,
        session_id=session_id,
        tags=["Production", "FinRisk-AI"],
        metadata={"client_version": "1.2.0"}
    )
    
    docs = retrieve_docs(question)
    answer = call_llm(question, docs)
    return answer

# Chạy thử
if __name__ == "__main__":
    result = handle_user_request(
        user_id="analyst_007", 
        session_id="session_finance_q3", 
        question="Doanh thu năm 2023 của công ty là bao nhiêu?"
    )
    print("Kết quả:", result)
```

---

### PHƯƠNG PHÁP 2: Tích hợp với LangChain / LangGraph (Tự động 100%)

Dùng `CallbackHandler` của Langfuse. Nó tự động bắt trọn vẹn toàn bộ các Node, Tool Calls, và LLM Generations trong StateGraph mà **không cần sửa một dòng logic code nào**.

```python
import os
from langfuse.callback import CallbackHandler
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

# 1. Khởi tạo Callback Handler
langfuse_handler = CallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

# 2. Xây dựng LangGraph như bình thường
# (Giả sử có workflow = StateGraph(...))
app = workflow.compile()

# 3. Thực thi Graph và truyền Handler vào config callbacks
response = app.invoke(
    {"question": "Tính chỉ số rủi ro tín dụng cho cty ABC"},
    config={
        "callbacks": [langfuse_handler],
        "metadata": {
            "user_id": "auditor_12",
            "environment": "production"
        },
        "tags": ["LangGraph-Execution", "Credit-Rating"]
    }
)
```

---

### PHƯƠNG PHÁP 3: OpenAI SDK Wrapper (Drop-in Replacement)

Nếu code của bạn viết thuần bằng thư viện `openai`, chỉ cần đổi 1 dòng import:

```python
# THAY VÌ: from openai import OpenAI
from langfuse.openai import OpenAI

client = OpenAI()

# Mọi lệnh create() sẽ tự động được trace lên Langfuse!
response = client.chat.completions.create(
    name="risk_assessment_prompt",
    model="gpt-4o",
    messages=[{"role": "user", "content": "Phân tích rủi ro nợ xấu năm 2024."}],
    metadata={"department": "Risk-Management"}
)
```

---

## 5. PROMPT MANAGEMENT & VERSIONING (QUẢN TRỊ PROMPT TẬP TRUNG)

### 💡 Vấn đề:
Nếu bạn hardcode System Prompt trong mã nguồn Python:
* Mỗi lần chỉnh sửa câu lệnh prompt, bạn phải **commit code, build Docker image, chạy CI/CD và deploy lại**.
* Không theo dõi được prompt version 1 hay version 2 cho kết quả tốt hơn.

### 🛠️ Giải pháp với Langfuse Prompt Management:
Prompt được lưu trên giao diện web của Langfuse. Code Python chỉ việc "kéo" prompt về theo nhãn (Tag) hoặc Version:

```python
from langfuse import Langfuse

langfuse = Langfuse()

# Lấy phiên bản prompt mới nhất có nhãn 'production'
prompt_template = langfuse.get_prompt("financial_credit_analyst", label="production")

# Biên dịch prompt với biến số thực tế
compiled_prompt = prompt_template.compile(
    company_name="Công ty Cổ phần ABC",
    debt_ratio="1.8"
)

print(compiled_prompt)
# -> Trả về prompt hoàn chỉnh được format sẵn!
```

---

## 6. PHÂN TÍCH CHI PHÍ (COST) & THEO DÕI PROMPT CACHING

Langfuse tự động tính tiền chi tiết đến từng phần nghìn cent dựa trên bảng giá của hơn 100+ mô hình hàng đầu thế giới:

```
┌─────────────────────────────────────────────────────────────┐
│                 BẢNG CHI PHÍ GENERATION                     │
├─────────────────────────────────────────────────────────────┤
│ Model: gpt-4o-2024-08-06                                    │
│ Input Tokens:  1,200 tokens  ($2.50 / 1M)  -> $0.00300      │
│ Cached Tokens: 4,000 tokens  ($1.25 / 1M)  -> $0.00500      │
│ Output Tokens:   350 tokens  ($10.0 / 1M)  -> $0.00350      │
│ ----------------------------------------------------------- │
│ TOTAL COST:                                -> $0.01150 USD  │
│ LATENCY: 850ms | TTFT (Time to First Token): 180ms          │
└─────────────────────────────────────────────────────────────┘
```

> 💡 **Prompt Caching:** Khi bạn nhồi tài liệu BCTC dài 50 trang vào Prompt, nhờ có Prompt Caching (OpenAI / Anthropic), các token đọc lại được giảm giá tới **50% - 80%**. Langfuse tự động hiển thị số lượng `Cached Tokens` này trên Dashboard để bạn giám sát ngân sách.

---

## 7. PIPELINE CHẤM ĐIỂM (SCORES & EVALUATION)

Để biết hệ thống chạy có tốt không, Langfuse hỗ trợ gắn **Score** từ 3 nguồn:

### 1. User Feedback (Đánh giá từ người dùng cuối)
Người dùng bấm nút Like/Dislike trên giao diện Web:
```python
from langfuse import Langfuse
langfuse = Langfuse()

# Gửi điểm phản hồi từ người dùng
langfuse.score(
    trace_id="id_cua_trace_vua_tao",
    name="user_feedback",
    value=1,            # 1 = Like (Hài lòng), 0 = Dislike
    comment="Thông tin số liệu rất chính xác và đầy đủ."
)
```

### 2. Tích hợp Ragas / DeepEval (Đánh giá Tự động)
Sau khi Ragas tính xong các metric như `Faithfulness`, bạn đẩy ngược điểm số vào Trace trên Langfuse:
```python
langfuse.score(
    trace_id="id_cua_trace_vua_tao",
    name="ragas_faithfulness",
    value=0.95,
    comment="Kiểm tra không phát hiện ảo giác."
)
```

### 3. LLM-as-a-Judge tự động trên giao diện Langfuse
Bạn có thể thiết lập trực tiếp trên web Langfuse: Mỗi khi có 1 Trace mới, hệ thống tự động gọi GPT-4o chấm điểm theo thang điểm 1-5 sao dựa trên bộ tiêu chuẩn định sẵn mà không cần viết code!

---

## 8. BỘ CHECKLIST VÀNG CHO PRODUCTION

1. **Bảo mật thông tin nhạy cảm (Data Masking / PII):**
   * Sử dụng hook tiền xử lý để ẩn số thẻ tín dụng, số CCCD, mật khẩu trước khi gửi payload lên Langfuse:
     ```python
     # Thay thế số thẻ thành **** trước khi update trace
     ```
2. **Asynchronous Flush (Không làm chậm hệ thống):**
   * SDK của Langfuse chạy luồng nền (Background worker thread). Đặt `LANGFUSE_FLUSH_INTERVAL=1.0` để gom các log gửi đi theo lô (batch), không gây ảnh hưởng đến thời gian phản hồi của người dùng.
3. **Tags & Metadata chiến lược:**
   * Luôn gắn `tags=["production", "release-v2.1"]` và `user_id` để dễ dàng lọc, tìm kiếm và phân tích người dùng nào đang tiêu tốn nhiều tiền API nhất.
