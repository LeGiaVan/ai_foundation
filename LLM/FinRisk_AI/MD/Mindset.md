Đây là câu hỏi về **tư duy**, không phải kỹ thuật — và đây là câu hỏi quan trọng nhất trước khi bắt đầu.

---

## 🧠 Tư duy để chinh phục dự án lớn trong thời đại AI

---

### 1. Từ bỏ tư duy "Hiểu hết rồi mới code"

Dự án này rộng vì nó **tích hợp** nhiều công nghệ, không phải vì mỗi công nghệ là khó. Hầu hết các layer đã có thư viện làm sẵn:

```
LangGraph        → đã lo State Machine, HITL, Checkpointer
Qdrant SDK       → đã lo Hybrid Search, Indexing
Langfuse SDK     → đã lo Tracing, Cost tracking, Prompt registry
FastAPI          → đã lo Async, Validation, SSE Streaming
Docker Compose   → đã lo Orchestration, Health check, Restart policy
```

**Việc của bạn là: kết nối chúng đúng chỗ, không phải tái phát minh chúng.**

---

### 2. Tư duy "Vertical Slice" — Luôn có thứ chạy được

Sai lầm phổ biến: code từng layer nằm ngang (xây hết API → xây hết Agent → xây hết DB).

Đúng hơn: cắt hệ thống theo chiều **dọc** — mỗi slice phải chạy end-to-end từ đầu đến cuối:

```
Slice 1 (Tuần 1): Input câu hỏi → LLM trả lời thô
                  → Chạy được. Chứng minh hệ thống sống.

Slice 2 (Tuần 2): Thêm Qdrant → Tìm context → Trả lời có nguồn
                  → RAG cơ bản hoạt động.

Slice 3 (Tuần 3): Thêm LangGraph → Phân loại ý định → Dispatch đúng Agent
                  → Multi-Agent hoạt động.

Slice 4 (Tuần 4): Thêm Langfuse → Xem trace trên UI
                  → Observability hoạt động.

Slice 5 (Tuần 5): Bọc FastAPI → Expose endpoint /analyze
                  → API hoạt động.

Slice 6 (Tuần 6): Đẩy lên VPS → Nginx + SSL
                  → Hệ thống live thực tế.
```

> Sau mỗi slice, bạn có một **sản phẩm chạy được** — không phải đống code dở dang.

---

### 3. Quy trình làm việc thực tế với AI (Vibe Coding đúng cách)

```
1. ĐỌC DOCS CHÍNH THỨC (10–20 phút)
   → Hiểu API surface của thư viện (không đọc toàn bộ, chỉ đọc Getting Started)

2. YÊU CẦU AI VIẾT BOILERPLATE
   → "Viết LangGraph StateGraph với 3 nodes: RAG, Math, Risk. Dùng TypedDict State."
   → Claude / Copilot gen ra ~80% code đúng

3. CHẠY VÀ QUAN SÁT LỖI
   → Không đọc code từng dòng trước khi chạy
   → Chạy → lỗi → hỏi AI → fix → chạy lại

4. HIỂU SAU KHI CHẠY ĐƯỢC
   → Khi code đã chạy, mới đọc lại từng dòng để hiểu "tại sao nó hoạt động"
   → Đây là lúc kiến thức thực sự ngấm vào não

5. CUSTOMIZE CHO BÀI TOÁN CỦA MÌNH
   → Thay tên node, thêm logic tài chính, sửa điều kiện routing
   → Lúc này bạn đang thật sự "code" chứ không phải copy
```

---

### 4. Quy tắc 80/20 — Biết gì cần hiểu sâu, gì chỉ cần biết cách dùng

| Phải hiểu sâu (20%) | Chỉ cần biết cách dùng (80%) |
| :--- | :--- |
| LangGraph State + Conditional Edges | Nginx config syntax |
| Qdrant Hybrid Search concept | Docker Compose YAML |
| Ragas metric là gì và tại sao | GitHub Actions workflow |
| Langfuse Trace/Span/Generation | Certbot SSL command |
| FastAPI Dependency Injection | Redis cache pattern |

> Nếu có gì không hiểu ở cột phải → Google + hỏi AI → Copy → Chạy được → Tiếp tục. Không cần dừng lại học sâu.

---

### 5. Một câu hỏi để unlock khi bị tắc

Mỗi khi không biết bắt đầu từ đâu, hỏi bản thân:

> **"Phiên bản ngu nhất, xấu nhất của tính năng này trông như thế nào?"**

- Agent phức tạp quá? → Bắt đầu bằng 1 hàm Python gọi thẳng LLM.
- RAG pipeline phức tạp quá? → Bắt đầu bằng 1 list cứng 5 câu context.
- CI/CD phức tạp quá? → Bắt đầu bằng 1 file `deploy.sh` chạy tay qua SSH.

**Sau khi phiên bản ngu đó chạy được → nâng dần lên.** Không bao giờ thiết kế hệ thống hoàn hảo từ ngày đầu.