# Giai đoạn 1 — Tuần 2-3: FastAPI + LLM API Integration

## Mục tiêu tổng thể

Sau giai đoạn này, bạn có thể tự xây **1 REST API bằng FastAPI, đóng vai trò "lớp bọc" (wrapper) cho Claude API**, hỗ trợ: gọi LLM cơ bản, structured output, streaming, và tối ưu chi phí bằng prompt caching. Đây là kỹ năng nền tảng của gần như mọi sản phẩm AI backend thực tế.

**Sản phẩm đầu ra (capstone project):** 1 "Chat API" hoàn chỉnh — chi tiết ở cuối tài liệu.

## Tiền đề (đã học ở Giai đoạn 0)

Type hint, async/await, coroutine, event loop, class/OOP, .env. Nếu quên phần nào, ôn nhanh trước khi bắt đầu — cả tuần 2-3 dùng lại liên tục các khái niệm này (đặc biệt: Pydantic model = class, endpoint async = `async def`, API key đọc từ `.env`).

---

## TUẦN 2 — FastAPI nền tảng + Anthropic API cơ bản

### Ngày 1: FastAPI cơ bản — path/query parameter, request body

**Mục tiêu:** Viết được 1 API có GET/POST, phân biệt được 3 cách nhận dữ liệu đầu vào.

**Khái niệm cần nắm:**
- Endpoint là gì, cách FastAPI gắn hàm Python vào 1 địa chỉ + method
- **Path parameter** — dữ liệu nằm trong URL: `/items/{item_id}`
- **Query parameter** — dữ liệu sau dấu `?`: `/items?skip=0&limit=10`
- **Request body** — dữ liệu gửi kèm trong POST/PUT, thường là JSON

**Tài liệu đọc:**
- FastAPI tutorial — "First Steps" → "Path Parameters" → "Query Parameters" → "Request Body" (4 mục đầu tiên của https://fastapi.tiangolo.com/tutorial/)

**Bài tập:**
1. Endpoint `GET /` trả về `{"message": "Hello"}`
2. Endpoint `GET /items/{item_id}` — trả về `{"item_id": item_id}`
3. Endpoint `GET /items` với query param `skip: int = 0`, `limit: int = 10` — trả về `{"skip": skip, "limit": limit}`
4. Endpoint `POST /items` nhận request body dạng dict thô (chưa dùng Pydantic) — trả lại nguyên dữ liệu đó

**Tiêu chí hoàn thành:** Chạy được cả 4 endpoint, gọi thử qua `/docs`, phân biệt rõ path param khác query param khác body ở chỗ nào (cú pháp khai báo trong hàm Python).

---

### Ngày 2: Pydantic model cho request body + uvicorn + Swagger UI

**Mục tiêu:** Thay request body thô bằng Pydantic model để có validate tự động; hiểu rõ uvicorn chạy gì, Swagger UI dùng để làm gì.

**Khái niệm cần nắm:**
- **Pydantic model** làm request body — FastAPI tự validate, tự parse JSON → object
- **uvicorn** — ASGI server, chương trình thực sự "chạy" app FastAPI của bạn (`uvicorn main:app --reload`); `--reload` để tự nạp lại khi sửa code
- **Swagger UI (`/docs`)** — trang tài liệu tự sinh từ JSON Schema của Pydantic model, có nút "Try it out"
- Còn có `/redoc` — 1 kiểu tài liệu khác, tự sinh song song

**Tài liệu đọc:**
- FastAPI tutorial — "Request Body" (phần dùng Pydantic model, không chỉ dict) + "Running the app" (uvicorn)
- freeCodeCamp video — đoạn build endpoint POST + chạy uvicorn (khoảng 30-60 phút đầu video)

**Bài tập:**
1. Định nghĩa Pydantic model `Item` (`name: str`, `price: float`, `description: str | None = None`)
2. Endpoint `POST /items` nhận `Item`, trả về `Item` kèm thêm field `total_with_tax` (tính = `price * 1.1`)
3. Chủ động gửi sai kiểu (`price` là chữ) qua `/docs`, đọc và giải thích lỗi 422 trả về
4. So sánh `/docs` và `/redoc` — 2 giao diện có gì giống/khác

**Tiêu chí hoàn thành:** Tự chạy được `uvicorn`, tự đọc hiểu lỗi 422 mà không cần tra cứu thêm.

---

### Ngày 3: Dependency Injection (`Depends`)

**Mục tiêu:** Hiểu và dùng được `Depends()` cho 2 việc phổ biến nhất: chia sẻ logic dùng lại nhiều lần, và xác thực (auth) đơn giản.

**Khái niệm cần nắm:**
- **Dependency injection** — thay vì mỗi endpoint tự viết logic lặp lại (vd: kiểm tra API key, tạo kết nối client), bạn viết 1 hàm riêng, "tiêm" (inject) vào endpoint qua `Depends(...)`
- FastAPI tự động gọi hàm dependency trước khi chạy endpoint, đưa kết quả vào tham số

**Ví dụ tối thiểu để hình dung trước khi đọc tài liệu:**
```python
from fastapi import Depends

def get_query_limit(limit: int = 10) -> int:
    return min(limit, 100)   # giới hạn tối đa, dùng lại được ở nhiều endpoint

@app.get("/items")
def list_items(limit: int = Depends(get_query_limit)):
    return {"limit": limit}
```

**Tài liệu đọc:**
- FastAPI tutorial — "Dependencies" (mục "First Steps" + "Classes as Dependencies" trong phần Dependencies)

**Bài tập:**
1. Viết dependency `verify_api_key` — đọc header `X-API-Key`, so sánh với 1 giá trị cố định (giả lập), nếu sai thì raise `HTTPException(401)`
2. Áp dụng `Depends(verify_api_key)` cho toàn bộ endpoint trong app (dùng tham số `dependencies=[Depends(...)]` ở cấp `app` hoặc `router`)
3. Viết dependency `get_settings()` đọc `.env` (dùng lại kiến thức Giai đoạn 0), trả về object chứa `ANTHROPIC_API_KEY` — dependency này sẽ dùng lại ở Ngày 4

**Tiêu chí hoàn thành:** Giải thích được bằng lời (không nhìn code): "Dependency injection giải quyết vấn đề gì, nếu không dùng nó thì code sẽ tệ ra sao."

---

### Ngày 4: Anthropic API fundamentals — gọi LLM cơ bản

**Mục tiêu:** Gọi được Claude qua API thật, hiểu cấu trúc 1 request/response, phân biệt system prompt và user prompt.

**Khái niệm cần nắm:**
- Cấu trúc cơ bản của Messages API: `model`, `max_tokens`, `messages` (list các `{role, content}`)
- **System prompt** — chỉ dẫn "vai trò/quy tắc chung" cho AI trong suốt cuộc hội thoại, tách riêng khỏi `messages`
- **User prompt** — nội dung câu hỏi/yêu cầu cụ thể của người dùng, nằm trong `messages` với `role: "user"`
- Khác biệt: system prompt định hình "AI là ai, phải cư xử thế nào"; user prompt là "việc cụ thể cần làm ngay bây giờ"

**Tài liệu đọc:**
- Anthropic Courses — module đầu tiên: **API fundamentals** (làm theo đúng thứ tự trong repo, đây là môn đầu tiên trong 5 môn)
- Anthropic docs — phần giới thiệu Messages API (tra trong docs chính thức nếu repo có link)

**Bài tập:**
1. Cài `anthropic` SDK, đọc API key từ `.env` (dùng dependency `get_settings()` viết ở Ngày 3)
2. Viết 1 hàm Python (chưa cần FastAPI) gọi Claude, có `system` + 1 `user message`, in ra câu trả lời
3. Thử đổi `system` prompt (vd: "trả lời như 1 hải tặc" vs "trả lời trang trọng, ngắn gọn") với cùng 1 câu hỏi, so sánh kết quả — tự cảm nhận system prompt ảnh hưởng gì
4. Bọc hàm gọi Claude ở bước 2 vào 1 endpoint FastAPI: `POST /chat` — nhận `{"message": str}`, trả về `{"reply": str}`

**Tiêu chí hoàn thành:** `POST /chat` chạy được qua `/docs`, gọi Claude thật, có tách `system` prompt riêng.

---

### Ngày 5: Tokenization, context window

**Mục tiêu:** Hiểu vì sao "độ dài" trong AI được đo bằng token chứ không phải ký tự/từ, và context window ảnh hưởng gì tới thiết kế API.

**Khái niệm cần nắm:**
- **Token** — đơn vị nhỏ mà LLM "đọc/viết", không phải lúc nào cũng là 1 từ (1 từ tiếng Việt có dấu thường tách thành nhiều token hơn tiếng Anh)
- **Context window** — tổng số token tối đa mà model có thể "nhìn  thấy" trong 1 lần gọi (gồm cả system prompt + toàn bộ lịch sử hội thoại + câu hỏi mới + phần trả lời)
- Hệ quả thực tế: hội thoại càng dài, gửi lại càng nhiều token cũ → tốn phí + có thể vượt giới hạn context window → cần chiến lược cắt bớt lịch sử cũ

**Tài liệu đọc:**
- Anthropic Courses — tiếp tục module **API fundamentals** (phần nói về token counting/context window nếu có trong module này)
- Anthropic docs — trang riêng về token counting (tra cứu trong docs chính thức)

**Bài tập:**
1. Dùng SDK, thử API đếm token (token counting) cho vài đoạn văn tiếng Việt và tiếng Anh cùng độ dài chữ — so sánh số token
2. Viết hàm ước lượng: với 1 danh sách tin nhắn (list message), tính tổng token gần đúng trước khi gửi request
3. (Thực hành tư duy, không cần code phức tạp) Thiết kế 1 chiến lược đơn giản: nếu lịch sử hội thoại vượt quá N token, cắt bớt các tin nhắn cũ nhất trước khi gửi

**Tiêu chí hoàn thành:** Giải thích được sự khác nhau giữa "số ký tự", "số từ", "số token" bằng ví dụ cụ thể tự đếm được.

### Checkpoint cuối tuần 2 (mini project)

Ghép lại toàn bộ tuần 2 thành 1 app FastAPI có:
- `POST /chat` — nhận `message: str`, optional `system_prompt: str`, gọi Claude, trả về `reply`
- Dùng `Depends()` để inject API key/client Claude (không tạo client mới mỗi request)
- Có validate bằng Pydantic cho cả request lẫn response
- Test được đầy đủ qua `/docs`

---

## TUẦN 3 — Prompt Engineering + Advanced API Features

### Ngày 6: Few-shot prompting

**Mục tiêu:** Biết cách "dạy" AI làm đúng định dạng/phong cách mong muốn bằng cách cho ví dụ mẫu, thay vì chỉ mô tả bằng lời.

**Khái niệm cần nắm:**
- **Zero-shot** — chỉ mô tả yêu cầu, không có ví dụ (cách làm mặc định, hay dùng nhất)
- **Few-shot** — đưa kèm vài ví dụ (input → output mẫu) trong prompt, giúp AI "bắt chước" đúng định dạng/phong cách, đặc biệt hữu ích khi mô tả bằng lời khó truyền đạt chính xác

**Tài liệu đọc:**
- Anthropic Courses — module **Prompt engineering interactive tutorial** (chương về few-shot / multishot prompting)
- Anthropic prompting best practices — mục "Use examples (multishot prompting)"

**Bài tập:**
1. Viết prompt zero-shot yêu cầu Claude phân loại cảm xúc bình luận (tích cực/tiêu cực/trung lập), thử vài câu, quan sát định dạng trả lời (có nhất quán không)
2. Viết lại thành few-shot — thêm 3 ví dụ mẫu (input → output đúng định dạng bạn muốn, vd: chỉ trả 1 từ, viết hoa) — so sánh độ nhất quán của output
3. Đưa phần few-shot này vào endpoint `POST /classify` trong FastAPI, `system_prompt` cố định gồm các ví dụ mẫu

**Tiêu chí hoàn thành:** Chỉ ra được ít nhất 1 trường hợp cụ thể few-shot cho kết quả nhất quán hơn zero-shot rõ rệt.

---

### Ngày 7: Chain-of-thought (CoT)

**Mục tiêu:** Biết khi nào và làm sao yêu cầu AI "suy nghĩ từng bước" để tăng độ chính xác với bài toán cần suy luận.

**Khái niệm cần nắm:**
- **Chain-of-thought** — kỹ thuật yêu cầu AI trình bày các bước suy luận trung gian trước khi đưa ra câu trả lời cuối, thay vì trả lời ngay
- Phù hợp cho: toán, logic, phân tích nhiều bước. Không cần thiết (thậm chí lãng phí token) cho: câu hỏi sự kiện đơn giản, phân loại ngắn

**Tài liệu đọc:**
- Anthropic Courses — tiếp module **Prompt engineering interactive tutorial** (chương chain-of-thought / "Let Claude think")
- Anthropic prompting best practices — mục "Chain of thought (CoT)"

**Bài tập:**
1. Cho Claude 1 bài toán đố (vd: bài toán tuổi tác, hoặc logic đơn giản có nhiều bước), hỏi trực tiếp không yêu cầu suy luận — quan sát có sai không
2. Hỏi lại, thêm chỉ dẫn "hãy suy nghĩ từng bước trước khi trả lời" — so sánh độ chính xác
3. Thử tách riêng: yêu cầu AI đưa phần suy luận vào 1 vùng riêng (vd: thẻ `<thinking>...</thinking>`), câu trả lời cuối vào vùng khác (vd: `<answer>...</answer>`) — để dễ tách phần "suy luận" ra khỏi phần "kết quả" khi code xử lý

**Tiêu chí hoàn thành:** Có ít nhất 1 ví dụ cụ thể cho thấy CoT sửa được câu trả lời sai thành đúng.

---

### Ngày 8: Structured output / JSON mode

**Mục tiêu:** Kết hợp Pydantic (đã học kỹ trước đó) với Claude để ép AI trả lời đúng cấu trúc JSON mong muốn, parse thẳng thành object Python.

**Khái niệm cần nắm:**
- Ôn nhanh: Pydantic tự sinh JSON Schema từ class → dùng schema đó "ra luật chơi" cho AI
- Với Claude cụ thể, cách phổ biến để ép structured output là dùng **tool use** — định nghĩa 1 "tool" có input schema đúng hình dạng bạn muốn, buộc Claude luôn gọi tool đó thay vì trả lời tự do bằng văn bản
- Vẫn cần bước validate/parse sau khi nhận kết quả (không phải lúc nào cũng đảm bảo tuyệt đối 100%)

**Tài liệu đọc:**
- Anthropic Courses — module **Tool use** (môn thứ 5 trong danh sách 5 khóa) — tập trung phần dùng tool để lấy structured output, không chỉ để "gọi hàm thật"
- Anthropic docs — trang Tool use / structured outputs (tra trong docs chính thức)

**Bài tập:**
1. Định nghĩa Pydantic model `ProductInfo` (`name: str`, `price: float`, `category: str`)
2. Viết 1 đoạn văn bản mô tả sản phẩm bằng lời tự nhiên (vd: quảng cáo), yêu cầu Claude "trích xuất" thông tin theo đúng cấu trúc `ProductInfo`
3. Dùng cơ chế tool use (hoặc structured output nếu SDK hỗ trợ trực tiếp) để nhận kết quả, sau đó `ProductInfo(**result)` để parse — thử với input "khó" (mô tả mơ hồ, thiếu giá) xem hành vi ra sao
4. Bọc thành endpoint `POST /extract` trong FastAPI: nhận đoạn văn bản, trả về `ProductInfo` đã trích xuất

**Tiêu chí hoàn thành:** `POST /extract` chạy được, trả về đúng JSON Schema của `ProductInfo`, có xử lý khi Claude trả thiếu trường (báo lỗi rõ ràng, không để app crash).

---

### Ngày 9: Streaming response

**Mục tiêu:** Trả lời "từng chữ một" (như giao diện chat thật) thay vì đợi Claude trả lời xong hết mới hiển thị — cả ở phía gọi Claude lẫn phía trả về cho client của chính API bạn.

**Khái niệm cần nắm:**
- **Streaming từ Claude** — thay vì đợi hết response, SDK cho phép nhận từng phần nhỏ (chunk) ngay khi model sinh ra, dùng `async with client.messages.stream(...)`
- **Streaming response ở FastAPI** — dùng `StreamingResponse`, cho phép API của bạn cũng trả dữ liệu dần dần cho client, không đợi toàn bộ xử lý xong
- Kết hợp cả 2: FastAPI nhận từng chunk từ Claude (async), rồi lập tức "chuyển tiếp" (forward) từng chunk đó ra client của mình — tạo hiệu ứng gõ chữ thời gian thực

**Tài liệu đọc:**
- Anthropic Courses — phần streaming trong module **API fundamentals** (nếu có) hoặc phần liên quan trong **Real world prompting**
- FastAPI docs — mục về `StreamingResponse` (không nằm trong tutorial cơ bản, tra trong "Advanced User Guide" trên trang FastAPI)
- Anthropic docs — trang Streaming Messages (tra trong docs chính thức)

**Bài tập:**
1. Viết 1 script Python (chưa cần FastAPI) gọi Claude streaming, in ra từng chunk ngay khi nhận được (không đợi xong)
2. Viết endpoint `POST /chat/stream` trong FastAPI, dùng `StreamingResponse`, forward từng chunk từ Claude ra ngoài
3. Test bằng `curl -N` (flag `-N` để tắt buffering, thấy rõ dữ liệu về dần dần) thay vì `/docs` (Swagger UI không hiển thị tốt streaming)
4. So sánh trải nghiệm: gọi `/chat` (non-streaming) vs `/chat/stream` với cùng 1 câu hỏi dài — cảm nhận độ trễ tới ký tự đầu tiên khác nhau ra sao

**Tiêu chí hoàn thành:** `POST /chat/stream` chạy được, quan sát được dữ liệu về theo từng chunk qua `curl -N`, giải thích được vì sao Swagger UI không phải công cụ tốt để test streaming.

---

### Ngày 10: Prompt caching

**Mục tiêu:** Hiểu cơ chế cache phần prompt tĩnh (system prompt dài, tài liệu tham chiếu cố định...) để giảm chi phí/độ trễ khi phần đó lặp lại giữa nhiều request.

**Khái niệm cần nắm:**
- Prompt caching hoạt động theo kiểu **cache tiền tố (prefix cache)** — phần đầu request giống hệt nhau giữa các lần gọi thì được "nhớ lại", không phải xử lý lại từ đầu
- Cần đánh dấu rõ ràng phần nào nên cache (thường là: system prompt dài, tài liệu tham chiếu cố định) — phần đó nên đặt ở **đầu** request, phần thay đổi mỗi lần (câu hỏi người dùng) đặt **sau**
- Cache có thời gian sống giới hạn (hết hạn sẽ phải tính lại từ đầu) — cụ thể thời hạn và cách khai báo, tra trong docs chính thức vì có thể thay đổi theo thời gian
- Phù hợp nhất khi: system prompt dài + lặp lại nhiều lần (chatbot có few-shot examples dài, RAG có tài liệu tham chiếu cố định)

**Tài liệu đọc:**
- Anthropic docs — trang Prompt caching (tra trong docs chính thức, tìm mục "build-with-claude/prompt-caching"; cú pháp cụ thể có thể đã cập nhật nên đọc bản mới nhất thay vì theo trí nhớ)

**Bài tập:**
1. Viết 1 system prompt dài (vd: dán nguyên 1 đoạn tài liệu vài trăm từ làm ngữ cảnh cố định)
2. Gọi Claude 2 lần liên tiếp với system prompt đó + câu hỏi khác nhau, **không** bật cache — ghi lại thời gian phản hồi
3. Bật cache cho phần system prompt tĩnh theo đúng cú pháp trong docs, gọi lại 2 lần — so sánh thời gian phản hồi lần gọi thứ 2 trở đi
4. Đọc phần `usage` trong response (số token được tính là cache hit / cache miss) — tự xác nhận cache có thực sự hoạt động không, đừng chỉ dựa vào cảm giác "có vẻ nhanh hơn"

**Tiêu chí hoàn thành:** Chỉ ra được bằng số liệu cụ thể (từ trường `usage` trong response) rằng cache đã được dùng ở lần gọi thứ 2.

---

### Ngày 11-13: Capstone Project — "Chat API"

Ghép toàn bộ kiến thức 2 tuần thành 1 dự án hoàn chỉnh. Đây là bài kiểm tra thực sự — nếu làm được hết, coi như đã nắm chắc Giai đoạn 1.

**Yêu cầu chức năng:**

| Endpoint | Method | Mô tả |
|---|---|---|
| `/health` | GET | Kiểm tra server sống, không cần auth |
| `/chat` | POST | Gửi 1 tin nhắn, nhận lại câu trả lời đầy đủ (non-streaming) |
| `/chat/stream` | POST | Giống trên, nhưng trả lời theo kiểu streaming |
| `/extract` | POST | Nhận đoạn văn bản tự do, trả về dữ liệu có cấu trúc (Pydantic model tự chọn chủ đề, vd: trích thông tin liên hệ, trích ý chính...) |

**Yêu cầu kỹ thuật (map đúng từng khái niệm đã học):**
- Toàn bộ request/response dùng **Pydantic model**, không dùng dict thô
- API key đọc từ **`.env`**, không hardcode
- Dùng **`Depends()`** để inject Claude client (tạo 1 lần, dùng lại nhiều request) và xác thực đơn giản (API key riêng của chính app bạn, khác API key gọi Claude)
- `/chat` có tham số **system prompt** tùy chỉnh được (không hardcode cứng)
- `/extract` áp dụng **structured output** (tool use + Pydantic)
- `/chat/stream` dùng **streaming response** thật sự (kiểm chứng bằng `curl -N`)
- System prompt dài (nếu có, vd: hướng dẫn hành vi chi tiết) áp dụng **prompt caching**
- Toàn bộ endpoint hiện đầy đủ, đúng, dễ hiểu trên **Swagger UI (`/docs`)** — có mô tả (`description`) rõ ràng cho từng field Pydantic

**Không bắt buộc nhưng nên thử nếu còn thời gian:**
- Áp dụng few-shot / chain-of-thought vào `/extract` nếu bài toán trích xuất cần suy luận nhiều bước
- Ước lượng và giới hạn context window (từ Ngày 5) — nếu tin nhắn quá dài, cắt bớt hoặc báo lỗi rõ ràng thay vì để lỗi từ Claude API dội ngược lên

### Ngày 14: Review và tổng kết

- Đọc lại toàn bộ code capstone, tự đặt câu hỏi: "Nếu bỏ Pydantic đi, chỗ nào sẽ dễ vỡ nhất?", "Nếu bỏ Depends() đi, code sẽ lặp lại ở đâu?"
- Viết 1 file README ngắn mô tả API (coi như luyện tập thêm kỹ năng tài liệu hoá — không bắt buộc dùng công cụ gì cầu kỳ)
- Tự đối chiếu với bảng "Definition of Done" bên dưới trước khi coi Giai đoạn 1 hoàn tất

---

## Definition of Done — Tự kiểm tra trước khi qua Giai đoạn 2

Với mỗi mục, tự hỏi: "Tôi có thể giải thích + viết code minh hoạ ngay mà không cần tra cứu không?"

- [ ] Phân biệt được path parameter / query parameter / request body chỉ bằng cách nhìn khai báo tham số hàm
- [ ] Giải thích được Pydantic model giúp gì khác so với dict thô trong request body
- [ ] Biết `uvicorn` dùng để làm gì, `--reload` khác gì khi không có nó
- [ ] Đọc hiểu Swagger UI, biết cách dùng "Try it out" để test nhanh
- [ ] Giải thích được dependency injection giải quyết vấn đề gì, viết được 1 dependency dùng `Depends()`
- [ ] Viết được request gọi Claude cơ bản, phân biệt rõ system prompt và user prompt
- [ ] Giải thích được token khác ký tự/từ ở điểm nào, context window ảnh hưởng gì tới thiết kế app
- [ ] Viết được 1 few-shot prompt và giải thích khi nào nó hữu ích hơn zero-shot
- [ ] Viết được prompt yêu cầu chain-of-thought và giải thích khi nào nên/không nên dùng
- [ ] Xây được 1 endpoint trả về structured output đúng theo Pydantic model
- [ ] Xây được 1 endpoint streaming, giải thích tại sao không test được bằng Swagger UI
- [ ] Giải thích cơ chế prefix cache của prompt caching, chỉ ra bằng số liệu `usage` là cache có hoạt động không

---

## Tổng hợp tài liệu tham khảo (dùng xuyên suốt 2 tuần)

| Nguồn | Dùng cho |
|---|---|
| [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/) | Ngày 1-3 (path/query/body, Pydantic, dependency injection) |
| [freeCodeCamp FastAPI video](https://www.youtube.com/watch?v=VirndPTeRaw) | Ngày 1-2, xem song song hoặc khi cần hình dung trực quan hơn văn bản |V
| [Anthropic Courses repo](https://github.com/anthropics/courses) | Ngày 4-10 — học đúng thứ tự: API fundamentals → Prompt engineering tutorial → Real world prompting → Prompt evaluations → Tool use |
| [Anthropic Prompting best practices](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) | Ngày 6-7 (few-shot, chain-of-thought), tra cứu nhanh khi cần |

**Lưu ý khi tra cứu docs chính thức của Anthropic:** cú pháp API (đặc biệt streaming, prompt caching, tool use) có thể thay đổi theo thời gian — luôn ưu tiên đọc bản docs mới nhất tại thời điểm bạn học, thay vì chỉ tin theo ví dụ cố định trong tài liệu cũ.