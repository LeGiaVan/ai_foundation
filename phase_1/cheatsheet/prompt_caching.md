# 🚀 Cẩm nang toàn tập: Prompt Caching (Anthropic Claude)

Prompt Caching là một tính năng cực kỳ mạnh mẽ giúp **giảm chi phí** và **tăng tốc độ phản hồi** (giảm latency) khi bạn phải gửi đi gửi lại một lượng lớn bối cảnh (context) cho LLM.

---

## 1. Bản chất của Prompt Caching (Cơ chế Prefix-based)

Bạn không thể chọn bất kỳ đoạn text nào ở giữa hội thoại để cache. Caching hoạt động theo nguyên tắc **Khớp tiền tố (Prefix-based)**.

Nghĩa là: Model sẽ đọc request của bạn từ trên xuống dưới. Tại điểm bạn đặt cờ (breakpoint) yêu cầu cache, model sẽ lấy **toàn bộ mọi thứ từ điểm đó trở ngược lên trên cùng** để tạo thành một "chìa khóa" (Cache Key). 
Ở lần gọi tiếp theo, nếu mọi thứ từ trên cùng đến điểm đó **GIỐNG NHAU Y ĐÚC 100%**, cache sẽ được kích hoạt (Cache Hit). Chỉ cần sai khác 1 dấu phẩy, toàn bộ cache sẽ bị vô hiệu hóa (Cache Miss).

**Thứ tự ưu tiên từ trên xuống dưới của một Request:**
1. `tools` (Định nghĩa công cụ)
2. `system` (Hướng dẫn hệ thống)
3. `messages` (Lịch sử hội thoại - User/Assistant)

---

## 2. Chi phí thực tế & Lợi ích kinh tế (Ví dụ với Claude 3.5 Sonnet)

Đầu tiên, bạn cần nắm rõ bảng giá (tính trên 1 Triệu tokens):
- **Base Input (Gửi bình thường không cache):** `$3.00`
- **Cache Creation (Tạo cache mới lần đầu):** `$3.75` (Đắt hơn 25% so với bình thường, xem như phí "khởi tạo")
- **Cache Read (Đọc lại từ cache đã có):** `$0.30` (**RẺ HƠN 90%** so với bình thường)

- **Tuổi thọ (TTL):** Cache tồn tại trong **5 phút** kể từ lần truy cập cuối cùng. Mỗi lần bạn dùng lại (Cache Hit), đồng hồ 5 phút sẽ được reset.

### 💸 Bài toán Ví dụ Thực tế:
Giả sử bạn build một hệ thống RAG nội bộ. Mỗi lần User hỏi, bạn phải đính kèm 1 tài liệu dài **100.000 tokens** vào System Prompt. Có 10 câu hỏi được gửi liên tiếp.

**Kịch bản 1: KHÔNG dùng Prompt Caching**
- Mỗi lần hỏi gửi lại 100k tokens x 10 lần = 1.000.000 tokens.
- **Tổng chi phí Input:** `$3.00`
- **Tốc độ:** Rất chậm vì model phải đọc đi đọc lại tài liệu 100k tokens tới 10 lần.

**Kịch bản 2: CÓ dùng Prompt Caching**
- **Câu hỏi 1 (Lần đầu - Cache Miss):** Model đọc 100k tokens và ghi vào Cache. Phí tạo cache: 100k * ($3.75 / 1M) = `$0.375`
- **9 Câu hỏi sau (Cache Hit):** Tài liệu đã nằm trong Cache, bạn chỉ trả phí đọc (Read). Phí: 900k * ($0.30 / 1M) = `$0.27`
- **Tổng chi phí Input:** $0.375 + $0.27 = `$0.645`
- **Kết luận:** Bạn giảm từ $3.00 xuống còn $0.645 (Tiết kiệm gần **80% tổng chi phí**), và ở 9 câu hỏi sau, thời gian bắt đầu trả lời (TTFB) sẽ diễn ra *gần như ngay lập tức*!

---

## 3. Điều kiện & Giới hạn

1. **Số lượng điểm đánh dấu (Breakpoints):** Tối đa **4 điểm** (`cache_control`) trong một request.
2. **Số Token tối thiểu để kích hoạt Cache:**
   - Claude 3.5 Sonnet / Haiku: Cần ít nhất **1024 tokens**.
   - Claude 3 Opus: Cần ít nhất **2048 tokens**.
   *(Nếu bạn đánh dấu cache cho đoạn text ngắn hơn mức này, API vẫn chạy bình thường nhưng cache sẽ không được tạo).*

---

## 4. Cách triển khai Code (Python)

Để đánh dấu một điểm cần Cache, bạn thêm object `{"cache_control": {"type": "ephemeral"}}` vào cuối của khối dữ liệu (block) đó.

### Ví dụ 1: Caching System Prompt (Phổ biến nhất)
Thường dùng khi bạn có một bộ tài liệu công ty hoặc hướng dẫn system rất dài.

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "<tài_liệu_công_ty_rất_dài_hơn_1024_tokens>...",
            # Đặt cờ cache ở đây! Nó sẽ cache toàn bộ nội dung system này
            "cache_control": {"type": "ephemeral"}
        }
    ],
    messages=[
        {"role": "user", "content": "Tóm tắt chương 1 giúp tôi"}
    ]
)
```

### Ví dụ 2: Caching Tools
Nếu bạn cung cấp cho LLM hàng chục công cụ (Tools) phức tạp, việc này cũng ngốn rất nhiều Token.

```python
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=[
        {
            "name": "get_weather",
            "description": "Lấy thời tiết...",
            "input_schema": {...}
        },
        {
            "name": "search_db",
            "description": "Tìm kiếm cơ sở dữ liệu...",
            "input_schema": {...},
            # Đặt cờ cache ở công cụ CÙỐI CÙNG trong danh sách.
            # Model sẽ cache toàn bộ danh sách tools ở trên.
            "cache_control": {"type": "ephemeral"}
        }
    ],
    messages=[{"role": "user", "content": "Thời tiết hôm nay?"}]
)
```

### Ví dụ 3: Caching Lịch sử hội thoại (Messages)
Thường dùng trong ứng dụng Chat dài, hoặc kỹ thuật Few-shot examples.

```python
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system="Bạn là AI trợ lý.",
    messages=[
        {"role": "user", "content": "Xin chào"},
        {"role": "assistant", "content": "Chào bạn, tôi giúp gì được?"},
        {
            "role": "user", 
            "content": [
                {
                    "type": "text", 
                    "text": "Câu hỏi mới của tôi là...",
                    # Đánh dấu cache ở tin nhắn gần nhất.
                    # Nó sẽ cache TẤT CẢ system + tools + lịch sử chat từ trên xuống dưới điểm này.
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        }
    ]
)
```

---

## 5. Làm sao biết Cache có hoạt động hay không?

Khi nhận được `response` từ Anthropic, bạn hãy kiểm tra thuộc tính `usage`:

```python
print(response.usage)
```

**Kết quả trả về:**
```json
{
  "input_tokens": 50,                      // Phần text KHÔNG được cache (ví dụ: câu hỏi mới của User)
  "cache_read_input_tokens": 1800,         // Tuyệt vời! Bạn vừa đọc 1800 tokens từ cache với giá rẻ bèo.
  "cache_creation_input_tokens": 0,        // Bằng 0 nghĩa là không tạo cache mới (đã dùng lại cache cũ).
  "output_tokens": 200
}
```
**Công thức:** Tổng số Input = `input_tokens` + `cache_read_input_tokens` + `cache_creation_input_tokens`.

---

## 6. Điều gì làm MẤT Cache (Cache Invalidation) - Kẻ thù của dân Dev

Bạn tốn tiền tạo Cache, nhưng lần gọi tiếp theo bị **Cache Miss** (Mất trắng, tạo lại từ đầu). Lý do là vì bạn đã phá vỡ nguyên tắc **"Khớp tiền tố"**:

| Hành động thay đổi | Tools Cache | System Cache | Messages Cache |
|---|---|---|---|
| **Sửa/Thêm/Xóa định nghĩa Tool** | ❌ Mất toàn bộ | ❌ Mất toàn bộ | ❌ Mất toàn bộ |
| **Sửa Text ở System Prompt** | ✅ Tools giữ lại | ❌ Mất System | ❌ Mất Messages |
| **Đổi tham số `tool_choice`** | ✅ Tools giữ lại | ✅ System giữ lại | ❌ Mất Messages |
| **User Upload ảnh mới vào chat** | ✅ Tools giữ lại | ✅ System giữ lại | ❌ Mất Messages |

**👉 Rút ra bài học cốt lõi:**
Những thứ gì CỐ ĐỊNH NHẤT (ít bị thay đổi nhất qua các lượt chat) thì phải đẩy lên TẦNG TRÊN CÙNG (Tools -> System -> Few-shot Messages). Những thứ hay thay đổi (câu hỏi của User) thì đẩy xuống DƯỚI CÙNG (sau điểm `cache_control`).

---

## 7. Áp dụng thực tế vào RAG (Truy xuất tài liệu)

Trong hệ thống RAG (Retrieval-Augmented Generation), mỗi khi User đặt câu hỏi, hệ thống sẽ tìm ra các mảnh tài liệu (chunks) khác nhau. Do tài liệu thay đổi liên tục, nếu bạn nhét tài liệu vào System Prompt, bạn sẽ phá nát Cache!

**Cấu trúc chuẩn cho RAG với Prompt Caching:**

```python
def rag_query(question: str, retrieved_chunks: list[str]):
    # 1. System Prompt CỐ ĐỊNH (Luôn được Cache)
    system_prompt = [
        {
            "type": "text",
            "text": "Đây là hướng dẫn hành vi dài 2000 tokens không bao giờ đổi...",
            "cache_control": {"type": "ephemeral"} # Đặt cờ cache 1 ở đây
        }
    ]
    
    # 2. Nội dung thay đổi (KHÔNG Cache)
    context_text = "\n".join(retrieved_chunks)
    
    # 3. Lắp ráp tin nhắn
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        system=system_prompt, 
        messages=[
            {
                "role": "user",
                "content": f"Dựa vào tài liệu sau: {context_text}\n\nTrả lời: {question}"
            }
        ]
    )
    return response
```
*Với cách này, hướng dẫn hành vi (system_prompt) sẽ luôn ăn Cache, còn tài liệu được nạp động vào thông qua câu hỏi của User ở dưới cùng!*
