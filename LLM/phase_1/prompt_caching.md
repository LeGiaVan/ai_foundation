# Chi tiết về Prompt Caching (Token Caching) — dựa trên tài liệu chính thức Claude API

Tài liệu này giải thích chi tiết cơ chế hoạt động của Prompt Caching (Bộ nhớ đệm cho câu lệnh) của Claude API. Mục tiêu là giúp hiểu rõ cách hệ thống lưu trữ và tái sử dụng context, từ đó tối ưu hóa chi phí và tốc độ phản hồi.

## 1. Cơ chế hoạt động cốt lõi

Khi bạn gửi request có đánh dấu cờ `cache_control` tại một vị trí cụ thể, hệ thống sẽ thực hiện các bước sau:
1. **Quét tiền tố (Prefix matching):** Hệ thống kiểm tra xem toàn bộ chuỗi văn bản tính từ đầu request cho đến vị trí bạn đặt **cache breakpoint** (điểm đánh dấu) đã từng xuất hiện trong một request gần đây hay chưa.
2. **Cache Hit (Trúng cache):** Nếu đoạn văn bản này đã tồn tại trong bộ nhớ đệm, mô hình sẽ tái sử dụng ngay lập tức các trạng thái tính toán trước đó. Điều này giúp giảm thiểu thời gian xử lý (nhanh hơn) và giảm chi phí API (rẻ hơn đáng kể).
3. **Cache Miss (Trượt cache):** Nếu không tìm thấy, hệ thống buộc phải xử lý toàn bộ đoạn văn bản từ đầu như bình thường. Tuy nhiên, sau khi xử lý, nó sẽ **ghi lại (cache write)** kết quả tính toán vào bộ nhớ đệm để các request tương lai có thể tái sử dụng.

### Cấu trúc phân cấp nghiêm ngặt: `tools → system → messages`
Đây là nguyên tắc quan trọng nhất. Claude xử lý các thành phần của prompt theo thứ tự từ trên xuống dưới. Do cơ chế cache yêu cầu phần tiền tố (prefix) phải giống hệt nhau, mọi thay đổi ở phần "trên" sẽ vô hiệu hóa (invalidate) toàn bộ cache của phần "dưới":

- **Nếu đổi `tools` (ví dụ: thêm công cụ mới):** Toàn bộ bộ nhớ đệm bị xóa sạch. Hệ thống phải đọc lại từ đầu `tools`, `system`, và `messages`.
- **Nếu đổi `system`, giữ nguyên `tools`:** Cache của `tools` vẫn được tái sử dụng. Nhưng cache của `system` và `messages` bị vô hiệu hóa.
- **Nếu đổi `messages`, giữ nguyên `tools` và `system`:** Đây là kịch bản tối ưu nhất. Hệ thống tái sử dụng toàn bộ định nghĩa công cụ và hướng dẫn hệ thống, chỉ tốn chi phí xử lý câu hỏi mới.

> **💡 Ví dụ minh họa trực quan:**
> Hãy tưởng tượng AI là một người đầu bếp. `Tools` là danh sách đồ dùng nhà bếp, `System` là cuốn cẩm nang nấu ăn, và `Messages` là đơn đặt món. Đầu bếp luôn đọc và học thuộc lòng theo thứ tự từ trên xuống. Nếu bạn chỉ thay đổi đơn đặt món (Messages), đầu bếp vẫn nhớ công cụ và cẩm nang. Nhưng nếu bạn thay đổi cẩm nang (System), đầu bếp buộc phải học lại cẩm nang mới và quên luôn đơn đặt món cũ. Dĩ nhiên, nếu bạn đổi bộ công cụ nhà bếp (Tools), đầu bếp sẽ phải học lại mọi thứ từ đầu.

## 2. Hai cách bật caching

### Cách 1: Automatic caching (Gắn cờ tự động - Khuyên dùng)

**Đặc điểm:** Đơn giản, hệ thống tự quản lý. Lý tưởng cho các ứng dụng Chatbot (hội thoại nhiều lượt).

```python
response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=1024,
    cache_control={"type": "ephemeral"},  # Đặt cờ ở cấp độ cao nhất (TOP-LEVEL)
    system="Bạn là trợ lý phân tích văn học...",
    messages=[
        {"role": "user", "content": "Phân tích chủ đề chính trong Kiêu hãnh và Định kiến"}
    ]
)
```

**Cách hoạt động chi tiết:** 
Khi bạn đặt `cache_control` ở cấp độ cao nhất của request, hệ thống sẽ **tự động** phân tích toàn bộ prompt và tìm ra vị trí tối ưu nhất ở dưới cùng (khối văn bản cuối cùng có thể cache) để đặt breakpoint.
Hơn nữa, khi hội thoại dài ra qua nhiều lượt chat (context tăng dần), điểm breakpoint này sẽ **tự động trượt xuống dưới** để bao gồm cả lịch sử chat mới nhất. Bạn không cần phải tự mình viết code theo dõi xem nên đặt cache ở đoạn nào.

> **Ví dụ minh họa cơ chế "Tự động trượt":**
> Hãy tưởng tượng bạn đang viết ứng dụng Chat, và bạn luôn truyền `cache_control={"type": "ephemeral"}` ở cấp độ cao nhất trong mọi request.
> 
> **💬 Lượt chat 1:**
> 1. `System:` "Bạn là trợ lý..."
> 2. `User:` "Hà Nội ở đâu?"
> ➔ *Hệ thống tự động đánh dấu (đặt breakpoint) ở cuối dòng số 2. Toàn bộ Lượt 1 được lưu vào Cache.*
> 
> **💬 Lượt chat 2 (Gửi kèm lịch sử của lượt 1):**
> 1. `System:` "Bạn là trợ lý..."  *(✅ Trúng cache)*
> 2. `User:` "Hà Nội ở đâu?" *(✅ Trúng cache)*
> 3. `Assistant:` "Hà Nội ở Việt Nam." *(Nội dung mới)*
> 4. `User:` "Có đặc sản gì?" *(Nội dung mới)*
> ➔ *Hệ thống nhận ra đoạn hội thoại đã dài ra. Nó tự động dời (trượt) điểm breakpoint từ dòng 2 xuống cuối dòng 4. Lịch sử cũ không cần xử lý lại, và phần lịch sử mới tự động được ghi nhớ thêm vào Cache.*

### Cách 2: Explicit breakpoints (Gắn cờ tường minh/thủ công)

**Đặc điểm:** Cho phép kiểm soát chi tiết. Phù hợp khi prompt của bạn chứa nhiều đoạn văn bản khác nhau, trong đó có đoạn hoàn toàn tĩnh (tài liệu tham khảo dài) và đoạn thường xuyên thay đổi (dữ liệu theo ngày).

```python
response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": long_fixed_instructions,         # Đoạn hướng dẫn cố định, rất dài
            "cache_control": {"type": "ephemeral"}   # Đánh dấu breakpoint NGAY TẠI ĐÂY
        },
        {
            "type": "text",
            "text": daily_changing_data              # Dữ liệu thay đổi mỗi ngày (không đánh dấu)
        }
    ],
    messages=[{"role": "user", "content": user_question}]
)
```

**Cách hoạt động chi tiết:** 
Thay vì để hệ thống tự quyết, bạn nhúng thẳng cờ `cache_control` vào ngay khối văn bản cụ thể mà bạn muốn ghi nhớ. Trong ví dụ trên, hệ thống sẽ luôn lưu trữ phần `long_fixed_instructions`. Phần `daily_changing_data` thay đổi mỗi ngày sẽ được xử lý riêng. Điều này giúp bảo vệ cache của phần hướng dẫn dài không bị phá vỡ bởi dữ liệu hàng ngày.

**Lưu ý:** Bạn chỉ được phép sử dụng **tối đa 4 breakpoint tường minh** trong 1 request. Bạn cũng có thể kết hợp cả 2 cách (tuy nhiên, cờ automatic sẽ chiếm mất 1 trong 4 slot này).

## 3. Ngưỡng token tối thiểu để cache hoạt động

Một điểm kỹ thuật rất quan trọng nhưng thường bị bỏ sót: **Nếu đoạn prompt bạn muốn cache quá ngắn, hệ thống sẽ phớt lờ cờ `cache_control` và không báo lỗi.** Nó chỉ âm thầm xử lý như một request bình thường.

| Model | Ngưỡng độ dài tối thiểu |
|---|---|
| Claude Opus 5, Fable 5, Mythos 5 | 512 token |
| Claude Sonnet 5, Sonnet 4.6, Opus 4.8 | 1,024 token |
| Claude Opus 4.6, Opus 4.5 | 4,096 token |
| Claude Haiku 4.5 | 4,096 token |

**Hệ quả:** Nếu bạn dùng `claude-sonnet-5` và gắn cờ cho một system prompt chỉ có 300 token, việc cache sẽ hoàn toàn vô nghĩa. Để kiểm tra tính năng cache có đang chạy hay không, hãy xem hai thông số `cache_creation_input_tokens` (số lượng token mới được ghi) và `cache_read_input_tokens` (số lượng token được đọc từ cache) trong object phản hồi của API. Nếu cả hai đều bằng 0, prompt của bạn chưa đạt ngưỡng độ dài tối thiểu.

## 4. Cơ chế "Lookback Window" (Cửa sổ quét ngược)

Đây là phần phức tạp và dễ gây hiểu nhầm nhất. Hệ thống không tự động cache toàn bộ phần "nội dung ổn định" từ đầu đến cuối một cách chung chung. Nó chỉ tìm kiếm lại các **điểm ghi nhớ (entries)** đã được tạo ra từ trước đó, và nó chỉ quét lùi lại tối đa **20 block văn bản (messages/blocks)**.

**Ví dụ minh họa:**
- **Lượt 1**: Request có 10 block. Bạn đặt breakpoint ở block 10. Hệ thống không tìm thấy cache cũ nên tạo một entry mới tại block 10.
- **Lượt 2**: Hội thoại dài ra thành 15 block. Bạn đặt breakpoint ở block 15. Hệ thống quét ngược từ block 15 lùi về block 10. Nó **tìm thấy** entry ở block 10 (trúng cache). Hệ thống chỉ cần đọc mới từ block 11-15, và tạo một entry mới tại block 15.
- **Lượt 3**: Hội thoại tăng vọt lên 35 block. Bạn đặt breakpoint ở block 35. Hệ thống quét ngược 20 vị trí (từ 35 lùi về 16). Nó **không tìm thấy** entry ở block 15, vì block 15 đã nằm ngoài cửa sổ quét 20 block. Kết quả: **Trượt cache hoàn toàn**. Bạn phải trả tiền để AI xử lý lại toàn bộ 35 block từ đầu.

**Giải pháp:** Nếu ứng dụng của bạn chèn thêm một lượng lớn dữ liệu (nhiều hơn 20 blocks) trong một lượt chat, cache sẽ liên tục bị đứt gãy. Bạn cần bổ sung thêm các Explicit breakpoint ở giữa để "đón đầu" bộ quét này.

## 5. Lỗi phổ biến nhất: Kẹp nội dung động vào trước Breakpoint

Lỗi này xảy ra khi bạn vô tình đưa dữ liệu thay đổi liên tục vào phần văn bản đứng trước dấu `cache_control`.

```python
# ❌ CÁCH LÀM SAI — Breakpoint bị vô hiệu hóa bởi nội dung động
system = [
    {"type": "text", "text": long_static_context},        # Đoạn 1: Rất dài, ổn định
    {"type": "text", "text": f"Thời gian: {now()}",       # Đoạn 2: Thay đổi TỪNG GIÂY
     "cache_control": {"type": "ephemeral"}}              # Breakpoint đặt SAI chỗ
]
```
**Lý do sai:** Cơ chế cache băm (hash) toàn bộ nội dung từ đầu cho đến breakpoint. Dù `long_static_context` giống hệt nhau, nhưng vì `now()` liên tục thay đổi, chuỗi mã băm sẽ luôn khác biệt mỗi khi bạn gọi API. Do đó, request này **không bao giờ** trúng cache.

```python
# ✅ CÁCH LÀM ĐÚNG — Tách biệt phần tĩnh và động
system = [
    {"type": "text", "text": long_static_context,
     "cache_control": {"type": "ephemeral"}},               # Đặt breakpoint ở khối cuối cùng KHÔNG ĐỔI
    {"type": "text", "text": f"Thời gian: {now()}"}         # Phần động để phía sau, KHÔNG đánh dấu
]
```

## 6. Phân tích chi phí (Giá cả)

Cache write (ghi vào bộ nhớ) luôn đắt hơn, nhưng Cache read (đọc từ bộ nhớ) lại rẻ hơn đáng kinh ngạc. Lấy ví dụ với model Claude Sonnet 5:

| Loại xử lý | Giá trên 1 triệu token | Ghi chú |
|---|---|---|
| Input thông thường | $2 | Giá tiêu chuẩn khi không dùng cache |
| Ghi Cache (Lưu trong 5 phút) | $2.50 (×1.25) | Đắt hơn 25% so với giá tiêu chuẩn |
| Ghi Cache (Lưu trong 1 giờ) | $4 (×2) | Đắt gấp đôi giá tiêu chuẩn |
| **Đọc Cache (Sử dụng lại)** | **$0.20 (×0.1)** | **Rẻ hơn 10 lần so với input thường** |
| Output (Văn bản sinh ra) | $10 | Không bị ảnh hưởng bởi caching |

**Kết luận tài chính:** Chi phí lần đầu tiên ghi cache đắt hơn một chút (25%), nhưng những lần đọc lại tiếp theo rẻ hơn 10 lần. Điểm hòa vốn diễn ra ngay ở **lần gọi API thứ 2**. Nếu prompt của bạn được sử dụng từ 2 lần trở lên, bạn chắc chắn tiết kiệm được tiền.

## 7. Thời gian sống của Cache (TTL) và Gia hạn

```python
"cache_control": {"type": "ephemeral", "ttl": "1h"}
```

- **Mặc định là 5 phút:** Lưu ý quan trọng, thời gian 5 phút được tính từ **thời điểm bắt đầu** xử lý request, không phải thời điểm kết thúc. Nếu request của bạn mất 4 phút để AI trả lời, bạn chỉ còn khoảng 1 phút để gửi request tiếp theo hòng tận dụng bộ nhớ đệm này.
- **Gia hạn tự động:** Mỗi khi một request trúng cache (đọc thành công), TTL sẽ được tự động đếm lại từ đầu hoàn toàn miễn phí.
- **Tùy chọn 1 giờ (`ttl`: "1h"):** Hữu ích cho các luồng Agent chạy ngầm tốn thời gian giữa các bước, hoặc các hệ thống chatbot nơi người dùng có thể mất nhiều hơn 5 phút để gõ câu phản hồi tiếp theo. Lưu ý phí ghi cache 1 giờ sẽ đắt hơn (xem bảng giá).

## 8. Theo dõi và Giám sát hiệu suất Cache

Sau khi gọi API, bạn có thể kiểm tra hiệu suất cache thông qua đối tượng `usage` trả về:

```python
print(response.usage)
# {
#   "input_tokens": 50,                      # Token động, xử lý bình thường (Sau breakpoint)
#   "cache_read_input_tokens": 1800,         # Token TÁI SỬ DỤNG (Giá cực rẻ)
#   "cache_creation_input_tokens": 248,      # Token MỚI GHI VÀO cache (Giá đắt hơn 1 chút)
#   "output_tokens": 503
# }
```
**Cách đánh giá:** Trong các lượt chat tiếp theo, nếu chỉ số `cache_read_input_tokens` ở mức cao và `cache_creation_input_tokens` rất thấp hoặc bằng 0, nghĩa là chiến lược thiết lập cache của bạn đang hoạt động cực kỳ hoàn hảo.

## 9. Điều gì làm mất Cache (Invalidation triggers)

Bảng dưới đây tóm tắt các thao tác làm vỡ cấu trúc cache:

| Hành động thay đổi | Cache của Tools | Cache của System | Cache của Messages |
|---|---|---|---|
| Sửa đổi định nghĩa Tool (Tool definition) | ✘ (Mất) | ✘ (Mất) | ✘ (Mất) |
| Bật/tắt Web Search nội bộ | ✓ (Giữ) | ✘ (Mất) | ✘ (Mất) |
| Đổi `tool_choice` (Bắt buộc dùng tool) | ✓ (Giữ) | ✓ (Giữ) | ✘ (Mất) |
| Thêm hoặc xóa hình ảnh | ✓ (Giữ) | ✓ (Giữ) | ✘ (Mất) |

*(Lưu ý: Thay đổi định nghĩa công cụ là thao tác phá hủy diện rộng nhất).*

## 10. Ứng dụng thực tế

### Áp dụng vào hệ thống RAG (Retrieval-Augmented Generation)

```python
FIXED_SYSTEM_PROMPT = """Bạn là trợ lý trả lời dựa trên tài liệu công ty... [rất dài]"""

def rag_query(question: str, retrieved_chunks: list[str]):
    # Các đoạn chunk được search ra thay đổi liên tục -> Đặt ở vị trí Messages
    context = "\n\n".join(retrieved_chunks)  
    
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1000,
        system=[
            {
                "type": "text",
                "text": FIXED_SYSTEM_PROMPT,          
                "cache_control": {"type": "ephemeral"} # Cache phần System tĩnh
            }
        ],
        messages=[
            {"role": "user", "content": f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {question}"}
        ]
    )
    return response.content[0].text
```
Nhờ kiến trúc này, toàn bộ khối `FIXED_SYSTEM_PROMPT` được cache ổn định (chỉ tốn $0.20/MTok cho các câu hỏi sau), trong khi `context` thay đổi theo từng câu hỏi được xử lý bình thường.

### Pre-warming (Làm nóng trí nhớ trước)

Nếu bạn biết trước hệ thống sắp có lượng lớn người dùng truy cập (ví dụ: mở ứng dụng buổi sáng), bạn có thể chủ động "làm nóng" bộ nhớ đệm trước để tránh độ trễ cao cho người dùng đầu tiên.

```python
def prewarm_cache():
    client.messages.create(
        model="claude-sonnet-5",
        max_tokens=0,                          # Đặt bằng 0 để không sinh output, chỉ ghi cache
        system=[{
            "type": "text",
            "text": FIXED_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"}
        }],
        messages=[{"role": "user", "content": "warmup"}]  # Chuỗi placeholder, hệ thống không trả lời
    )

# Chạy ngầm hàm này mỗi < 5 phút để giữ cho cache luôn "ấm" và sẵn sàng.
prewarm_cache()
```