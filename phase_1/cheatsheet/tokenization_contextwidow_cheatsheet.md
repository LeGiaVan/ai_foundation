# 🧠 Tokens & Context Windows — Cheatsheet

> Tóm tắt nhanh dựa trên bài viết *"Tokens, Context Windows, and Why They Matter"* (dev.to, Satinath Mondal)

---

## 1. Token là gì?

- ❌ Sai lầm phổ biến: nghĩ **1 token = 1 từ**
- ✅ Thực tế: token là **đơn vị subword** — một từ có thể là nhiều token, hoặc nhiều từ ghép lại thành 1 token
- Ví dụ minh họa:
  - `"hello"` → 1 token
  - `"Hello"` (viết hoa) → token **khác** với `"hello"`
  - `"tokenization"` → 2 token (`token` + `ization`)
  - `"12345"` → thường tách thành nhiều token số
  - Text đa ngôn ngữ (tiếng Trung, tiếng Pháp...) thường tốn nhiều token hơn tiếng Anh

**Tại sao quan trọng?**
| Lý do | Ý nghĩa |
|---|---|
| 💰 Chi phí | API tính tiền theo **token**, không theo từ |
| 📏 Giới hạn context | Mỗi model có **trần token tối đa** (input + output) |
| 🎯 Chất lượng | Nhiều token hơn ≠ câu trả lời tốt hơn (viết súc tích tiết kiệm token hơn) |

---

## 2. Tokenization hoạt động thế nào?

Hầu hết LLM hiện đại (GPT-4, Claude, Llama) dùng thuật toán **BPE (Byte-Pair Encoding)**:

1. Bắt đầu từ từng **ký tự** riêng lẻ
2. Tìm **cặp ký tự xuất hiện thường xuyên nhất** và gộp lại
3. Lặp lại quá trình gộp cho đến khi đạt bộ từ vựng mong muốn
4. Ánh xạ kết quả cuối thành **token ID**

⚠️ **Lưu ý quan trọng**: Mỗi model dùng tokenizer riêng — không được giả định số token của model này áp dụng cho model khác.

---

## 3. Context Window — Giới hạn cần nhớ

**Context window** = tổng số token tối đa model xử lý được trong 1 request (input + output cộng lại).

| Model | Context Window | Ghi chú |
|---|---|---|
| GPT-4o / GPT-4o-mini | 128K | ~300 trang |
| Claude 3.5 Sonnet / Haiku | 200K | ~500 trang |
| Gemini 2.0 Flash | 1M | ~2.500 trang |
| Gemini 1.5 Pro | 2M | ~5.000 trang |
| Llama 3.3 70B, Qwen 2.5 72B, DeepSeek V3 | 128K | Open-source bắt kịp |

*(Ước lượng: 1 trang ≈ 400–450 token — số liệu tại thời điểm bài viết, tháng 12/2025)*

**Vùng cảnh báo khi dùng context:**
- 🟢 < 50%: ổn
- 🟡 50–70%: cẩn thận
- 🟠 70–90%: cảnh báo
- 🔴 > 90%: nguy hiểm, dễ lỗi

---

## 4. 4 chiến lược xử lý văn bản dài hơn context window

| Chiến lược | Dùng khi nào | Ưu điểm | Nhược điểm |
|---|---|---|---|
| **Chunking + Summarization** | Cần bao phủ toàn bộ tài liệu | Đầy đủ nội dung | Chậm, có thể mất liên kết ngữ cảnh giữa các đoạn |
| **RAG** (Retrieval-Augmented Generation) | Hỏi-đáp trên tài liệu lớn | Nhanh, mở rộng tốt | Cần vector database, setup phức tạp |
| **Map-Reduce** | Xử lý song song, trích xuất dữ liệu có cấu trúc | Chạy song song, tổng hợp kết quả tốt | Tốn chi phí API hơn, code phức tạp hơn |
| **Sliding Window / Streaming** | Chatbot, hội thoại liên tục | Đơn giản, real-time | Mất ngữ cảnh cũ, không hợp cho tài liệu dài |

### Chi tiết nhanh từng chiến lược

**Chunking & Summarization**
- Cắt văn bản thành các đoạn (chunk) vừa với giới hạn token
- "Smart chunking": cắt theo ranh giới tự nhiên (đoạn văn, câu) + có overlap để giữ ngữ cảnh
- Tóm tắt lặp (progressive summarization): tóm tắt từng chunk → gộp lại → tóm tắt tiếp nếu vẫn còn quá dài

**RAG**
- Bước 1: Chia nhỏ tài liệu → tạo embedding cho từng đoạn → lưu vào vector DB (vd Pinecone)
- Bước 2: Khi có câu hỏi → tạo embedding câu hỏi → tìm các đoạn liên quan nhất
- Bước 3: Ghép các đoạn liên quan (trong giới hạn token budget) làm context → đưa vào prompt để LLM trả lời

**Map-Reduce**
- **Map**: áp dụng 1 hàm xử lý (tóm tắt, trích xuất entity...) lên từng chunk, chạy song song
- **Reduce**: gộp kết quả các chunk lại thành kết quả cuối cùng

**Sliding Window (hội thoại)**
- Giữ N tin nhắn gần nhất, loại bỏ tin nhắn cũ khi vượt giới hạn token
- Biến thể nâng cao: tóm tắt các tin nhắn cũ thay vì xóa hẳn, giữ system message + summary + tin nhắn gần đây

---

## 5. Tối ưu chi phí (Cost Optimization)

1. **Token Caching** — lưu cache câu trả lời cho các prompt lặp lại, tránh gọi API thừa
2. **Prompt Compression** — dùng model rẻ (vd mini) để nén prompt dài mà vẫn giữ ý chính
3. **Smart Model Routing** — định tuyến theo độ phức tạp:
   - Task đơn giản / ít token → model rẻ (mini)
   - Task phức tạp / nhiều token → model mạnh hơn
   - Task cần suy luận sâu → model chuyên về reasoning

---

## 6. Best Practices khi lên Production

- ✅ **Luôn dùng tokenizer thật** của model để đếm token, không đếm bằng số từ
- ✅ **Chừa khoảng đệm (safety margin)** cho phần trả lời — không dùng hết 100% context cho input
- ✅ **Tính overhead định dạng tin nhắn** (mỗi message có vài token phụ phí cho role, format...)
- ✅ **Theo dõi ngân sách token** (token budget) theo ngày/tháng để tránh vượt chi phí
- ✅ **Có cơ chế fallback** khi gặp lỗi vượt context (nén prompt, chuyển sang chunking...)
- ✅ **Test với dữ liệu thực tế** (kích thước tài liệu thật) trước khi go-live

---

## 7. Lỗi thường gặp (Pitfalls) ❌

| Lỗi | Vì sao sai | Cách đúng |
|---|---|---|
| Đếm số từ thay vì token | `text.split()` không phản ánh đúng số token thực | Dùng tokenizer của model (vd `tiktoken`) |
| Dùng hết context cho output | `max_tokens = limit - prompt_tokens` có thể vượt giới hạn thực tế | Luôn trừ thêm safety margin |
| Bỏ qua overhead định dạng | Mỗi message có thêm vài token cho role/format | Cộng thêm token overhead khi tính tổng |

---

## 8. Ghi nhớ nhanh (Key Takeaways)

- 🔑 Token ≠ từ — luôn dùng tokenizer đúng của model
- 🔑 Mỗi model có tokenizer và giới hạn context riêng
- 🔑 Luôn chừa chỗ cho phần output khi tính token budget
- 🔑 Chọn chiến lược phù hợp: RAG cho Q&A, Map-Reduce cho xử lý song song, Chunking cho tóm tắt toàn văn, Sliding Window cho chat
- 🔑 Cache, nén prompt, và định tuyến model thông minh để tiết kiệm chi phí

---

*Cheatsheet được tổng hợp và diễn giải lại từ bài blog gốc trên DEV Community (đăng 30/12/2025). Số liệu về giá và giới hạn model có thể đã thay đổi — nên kiểm tra tài liệu chính thức của nhà cung cấp model để có thông tin mới nhất.*