# Cẩm Nang Tư Duy Thiết Kế Hệ Thống AI (System Design)

Khi chuyển từ việc "code chạy được trên máy mình" (1 user) sang "hệ thống phục vụ hàng ngàn người" (Production), bạn cần trang bị một bộ tư duy hoàn toàn mới. Đối với các ứng dụng AI Backend (gọi API LLM), thách thức còn lớn hơn vì các tác vụ này thường rất chậm và phụ thuộc vào bên thứ 3 (như Groq, OpenAI).

Dưới đây là 5 trụ cột tư duy cốt lõi bạn cần nắm vững:

---

## 1. Tư duy Không Chặn (Non-Blocking) & Bất Đồng Bộ (Asynchronous)

**Vấn đề:** 
Một request xử lý AI thường mất 3-10 giây. Nếu server của bạn xử lý đồng bộ (Synchronous), khi User A đang chờ kết quả, User B gửi request tới sẽ bị "treo" ở ngoài cửa. 100 User gửi cùng lúc = Server sập.

**Tư duy:**
- Phải luôn giữ cho "Cửa chính" (Event Loop của FastAPI) được thông thoáng.
- Bất cứ khi nào phải **chờ đợi** (chờ AI trả lời, chờ Database đọc dữ liệu, chờ tải file), phải dùng `await` để "nhường" CPU đi phục vụ người khác (như User B) trong lúc chờ.
- **Quy tắc vàng:** Không bao giờ dùng các thư viện đồng bộ (như `requests`, `time.sleep`, `Groq()`) bên trong `async def`. Hãy dùng `httpx`, `asyncio.sleep`, `AsyncGroq()`.

---

## 2. Tư duy Quản Lý "Nút Cổ Chai" (Bottlenecks & Rate Limits)

**Vấn đề:**
Kể cả khi FastAPI của bạn có thể nhận 10,000 request cùng lúc (vì nó bất đồng bộ), thì bên thứ 3 (Groq, OpenAI, Database) lại không chịu nổi và sẽ chặn bạn (Lỗi 429 - Rate Limit) hoặc sập.

**Tư duy:**
Bạn phải chủ động "phanh" hệ thống của mình lại trước khi đối tác chặn bạn.
- **Cấp độ 1 (Semaphore):** Dùng `asyncio.Semaphore` để giới hạn số lượng request gọi ra ngoài *trên một server*. (Ví dụ: Chỉ cho phép gọi LLM tối đa 10 luồng cùng lúc).
- **Cấp độ 2 (Hàng đợi - Message Queue):** Khi số lượng User tăng vọt (Spike traffic), thay vì xử lý ngay, hãy tống các request vào một "Phòng chờ" (như **Redis, RabbitMQ, Celery**). Sẽ có các "Worker" chạy ngầm, từ từ lấy từng task trong phòng chờ ra xử lý và trả kết quả sau. User sẽ nhận được thông báo: *"Yêu cầu của bạn đang được xử lý, vui lòng chờ..."*.

---

## 3. Tư duy Không Trạng Thái (Stateless)

**Vấn đề:**
Nếu bạn lưu lịch sử chat của User vào một biến toàn cục trong RAM (ví dụ: `chat_history = []` trong `main.py`). Khi lượng User đông lên, bạn phải thuê thêm 2, 3 máy chủ (Server 1, Server 2). Lúc này, Request 1 của User vào Server 1, Request 2 của User lại bay vào Server 2 -> Server 2 không biết User này là ai vì RAM không dùng chung!

**Tư duy:**
- API Backend phải hoàn toàn **Stateless (Không lưu trạng thái trong RAM)**.
- Mọi dữ liệu cần nhớ (Lịch sử chat, phiên đăng nhập, file đã tải lên) đều phải được ném ra một nơi lưu trữ dùng chung: **Database (PostgreSQL, MongoDB)** hoặc **Cache (Redis)**.
- Khi có Request tới, Server sẽ lấy ID của User, chạy ra Database móc lịch sử chat về, gửi cho LLM, rồi lại cất kết quả vào Database. Server không nhớ gì cả sau khi xử lý xong.

---

## 4. Tư duy Chấp Nhận Thất Bại (Design for Failure)

**Vấn đề:**
Mạng internet không ổn định. API của Groq thỉnh thoảng sẽ bị ngỏm vài giây. Trả kết quả JSON đôi khi bị hỏng format. Nếu code của bạn mặc định mọi thứ luôn suôn sẻ, hệ thống sẽ liên tục crash.

**Tư duy:**
Hãy thiết kế với tâm thế: **"Kiểu gì nó cũng sẽ lỗi, quan trọng là hệ thống tự cứu mình thế nào"**.
- **Retry Mechanism:** Nếu gọi LLM lỗi do mạng, đừng báo lỗi cho User ngay. Hãy âm thầm thử lại (Retry) 3 lần với khoảng cách thời gian tăng dần (Exponential Backoff).
- **Graceful Degradation:** Nếu AI chính bị sập (Groq chết), hệ thống tự động chuyển sang gọi AI dự phòng (OpenAI/Gemini). Nếu tất cả đều chết, trả về một câu thông báo lịch sự cho User thay vì lỗi 500 nổ tung màn hình.
- **Sử dụng `gather(return_exceptions=True)`** như đã học để một task phụ chết không kéo theo cả hệ thống chết.

---

## 5. Tư duy Trải Nghiệm Người Dùng (UX) Khỏa Lấp Sự Chậm Trễ

**Vấn đề:**
Generative AI bản chất là chậm. Việc bắt User nhìn màn hình trắng bóc quay mòng mòng trong 10 giây là một thảm họa UX. Họ sẽ F5 liên tục (càng làm server quá tải).

**Tư duy:**
- **Streaming:** Luôn luôn dùng Streaming (trả từng chữ như ChatGPT) để User thấy "hệ thống đang làm việc". 
- **Tiến trình (Progress Bar):** Với tác vụ dài như Map-Reduce (như đồ án Capstone của bạn), hãy dùng **WebSocket** để bắn tín hiệu về cho Frontend: *"Đang đọc trang 1/10...", "Đang tóm tắt phần 1...", "Đang tổng hợp..."*.
- Đừng để User phải đoán xem hệ thống có bị treo hay không.

---

### Bức Tranh Tổng Thể (Architecture)

Một hệ thống AI Production tiêu chuẩn thường có hình hài thế này:

1. **Client (Web/App)** gửi Request tới Server.
2. **Nginx / API Gateway:** Đứng ngoài cùng làm bảo vệ, chặn bớt spam (Rate limiting IP) và chia đều request (Load Balancing) cho các máy chủ FastAPI.
3. **FastAPI (Stateless):** Tiếp nhận request (Non-blocking). Nếu là tác vụ nhanh (Chat), xử lý luôn. Nếu tác vụ siêu chậm (Tóm tắt sách 1000 trang), ném vào Celery Queue rồi báo Client "Đang xử lý nhé".
4. **Celery Workers:** Từ từ lôi sách ra tóm tắt (có Semaphore bảo vệ chống quá tải Groq API).
5. **Database (PostgreSQL) / Cache (Redis):** Nơi lưu trữ mọi trạng thái, lịch sử, kết quả.
6. **WebSockets:** Kênh để Backend bắn từng token (Streaming) hoặc thông báo tiến độ về lại cho Client.
