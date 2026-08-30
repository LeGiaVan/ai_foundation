# Cheat Sheet WebSocket (tiếng Việt)

## 1. WebSocket là gì?
- **Kênh giao tiếp full‑duplex** qua một kết nối TCP duy nhất.
- Quá trình bắt đầu bằng một yêu cầu HTTP `Upgrade: websocket`; sau đó giao thức chuyển sang khung dữ liệu nhị phân.
- Thích hợp cho các **kịch bản thời gian thực**: chat, thông báo, streaming phản hồi LLM, chỉnh sửa cộng đồng.

---

## 2. Các khái niệm cốt lõi
| Khái niệm | Mô tả |
|---|---|
| **Connection** | Kết nối liên tục được mở sau khi handshake thành công. |
| **Message Types** | Khung `text` (UTF‑8) và `binary`. |
| **Ping/Pong** | Khung keep‑alive; server trả `pong` tự động khi nhận `ping`. |
| **Close** | Đóng kết nối một cách lịch sự, kèm mã lỗi và lý do (tùy chọn). |
| **Sub‑protocols** | Các chuỗi tùy chọn được thương lượng trong handshake (ví dụ `graphql-ws`). |

---

## 3. Server FastAPI sử dụng WebSocket (Python)
```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List

app = FastAPI()

class ConnectionManager:
    """Quản lý các kết nối đang hoạt động để có thể broadcast."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active_connections.remove(ws)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/chat")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_text()          # nhận tin nhắn từ client
            # Xử lý tùy ý (ví dụ gọi LLM) – ở đây chỉ echo lại
            await manager.broadcast(f"Người dùng nói: {data}")
    except WebSocketDisconnect:
        manager.disconnect(ws)
        await manager.broadcast("Có người rời khỏi phòng chat")
```
**Điểm quan trọng**
- Phải gọi `await ws.accept()` trước khi gửi/nhận bất kỳ dữ liệu nào.
- Bắt `WebSocketDisconnect` để giải phóng tài nguyên.
- Mẫu `ConnectionManager` giúp broadcast tới tất cả client đang kết nối.

### Streaming phản hồi LLM (giống API `/ask/stream` của bạn)
```python
@app.post("/ask/stream")
async def ask_stream(request: AskRequest, ws: WebSocket):
    await ws.accept()
    async for token in rag_pipeline.ask_question_stream(request.question):
        await ws.send_text(token)   # gửi từng token cho client
    await ws.close()
```
> **Mẹo:** giữ endpoint **non‑blocking**; generator sẽ yield token và FastAPI stream chúng qua WebSocket.

---

## 4. Client JavaScript (trình duyệt)
```html
<script>
  const socket = new WebSocket("ws://127.0.0.1:8000/ws/chat");

  socket.addEventListener('open', () => {
    console.log('WebSocket đã mở');
    socket.send('Xin chào server!');
  });

  socket.addEventListener('message', event => {
    console.log('Nhận được:', event.data);
    // Thêm vào UI, ví dụ hiển thị trong khung chat
  });

  socket.addEventListener('close', () => console.log('WebSocket đã đóng'));
  socket.addEventListener('error', err => console.error('Lỗi WS', err));

  // Tự động gửi ping mỗi 30s để giữ kết nối (trình duyệt thường tự làm)
  setInterval(()=> socket.send('ping'), 30000);
</script>
```
**Mẫu tự‑reconnect**
```js
function connect(){
  const ws = new WebSocket('ws://127.0.0.1:8000/ws/chat');
  ws.onclose = () => setTimeout(connect, 2000); // tự kết nối lại sau 2s
  // … các handler khác …
}
connect();
```

---

## 5. Client Python (asyncio)
```python
import asyncio, websockets

async def hello():
    async with websockets.connect('ws://127.0.0.1:8000/ws/chat') as ws:
        await ws.send('Chào từ Python!')
        async for msg in ws:
            print('Nhận được:', msg)

asyncio.run(hello())
```
- Thư viện `websockets` tự xử lý ping/pong.
- Dùng `async for` để lắng nghe vô hạn.

---

## 6. Xác thực & ủy quyền
1. **Token trong query string**
   ```python
   @app.websocket("/ws/chat")
   async def ws_endpoint(ws: WebSocket, token: str = Depends(auth.validate_token)):
       ...
   ```
2. **Header trong handshake** (hỗ trợ bởi Starlette/FastAPI)
   ```python
   @app.websocket("/ws/protected")
   async def protected(ws: WebSocket):
       if not await validate(ws.headers.get('authorization')):
           await ws.close(code=1008)   # Policy Violation
           return
   ```
3. **Sub‑protocol** – thương lượng một sub‑protocol dựa trên JWT nếu cần luồng xác thực phức tạp.

---

## 7. Triển khai & tối ưu cho môi trường production
| Vấn đề | Giải pháp |
|---|---|
| **Mở rộng ngang** | Đặt reverse proxy (NGINX, Traefik) hỗ trợ `Upgrade` và `Connection: upgrade`. |
| **Sticky sessions** | Khi chạy nhiều worker của Uvicorn, bật **session affinity** để mỗi client luôn tới cùng một worker (cần cho `ConnectionManager` trong bộ nhớ). |
| **Broadcast qua Pub/Sub** | Dùng Redis Pub/Sub hoặc RabbitMQ; mỗi worker đăng ký kênh và phát tin tới các WebSocket cục bộ. |
| **Back‑pressure** | `await ws.send_text(msg)` sẽ chờ cho tới khi buffer có chỗ, tránh OOM. |
| **Kiểm tra sức khỏe** | Cung cấp endpoint HTTP `/health`; bổ sung script kiểm tra `ping`/`pong` qua WebSocket. |
| **TLS** | Trong production dùng `wss://` (HTTPS + WebSocket). Cài cert từ Let’s Encrypt hoặc nhà cung cấp cloud. |

---

## 8. Những lỗi phổ biến & cách debug
- **Quên `await ws.accept()`** → client nhận lỗi “connection closed before handshake”.
- **Kết hợp async và sync** – nếu thực hiện công việc nặng CPU (như embedding) hãy chuyển sang thread pool (`run_in_threadpool`).
- **Ping timeout** – trình duyệt sẽ đóng sau ~60s không hoạt động; gửi ping định kỳ hoặc cấu hình server `ping_interval`. 
- **Thứ tự tin nhắn** – WebSocket bảo đảm thứ tự, nhưng nếu truyền qua hàng đợi (queue) bạn phải tự đảm bảo. 
- **Xử lý lỗi** – luôn bọc `receive_*` trong `try/except WebSocketDisconnect` để tránh exception không được bắt.

---

## 9. Tham chiếu nhanh (cheat sheet)
```text
GET /ws/chat   -> Upgrade -> WebSocket
ws.send(data)  -> gửi text hoặc binary
ws.receive()   -> await khung tới
ws.close([code, reason])
Ping/Pong      -> tự động, có thể cấu hình interval
```

---

### TL;DR (tóm tắt)
- **Server:** FastAPI `@app.websocket`, `ConnectionManager` để broadcast, dùng generator async cho streaming LLM.
- **Client (JS):** `new WebSocket(url)`, `onmessage`, mẫu tự‑reconnect.
- **Auth:** kiểm tra token trong query hoặc header, đóng kết nối với code `1008` nếu không hợp lệ.
- **Scale:** dùng reverse proxy có `sticky sessions` hoặc Redis Pub/Sub.
- **Debug:** luôn gọi `accept()`, bắt `WebSocketDisconnect`, theo dõi ping/pong.

Bạn có thể sao chép các đoạn code trên vào dự án của mình để thêm tính năng thời gian thực cho API.
