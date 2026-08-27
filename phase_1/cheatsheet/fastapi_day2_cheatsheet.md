# 📖 Tổng hợp kiến thức Ngày 2: Pydantic & Uvicorn

Tiếp nối Ngày 1, hôm nay chúng ta sẽ nâng cấp Request Body bằng Pydantic và tìm hiểu các công cụ đi kèm với FastAPI.

---

## 1. Pydantic Model (Bảo vệ Request Body)

Ở Ngày 1, khi nhận dữ liệu từ người dùng gửi lên qua POST/PUT, chúng ta dùng kiểu `dict` (từ điển thô). Cách này rất rủi ro vì người dùng có thể gửi thiếu dữ liệu hoặc sai kiểu (ví dụ: gửi chữ thay vì số).

**Cách giải quyết:** Tạo một class kế thừa từ `BaseModel` của thư viện Pydantic.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Định nghĩa cấu trúc dữ liệu mong muốn
class Item(BaseModel):
    name: str
    price: float
    description: str | None = None # Có cũng được, không có thì bằng None

@app.post("/items")
def create_item(item: Item):
    # Trả về dữ liệu gốc kèm theo tính toán mới (Ví dụ thêm thuế 10%)
    return {
        "name": item.name,
        "price": item.price,
        "total_with_tax": item.price * 1.1,
        "description": item.description
    }
```

**3 Siêu năng lực của Pydantic:**
1. **Ép kiểu tự động:** Khách gửi `price` là `"50.5"` (kiểu chữ), Pydantic tự đổi thành `50.5` (kiểu số thập phân).
2. **Gợi ý code (Auto-complete):** Khi gõ `item.`, editor sẽ gợi ý ra `name, price, description` (dùng dict thô không bao giờ có tính năng này).
3. **Bắt lỗi tự động:** Đọc tiếp phần 3!

---

## 2. Uvicorn là gì?

FastAPI chỉ là một "bản thiết kế". Để bản thiết kế đó chạy được và nhận request từ trình duyệt, bạn cần một **Web Server**. Đó chính là **Uvicorn** (một ASGI server chuyên trị code bất đồng bộ).

**Cú pháp khởi chạy:**
```bash
uvicorn main:app --reload
```
- `main`: Là tên file code của bạn (ví dụ file là `main.py`).
- `app`: Là tên biến bạn khai báo `app = FastAPI()`.
- `--reload`: (Rất quan trọng khi code) Lệnh này bảo Uvicorn hãy "theo dõi" file. Mỗi lần bạn bấm `Ctrl + S` lưu file, server sẽ tự động reset để cập nhật code mới, bạn không cần tắt đi bật lại.

---

## 3. Lỗi 422 (Đặc sản của FastAPI)

Nếu bạn khai báo `price: float` nhưng người dùng cố tình gửi chữ (ví dụ: `price: "mười nghìn"`), app của bạn có bị sập không?
**Câu trả lời là KHÔNG.**

FastAPI sẽ tự động chặn request đó lại và trả về cho người dùng mã lỗi **422 Unprocessable Entity** kèm theo thông báo cực kỳ chi tiết báo chính xác họ đã nhập sai ở trường nào. Bạn (lập trình viên) không cần phải viết bất kỳ dòng code `if/else` nào để kiểm tra (Validate) dữ liệu nữa!

---

## 4. Swagger UI (`/docs`) vs Redoc (`/redoc`)

Ngay khi bạn code xong, FastAPI tự động tạo ra 2 trang tài liệu (Documentation) tuyệt đẹp dựa trên Pydantic class của bạn.

1. **Vào `http://127.0.0.1:8000/docs` (Swagger UI):**
   - Đây là công cụ đắc lực nhất của lập trình viên Backend. 
   - Nó có nút **"Try it out"** giúp bạn gửi dữ liệu giả (Test API) trực tiếp trên trình duyệt mà không cần code Front-end hay dùng Postman.
   
2. **Vào `http://127.0.0.1:8000/redoc` (Redoc):**
   - Không có nút "Try it out" để test.
   - Nhưng giao diện đọc chữ được sắp xếp tĩnh, khoa học và gọn gàng hơn. Thường dùng để gửi cho các đối tác hoặc lập trình viên khác đọc.

---
## 🎯 Lời khuyên cho Bài tập Ngày 2:
Hãy copy đoạn code ví dụ ở phần 1, chạy Uvicorn, mở `/docs` và cố tình gửi chữ vào ô `price` để tận mắt xem lỗi 422 trông như thế nào nhé!
