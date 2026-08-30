# 📖 Tổng hợp kiến thức Ngày 1: FastAPI Nền tảng

Thay vì phải lên web đọc tài liệu, đây là toàn bộ lý thuyết và ví dụ thực tế bạn cần nắm vững cho **Ngày 1**.

---

## 1. Endpoint là gì?
**Endpoint** (Điểm cuối) là điểm giao tiếp giữa Client (người dùng/trình duyệt) và Server (ứng dụng của bạn).
Trong FastAPI, một Endpoint được tạo ra bằng cách kết hợp **1 Địa chỉ URL** (Path) và **1 Hành động** (HTTP Method), sau đó gắn vào một hàm Python.

**Các HTTP Method phổ biến:**
- `GET`: Lấy dữ liệu về (Đọc).
- `POST`: Gửi dữ liệu lên để tạo mới (Thêm).
- `PUT` / `PATCH`: Cập nhật dữ liệu (Sửa).
- `DELETE`: Xóa dữ liệu (Xóa).

**Cách viết trong FastAPI:**
```python
from fastapi import FastAPI

app = FastAPI()

# Đây là 1 Endpoint: Method GET + Path "/"
@app.get("/")
def say_hello():
    return {"message": "Xin chào FastAPI!"}
```

---

## 2. Phân biệt 3 cách nhận dữ liệu đầu vào

Có 3 vị trí mà người dùng có thể nhét dữ liệu vào để gửi cho Server. Bạn cần phân biệt rõ khi nào dùng cái nào.

### 2.1. Path Parameter (Dữ liệu nằm trực tiếp trong URL)
- **Đặc điểm:** Dùng để **xác định danh tính** của một tài nguyên cụ thể (Ví dụ: Lấy sinh viên có ID là 5). Nó là thành phần **bắt buộc** phải có trong đường dẫn.
- **Cú pháp:** Đặt tên biến trong dấu ngoặc nhọn `{}` trên đường dẫn, và khai báo đúng tên đó trong tham số hàm.

```python
# Ví dụ: GET /users/5
@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"id_nguoi_dung": user_id}
```
> [!TIP]
> Bạn khai báo `user_id: int`, FastAPI sẽ tự động ép kiểu từ chuỗi (URL luôn là chuỗi) sang số nguyên cho bạn!

### 2.2. Query Parameter (Dữ liệu nằm sau dấu `?`)
- **Đặc điểm:** Dùng để **lọc, phân trang, hoặc tìm kiếm** (Ví dụ: Lấy 10 sinh viên, bỏ qua 5 người đầu tiên). Nó thường là **không bắt buộc** (Optional).
- **Cú pháp:** FastAPI quy ước: Bất kỳ tham số nào có trong hàm Python **NHƯNG KHÔNG CÓ TRONG ĐƯỜNG DẪN `{}`**, thì nó tự động được coi là Query Parameter.

```python
# Ví dụ: GET /items?skip=0&limit=10
@app.get("/items")
def get_items(skip: int = 0, limit: int = 10):
    return {"bo_qua": skip, "lay_toi_da": limit}
```
> [!NOTE]
> Khác biệt mấu chốt: Path param nằm trong URL `{}`. Query param KHÔNG CÓ trong URL, người dùng tự truyền qua dấu `?`.

### 2.3. Request Body (Dữ liệu giấu kín bên trong)
- **Đặc điểm:** Dùng khi muốn gửi **dữ liệu lớn, phức tạp hoặc nhạy cảm** (như thông tin đăng ký tài khoản, nội dung bài viết). Dữ liệu này thường gửi qua phương thức `POST` hoặc `PUT`, được mã hóa dưới dạng JSON.
- **Cú pháp (Dạng thô - Dictionary):** Khai báo tham số kiểu `dict`. *(Sang ngày 2 bạn sẽ học dùng Pydantic thay cho dict)*.

```python
# Ví dụ: POST /items (Kèm theo cục JSON gửi ngầm)
@app.post("/items")
def create_item(payload: dict):
    # Trả lại y nguyên những gì người dùng gửi lên
    return {"da_nhan_duoc": payload}
```

---

## 🎯 Tóm tắt nhanh:
1. `GET /users/5` ➔ **Path Param** (Danh tính cụ thể, bắt buộc).
2. `GET /users?age=18` ➔ **Query Param** (Bộ lọc, tuỳ chọn).
3. `POST /users` kèm JSON ➔ **Request Body** (Dữ liệu lớn, ẩn, tạo mới).
