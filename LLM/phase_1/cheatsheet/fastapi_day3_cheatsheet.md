# 📖 Tổng hợp kiến thức Ngày 3: Dependency Injection (`Depends`)

Chào mừng bạn đến với một trong những vũ khí mạnh mẽ nhất của FastAPI: **Dependency Injection** (Tiêm phụ thuộc). Nghe tên thì nguy hiểm nhưng bản chất lại cực kỳ đơn giản và thực dụng.

---

## 1. Dependency Injection là gì? Giải quyết vấn đề gì?

Hãy tưởng tượng bạn có 50 cái API (endpoint). Trong đó có 30 API yêu cầu người dùng phải đăng nhập (kiểm tra Token/API Key), và 20 API còn lại yêu cầu phải kết nối vào Database.

**Nếu không dùng Dependency Injection (Code tệ):**
Bạn phải viết đi viết lại đoạn code `if token == "sai": báo_lỗi()` ở cả 30 cái API. Nếu một ngày logic kiểm tra token thay đổi, bạn phải sửa thủ công ở 30 chỗ khác nhau! Quá mệt mỏi!

**Nếu dùng Dependency Injection (Code chuẩn):**
Bạn chỉ cần viết logic kiểm tra Token đúng **1 lần duy nhất** vào một hàm riêng biệt. Sau đó, ở bất kỳ API nào cần kiểm tra Token, bạn chỉ việc "tiêm" (inject) hàm đó vào bằng từ khóa `Depends`. FastAPI sẽ tự động chạy hàm đó trước khi chạy API của bạn.

---

## 2. Cách dùng `Depends()` cơ bản

Cú pháp: `tham_số: kiểu_dữ_liệu = Depends(tên_hàm_phụ_thuộc)`

**Ví dụ 1: Chia sẻ logic tính toán**
```python
from fastapi import FastAPI, Depends

app = FastAPI()

# 1. Viết một hàm phụ thuộc (Hàm này lấy limit từ người dùng và giới hạn tối đa là 100)
def get_query_limit(limit: int = 10) -> int:
    return min(limit, 100)

# 2. "Tiêm" hàm đó vào Endpoint
@app.get("/items")
def list_items(gioi_han: int = Depends(get_query_limit)):
    # Lúc này, biến 'gioi_han' chính là kết quả trả về của hàm 'get_query_limit'
    return {"so_luong_lay": gioi_han}
```

---

## 3. Ứng dụng số 1: Xác thực người dùng (Kiểm tra API Key)

Đây là ứng dụng phổ biến nhất của `Depends()`. Thay vì dùng Query Param hay Body, API Key thường được gửi ngầm qua **Header** của HTTP Request. 

Để lấy Header trong FastAPI, ta dùng biến `Header()` thay vì khai báo kiểu thông thường.

```python
from fastapi import FastAPI, Depends, Header, HTTPException

app = FastAPI()

# 1. Hàm kiểm tra API Key
# Khai báo x_api_key: str = Header(...) nghĩa là: FastAPI hãy tự moi cái header tên "x-api-key" ra đây
def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != "MAT_KHAU_BI_MAT_123":
        # Ném lỗi 401 Unauthorized (Không có quyền truy cập)
        raise HTTPException(status_code=401, detail="API Key không hợp lệ hoặc bị thiếu!")
    
    return x_api_key

# 2. Tiêm vào API cần bảo vệ
@app.get("/thong-tin-mat")
def get_secret_data(api_key: str = Depends(verify_api_key)):
    return {"thong_diep": "Nếu bạn thấy dòng này, bạn đã nhập đúng khóa!"}
```
*Ghi chú: Khi mở `/docs`, FastAPI đủ thông minh để tự động tạo ra một ô nhập Header riêng biệt để bạn thử API Key!*

---

## 4. Bảo vệ toàn bộ Server cùng lúc

Nếu bạn muốn BẤT KỲ ai gọi vào server cũng phải có API Key (áp dụng cho toàn bộ endpoint), bạn không cần phải copy `Depends` dán vào từng hàm một. Hãy dán nó ngay lúc khai báo `app`!

```python
from fastapi import FastAPI, Depends

# Tiêm phụ thuộc vào cấp cao nhất (Toàn bộ app)
app = FastAPI(dependencies=[Depends(verify_api_key)])

@app.get("/api-1")
def api_mot():
    return "Đã được bảo vệ"

@app.get("/api-2")
def api_hai():
    return "Cũng đã được bảo vệ"
```

---

## 5. Đọc biến môi trường (`.env`) sạch sẽ nhất

Ở Ngày 4, bạn sẽ bắt đầu dùng API Key của Anthropic (Claude). API Key này không được hard-code mà phải nằm trong file `.env`. Hãy viết một `Depends()` để đọc nó nhé.

*(Cần cài đặt: `pip install pydantic-settings` và `python-dotenv`)*

```python
from pydantic_settings import BaseSettings
from fastapi import FastAPI, Depends

# 1. Pydantic Settings sẽ tự động tìm file .env và đọc các biến có tên tương ứng
class Settings(BaseSettings):
    anthropic_api_key: str = "default_key_neu_khong_thay"

    class Config:
        env_file = ".env"

# 2. Khởi tạo setting
def get_settings():
    return Settings()

app = FastAPI()

# 3. Tiêm vào Endpoint
@app.get("/chat")
def chat_with_ai(settings: Settings = Depends(get_settings)):
    # Bây giờ bạn có thể dùng API Key một cách cực kỳ an toàn
    return {"api_key_dang_dung": settings.anthropic_api_key}
```

---
## 🎯 Lời khuyên
Khi làm bài tập Ngày 3, bạn hãy copy đoạn code xác thực API Key ở phần 3, chạy bằng Uvicorn và lên `/docs` trải nghiệm cảm giác bị FastAPI đuổi ra ngoài (lỗi 401) vì nhập sai API Key nhé!
