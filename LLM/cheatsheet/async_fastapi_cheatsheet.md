# 🚀 THE ULTIMATE CHEATSHEET: ASYNCIO & FASTAPI

Bản tổng hợp chi tiết và đầy đủ nhất về Lập trình Bất đồng bộ (Asynchronous Programming) trong Python và cách áp dụng chuẩn mực trong FastAPI.

---

## 1. BẢN CHẤT CỦA ĐỒNG BỘ VÀ BẤT ĐỒNG BỘ

### 1.1. Sync (Đồng bộ - Blocking)
- **Cơ chế:** Làm tuần tự từng việc một. Việc A phải hoàn thành 100% thì việc B mới được phép bắt đầu.
- **Vấn đề:** Nếu việc A là một tác vụ tốn thời gian (I/O Bound) như: tải file từ internet, truy vấn Database, đọc/ghi ổ cứng... thì CPU sẽ "ngồi chơi xơi nước", làm cho toàn bộ chương trình bị "đóng băng" (treo/block) chờ đợi.
- **Ví dụ đời sống:** Bạn ra quán gọi phở. Đầu bếp nấu phở cho bạn xong bưng ra, bạn ăn xong tính tiền, thì đầu bếp mới bắt đầu hỏi người khách thứ 2 ăn gì. 

### 1.2. Async (Bất đồng bộ - Non-blocking)
- **Cơ chế:** Làm luân phiên nhiều việc trên cùng 1 luồng (Thread). Nếu việc A phải đợi, hệ thống sẽ "cất" việc A đi và rảnh tay làm việc B. Khi nào việc A đợi xong thì lấy ra làm tiếp.
- **Ưu điểm:** Khai thác tối đa sức mạnh của CPU. Xử lý hàng nghìn kết nối cùng lúc mà không tốn thêm bộ nhớ để tạo Thread mới.
- **Ví dụ đời sống:** Đầu bếp quán phở. Nhận order của khách 1 -> cho bánh phở vào chần. Trong lúc đợi bánh phở mềm (chờ đợi), đầu bếp quay sang nhận order của khách 2 và thái thịt. Bánh phở mềm xong thì quay lại vớt ra bát.

---

## 2. EVENT LOOP (Trái tim của Asyncio)

- **Định nghĩa:** Event Loop (Vòng lặp sự kiện) là một vòng lặp `while True` chạy vô tận dưới nền. 
- **Nhiệm vụ:** Nó giống như một người Quản lý. Nó liên tục đi hỏi các tác vụ (Tasks): *"Tác vụ này đang tính toán hay đang chờ? Nếu đang chờ thì tao cất qua một bên, nếu tính xong rồi thì tao lấy kết quả trả về"*.
- **Quy tắc tử thần:** Event Loop chạy trên **ĐÚNG 1 LUỒNG (Single-Thread)**. Do đó, nếu có bất kỳ một tác vụ nào tính toán quá lâu hoặc bị "ngủ" (`time.sleep`), Event Loop sẽ bị kẹt cứng ở đó, các tác vụ khác vĩnh viễn không được gọi.

---

## 3. CẶP BÀI TRÙNG: `async` VÀ `await`

Đây là 2 từ khóa bắt buộc phải có để viết code bất đồng bộ trong Python hiện đại.

### `async def` (Khai báo Coroutine)
Dùng để định nghĩa một hàm bất đồng bộ.
```python
async def nau_com():
    return "Cơm đã chín!"

# Lưu ý: Gọi hàm async KHÔNG làm hàm chạy, mà nó trả về một Coroutine Object.
print(nau_com()) # Báo lỗi hoặc in ra: <coroutine object nau_com at 0x...>
```

### `await` (Chờ đợi thông minh)
Chỉ được phép sử dụng `await` bên trong một hàm `async def`.
Nó có ý nghĩa: *"Hệ thống ơi, chỗ này phải đợi đấy. Cứ tạm dừng hàm này ở đây, đi làm việc khác đi. Khi nào có kết quả thì đánh thức tao dậy chạy tiếp"*.

```python
import asyncio

async def nau_com():
    print("Đang cắm cơm...")
    await asyncio.sleep(2)  # Hệ thống rảnh tay trong 2 giây này
    return "Cơm chín!"

async def main():
    ket_qua = await nau_com()  # Gọi và ĐỢI hàm nấu cơm chạy xong
    print(ket_qua)

asyncio.run(main()) # Cách để khởi động Event Loop
```

---

## 4. CÁC HÀM QUAN TRỌNG NHẤT CỦA `asyncio`

1. **`asyncio.run(coroutine)`**: Khởi động Event Loop và chạy hàm async. Thường chỉ dùng 1 lần duy nhất ở file chạy chính. (FastAPI đã tự động làm việc này cho bạn dưới nền).
2. **`asyncio.sleep(giây)`**: Phiên bản Async của `time.sleep()`. Dùng để chờ đợi mà không làm treo hệ thống.
3. **`asyncio.gather(*coroutines)`**: Chạy **SONG SONG** nhiều tác vụ cùng một lúc và thu thập kết quả về thành 1 danh sách (List).

```python
async def main():
    # Chạy 3 tác vụ song song, tổng thời gian đợi = thời gian của tác vụ lâu nhất!
    results = await asyncio.gather(
        nau_com(),   # mất 2s
        luoc_rau(),  # mất 1s
        ran_trung()  # mất 2s
    )
    print(results)   # ['Cơm chín', 'Rau chín', 'Trứng chín'] sau đúng 2 giây.
```

---

## 5. BÍ KÍP "SỐNG CÒN" TRONG FASTAPI

Khi sử dụng FastAPI, câu hỏi lớn nhất luôn là: **"Nên đặt `async def` hay `def` thường cho API?"**

### Quy tắc 1: Nếu dùng `async def`
- **Bắt buộc:** Bên trong hàm chỉ được chứa các logic tính toán cực nhanh, hoặc gọi các thư viện **CÓ HỖ TRỢ ASYNC** (dùng bằng chữ `await`). Ví dụ: `httpx` (thay cho requests), `Motor` (MongoDB), `asyncpg` (PostgreSQL), `aiofiles` (Đọc ghi file).
- **Tác hại:** Nếu lỡ tay nhét một hàm Sync chậm chạp (như `time.sleep()`, vòng lặp hàng triệu lần, `requests.get`) vào đây, **TOÀN BỘ SERVER SẼ BỊ TREO**, các User đến sau sẽ bị Timeout.

### Quy tắc 2: Nếu dùng `def` thường
- **Cơ chế thần kỳ của FastAPI:** Khi thấy bạn khai báo API bằng `def` thường, FastAPI tự hiểu rằng *"Thằng chả này đang viết code Sync chậm chạp đây"*. Nó sẽ **TỰ ĐỘNG** ném toàn bộ hàm của bạn sang một **Thread Pool phụ** (Luồng khác) để chạy.
- **Kết quả:** Event Loop chính vẫn rảnh tay đi đón khách khác. Server KHÔNG BỊ TREO.
- **Lời khuyên:** Nếu bạn dùng thư viện cũ (SQLAlchemy bản cũ, Pandas, OpenCV, Machine Learning), cứ mạnh dạn khai báo `def` thường!

---

## 6. TUYỆT CHIÊU: `run_in_executor` (Trộn lẫn Sync và Async)

Bạn đang viết một API `async def` cực xịn xò để lấy dữ liệu từ DB, nhưng đột nhiên sếp bắt bạn gọi một hàm nhận diện khuôn mặt bằng OpenCV (cực nặng và không hỗ trợ Async). Bạn phải làm sao? 
-> **Hãy dùng `run_in_executor` để "đá" phần nặng nhọc đó sang Thread khác!**

```python
import asyncio
import time
from fastapi import FastAPI

app = FastAPI()

# 1. Hàm Sync tốn CPU, KHÔNG có async
def xu_ly_anh_nang_ne(image_name: str) -> str:
    print(f"Bắt đầu xử lý {image_name}...")
    time.sleep(3)  # Giả lập xử lý nặng
    return f"Đã xử lý xong {image_name}"

# 2. Hàm Async mượt mà
@app.get("/analyze")
async def analyze_api():
    loop = asyncio.get_event_loop()
    
    # Bước 1: Có thể fetch data nhanh gọn (await asyncio.sleep)
    
    # Bước 2: Đẩy việc nặng sang ThreadPool để KHÔNG block Event Loop
    # Tham số 1: None (Sử dụng ThreadPool mặc định, tối đa ~40 luồng)
    # Tham số 2: Tên hàm Sync
    # Tham số 3 trở đi: Các tham số truyền cho hàm Sync
    ket_qua = await loop.run_in_executor(None, xu_ly_anh_nang_ne, "avatar.jpg")
    
    return {"message": ket_qua}
```

---

## 7. CÁC LỖI THƯỜNG GẶP (TROUBLESHOOTING)

1. **Lỗi `RuntimeWarning: coroutine 'xyz' was never awaited`**
   - *Nguyên nhân:* Gọi hàm `async` nhưng quên thêm chữ `await` ở đằng trước.
   - *Hậu quả:* Hàm đó chưa hề được chạy, và biến nhận được chỉ là một `Coroutine Object`.
   - *Cách sửa:* Thêm `await` trước lời gọi hàm.

2. **Server bị đơ/treo khi có nhiều người truy cập**
   - *Nguyên nhân:* Gọi hàm Blocking (như `requests.get` hoặc `time.sleep`) bên trong một hàm `async def` của FastAPI.
   - *Cách sửa:* Đổi hàm API thành `def` thường, HOẶC dùng `run_in_executor` để bọc hàm Blocking đó lại, HOẶC thay bằng thư viện có hỗ trợ Async (như `httpx.get`, `asyncio.sleep`).

3. **Lỗi `RuntimeError: This event loop is already running`**
   - *Nguyên nhân:* Cố tình gọi `asyncio.run()` bên trong một API của FastAPI.
   - *Cách sửa:* FastAPI đã tự động chạy Event Loop cho bạn rồi. Bạn chỉ cần dùng `await` là đủ, tuyệt đối không dùng `asyncio.run()` bên trong code API.
