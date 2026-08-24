# Python Virtual Environment & Environment Variables Cheat Sheet

> [!TIP]
> Sử dụng môi trường ảo (virtual environment) giúp cô lập các thư viện (packages) cho từng dự án, tránh xung đột phiên bản giữa các dự án khác nhau trên cùng một máy tính.

## 1. Khởi tạo Môi trường ảo (venv)

```bash
# Di chuyển vào thư mục dự án của bạn
cd path/to/your/project

# Tạo một môi trường ảo có tên là "venv" (tên thư mục)
python -m venv venv

# Hoặc trên một số hệ thống (macOS/Linux) cần chỉ định python3:
python3 -m venv venv
```

## 2. Kích hoạt (Activate) Môi trường ảo

Bạn cần kích hoạt môi trường ảo trước khi cài đặt thư viện hay chạy mã nguồn dự án.

```bash
# Trên Windows (Command Prompt)
venv\Scripts\activate.bat

# Trên Windows (PowerShell)
venv\Scripts\Activate.ps1

# Trên macOS và Linux
source venv/bin/activate
```
> Khi được kích hoạt thành công, bạn sẽ thấy `(venv)` xuất hiện ở đầu dòng lệnh terminal của bạn.

## 3. Quản lý Thư viện (Packages) với pip

> [!IMPORTANT]
> Hãy chắc chắn rằng môi trường ảo **đã được kích hoạt** trước khi chạy các lệnh `pip`.

```bash
# Cài đặt một thư viện
pip install <tên-thư-viện>
# Ví dụ: pip install requests

# Cài đặt một phiên bản cụ thể
pip install <tên-thư-viện>==<phiên-bản>
# Ví dụ: pip install fastapi==0.103.1

# Nâng cấp một thư viện
pip install --upgrade <tên-thư-viện>

# Xem danh sách các thư viện đã cài đặt trong môi trường hiện tại
pip list

# Xóa (gỡ cài đặt) một thư viện
pip uninstall <tên-thư-viện>
```

## 4. Làm việc với `requirements.txt`

Tệp `requirements.txt` dùng để lưu trữ danh sách các thư viện dự án yêu cầu, giúp người khác (hoặc khi deploy) cài đặt lại môi trường một cách dễ dàng.

```bash
# Xuất danh sách các thư viện đang dùng ra file requirements.txt
pip freeze > requirements.txt

# Cài đặt toàn bộ thư viện từ file requirements.txt (khi clone dự án từ GitHub về)
pip install -r requirements.txt
```

## 5. Hủy kích hoạt (Deactivate)

Khi bạn đã làm việc xong và muốn thoát khỏi môi trường ảo để trở về môi trường Python mặc định của hệ thống.

```bash
deactivate
```

---

## 6. Biến Môi Trường (Environment Variables) với `.env`

Để bảo mật thông tin nhạy cảm (API Keys, Database passwords...), không bao giờ được viết code trực tiếp (hardcode) hoặc đẩy những thông tin này lên GitHub. Thay vào đó, hãy dùng file `.env`.

**Cài đặt thư viện `python-dotenv`:**
```bash
pip install python-dotenv
```

**Cách sử dụng:**

**Bước 1:** Tạo tệp `.env` trong thư mục gốc của dự án:
```env
# Nội dung file .env (không có khoảng trắng quanh dấu =)
SECRET_KEY=my_super_secret_key
DATABASE_URL=postgres://user:pass@localhost:5432/mydb
DEBUG=True
```

**Bước 2:** Đảm bảo `.env` KHÔNG bị đẩy lên GitHub bằng cách thêm vào `.gitignore`:
```text
# Nội dung file .gitignore
venv/
.env
__pycache__/
```

**Bước 3:** Đọc biến môi trường trong code Python:
```python
import os
from dotenv import load_dotenv

# Tải các biến môi trường từ file .env vào biến hệ thống
load_dotenv()

# Đọc giá trị (trả về chuỗi string, nếu không có trả về None)
secret_key = os.getenv("SECRET_KEY")
db_url = os.getenv("DATABASE_URL")
debug_mode = os.getenv("DEBUG")

print(f"Key: {secret_key}")
```
