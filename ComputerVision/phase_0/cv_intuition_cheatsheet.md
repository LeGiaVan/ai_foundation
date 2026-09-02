# Cheatsheet: Bản chất cốt lõi của Computer Vision qua góc nhìn trực quan

Tài liệu này tổng hợp lại các khái niệm nền tảng trong lập trình Computer Vision, sử dụng các phép ẩn dụ đời thực để giúp bạn hình thành "tư duy Tensor" thay vì phải nhớ các công thức toán học khô khan.

---

## 1. Số hóa điểm ảnh (Pixel) và Giới hạn 0-255
Mọi hình ảnh trên máy tính đều được cấu tạo từ các điểm ảnh (pixel).
- **Phần cứng 8-bit:** Máy tính phân bổ đúng 8-bit bộ nhớ cho 1 điểm màu. Nhờ vậy nó có thể chứa được tối đa $2^8 = 256$ trạng thái khác nhau.
- **Giá trị:** Vì máy tính đếm từ `0`, nên giá trị của pixel sẽ chạy từ `0` (Đen tuyền / Tối nhất) đến `255` (Sáng chói nhất).
- **Kiểu dữ liệu gốc:** Khi đọc ảnh bằng OpenCV, kiểu dữ liệu luôn là `uint8` (Số nguyên dương 8-bit).

---

## 2. Chiều dữ liệu (Dimensions) - Tư duy Tensor
Hãy hình dung các chiều dữ liệu (Shape) như các lớp thông tin được xếp chồng lên nhau:

| Loại dữ liệu | Số chiều (Dimensions) | Phép ẩn dụ thực tế |
| :--- | :--- | :--- |
| **1 Điểm ảnh (Pixel)** | 0D (Scalar) | 1 con số vô hướng. |
| **Ảnh Grayscale (Xám)** | 2D `(H, W)` | **1 Bảng Excel**. Các hàng là Chiều cao (H), các cột là Chiều rộng (W). Mỗi ô Excel chứa 1 con số độ sáng (0-255). |
| **Ảnh Color (Màu)** | 3D `(H, W, C)` | **Khối Rubik 3D**. Có 3 bảng Excel xếp chồng lên nhau tương ứng với 3 kênh màu (Channels). |
| **Batch ảnh (Train AI)** | 4D `(N, C, H, W)` | **Một Thùng phuy lớn**. Trong thùng phuy chứa N (Batch Size) các khối Rubik 3D giống hệt nhau. GPU luôn xử lý nguyên cả thùng này cùng 1 lúc thay vì nhặt từng khối Rubik ra. |

---

## 3. Lịch sử của OpenCV và Bẫy BGR
> [!WARNING]
> Luôn luôn convert màu sang RGB trước khi đưa ảnh vào các hàm vẽ (`matplotlib`) hoặc các Mô hình AI!

- **Tại sao OpenCV dùng BGR?** Vào thập niên 90 - 2000, các nhà sản xuất phần cứng máy ảnh và màn hình chuộng định dạng BGR. OpenCV được viết vào thời đó nên chọn BGR làm gốc để tối ưu tốc độ đọc từ phần cứng.
- **Tại sao phải đổi sang RGB?** Gần như 100% các công nghệ hiện đại ngày nay (Thư viện Deep Learning, PyTorch, YOLO) đều mặc định đọc ảnh dưới chuẩn RGB (Đỏ - Xanh lá - Xanh dương). Nếu quên dòng code `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`, quả táo màu đỏ sẽ bị AI nhìn thành màu xanh lam thẫm, dẫn đến phán đoán sai lệch.

---

## 4. Vòng lặp `for` vs Vectorization
> [!TIP]
> Quy tắc sống còn: Thấy bản thân đang viết vòng lặp `for` duyệt qua các mảng dữ liệu ảnh, hãy dừng lại ngay lập tức và tìm cách Vectorize!

**Sự khác biệt về cách thức hoạt động:**
- **Vòng lặp `for` (Thông dịch):** Giống như một vị Sếp (Python) mắc bệnh quản lý vi mô. Sếp đi tới gặp Công nhân (CPU), chỉ đạo nhặt từng con số lên, tính toán, cất đi, rồi lại qua con số tiếp theo. Hàng triệu vòng lặp làm lãng phí 99% thời gian vào việc "giao tiếp và kiểm tra lệnh".
- **Vectorization (Biên dịch bằng C):** Lệnh được đóng gói thành 1 cục (ví dụ: `img / 255.0`). Sếp vứt thẳng cả cục dữ liệu khổng lồ đó cho một đoạn code C đã được biên dịch sẵn phần lõi. Đoạn code C này sử dụng tập lệnh phần cứng **SIMD** để tính toán song song hàng vạn phép tính trong 1 chớp mắt mà không cần Sếp xen vào. Nhanh hơn vòng lặp gấp hàng trăm lần!

---

## 5. Phép thuật Broadcasting
Broadcasting là cơ chế để NumPy tự động tính toán giữa 2 mảng không cùng kích thước mà không bị lỗi.

**Phép ẩn dụ "Chiếc điều khiển và Màn hình Tivi":**
- Nếu bạn có bức ảnh tivi `224x224x3` và một vector `[5, 10, 15]` có shape là `(3,)`.
- **Cơ chế hoạt động:** Vector nhỏ bé đóng vai trò như một sóng tín hiệu từ chiếc *Điều khiển từ xa*. Khi thực hiện phép trừ, NumPy đóng vai trò như tháp truyền hình, "phát sóng" tín hiệu giảm 5, 10, 15 xuống **tất cả** 50.000 bóng đèn LED trên tivi cùng một lúc.
- **Quy tắc khớp Shape:** So sánh 2 shape từ Phải sang Trái. Các con số phải BẰNG NHAU hoặc một bên PHẢI LÀ 1. Nếu thỏa mãn, NumPy sẽ tự "photocopy" bên nhỏ để lấp đầy kích thước của bên to.

---

## 6. Tiền xử lý kinh điển: Normalization (Chuẩn hóa)
Bất cứ bức ảnh nào trước khi đưa vào mô hình AI đều phải qua một công thức:
```python
normalized_img = (img / 255.0 - mean) / std
```

### Tại sao phải Normalize? (Ẩn dụ Chấm điểm thi)
Nếu không chuẩn hóa, AI sẽ bị "lóa mắt" bởi các con số lớn (điểm môn Toán 10.000 điểm) và bỏ qua các con số nhỏ (điểm Tiếng Anh 10 điểm). Với ảnh, AI sẽ bị bối rối giữa những tấm ảnh quá sáng (giá trị gần 255) và quá tối (giá trị gần 0). 
- Phép trừ đi Trung bình (`- mean`) giúp **kéo tâm của mọi bức ảnh về mốc số 0**.
- Phép chia cho Độ lệch chuẩn (`/ std`) giúp **ép khoảng cách dữ liệu dao động chật hẹp quanh 1.0**.
Nhờ đó AI học mượt mà hơn, tập trung vào đặc trưng hình học của vật thể chứ không phải độ chói của ánh sáng.

### Tại sao lại dùng 6 con số Mean / Std cố định?
```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```
> [!IMPORTANT]
> Đây là các thông số thống kê của 1.2 triệu bức ảnh từ bộ dữ liệu **ImageNet**.

Vì chúng ta thường sử dụng các mô hình pre-trained (đã được đào tạo trước bằng ImageNet thông qua kỹ thuật Transfer Learning), nên chúng ta BẮT BUỘC phải "chuẩn hóa" bức ảnh của mình theo đúng thói quen và lăng kính mà mô hình đó đã từng học. Nếu không làm vậy, mô hình sẽ bị "mù màu" và dự đoán sai.
