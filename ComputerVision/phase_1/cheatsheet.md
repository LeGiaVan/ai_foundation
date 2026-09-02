# Cheatsheet — Giai đoạn 1: Image Processing nền tảng (Tuần 2-3)

> Domain xuyên suốt: pipeline này chính là "tiền xử lý" nằm ngay trước model deep learning trong hệ thống kiểm tra chất lượng bằng camera. Nắm chắc phần này giúp bạn tự debug được 80% lỗi "model không detect được lỗi trên sản phẩm" — vì rất nhiều lỗi thực ra nằm ở bước tiền xử lý chứ không phải ở model.

---

## 1. Pixel & Channel

**Khái niệm cốt lõi:** ảnh là ma trận số nguyên (thường `uint8`, giá trị 0-255).

| Loại ảnh | Shape (OpenCV/numpy) | Ý nghĩa |
|---|---|---|
| Grayscale | `(H, W)` | 1 giá trị độ sáng / pixel |
| Color (BGR) | `(H, W, 3)` | 3 channel: Blue, Green, Red |
| Color + alpha | `(H, W, 4)` | thêm channel trong suốt |

```python
import cv2
img = cv2.imread("part.jpg")   # đọc ảnh -> shape (H, W, 3), dtype uint8
print(img.shape, img.dtype)
print(img[100, 200])           # pixel tại row=100, col=200 -> [B, G, R]
```

⚠️ **Bẫy kinh điển:** OpenCV đọc ảnh theo thứ tự **BGR**, không phải RGB. Nếu đưa thẳng vào matplotlib (`plt.imshow`) hoặc PyTorch (thường train theo RGB) mà không đổi thứ tự, màu sẽ bị lệch → ảnh hưởng đến augmentation dựa trên màu (color jitter) và có thể khiến model học sai đặc trưng màu của lỗi (VD: vết ố vàng bị hiểu nhầm màu khác).

```python
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # đổi BGR -> RGB khi cần
```

---

## 2. Color space

| Color space | Dùng khi nào trong QC nhà máy |
|---|---|
| **Grayscale** | Edge detection, thresholding hình học, khi màu không quan trọng (VD: kiểm tra vết nứt, méo hình) |
| **HSV** | Tách lỗi theo **màu sắc** ổn định hơn RGB dưới ánh sáng thay đổi (VD: phát hiện vết ố, đổi màu bề mặt, phân loại theo màu sản phẩm) |
| **LAB** | Khi cần đo sai lệch màu chính xác (color difference), ít bị ảnh hưởng bởi độ sáng hơn RGB |

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)   # H: 0-179, S: 0-255, V: 0-255 (OpenCV convention)
```

**Vì sao HSV tốt hơn RGB khi ánh sáng thay đổi:** trong HSV, kênh **Hue (màu sắc thuần)** tách biệt khỏi **Value (độ sáng)**. Khi đèn nhà máy nhấp nháy/thay đổi cường độ, Value thay đổi nhưng Hue của một vết lỗi màu vẫn tương đối ổn định → threshold theo Hue robust hơn threshold theo RGB.

**Giải nghĩa đơn giản:**
- **HSV** (giống cách con người miêu tả màu):
  - **H (Hue - Màu sắc)**: Bản chất nó là màu gì? (Đỏ, cam, vàng, lục...). Tưởng tượng một vòng tròn màu sắc, H chính là góc trên vòng tròn đó (OpenCV dùng thang 0-179 thay vì 0-360).
  - **S (Saturation - Độ bão hòa)**: Màu đó đậm hay phai? (VD: Đỏ tươi thì S cao, đỏ nhạt thành hồng thì S thấp).
  - **V (Value - Độ sáng)**: Màu đó sáng chói hay tối thui?
- **LAB** (Mô phỏng chính xác sinh học mắt người):
  - **L (Lightness)**: Độ sáng/tối thuần túy, hoàn toàn độc lập với màu.
  - **A**: Chạy từ Xanh lục (Green) sang Đỏ (Red).
  - **B**: Chạy từ Xanh lam (Blue) sang Vàng (Yellow).
  - *Tại sao chia A và B như vậy?* Vì võng mạc người hoạt động theo các cặp màu đối lập (bạn không bao giờ thấy màu "xanh lục ngả đỏ"). Đo khoảng cách giữa 2 pixel trong không gian LAB giống hệt như cách mắt người cảm nhận độ lệch màu.

```python
# Ví dụ: tách vùng màu đỏ (VD: linh kiện lỗi được đánh dấu đỏ) trong HSV
lower_red = (0, 120, 70)
upper_red = (10, 255, 255)
mask = cv2.inRange(hsv, lower_red, upper_red)
```

---

## 3. Convolution & Kernel

**Định nghĩa trực giác:** một kernel (ma trận nhỏ, VD 3x3) trượt qua từng vị trí trên ảnh, tại mỗi vị trí tính **dot product** giữa kernel và vùng ảnh tương ứng → ra 1 giá trị pixel mới.

```
output(x,y) = Σ Σ kernel(i,j) * input(x+i, y+j)
```

Đây chính là phép toán nền tảng cho **cả** filter cổ điển (blur, sharpen, edge) **lẫn** CNN ở Giai đoạn 2 — khác biệt duy nhất là kernel filter cổ điển do người thiết kế sẵn (fixed), còn kernel CNN được học từ dữ liệu.

**Các tham số quan trọng (sẽ gặp lại ở Giai đoạn 2):**
- **Kernel size**: 3x3, 5x5... — kernel càng lớn, vùng ảnh hưởng (receptive field) càng rộng, ảnh càng mượt/mất chi tiết.
- **Stride**: bước nhảy của kernel. Stride=1 giữ nguyên kích thước ảnh (nếu có padding phù hợp).
- **Padding**: thêm viền (thường là 0) quanh ảnh để kernel xử lý được cả pixel biên. `'valid'` = không pad (ảnh output nhỏ hơn input), `'same'` = pad để output = input.

```python
kernel = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])   # kernel sharpen
sharpened = cv2.filter2D(img, -1, kernel)
```

---

## 4. Gaussian Blur

**Mục đích:** làm mượt ảnh, giảm nhiễu (noise) trước khi edge detection hoặc thresholding — bước gần như bắt buộc trong pipeline QC vì camera công nghiệp luôn có nhiễu cảm biến.

**Cách hoạt động:** kernel có giá trị phân phối theo hàm Gaussian (trọng số pixel trung tâm cao nhất, giảm dần ra biên) — đây là lý do bạn cần hiểu khái niệm **phân phối Gaussian** ở Giai đoạn 0.

```python
blurred = cv2.GaussianBlur(img, (5, 5), sigmaX=0)
# (5,5): kernel size (phải là số lẻ)
# sigmaX=0: OpenCV tự tính sigma dựa trên kernel size
```

| Tham số | Ảnh hưởng |
|---|---|
| Kernel size lớn | Mượt hơn, nhưng mất chi tiết nhỏ (có thể xóa mất vết lỗi nhỏ!) |
| Kernel size nhỏ | Giữ chi tiết, nhưng lọc nhiễu kém |
| sigma lớn | Blur mạnh hơn dù kernel size giữ nguyên |

⚠️ **Lưu ý QC thực tế:** blur quá mạnh có thể **xóa mất defect nhỏ** (vết trầy, lỗ kim) trước khi model kịp thấy. Luôn tune kernel size theo kích thước lỗi nhỏ nhất cần phát hiện, không dùng giá trị mặc định mù quáng.

---

## 5. Edge Detection

### 5a. Sobel — đạo hàm theo hướng x/y

Chính là ứng dụng trực tiếp của khái niệm **đạo hàm** ở Giai đoạn 0: Sobel xấp xỉ tốc độ thay đổi cường độ pixel theo trục x và y. Nơi cường độ thay đổi đột ngột (biên vật thể) → đạo hàm lớn → là edge.

```python
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)   # đạo hàm theo x
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)   # đạo hàm theo y
magnitude = cv2.magnitude(sobel_x, sobel_y)             # độ lớn gradient = sqrt(gx² + gy²)
```

### 5b. Canny — pipeline edge detection hoàn chỉnh, hay dùng nhất

Canny = Gaussian blur → tính gradient (Sobel) → non-max suppression (giữ lại edge mỏng nhất) → double threshold (phân loại edge mạnh/yếu) → hysteresis (nối edge yếu vào edge mạnh nếu liền kề).

```python
import cv2

# 1. Luôn dùng ảnh grayscale cho Canny
gray = cv2.imread('product.jpg', cv2.IMREAD_GRAYSCALE)

# 2. QUAN TRỌNG: Blur ảnh để khử nhiễu trước khi tìm cạnh
# Nếu không blur, Canny rất nhạy cảm với các đốm nhiễu lấm tấm
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 3. Chạy Canny Edge Detection
# gradient > 150 -> chắc chắn là cạnh
# gradient < 50 -> chắc chắn vứt bỏ
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
```

| Tham số | Ý nghĩa |
|---|---|
| `threshold1` (lower) | Dưới ngưỡng này → chắc chắn không phải edge |
| `threshold2` (upper) | Trên ngưỡng này → chắc chắn là edge |
| Giữa hai ngưỡng | Chỉ giữ nếu nối liền với edge mạnh (hysteresis) |

**Mẹo tune threshold nhanh:** tỷ lệ `threshold2 : threshold1` thường 2:1 hoặc 3:1. Bắt đầu (50, 150) rồi chỉnh theo độ tương phản ảnh thực tế — ảnh nhà máy sáng đều thì threshold có thể cao hơn để giảm nhiễu.

---

## 6. Morphological Operations

Thao tác trên ảnh **nhị phân** (binary, thường sau threshold), dùng kernel gọi là **structuring element**.

| Phép toán | Hiệu ứng | Dùng khi nào |
|---|---|---|
| **Erosion** (ăn mòn) | Thu nhỏ vùng trắng, xóa nhiễu nhỏ | Loại bỏ đốm nhiễu lấm tấm sau threshold |
| **Dilation** (giãn nở) | Phình to vùng trắng, lấp lỗ hổng nhỏ | Nối liền các mảnh vỡ của cùng 1 vết lỗi bị đứt đoạn |
| **Opening** = Erosion → Dilation | Xóa nhiễu nhỏ mà không làm mất kích thước vật thể chính | Tiền xử lý trước khi đếm/đo vật thể |
| **Closing** = Dilation → Erosion | Lấp lỗ hổng nhỏ bên trong vật thể mà không làm phình to biên ngoài | Làm liền mạch vùng defect bị vỡ vụn |

```python
kernel = np.ones((3, 3), np.uint8)
eroded  = cv2.erode(binary_img, kernel, iterations=1)
dilated = cv2.dilate(binary_img, kernel, iterations=1)
opened  = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
closed  = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
```

**Tình huống QC thực tế:** ảnh binary sau threshold có vết trầy bị đứt thành nhiều đoạn nhỏ do nhiễu → dùng **closing** để nối lại thành 1 vùng liền → contour detection (mục 8) mới đếm được đúng là 1 lỗi thay vì 5 lỗi nhỏ giả.

---

## 7. Thresholding

Chuyển ảnh grayscale → ảnh nhị phân (0/255), bước bắt buộc trước contour detection.

### 7a. Simple threshold
```python
_, binary = cv2.threshold(gray, thresh=127, maxval=255, type=cv2.THRESH_BINARY)
```
Nhược điểm: một ngưỡng cố định (127) không robust khi ánh sáng thay đổi giữa các lần chụp.

### 7b. Otsu's threshold — tự động chọn ngưỡng tối ưu
```python
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```
Otsu tự tìm ngưỡng tách 2 lớp (foreground/background) sao cho phương sai trong từng lớp nhỏ nhất — tốt khi ảnh có phân bố sáng/tối rõ 2 nhóm (VD: vật thể sáng trên nền băng chuyền tối).

### 7c. Adaptive threshold — khi ánh sáng không đều trên cùng 1 ảnh
```python
binary = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, blockSize=11, C=2
)
```
Mỗi vùng nhỏ trong ảnh tự tính ngưỡng riêng dựa trên vùng lân cận (`blockSize`) — cực kỳ quan trọng trong nhà máy vì ánh sáng đèn thường không đều trên toàn khung hình (VD: góc camera bị bóng đổ).

**Bảng chọn nhanh:**

| Tình huống ánh sáng | Nên dùng |
|---|---|
| Ánh sáng đều, ổn định | Simple hoặc Otsu |
| Ánh sáng thay đổi theo vị trí trong ảnh | Adaptive threshold |
| Ánh sáng thay đổi theo thời gian (giữa các lần chụp) | Otsu (tự tính lại mỗi ảnh) + cân nhắc calibrate đèn định kỳ |

---

## 8. Contour Detection

**Contour** = đường viền khép kín bao quanh vùng có cùng cường độ/màu, tìm được trên ảnh binary.

```python
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)          # diện tích vùng lỗi -> lọc nhiễu nhỏ theo area
    x, y, w, h = cv2.boundingRect(cnt)    # bounding box quanh contour
    perimeter = cv2.arcLength(cnt, True)  # chu vi -> tính độ tròn, hình dạng bất thường

    if area > 50:   # lọc contour quá nhỏ (nhiễu)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
```

| Tham số `mode` | Ý nghĩa |
|---|---|
| `RETR_EXTERNAL` | Chỉ lấy contour ngoài cùng (bỏ qua lỗ bên trong) |
| `RETR_TREE` | Lấy toàn bộ hierarchy (contour lồng nhau) — cần khi lỗi có cấu trúc phức tạp (lỗ trong lỗ) |

**Ứng dụng QC trực tiếp:** đây là cách "detect lỗi" đơn giản nhất **không cần deep learning** — threshold vùng bất thường → tìm contour → lọc theo area/shape → coi là defect. Nhiều dây chuyền thực tế vẫn dùng pipeline classical này cho lỗi hình học rõ ràng (méo, thiếu chi tiết, lệch vị trí) vì nhanh, không cần training, dễ giải thích với QA — chỉ chuyển sang deep learning khi lỗi phức tạp/tinh vi hơn (Giai đoạn 3-5).

---

## 9. Histogram Equalization

**Mục đích:** tăng độ tương phản ảnh bị quá tối/quá sáng, giúp bước threshold/edge detection phía sau hoạt động tốt hơn.

```python
equalized = cv2.equalizeHist(gray)   # chỉ áp dụng cho ảnh grayscale 1 kênh
```

**Vấn đề của equalizeHist thường:** áp dụng đồng đều toàn ảnh, có thể làm nhiễu vùng đã đủ sáng bị "cháy" quá mức. Giải pháp:

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized_clahe = clahe.apply(gray)
```

**CLAHE (Contrast Limited Adaptive Histogram Equalization):** chia ảnh thành các ô nhỏ (`tileGridSize`), cân bằng histogram riêng từng ô, giới hạn độ tương phản (`clipLimit`) để tránh khuếch đại nhiễu quá mức — **lựa chọn mặc định nên dùng trong pipeline QC công nghiệp** thay vì `equalizeHist` thường, vì ảnh sáng nhà máy hiếm khi đồng đều toàn khung hình.

---

## 10. Checklist lỗi thường gặp (self-debug trước khi hỏi ai)

- [ ] Đổi màu bất thường khi hiển thị bằng matplotlib → quên `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`
- [ ] `cv2.GaussianBlur` báo lỗi kernel size → kernel size phải là **số lẻ** (VD: (5,5), không phải (4,4))
- [ ] Threshold ra toàn đen hoặc toàn trắng → kiểm tra lại `THRESH_BINARY` vs `THRESH_BINARY_INV`, hoặc ánh sáng ảnh input có vấn đề
- [ ] Contour tìm được quá nhiều mảnh vụn → thiếu bước morphological closing/opening trước đó
- [ ] `findContours` báo lỗi số lượng giá trị trả về không khớp → phiên bản OpenCV khác nhau trả về 2 hoặc 3 giá trị (`contours, hierarchy` vs `image, contours, hierarchy`), kiểm tra version đang dùng
- [ ] Edge detection ra toàn nhiễu vụn → thiếu Gaussian blur trước Canny, hoặc threshold Canny chưa phù hợp độ tương phản ảnh

---

## 11. Bài tập thực hành đề xuất (áp dụng ngay domain QC)

Xây 1 pipeline nhỏ, input là ảnh sản phẩm có vết lỗi giả lập (VD: vẽ tay 1 vết trầy trên ảnh mẫu), output là ảnh đã khoanh vùng lỗi bằng bounding box đỏ:

1. Đọc ảnh → chuyển grayscale
2. Gaussian blur giảm nhiễu
3. Adaptive threshold hoặc Otsu tách vùng bất thường
4. Morphological closing để nối vùng lỗi bị đứt đoạn
5. Tìm contour, lọc theo area tối thiểu
6. Vẽ bounding box + tính toán area/perimeter của từng lỗi, in ra console

→ Đây chính là "mini phiên bản classical" của toàn bộ hệ thống QC bạn sẽ xây ở Capstone (Giai đoạn 11), chỉ khác là sau này bounding box sẽ do model deep learning sinh ra thay vì contour detection thuần túy.

---

## 12. Câu hỏi tự kiểm tra (chuẩn bị sớm cho phỏng vấn)

- Convolution khác correlation ở điểm nào? (Gợi ý: convolution có lật kernel 180°, nhưng OpenCV `filter2D` thực chất làm correlation — không ảnh hưởng nhiều với kernel đối xứng)
- Vì sao Canny cần Gaussian blur trước khi tính gradient?
- Erosion và Dilation, cái nào dùng để loại nhiễu, cái nào dùng để nối liền vùng bị đứt?
- Tại sao nên dùng adaptive threshold thay vì threshold cố định trong môi trường ánh sáng nhà máy?
- HSV có lợi thế gì so với RGB khi ánh sáng thay đổi theo thời gian?