# 📌 Image Processing - Quick Reference Cheatsheet

> **Dành cho:** Ôn tập nhanh, tra cứu khi code, kiểm tra kiến thức  
> **Nguyên tắc:** Chỉ có từ khóa + ý chính - chi tiết bạn đã biết

---

## 1. PIXEL & CHANNEL

| Khái niệm | Công thức / Code |
|-----------|------------------|
| Image shape | `(H, W)` - grayscale<br>`(H, W, 3)` - BGR<br>`(H, W, 4)` - BGRA |
| dtype | `uint8` (0-255) |
| Read image | `cv2.imread()` |
| Access pixel | `img[row, col]` → `[B, G, R]` |
| ⚠️ **BGR trap** | OpenCV = BGR, not RGB |
| Fix | `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` |

---

## 2. COLOR SPACES

| Space | Use Case | Code |
|-------|----------|------|
| **GRAY** | Shape/edge, không cần màu | `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` |
| **HSV** | Màu sắc, ánh sáng thay đổi | `cv2.cvtColor(img, cv2.COLOR_BGR2HSV)` |
| **LAB** | Đo màu chính xác (Delta-E) | `cv2.cvtColor(img, cv2.COLOR_BGR2LAB)` |

```
HSV: H = Màu (0-179) | S = Độ đậm | V = Độ sáng
LAB: L = Sáng | A = Green→Red | B = Blue→Yellow
```

---

## 3. CONVOLUTION

| Khái niệm | Ý nghĩa |
|-----------|---------|
| Kernel | Ma trận nhỏ trượt trên ảnh |
| Operation | Dot product → 1 pixel mới |
| Kernel size | `3x3`, `5x5` - **phải số lẻ** |
| Stride | Bước nhảy (1 = giữ kích thước) |
| Padding | `same` (P = (k-1)/2) | `valid` (không pad) |

```python
cv2.filter2D(img, -1, kernel)
```

⚠️ **filter2D = correlation** (không lật kernel - không quan trọng với kernel đối xứng)

---

## 4. GAUSSIAN BLUR

| Mục đích | Giảm nhiễu |
|----------|------------|
| Code | `cv2.GaussianBlur(img, (5,5), sigmaX=0)` |
| Kernel | Phân phối Gaussian (trọng tâm cao nhất) |
| ⚠️ **Warning** | Blur quá mạnh → mất defect nhỏ |

| Kernel | Effect |
|--------|--------|
| `(3,3)` | Giữ chi tiết, lọc nhiễu kém |
| `(5,5)` | Cân bằng ✅ |
| `(7,7)` | Mượt mạnh, mất chi tiết |

---

## 5. EDGE DETECTION

### Sobel - Đạo hàm
```python
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
magnitude = cv2.magnitude(sobel_x, sobel_y)
```

### Canny - 5 bước

| # | Bước | Ý nghĩa |
|---|------|---------|
| 1 | Gaussian Blur | Khử nhiễu |
| 2 | Sobel | Tính gradient |
| 3 | Non-max suppression | Ép mỏng viền |
| 4 | Double threshold | Phân loại mạnh/yếu |
| 5 | Hysteresis | Nối biên yếu với biên mạnh |

```python
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
```

⚠️ **Tỷ lệ:** `threshold2 : threshold1 = 2:1` hoặc `3:1`

---

## 6. MORPHOLOGICAL OPERATIONS

| Operation | Formula | Dùng khi |
|-----------|---------|----------|
| **Erosion** | Ăn mòn | Xóa nhiễu nhỏ |
| **Dilation** | Giãn nở | Lấp lỗ hổng |
| **Opening** | Erosion → Dilation | Xóa nhiễu, tách vật dính |
| **Closing** | Dilation → Erosion | Nối đoạn đứt, lấp lỗ |

```python
kernel = np.ones((3,3), np.uint8)

eroded  = cv2.erode(binary, kernel)
dilated = cv2.dilate(binary, kernel)
opened  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closed  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
```

🎯 **Nhớ:** Opening = "mở" khoảng cách | Closing = "đóng" khoảng cách

---

## 7. THRESHOLDING

| Method | Code | Khi dùng |
|--------|------|----------|
| Simple | `cv2.threshold(gray, 127, 255, THRESH_BINARY)` | Ánh sáng đều |
| Otsu | `cv2.threshold(gray, 0, 255, THRESH_BINARY + THRESH_OTSU)` | 2 lớp rõ rệt |
| Adaptive | `cv2.adaptiveThreshold(gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2)` | Ánh sáng không đều |

```python
# Output luôn là binary (0/255)
```

---

## 8. CONTOUR DETECTION

```python
contours, hierarchy = cv2.findContours(binary, mode, method)
```

| Mode | Ý nghĩa |
|------|---------|
| `RETR_EXTERNAL` | Chỉ contour ngoài cùng ✅ |
| `RETR_TREE` | Tất cả, có hierarchy (lồng nhau) |
| `RETR_LIST` | Tất cả, không hierarchy |

| Method | Ý nghĩa |
|--------|---------|
| `CHAIN_APPROX_SIMPLE` | Chỉ giữ điểm cần thiết ✅ |
| `CHAIN_APPROX_NONE` | Giữ tất cả điểm |

```python
area = cv2.contourArea(cnt)
x, y, w, h = cv2.boundingRect(cnt)
perimeter = cv2.arcLength(cnt, True)
```

---

## 9. HISTOGRAM EQUALIZATION

| Method | Code | Ứng dụng |
|--------|------|----------|
| Global | `cv2.equalizeHist(gray)` | Ít dùng |
| CLAHE ✅ | `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))` | **Mặc định trong QC** |

**CLAHE params:**
- `clipLimit`: Giới hạn tương phản (↑ = contrast mạnh)
- `tileGridSize`: Số ô chia nhỏ (↑ = local hơn)

---

## 10. QUICK DECISION FLOW

```
Input Image
    │
    ├── Màu quan trọng? ── YES ──→ HSV/LAB
    │                         NO ──→ GRAY
    │
    ├── Gaussian Blur (3x3/5x5/7x5)
    │
    ├── CLAHE (nếu độ tương phản kém)
    │
    ├── Threshold
    │   ├── Ánh sáng đều? ── Otsu
    │   └── Ánh sáng không đều? ── Adaptive
    │
    ├── Morphology
    │   ├── Nhiễu nhỏ / dính nhau? ── Opening
    │   └── Đứt đoạn / lỗ hổng? ── Closing
    │
    ├── Detection
    │   ├── Cần edge/shape? ── Canny (50, 150)
    │   └── Cần vùng/blobs? ── findContours (RETR_EXTERNAL)
    │
    └── Filter by area (min_area)
         │
         └── Output: Bounding Boxes
```

---

## 11. COMMON BUGS - CHECKLIST

| Symptom | Root Cause | Fix |
|---------|------------|-----|
| Màu sai khi hiển thị | Quên BGR→RGB | `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` |
| GaussianBlur lỗi | Kernel size chẵn | Dùng số lẻ: `(5,5)` |
| Threshold ra toàn đen/trắng | `BINARY` vs `BINARY_INV` sai | Đảo lại flag |
| Quá nhiều contour rác | Thiếu morphology | Thêm Opening/Closing |
| findContours lỗi | OpenCV version khác | `contours, hierarchy = ...` (3.x+) |
| Canny toàn nhiễu | Thiếu Gaussian Blur | Blur trước Canny |
| CLAHE bị lưới ô | tileGridSize quá nhỏ | Tăng size: `(8,8)` → `(16,16)` |

---

## 12. KEYWORD SUMMARY

```
📷  OpenCV    🔵  BGR        ⚠️  RGB trap    🌫️  Blur
📐  Kernel    ➗  Convolution 🔄  Stride       📦  Padding
🎨  HSV       🏷️  LAB         ⬛  Grayscale   📊  Histogram
✂️  Threshold 📏  Otsu        📐  Adaptive    🔧  Morphology
🧹  Erosion   💨  Dilation    🔓  Opening     🔒  Closing
📏  Canny     📐  Sobel       📈  Gradient    🎯  Contour
📦  Bounding  📊  Area        📏  Perimeter   🚫  Noise
```

---

## 13. FORMULA REFERENCE

| Concept | Formula |
|---------|---------|
| Padding (same) | `P = (kernel_size - 1) / 2` |
| Grayscale | `0.114*B + 0.587*G + 0.299*R` |
| Gradient magnitude | `sqrt(Gx² + Gy²)` |
| Canny ratio | `threshold2 = 2×threshold1` đến `3×threshold1` |
| CLAHE tile | `tileGridSize = (8, 8)` cho HD |

---

## 14. CODE SNIPPETS - MOST USED

```python
# Read & Convert
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# CLAHE
clahe = cv2.createCLAHE(2.0, (8, 8))
enhanced = clahe.apply(blurred)

# Threshold
binary = cv2.adaptiveThreshold(enhanced, 255, 
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)

# Morphology
kernel = np.ones((3,3), np.uint8)
processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Contours
contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, 
                               cv2.CHAIN_APPROX_SIMPLE)

# Filter & Draw
for cnt in contours:
    if cv2.contourArea(cnt) > 50:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
```
