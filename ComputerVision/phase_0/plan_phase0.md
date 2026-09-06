# Cheatsheet — Giai đoạn 0: Python & Toán nền cho CV (Tuần 1)

> Đây là nền tảng cho **toàn bộ** roadmap 30 tuần. Không cần thành thạo như dân toán/CS, chỉ cần đủ trực giác + thao tác code để không bị khựng lại khi đọc paper hoặc code PyTorch/OpenCV ở các giai đoạn sau. Mỗi phần dưới đây đều có ghi chú "dùng ở đâu trong roadmap" để bạn thấy ngay giá trị thực tế, tránh cảm giác học toán suông.

---

## 1. NumPy Vectorization

### 1a. Array cơ bản — nền tảng để "đọc" mọi đoạn code CV

```python
import numpy as np

a = np.array([1, 2, 3])                    # vector 1D, shape (3,)
b = np.array([[1, 2], [3, 4]])              # ma trận 2D, shape (2, 2)
img = np.zeros((224, 224, 3), dtype=np.uint8)  # tensor 3D giả lập ảnh RGB

print(a.shape, a.dtype, a.ndim)
```

| Thuộc tính | Ý nghĩa | Vì sao quan trọng trong CV |
|---|---|---|
| `.shape` | Kích thước từng chiều | Đọc sai shape = nguồn lỗi #1 khi ghép pipeline (VD: đưa ảnh (H,W,C) vào model cần (C,H,W)) |
| `.dtype` | Kiểu dữ liệu (uint8, float32...) | Ảnh đọc từ OpenCV là `uint8` (0-255), nhưng model cần `float32` (0-1 hoặc chuẩn hoá) — quên convert là lỗi rất phổ biến |
| `.ndim` | Số chiều | Grayscale = 2D, color = 3D, batch ảnh = 4D |

### 1b. Vectorization thay vì loop — vì sao và tốc độ khác nhau thế nào

```python
# CÁCH CHẬM (loop thuần Python) — KHÔNG làm vậy với ảnh
img_float = np.zeros_like(img, dtype=np.float32)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        for k in range(img.shape[2]):
            img_float[i, j, k] = img[i, j, k] / 255.0

# CÁCH ĐÚNG (vectorized) — nhanh hơn hàng trăm lần
img_float = img.astype(np.float32) / 255.0
```

**Vì sao nhanh hơn:** NumPy thực thi phép toán bằng code C biên dịch sẵn, xử lý cả block dữ liệu cùng lúc (tận dụng SIMD của CPU), thay vì Python phải diễn giải (interpret) từng phép toán một trong vòng lặp. Với ảnh 224x224x3 ≈ 150,000 phần tử, loop Python có thể chậm hơn 100-1000 lần so với vectorized.

**Quy tắc thực hành:** nếu bạn thấy mình đang viết `for` loop để duyệt qua từng pixel/element của một mảng NumPy — **dừng lại**, gần như luôn có cách vectorize.

### 1c. Indexing & Slicing đa chiều

```python
crop = img[50:100, 30:80, :]        # cắt vùng ảnh: row 50-100, col 30-80, giữ mọi channel
red_channel = img[:, :, 2]          # lấy channel Red (OpenCV: BGR, index 2 = R)
img_rgb = img[:, :, ::-1]           # đảo thứ tự channel: BGR -> RGB
flipped = img[:, ::-1, :]           # lật ảnh theo chiều ngang (horizontal flip — dùng trong augmentation)
```

| Cú pháp | Ý nghĩa |
|---|---|
| `img[a:b, c:d]` | Slice theo khoảng — dùng để crop ROI (region of interest), rất hay dùng khi chỉ cần kiểm tra 1 vùng cụ thể trên sản phẩm |
| `img[:, :, ::-1]` | Đảo thứ tự phần tử theo chiều cuối — đổi BGR↔RGB |
| `img[::2, ::2]` | Lấy mỗi 2 pixel 1 lần — downsample ảnh nhanh (không nội suy, khác `cv2.resize`) |
| Boolean indexing: `img[mask]` | Lấy các pixel thoả điều kiện — dùng khi áp mask nhị phân (kết quả threshold/segmentation) lên ảnh gốc |

### 1d. Broadcasting

**Quy tắc:** NumPy tự động "nhân bản" mảng nhỏ hơn để khớp shape với mảng lớn hơn, miễn là các chiều tương thích (bằng nhau, hoặc một trong hai bằng 1).

```python
img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

mean = np.array([0.485, 0.456, 0.406])   # shape (3,)
std  = np.array([0.229, 0.224, 0.225])   # shape (3,)

normalized = (img.astype(np.float32) / 255.0 - mean) / std
# img: (224, 224, 3), mean/std: (3,) -> broadcast tự động áp cho từng channel
```

Đây chính xác là bước **normalize ảnh** bạn sẽ viết đi viết lại xuyên suốt Giai đoạn 2 trở đi (giá trị mean/std ở trên là chuẩn ImageNet, rất hay gặp khi dùng pretrained model).

⚠️ **Lỗi hay gặp:** shape `(224, 224, 3)` broadcast với `(3, 224, 224)` sẽ **báo lỗi hoặc cho kết quả sai âm thầm** — luôn kiểm tra chiều channel đang ở đầu hay cuối trước khi broadcast.

### 1e. reshape / transpose / expand_dims — chuyển đổi format HWC ↔ CHW

Đây là thao tác bạn sẽ dùng ở **mọi** bước chuyển từ ảnh (OpenCV/numpy quen dùng `(H, W, C)`) sang tensor đưa vào PyTorch (quen dùng `(C, H, W)`).

```python
img = np.zeros((224, 224, 3))          # (H, W, C) — format ảnh thông thường

chw = img.transpose(2, 0, 1)           # -> (C, H, W) — format PyTorch cần
print(chw.shape)                       # (3, 224, 224)

batch = np.expand_dims(chw, axis=0)    # -> (1, C, H, W) — thêm chiều batch
print(batch.shape)                     # (1, 3, 224, 224)
```

| Hàm | Tác dụng | Lưu ý |
|---|---|---|
| `.reshape(new_shape)` | Đổi shape, giữ nguyên dữ liệu và thứ tự phần tử | Chỉ dùng khi **không** cần đổi thứ tự trục — nhầm `reshape` với `transpose` là lỗi rất phổ biến (dữ liệu bị "xáo trộn" sai vị trí dù không báo lỗi) |
| `.transpose(axes)` | Đổi **thứ tự trục** thực sự | Dùng khi chuyển HWC↔CHW |
| `np.expand_dims(arr, axis)` | Thêm 1 chiều kích thước=1 | Dùng khi thêm chiều batch trước khi đưa vào model |
| `np.squeeze(arr)` | Bỏ các chiều kích thước=1 | Dùng khi lấy output model (1, H, W) về lại (H, W) để hiển thị |

### 1f. Aggregation — mean, std, sum theo axis

```python
img = np.random.rand(224, 224, 3)

mean_per_channel = img.mean(axis=(0, 1))   # trung bình theo H,W -> shape (3,) = mean từng channel
print(img.mean(), img.std(), img.min(), img.max())   # thống kê toàn ảnh — dùng để debug nhanh ảnh có bất thường không
```

**Mẹo debug thực tế:** khi pipeline cho kết quả lạ, việc đầu tiên nên làm là in `img.shape`, `img.dtype`, `img.min()`, `img.max()` — 90% lỗi lộ ra ngay ở bước này (VD: ảnh toàn giá trị 0, hoặc dtype sai khiến phép chia ra toàn 0).

---

## 2. Ma trận / Tensor

### 2a. Phân cấp khái niệm — ánh xạ trực tiếp sang dữ liệu CV

| Toán học | Số chiều | Ví dụ trong CV |
|---|---|---|
| Scalar | 0D | 1 giá trị pixel grayscale, 1 giá trị loss |
| Vector | 1D | Feature embedding của 1 ảnh (VD: vector 512 chiều output từ ResNet) |
| Ma trận (Matrix) | 2D | 1 ảnh grayscale `(H, W)` |
| Tensor | ≥3D | 1 ảnh màu `(H, W, C)`, batch ảnh `(N, C, H, W)`, hoặc feature map trong CNN |

**Kỹ năng cần luyện:** nhìn một dòng code `x.shape = (32, 3, 224, 224)` và **phản xạ ngay** đọc ra: batch 32 ảnh, 3 channel, 224x224 pixel — không cần suy nghĩ lâu. Kỹ năng này quan trọng hơn nhiều so với việc nhớ công thức toán.

### 2b. Phép nhân ma trận vs phép nhân element-wise

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

elementwise = A * B          # nhân từng phần tử tương ứng -> shape (2,2)
matmul = A @ B                # phép nhân ma trận thực sự (hoặc np.matmul(A, B))
```

| Phép toán | Ký hiệu NumPy | Dùng khi nào trong CV |
|---|---|---|
| Element-wise | `*`, `np.multiply` | Áp mask lên ảnh, nhân trọng số attention, tính loss element-wise |
| Matrix multiplication | `@`, `np.matmul`, `np.dot` (cho 2D) | Fully-connected layer (`output = W @ x + b`), phép biến đổi tuyến tính, tính similarity giữa các embedding |

⚠️ **Bẫy rất hay gặp với người mới:** dùng `*` khi ý định thực sự là `@` (hoặc ngược lại) — NumPy **không báo lỗi** nếu shape tình cờ hợp lệ cho cả hai phép, kết quả sai âm thầm rất khó phát hiện. Luôn tự hỏi: "mình đang muốn nhân từng phần tử, hay đang muốn biến đổi không gian vector?"

### 2c. Transpose & Identity matrix (khái niệm, không cần tính tay)

```python
A_T = A.T                          # chuyển vị: đổi hàng thành cột
I = np.eye(3)                      # ma trận đơn vị 3x3
```

Chỉ cần biết: transpose sẽ xuất hiện liên tục khi chuyển đổi shape (mục 1e), và ma trận đơn vị là nền tảng khái niệm cho ma trận nghịch đảo — sẽ gặp lại khi học camera calibration (Giai đoạn 7: tính ma trận intrinsic/extrinsic của camera).

---

## 3. Đại số tuyến tính cơ bản

### 3a. Dot product (tích vô hướng)

```python
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dot = np.dot(v1, v2)     # = 1*4 + 2*5 + 3*6 = 32
```

**Ý nghĩa hình học:** dot product đo mức độ "cùng hướng" giữa hai vector — dot product lớn khi hai vector gần như song song cùng chiều, bằng 0 khi vuông góc, âm khi ngược chiều.

**Vì sao quan trọng trong CV — 2 ứng dụng trực tiếp:**
1. **Convolution** (Giai đoạn 1 & 2): mỗi lần kernel trượt qua ảnh, phép tính thực chất là dot product giữa kernel (dàn phẳng thành vector) và vùng ảnh tương ứng.
2. **Cosine similarity** — so sánh độ giống nhau giữa 2 embedding (VD: so sánh ảnh sản phẩm hiện tại với ảnh mẫu "chuẩn" trong anomaly detection ở Giai đoạn 5):

```python
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
```

### 3b. Norm (L1, L2) — đo "độ lớn" của vector

```python
v = np.array([3, 4])
l2_norm = np.linalg.norm(v, ord=2)   # = sqrt(3² + 4²) = 5.0  (norm Euclid — phổ biến nhất)
l1_norm = np.linalg.norm(v, ord=1)   # = |3| + |4| = 7.0
```

| Norm | Công thức | Dùng ở đâu |
|---|---|---|
| **L2** | `sqrt(Σ x²)` | MSE loss (`(pred - target)²`), khoảng cách Euclid giữa 2 embedding (dùng trong PatchCore — Giai đoạn 5, so khoảng cách feature ảnh test với "ngân hàng" feature ảnh bình thường) |
| **L1** | `Σ |x|` | MAE loss, L1 regularization (khuyến khích weight thưa — sparse) |

### 3c. Eigenvalue / Eigenvector — chỉ cần trực giác hình học

**Định nghĩa trực giác (không cần chứng minh):** với một phép biến đổi tuyến tính (ma trận A), hầu hết vector khi nhân với A sẽ vừa bị **xoay** vừa bị **scale**. Nhưng có một số vector đặc biệt — **eigenvector** — chỉ bị **scale** (kéo dài/thu ngắn) mà **không đổi hướng**. Hệ số scale đó gọi là **eigenvalue**.

```python
A = np.array([[2, 0], [0, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(eigenvalues)    # [2. 3.]
```

**Vì sao cần biết (dù không cần tính tay):**
- **PCA (giảm chiều dữ liệu):** eigenvector của ma trận hiệp phương sai (covariance matrix) chỉ ra "hướng" mà dữ liệu biến thiên nhiều nhất — dùng để nén embedding, trực quan hoá feature space cao chiều.
- **Camera calibration & homography (Giai đoạn 7):** các phép biến đổi hình học giữa ảnh 2D và không gian 3D dựa trên phân tích ma trận biến đổi, eigenvalue/eigenvector là công cụ nền để hiểu các phép phân tích đó (SVD — Singular Value Decomposition — là mở rộng của eigen-decomposition).

👉 Xem lại video 3Blue1Brown (phụ đề Việt) tập "Eigenvectors and eigenvalues" nếu muốn hình dung trực quan — không cần làm bài tập tính tay phức tạp ở giai đoạn này.

### 3d. Determinant (chỉ cần biết khái niệm, không cần đào sâu)

Determinant đo "hệ số scale diện tích/thể tích" mà một phép biến đổi tuyến tính gây ra. Determinant = 0 nghĩa là phép biến đổi "làm sập" không gian xuống chiều thấp hơn (mất thông tin, ma trận không khả nghịch). Chỉ cần nhớ ý nghĩa này — sẽ gặp lại khi ma trận camera bị suy biến (degenerate) trong bài toán calibration.

---

## 4. Gradient / Đạo hàm

### 4a. Đạo hàm là gì — trực giác

Đạo hàm của hàm số tại một điểm = **độ dốc** (slope) của đồ thị tại điểm đó = tốc độ thay đổi của output khi input thay đổi một lượng cực nhỏ.

```
f'(x) = lim(h→0) [f(x+h) - f(x)] / h
```

Không cần tính giới hạn tay — chỉ cần nhớ: đạo hàm dương → hàm đang tăng, đạo hàm âm → hàm đang giảm, đạo hàm càng lớn → dốc càng đứng.

**Liên hệ trực tiếp Giai đoạn 1:** bộ lọc Sobel (edge detection) **chính là** phép xấp xỉ đạo hàm rời rạc của cường độ pixel theo trục x/y. Nơi đạo hàm (= độ dốc thay đổi cường độ) lớn nhất chính là biên vật thể (edge).

### 4b. Đạo hàm riêng (partial derivative) & Gradient vector

Khi hàm số phụ thuộc **nhiều biến** (VD: loss function phụ thuộc hàng triệu weight trong mạng neural), đạo hàm riêng theo 1 biến = đạo hàm tính theo biến đó, **giữ cố định tất cả biến còn lại**.

**Gradient** = vector chứa toàn bộ đạo hàm riêng theo từng biến:

```
∇f = [∂f/∂x1, ∂f/∂x2, ..., ∂f/∂xn]
```

Gradient chỉ ra **hướng tăng nhanh nhất** của hàm số tại điểm đang xét.

### 4c. Gradient Descent — trực giác "quả bóng lăn xuống dốc"

```python
# Giả lập gradient descent tối giản trên hàm f(x) = x²  (đạo hàm f'(x) = 2x)
x = 10.0            # điểm khởi tạo
lr = 0.1             # learning rate
for step in range(20):
    grad = 2 * x     # đạo hàm tại x
    x = x - lr * grad     # đi ngược hướng gradient để giảm f(x)
print(x)   # x tiến dần về 0 -- điểm cực tiểu của f(x) = x²
```

| Khái niệm | Ý nghĩa |
|---|---|
| **Gradient descent** | Cập nhật tham số theo hướng **ngược** với gradient (vì gradient chỉ hướng tăng, ta muốn giảm loss) |
| **Learning rate (lr)** | Bước nhảy mỗi lần cập nhật — quá lớn dễ "nhảy qua" điểm tối ưu (loss dao động/phân kỳ), quá nhỏ hội tụ rất chậm |
| **Learning rate scheduling** (sẽ gặp ở Giai đoạn 2) | Giảm dần lr theo thời gian train — bước đầu đi nhanh, bước sau đi chậm để hội tụ chính xác hơn |

### 4d. Chain rule — nền tảng của Backpropagation (Giai đoạn 2)

Khi một hàm là **hợp của nhiều hàm** (VD: input → conv layer 1 → conv layer 2 → ... → loss), đạo hàm của loss theo một weight ở layer đầu được tính bằng cách **nhân liên tiếp** đạo hàm qua từng layer trung gian:

```
∂Loss/∂w1 = ∂Loss/∂layer_n × ∂layer_n/∂layer_(n-1) × ... × ∂layer_1/∂w1
```

Đây chính là cơ chế **backpropagation**: lan truyền đạo hàm ngược từ loss về đầu mạng, layer nào cũng chỉ cần biết đạo hàm cục bộ của chính nó, nhân dồn lại theo chain rule. Ở Giai đoạn 0 này bạn **chưa cần** tự code backprop tay — chỉ cần hiểu đủ để không thấy `loss.backward()` trong PyTorch là "hộp đen ma thuật".

---

## 5. Xác suất cơ bản

### 5a. Phân phối (Distribution) — đặc biệt là Gaussian/Normal

Phân phối mô tả "khả năng xảy ra" của các giá trị. Phân phối **Gaussian (chuẩn)** là quan trọng nhất trong CV — đặc trưng bởi 2 tham số: **mean (μ)** và **độ lệch chuẩn (σ)**.

```python
import numpy as np
samples = np.random.normal(loc=0.0, scale=1.0, size=1000)   # loc = mean, scale = std
```

**3 nơi Gaussian xuất hiện xuyên suốt roadmap:**
1. **Gaussian blur** (Giai đoạn 1): trọng số kernel được lấy từ hàm mật độ Gaussian 2D.
2. **Khởi tạo weight** trong neural network (Giai đoạn 2): weight ban đầu thường được sample từ phân phối Gaussian quanh 0.
3. **PaDiM — anomaly detection** (Giai đoạn 5): giả định feature embedding của ảnh "bình thường" tuân theo phân phối Gaussian đa biến, ảnh bất thường sẽ có xác suất thấp dưới phân phối này (đo bằng khoảng cách Mahalanobis — mở rộng của z-score dùng ma trận hiệp phương sai).

### 5b. Kỳ vọng (Expectation) & Phương sai (Variance)

```python
mean = np.mean(samples)     # kỳ vọng mẫu — giá trị "trung tâm" của phân phối
var = np.var(samples)       # phương sai — mức độ phân tán quanh mean
std = np.std(samples)       # độ lệch chuẩn = sqrt(variance)
```

**Ứng dụng trực tiếp — chuẩn hoá ảnh (normalization):**

```python
normalized = (img - img.mean()) / img.std()
```

Trừ mean, chia std giúp dữ liệu về phân phối có tâm quanh 0, độ trải rộng chuẩn hoá — giúp mô hình hội tụ nhanh và ổn định hơn khi train (tránh gradient bùng nổ/biến mất do input có scale quá lớn/nhỏ khác nhau giữa các channel).

### 5c. Softmax & Cross-entropy — cầu nối giữa xác suất và classification

**Công thức Toán học:**
- **Softmax:** Biến đổi vector điểm số thô $z_i$ thành xác suất $p_i$:
  $$ p_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}} $$
- **Cross-entropy Loss:** Tính hình phạt khi dự đoán, với $y_i$ là nhãn thực tế (1 cho lớp đúng, 0 cho lớp sai) và $p_i$ là xác suất dự đoán:
  $$ L = -\sum_{i} y_i \log(p_i) $$
  *(Hiểu đơn giản: Đối với lớp đúng, Loss = $-\log(\text{xác suất dự đoán})$. Do đó: Đoán sai mà quá tự tin $\rightarrow$ phạt cực nặng; đoán đúng mà rụt rè $\rightarrow$ vẫn bị phạt; đoán đúng và tự tin 100% $\rightarrow$ Loss = 0).*

```python
def softmax(logits):
    exp = np.exp(logits - np.max(logits))   # trừ max để tránh tràn số (numerical stability)
    return exp / exp.sum()

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(probs)   # VD: [0.659, 0.242, 0.099] -- tổng = 1, mỗi giá trị là "xác suất" thuộc 1 lớp
```

**Vì sao quan trọng:** output cuối của model classification (Giai đoạn 2) là một vector logits (điểm số thô), softmax biến nó thành **phân phối xác suất** hợp lệ (tổng = 1, mọi giá trị ≥ 0) trên các lớp. **Cross-entropy loss** sau đó đo "khoảng cách" giữa phân phối dự đoán này và phân phối thật (one-hot label) — hiểu gốc rễ xác suất giúp bạn hiểu *tại sao* cross-entropy là loss chuẩn cho classification chứ không phải chỉ nhớ công thức.

### 5d. Precision/Recall — chưa cần sâu ở tuần này, chỉ cần biết tên

Ở Giai đoạn 5 (Anomaly Detection), bạn sẽ cần hiểu sâu **precision/recall trade-off** khi tune threshold theo chi phí bỏ sót lỗi. Tuần này chỉ cần nắm khái niệm nền: xác suất có điều kiện là gì, để khi gặp công thức Precision = TP/(TP+FP) sẽ không bỡ ngỡ hoàn toàn.

---

## 6. OOP Python cho pipeline xử lý ảnh

### 6a. Class & Object cơ bản

```python
class ImagePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def resize(self, img):
        import cv2
        return cv2.resize(img, self.target_size)

    def normalize(self, img):
        return img.astype(np.float32) / 255.0

preprocessor = ImagePreprocessor(target_size=(256, 256))
processed = preprocessor.normalize(preprocessor.resize(img))
```

**Vì sao dùng class thay vì hàm rời rạc:** class cho phép **giữ trạng thái** (VD: `target_size`) xuyên suốt nhiều lần gọi, và đóng gói các bước liên quan với nhau — nền tảng để code không trở thành mớ hàm rời rạc khó bảo trì khi pipeline phức tạp dần lên qua các giai đoạn.

### 6b. Dunder methods bắt buộc phải hiểu: `__init__`, `__call__`, `__len__`, `__getitem__`

Đây là phần **quan trọng nhất** của mục OOP, vì `torch.utils.data.Dataset` (sẽ dùng liên tục từ Giai đoạn 2 trở đi) **bắt buộc** phải implement đúng 2 dunder method sau:

```python
class DefectDataset:
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        # PyTorch DataLoader gọi len(dataset) để biết có bao nhiêu sample
        return len(self.image_paths)

    def __getitem__(self, idx):
        # PyTorch DataLoader gọi dataset[idx] để lấy 1 sample cụ thể
        import cv2
        img = cv2.imread(self.image_paths[idx])
        label = self.labels[idx]
        return img, label

dataset = DefectDataset(paths, labels)
print(len(dataset))       # gọi __len__ ngầm
img, label = dataset[0]   # gọi __getitem__ ngầm
```

| Dunder method | Được gọi khi nào | Vì sao bắt buộc cho Dataset PyTorch |
|---|---|---|
| `__init__` | Khi tạo object (`ImagePreprocessor(...)`) | Khởi tạo state ban đầu (đường dẫn ảnh, config...) |
| `__call__` | Khi gọi object như hàm (`preprocessor(img)`) | Nhiều pipeline augmentation (torchvision transforms) dùng pattern này để object có thể "gọi được" trực tiếp |
| `__len__` | Khi gọi `len(object)` | `DataLoader` cần biết tổng số sample để chia batch |
| `__getitem__` | Khi gọi `object[idx]` | `DataLoader` dùng để lấy từng sample theo index, hỗ trợ shuffle/batch tự động |

**Ví dụ `__call__` — pattern hay gặp trong augmentation pipeline:**

```python
class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        return (img - self.mean) / self.std

normalize = Normalize(mean=0.5, std=0.5)
result = normalize(img)   # gọi object như 1 hàm — nhờ __call__
```

### 6c. Inheritance — nền tảng đọc hiểu mọi kiến trúc CNN trong PyTorch

```python
import torch.nn as nn

class SimpleModel(nn.Module):          # kế thừa từ nn.Module — bắt buộc với mọi model PyTorch
    def __init__(self, num_classes):
        super().__init__()              # gọi __init__ của lớp cha (nn.Module) — KHÔNG được bỏ qua
        self.conv = nn.Conv2d(3, 16, kernel_size=3)
        self.fc = nn.Linear(16 * 222 * 222, num_classes)

    def forward(self, x):               # override method forward — nn.Module gọi ngầm khi model(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

**Điểm mấu chốt cần hiểu:**
- Mọi model trong PyTorch (kể cả ResNet, YOLO backbone...) đều kế thừa `nn.Module` theo đúng pattern trên.
- `super().__init__()` là bước bắt buộc — quên dòng này là lỗi runtime rất khó hiểu với người mới.
- Bạn override method `forward()` để định nghĩa "dữ liệu đi qua model như thế nào" — khi gọi `model(x)`, PyTorch tự động gọi `forward(x)` ở phía sau (thông qua `__call__` được định nghĩa sẵn trong `nn.Module`).

### 6d. Composition — tư duy thiết kế pipeline (dùng xuyên suốt đến tận Capstone)

Thay vì 1 class khổng lồ làm mọi việc, tách thành các object nhỏ độc lập, ghép lại với nhau:

```python
class ImagePipeline:
    def __init__(self, preprocessor, model, postprocessor):
        self.preprocessor = preprocessor
        self.model = model
        self.postprocessor = postprocessor

    def run(self, img):
        img = self.preprocessor(img)
        raw_output = self.model(img)
        result = self.postprocessor(raw_output)
        return result
```

**Vì sao composition tốt hơn 1 class làm hết mọi việc:**
- Dễ test riêng từng bước (VD: test preprocessor độc lập mà không cần chạy cả model).
- Dễ thay thế 1 thành phần (VD: đổi model detection sang model khác) mà không đụng vào phần còn lại.
- Đây chính xác là kiến trúc bạn sẽ xây ở **Capstone (Giai đoạn 11)**: Camera → Preprocessor → Model → Postprocessor → Tracker → Dashboard, mỗi khối là 1 object độc lập.

---

## 7. Checklist lỗi thường gặp (self-debug trước khi hỏi ai)

- [ ] `ValueError: operands could not be broadcast together` → kiểm tra lại shape 2 mảng đang cộng/nhân, đặc biệt kiểm tra chiều channel đang ở đầu hay cuối
- [ ] Dùng `reshape` nhưng dữ liệu ra "sai trật tự" dù không lỗi → đáng lẽ phải dùng `transpose`
- [ ] Nhầm `*` (element-wise) với `@` (matrix multiplication) → kết quả sai âm thầm, không báo lỗi nếu shape tình cờ khớp
- [ ] Ảnh sau khi normalize toàn giá trị 0 → quên `.astype(np.float32)` trước khi chia (chia số nguyên `uint8` cho 255 có thể làm tròn về 0)
- [ ] `TypeError` khi PyTorch `DataLoader` chạy → quên implement đúng `__len__`/`__getitem__` trong class `Dataset`
- [ ] Quên `super().__init__()` trong model kế thừa `nn.Module` → lỗi runtime khó hiểu khi gọi `model(x)`
- [ ] Loss là `NaN` ngay từ đầu train → thường do learning rate quá lớn (gradient descent "nhảy" ra ngoài vùng hội tụ) hoặc chưa chuẩn hoá input (mục 5b)

---

## 8. Bài tập thực hành đề xuất

Viết 1 class `ImagePipeline` áp dụng toàn bộ khái niệm trên, dùng ảnh thật:

1. `__init__`: nhận đường dẫn ảnh, target size
2. `load()`: đọc ảnh bằng OpenCV → in `shape`, `dtype`, `min`, `max` để luyện thói quen debug
3. `resize()`: resize về target size
4. `normalize()`: chuẩn hoá về [0,1], sau đó trừ mean/chia std (dùng broadcasting — mục 1d)
5. `to_chw()`: dùng `transpose` chuyển từ `(H,W,C)` sang `(C,H,W)`
6. Bọc toàn bộ pipeline trong `__call__` để có thể gọi `pipeline(img)` trực tiếp
7. Viết thêm class `SimpleDataset` với `__len__`/`__getitem__` nhận vào danh sách đường dẫn ảnh, gọi `ImagePipeline` bên trong `__getitem__`

→ Đây chính là bộ khung tối giản của `Dataset` class bạn sẽ mở rộng liên tục từ Giai đoạn 2 trở đi.

---

## 9. Câu hỏi tự kiểm tra (chuẩn bị sớm cho phỏng vấn)

- Vì sao dùng vectorization (NumPy) nhanh hơn loop Python thuần?
- Broadcasting hoạt động theo quy tắc nào? Cho ví dụ 1 trường hợp broadcast lỗi.
- `reshape` và `transpose` khác nhau ở điểm nào? Khi nào dùng nhầm sẽ gây lỗi âm thầm?
- Eigenvector/eigenvalue có ý nghĩa hình học gì? Liên hệ với PCA như thế nào?
- Gradient descent hoạt động ra sao? Learning rate ảnh hưởng thế nào nếu quá lớn/quá nhỏ?
- Chain rule liên quan gì đến backpropagation?
- Vì sao cần chuẩn hoá (normalize) ảnh trước khi đưa vào model, xét theo góc độ mean/variance?
- `__len__` và `__getitem__` được `DataLoader` của PyTorch dùng như thế nào?
- Vì sao mọi model PyTorch phải kế thừa `nn.Module` và gọi `super().__init__()`?