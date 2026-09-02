# Cheatsheet: Gradient & Đạo hàm trong AI/ML

> Phần này là **nền tảng bắt buộc** để hiểu cách mạng neural học. Không cần thành thạo giải tích — chỉ cần trực giác đủ dùng để không thấy `loss.backward()` là "hộp đen".

---

## Phần 1 — Đạo hàm là gì?

### Trực giác: "Độ dốc tại một điểm"

Hãy tưởng tượng bạn đứng trên một con dốc và nhìn xuống:
- **Đạo hàm** = độ nghiêng của mặt đất ngay dưới chân bạn tại điểm đó.
- Dốc đứng → đạo hàm lớn.
- Mặt phẳng → đạo hàm = 0.
- Dốc xuống → đạo hàm âm. Dốc lên → đạo hàm dương.

```
Đồ thị f(x) = x²

  f(x)
   |         *         *
   |       *             *
   |     *                 *
   |   *        ← đỉnh (f'=0)
   |  *                     *
   +----+----+----+----+----→ x
       -2   -1    0    1    2

  Tại x=2:  f'(2) = 4  → dốc lên (dương)
  Tại x=0:  f'(0) = 0  → đáy, không dốc
  Tại x=-2: f'(-2) = -4 → dốc xuống (âm)
```

### Quy tắc nhanh (không cần tính giới hạn)

| Đạo hàm | Ý nghĩa | Trong AI/ML có nghĩa là... |
|---|---|---|
| **> 0** (dương) | Hàm đang tăng | Nếu tăng weight này, loss sẽ tăng |
| **< 0** (âm) | Hàm đang giảm | Nếu tăng weight này, loss sẽ giảm |
| **= 0** | Đỉnh hoặc đáy | Có thể đã tìm được minima (loss thấp nhất) |
| Giá trị **lớn** | Dốc đứng | Cần thay đổi ít weight thôi, không thì "nhảy qua" đáy |

### Ứng dụng thực tế — Bộ lọc Sobel (Giai đoạn 1)

#### Vấn đề: Làm sao phát hiện biên (cạnh) của vật thể trong ảnh?

Biên vật thể = nơi màu sắc / độ sáng **thay đổi đột ngột**. Ví dụ: từ nền trắng sang vật thể đen, pixel chuyển từ 255 → 0 rất nhanh trong vài pixel liên tiếp.

Trong toán học, "thay đổi nhanh" chính xác là **đạo hàm lớn**. Vậy: **Tìm biên = Tính đạo hàm của cường độ pixel.**

---

#### Đạo hàm liên tục vs. Đạo hàm rời rạc

- **Liên tục** (toán học): `f'(x) = lim(h→0) [f(x+h) - f(x)] / h`
- **Rời rạc** (ảnh số — pixel): Không thể lấy h → 0 vì pixel là số nguyên! Thay vào đó dùng xấp xỉ đơn giản nhất:

```
f'(x) ≈ f(x+1) - f(x-1)
         ──────────────
                2
```
Tức là: lấy pixel bên phải trừ pixel bên trái, chia 2 → xấp xỉ đạo hàm tại vị trí đó.

---

#### Sobel Kernel là gì?

Sobel thực hiện phép tính trên theo **ma trận 3×3** trượt qua toàn bộ ảnh:

```
Sobel_X (đạo hàm theo chiều ngang):
┌─────────────────────┐
│  -1   0   +1        │
│  -2   0   +2        │ ← Trọng số lớn hơn ở hàng giữa
│  -1   0   +1        │   để lấy thêm thông tin từ hàng trên/dưới
└─────────────────────┘

Sobel_Y (đạo hàm theo chiều dọc):
┌─────────────────────┐
│  -1  -2   -1        │
│   0   0    0        │
│  +1  +2   +1        │
└─────────────────────┘
```

Khi kernel trượt qua 1 ô ảnh 3×3, nó **nhân từng phần tử rồi cộng lại** (dot product):
- Cột trái có dấu **âm** (trừ) → "lấy pixel bên trái"
- Cột phải có dấu **dương** (cộng) → "lấy pixel bên phải"
- Kết quả = pixel_phải − pixel_trái → đúng là đạo hàm!

---

#### Ví dụ số cụ thể

Giả sử một hàng pixel trong ảnh grayscale:

```
Vị trí:     x=0   x=1   x=2   x=3   x=4   x=5   x=6   x=7
Pixel:      [200,  200,  200,  200,    5,    5,    5,    5]
                                     ↑
                               Đây là biên! (200→5)
```

Tính đạo hàm rời rạc thủ công tại từng vị trí:

```
Tại x=1: (200 - 200) / 2 = 0       ← nền đồng đều, không có biên
Tại x=2: (200 - 200) / 2 = 0       ← nền đồng đều, không có biên
Tại x=3: (5   - 200) / 2 = -97.5   ← BIÊN! đạo hàm âm lớn = cường độ giảm đột ngột
Tại x=4: (5   - 200) / 2 = -97.5   ← vẫn trong vùng biên
Tại x=5: (5   -   5) / 2 = 0       ← nền mới đồng đều, không có biên
```

Kết quả sau khi lấy **giá trị tuyệt đối**: biên lộ rõ tại x=3 và x=4 với giá trị ≈97.5.

---

#### Code đầy đủ + giải thích từng dòng

```python
import cv2
import numpy as np

img = cv2.imread('anh.jpg', cv2.IMREAD_GRAYSCALE)
# Lưu ý: ảnh phải là grayscale (1 channel), không phải BGR 3 channel

# ── BƯỚC 1: Tính đạo hàm theo chiều NGANG (phát hiện cạnh dọc) ──
sobel_x = cv2.Sobel(
    img,         # ảnh đầu vào
    cv2.CV_64F,  # kiểu dữ liệu output: float64 (quan trọng! uint8 không lưu được số âm)
    1,           # dx=1: lấy đạo hàm bậc 1 theo x
    0,           # dy=0: không lấy theo y
    ksize=3      # kích thước kernel 3×3
)

# ── BƯỚC 2: Tính đạo hàm theo chiều DỌC (phát hiện cạnh ngang) ──
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# ── BƯỚC 3: Kết hợp cả hai hướng ──
# Gradient tổng hợp = sqrt(sobel_x² + sobel_y²)  ← chính là L2 norm của vector gradient!
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

# ── BƯỚC 4: Chuẩn hoá về [0,255] để hiển thị ──
magnitude = np.uint8(np.clip(magnitude, 0, 255))

# Nơi magnitude cao = cường độ thay đổi đột ngột = BIÊN vật thể
cv2.imshow('Bien vat the', magnitude)
cv2.waitKey(0)
```

#### Kết nối với Gradient Vector (Phần 2)

Nhận thấy công thức kết hợp `sqrt(sobel_x² + sobel_y²)` chính là **độ lớn (magnitude) của vector gradient 2D**:

```
∇I = [∂I/∂x, ∂I/∂y]   ← gradient của ảnh I tại 1 pixel
                           (∂I/∂x = sobel_x, ∂I/∂y = sobel_y)

|∇I| = sqrt((∂I/∂x)² + (∂I/∂y)²)  ← độ lớn gradient = độ "sắc nét" của biên
```

> Đây là lý do edge detection và gradient trong deep learning dùng **cùng một ngôn ngữ toán học** — chúng đều đang hỏi: "Hàm số đang thay đổi nhanh như thế nào và theo hướng nào?"

---

## Phần 2 — Đạo hàm riêng & Gradient Vector

### Vấn đề: Mạng neural có hàng triệu "núm vặn" (weights)

Hàm mất mát `L` không phụ thuộc vào 1 biến mà phụ thuộc vào **hàng triệu weight**:

```
L(w1, w2, w3, ..., w_n)
```

Muốn biết "vặn núm `w1` tăng lên một chút thì loss tăng hay giảm?", ta tính **đạo hàm riêng** theo `w1` — giả vờ tất cả các weight khác đứng im, chỉ thay đổi `w1`.

```python
# Ví dụ: loss = (w1 * x + w2 - y)²
# ∂L/∂w1 = 2 * (w1*x + w2 - y) * x   ← đạo hàm riêng theo w1
# ∂L/∂w2 = 2 * (w1*x + w2 - y)       ← đạo hàm riêng theo w2
```

### Gradient = "Bản đồ đầy đủ" của mọi đạo hàm riêng

Gradient `∇L` là một **vector** gom tất cả các đạo hàm riêng lại:

```
∇L = [∂L/∂w1,  ∂L/∂w2,  ∂L/∂w3,  ...,  ∂L/∂wn]
      ↑           ↑           ↑                ↑
   Vặn w1     Vặn w2      Vặn w3          Vặn wn
   thì loss   thì loss    thì loss         thì loss
   thay đổi   thay đổi    thay đổi         thay đổi
   bao nhiêu? bao nhiêu?  bao nhiêu?       bao nhiêu?
```

```python
import numpy as np

# Tính gradient đơn giản (1 layer, 2 weights)
# loss = (w1*x + w2 - y)^2
x, y = 2.0, 5.0   # 1 data point
w1, w2 = 1.0, 0.5  # khởi tạo weight ngẫu nhiên

pred = w1 * x + w2        # = 2.5
loss = (pred - y) ** 2    # = (2.5 - 5)^2 = 6.25

# Gradient (đạo hàm riêng)
grad_w1 = 2 * (pred - y) * x   # = 2 * (-2.5) * 2 = -10
grad_w2 = 2 * (pred - y)       # = 2 * (-2.5)     = -5

gradient = np.array([grad_w1, grad_w2])  # [-10, -5]
# Gradient âm → tăng w1, w2 sẽ GIẢM loss → đúng hướng cần đi!
```

---

## Phần 3 — Tại sao Gradient chỉ hướng tăng NHANH NHẤT?

> Đây là câu hỏi quan trọng nhất. Mình sẽ giải thích bằng ví dụ cụ thể, không dùng toán phức tạp.

### Ví dụ: Bạn đứng trên sườn núi

Giả sử loss function có dạng 2D (phụ thuộc 2 weight `w1`, `w2`):

```
       w2
        ^
        |         (cao, loss lớn)
        |    loss = 9  loss = 4  loss = 1
        |      ○---------○---------●  ← vị trí bạn đứng
        |                          |
        |    loss = 4  loss = 2    |
        |                          ↓ gradient
        |    loss = 1              ↓ (hướng dốc xuống nhanh nhất)
        +-----------------------------→ w1
               (thấp, loss nhỏ)
```

Giả sử bạn đứng tại `(w1=3, w2=2)` và gradient tính được là `[6, 4]`.

**Câu hỏi:** Nếu bước đi 1 bước nhỏ theo bất kỳ hướng nào, hướng nào làm loss **tăng nhiều nhất**?

### Thử nhiều hướng khác nhau:

```python
import numpy as np

# Gradient tại điểm hiện tại
gradient = np.array([6.0, 4.0])

# Thử các hướng khác nhau (mỗi hướng dài 1 đơn vị)
huong_theo_gradient     = np.array([0.832, 0.555])  # ĐỒNG HƯỚNG với gradient
huong_vuong_goc         = np.array([-0.555, 0.832]) # VUÔNG GÓC với gradient
huong_bat_ky            = np.array([0.6, 0.8])       # hướng bất kỳ

# Độ thay đổi loss khi đi theo mỗi hướng = dot product với gradient
delta_loss_theo_gradient   = np.dot(gradient, huong_theo_gradient)  # = 7.21
delta_loss_vuong_goc       = np.dot(gradient, huong_vuong_goc)      # ≈ 0
delta_loss_bat_ky          = np.dot(gradient, huong_bat_ky)         # = 6.8

print(f"Đi theo gradient:  loss tăng {delta_loss_theo_gradient:.2f}")  # 7.21 ← LỚN NHẤT
print(f"Đi vuông góc:      loss tăng {delta_loss_vuong_goc:.2f}")      # 0.0  ← không đổi
print(f"Đi hướng bất kỳ:  loss tăng {delta_loss_bat_ky:.2f}")          # 6.8  ← nhỏ hơn
```

**Kết luận:** Hướng đồng hướng với gradient luôn cho delta loss lớn nhất. **Đó chính là lý do gradient = hướng tăng nhanh nhất.**

### Giải thích trực giác (không cần toán):

Gradient `[6, 4]` có nghĩa là:
- Vặn `w1` tăng 1 đơn vị → loss tăng **6** đơn vị
- Vặn `w2` tăng 1 đơn vị → loss tăng **4** đơn vị

Hướng gradient `[6, 4]` chính là hướng "**tận dụng tối đa cả hai đòn bẩy cùng lúc theo đúng tỉ lệ của chúng**". Bất kỳ hướng nào khác đều bỏ phí ít nhất một phần sức mạnh của các đòn bẩy này.

> **Kết luận cốt lõi:**
> - Gradient **chỉ hướng TĂNG nhanh nhất** → để **GIẢM** loss, ta đi **NGƯỢC LẠI** hướng gradient.
> - Đây là toàn bộ bí quyết của Gradient Descent.

---

## Phần 4 — Gradient Descent: "Quả bóng lăn xuống dốc"

### Công thức cập nhật weight

```
w_mới = w_cũ - learning_rate × gradient
          ↑                         ↑
     Dấu trừ              Gradient chỉ hướng TĂNG
     = đi NGƯỢC lại       → trừ đi = đi XUỐNG dốc
```

### Code minh họa từng bước

```python
# Tìm minima của f(x) = x² — đạo hàm f'(x) = 2x
x = 10.0     # khởi tạo: đứng tại điểm x=10 (xa đáy)
lr = 0.1     # learning rate: bước mỗi lần đi

print(f"Ban đầu: x={x:.2f}, f(x)={x**2:.2f}")
for step in range(20):
    grad = 2 * x           # 1. Tính gradient tại vị trí hiện tại
    x = x - lr * grad      # 2. Bước NGƯỢC hướng gradient
    print(f"Bước {step+1:02d}: x={x:.4f}, f(x)={x**2:.6f}")

# Kết quả: x tiến dần về 0 (đáy của hàm x²)
# Bước 01: x=8.0000,  f(x)=64.0
# Bước 05: x=3.2768,  f(x)=10.74
# Bước 10: x=1.0737,  f(x)=1.153
# Bước 20: x=0.1153,  f(x)=0.013  ← gần đáy rồi!
```

### Learning Rate — "Kích thước bước chân"

```
                   ĐÁNH MẤT ĐÁY (lr quá lớn)
f(x)  *           ↗  ↙  ↗  ↙  ↗ (dao động không hội tụ)
      *     *
           *  *  
              * (đáy)

                   ĐI RẤT CHẬM (lr quá nhỏ)
f(x)  * * * * * * * * * → (mãi không đến đáy)

                   VỪA ĐẸP
f(x)  *
         *
               *
                     * (đến đáy trong vài bước)
```

| Learning Rate | Hệ quả | Biểu hiện khi train |
|---|---|---|
| **Quá lớn** (VD: 1.0) | Bước quá dài, bắn qua đáy | Loss dao động hoặc tăng dần |
| **Quá nhỏ** (VD: 1e-6) | Mỗi bước nhỏ xíu | Loss giảm rất chậm, training mất nhiều giờ |
| **Vừa** (VD: 1e-3) | Hội tụ ổn định | Loss giảm đều, mô hình học được |

---

## Phần 5 — Chain Rule & Backpropagation

### Vấn đề: Làm sao tính gradient cho lớp đầu trong mạng 100 lớp?

```
Input → Layer 1 → Layer 2 → ... → Layer 100 → Loss
                                               ↑
                        Phải tính đạo hàm của Loss
                        theo weight ở tận Layer 1!
```

### Chain Rule: "Nhân liên tiếp"

Nếu bạn có chuỗi hàm lồng nhau:
```
z = g(h(x))  →  dz/dx = dz/dh × dh/dx
```

**Ví dụ thực tế:** Giá vé máy bay phụ thuộc vào giá dầu, giá dầu phụ thuộc vào tỉ giá USD.
```
"Tỉ giá USD tăng 1% → dầu tăng bao nhiêu?" × "Dầu tăng 1% → vé tăng bao nhiêu?"
= "Tỉ giá USD tăng 1% → vé tăng bao nhiêu?"
```

### Áp dụng cho mạng neural

```
∂Loss/∂w1 = (∂Loss/∂Layer100) × (∂Layer100/∂Layer99) × ... × (∂Layer1/∂w1)
```

Mỗi layer chỉ cần tính đạo hàm cục bộ của chính nó — nhân dồn lại theo chain rule.

### Backpropagation trong PyTorch

```python
import torch
import torch.nn as nn

# Định nghĩa model đơn giản
model = nn.Linear(3, 1)   # 1 linear layer
criterion = nn.MSELoss()

x = torch.randn(4, 3)     # 4 samples, 3 features
y = torch.randn(4, 1)     # 4 labels

# --- FORWARD PASS ---
pred = model(x)            # tính prediction
loss = criterion(pred, y)  # tính loss

# --- BACKWARD PASS (Backpropagation) ---
loss.backward()  # PyTorch tự động chạy chain rule, tính gradient cho MỌI weight

# Xem gradient đã được tính
print(model.weight.grad)   # gradient của loss theo weight của linear layer
print(model.bias.grad)     # gradient của loss theo bias

# --- CẬP NHẬT WEIGHT ---
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer.step()           # cập nhật: weight -= lr × gradient
optimizer.zero_grad()      # QUAN TRỌNG: reset gradient về 0 cho lần tính tiếp
```

> ⚠️ **Lỗi hay gặp:** Quên gọi `optimizer.zero_grad()` trước mỗi batch → gradient bị cộng dồn từ batch trước → kết quả training sai.

---

## Phần 6 — Các vấn đề thực tế với Gradient

### Vanishing Gradient (Gradient biến mất)

```
Mạng 50 lớp:
∂Loss/∂w1 = 0.9 × 0.9 × 0.9 × ... × 0.9  (50 lần)
           = 0.9^50 ≈ 0.005 → quá nhỏ, layer đầu không học được
```

**Biểu hiện:** Loss ở output giảm, nhưng các layer đầu hầu như không thay đổi.
**Giải pháp:** Dùng activation function ReLU (thay vì Sigmoid/Tanh), dùng Batch Normalization, dùng Residual Connection (ResNet).

### Exploding Gradient (Gradient bùng nổ)

```
∂Loss/∂w1 = 2.0 × 2.0 × ... × 2.0  (50 lần)
           = 2^50 ≈ 10^15 → khổng lồ, weight update không kiểm soát được
```

**Biểu hiện:** Loss ra `NaN` ngay từ những batch đầu.
**Giải pháp:** Gradient Clipping, giảm learning rate, chuẩn hoá input.

### Gradient Clipping (hay dùng với RNN/LSTM)

```python
# Sau loss.backward(), trước optimizer.step()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Nếu gradient quá lớn, scale xuống để norm = 1.0
```

---

## Phần 7 — Tổng kết nhanh

| Khái niệm | Định nghĩa | Trong PyTorch |
|---|---|---|
| **Đạo hàm** `f'(x)` | Độ dốc tại một điểm | Tự tính hoặc `torch.autograd` |
| **Đạo hàm riêng** `∂L/∂w` | Đạo hàm theo 1 weight, các weight khác cố định | Tự động sau `loss.backward()` |
| **Gradient** `∇L` | Vector gom tất cả đạo hàm riêng | `param.grad` sau `loss.backward()` |
| **Gradient chỉ hướng tăng nhanh nhất** | Vì đồng hướng gradient cho dot product lớn nhất | — |
| **Gradient Descent** | `w = w - lr × ∇L` | `optimizer.step()` |
| **Learning Rate** | Kích thước bước mỗi lần cập nhật | `lr=0.001` trong SGD/Adam |
| **Chain Rule** | Đạo hàm hàm lồng nhau = nhân các đạo hàm riêng | Tự động trong `backward()` |
| **Backpropagation** | Chain rule áp dụng cho mạng neural | `loss.backward()` |

---

## Phần 8 — Checklist lỗi thường gặp

- [ ] Loss ra `NaN` ngay từ đầu → learning rate quá lớn hoặc chưa normalize input
- [ ] Loss không giảm → learning rate quá nhỏ, hoặc quên `optimizer.step()`
- [ ] Loss giảm rồi lại tăng lên → learning rate hơi lớn, hoặc cần dùng lr scheduling
- [ ] Gradient luôn bằng 0 (`param.grad = None`) → phép toán không hỗ trợ autograd (VD: dùng `numpy` thay vì `torch`)
- [ ] Quên `optimizer.zero_grad()` → gradient cộng dồn từ batch trước → training sai

---

## Phần 9 — Tài liệu tham khảo

- **3Blue1Brown:** "Gradient descent, how neural networks learn" (YouTube — trực quan nhất)
- **3Blue1Brown:** "What is backpropagation really doing?" (YouTube)
- **CS231n Stanford:** Lecture 3 — Gradient & Backpropagation (slides miễn phí)
- **PyTorch docs:** `torch.autograd` — cách autograd tính gradient
