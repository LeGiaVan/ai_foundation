# Cheatsheet — Giai đoạn 2: Deep Learning & Các Bài toán Thị giác Cốt lõi (Tuần 4-11)

> Domain xuyên suốt: đây là giai đoạn chuyển từ "tiền xử lý cổ điển" (Giai đoạn 1) sang deep learning — model sẽ **tự học** đặc trưng từ dữ liệu thay vì người thiết kế kernel bằng tay. Toàn bộ nội dung 3 phần (Classification → Detection → Segmentation) đều được gắn vào mục tiêu cuối: **hệ thống QC sản xuất** bạn sẽ xây ở Capstone (Giai đoạn 11). Mỗi bài toán là một "bậc thang": classification cho biết *có vấn đề hay không*, detection cho biết *vấn đề ở đâu*, segmentation cho biết *vấn đề chiếm vùng nào chính xác*.

---

# PHẦN A — DEEP LEARNING CHO VISION (Tuần 4-6)

> Mục tiêu sau 3 tuần: build được một model classification chuẩn hoá (train từ đầu CNN nhỏ + fine-tune pretrained model), hiểu được vòng đời huấn luyện (forward/backward, hyperparameter, overfitting), và biết cách đánh giá chính xác. Đây là nền móng bắt buộc trước khi sang detection/segmentation.

---

## A1. CNN (Convolutional Neural Network) — vì sao lại là convolution

### A1a. Từ ý tưởng (Giai đoạn 1) lên kiến trúc học được

Ở Giai đoạn 1 bạn đã thấy convolution với kernel **do con người thiết kế** (Sobel, Gaussian, sharpen). Trong CNN, **kernel là tham số học được** — máy tự tìm ra bộ lọc nào tốt nhất cho bài toán từ dữ liệu.

```
Ảnh đầu vào (H, W, C)  →  [Conv 3x3]  →  Feature map  →  [Pooling]  →  ...  →  FC layer  →  logits →
```

**Ba ý tưởng quyết định của CNN (khác với MLP thuần):**

| Ý tưởng | Giải thích | Vì sao quan trọng với ảnh |
|---|---|---|
| **Local connectivity** | Mỗi neuron chỉ nhìn một vùng nhỏ của ảnh (receptive field), không nhìn toàn ảnh | Ảnh có tính cục bộ: 1 pixel cạnh tranh, 1 đặc trưng lặp lại ở mọi vị trí |
| **Weight sharing** | Một kernel được dùng **đi lại** trên toàn bộ ảnh (shared weights) | Giảm mạnh số tham số — model nhẹ, ít overfit; đồng thời bất biến dịch chuyển (translation invariance) |
| **Hierarchical features** | Layer đầu học cạnh/góc → layer giữa học cấu trúc (mắt, bánh xe) → layer cuối học đối tượng hoàn chỉnh | Giống cách vỏ não thị giác người xử lý từ chi tiết lên tổng thể |

### A1b. Các thuật ngữ cốt lõi — kèm công thức

| Thuật ngữ | Khái niệm | Công thức / Cách hiểu |
|---|---|---|
| **Kernel (filter)** | Ma trận trọng số nhỏ (3x3, 5x5...) trượt trên ảnh | Kích thước thường là số lẻ để có tâm đối xứng |
| **Stride** | Bước nhảy của kernel mỗi lần trượt | `Stride = 1` giữ nguyên resolution (nếu padding phù hợp); `Stride = 2` giảm ảnh đi 1 nửa (dùng thay pooling) |
| **Padding** | Thêm viền 0 quanh ảnh | `P = (K - 1) / 2` để output = input (same padding) |
| **Feature map** | Ảnh output sau khi áp 1 kernel | Mỗi kernel sinh 1 feature map → `num_filters` kernel sinh `num_filters` channel |
| **Convolution output size** | Kích thước feature map | `O = (W - K + 2P) / S + 1` (cho 1 chiều) |

```python
import torch
import torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
x = torch.randn(1, 3, 224, 224)      # (batch, channel, H, W)
out = conv(x)
print(out.shape)                      # torch.Size([1, 16, 224, 224]) — 16 feature map, same size
```

**Ví dụ tính output size:**

```python
# Input 32x32, kernel 3x3, stride 1, padding 0  ->  O = (32 - 3 + 0)/1 + 1 = 30
# Input 32x32, kernel 5x5, stride 2, padding 0  ->  O = (32 - 5 + 0)/2 + 1 = 14.5 -> 14 (floor)
def out_size(W, K, P, S):
    return (W - K + 2 * P) // S + 1
print(out_size(32, 3, 0, 1))   # 30
print(out_size(32, 5, 0, 2))   # 14
```

### A1c. Pooling — đặc trưng chính + giảm chiều

**Max pooling** = lấy giá trị lớn nhất trong từng ô (VD 2x2, stride 2). **Average pooling** = lấy trung bình.

```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)
print(pool(torch.randn(1, 16, 224, 224)).shape)   # torch.Size([1, 16, 112, 112]) — giảm 1 nửa
```

**Vì sao pooling hữu ích:**
1. Giảm kích thước feature map → giảm số tham số, tốc độ, tránh overfit.
2. Giữ lại "feature mạnh nhất" trong vùng → bất biến vị trí tương đối nhỏ (1 vết lỗi dịch vài px vẫn được pooling giữ lại).
3. Mở rộng receptive field từng bước: layer sau "nhìn" một vùng thực tế trên ảnh rộng hơn.

⚠️ **Xu hướng hiện đại:** nhiều kiến trúc (ResNet, EfficientNet) thay pooling bằng **convolution stride=2** — ít mất thông tin hơn, nhưng khái niệm pooling vẫn cần hiểu vì nó xuất hiện trong Global Average Pooling (GAP) ở cuối mọi mạng classification.

### A1d. Activation — đưa phi tuyến vào

Không có activation phi tuyến, chồng nhiều conv layer chỉ tương đương 1 layer tuyến tính. Ba activation chính:

| Activation | Công thức | Đặc điểm |
|---|---|---|
| **ReLU** | `max(0, x)` | Mặc định cho mọi CNN hiện đại — nhanh, tránh vanishing gradient |
| **Sigmoid** | `1 / (1 + e^-x)` | Dùng ở output nhị phân (xác suất 0-1), giảm gradient ở 2 đầu |
| **Softmax** | `e^z / Σ e^z` | Dùng ở output **multi-class** — biến logits thành xác suất tổng = 1 |

```python
import torch.nn.functional as F
x = torch.tensor([2.0, 1.0, 0.1])
print(F.softmax(x, dim=0))   # xác suất mỗi lớp, tổng = 1
print(F.relu(torch.tensor([-1.0, 0.0, 2.0])))   # [0., 0., 2.]
```

**Minh hoạ ReLU phi tuyến trong CNN:** một layer `Conv→ReLU→Conv` biểu diễn được các ranh giới quyết định cong/phức tạp (đặc biệt hữu ích để phân tách vết lỗi có hình dạng lồi lõm trên nền có kết cấu), còn `Conv→Conv` chỉ là một phép tuyến tính.

---

## A2. Backpropagation & Cách PyTorch tự học

### A2a. Nhắc ngắn từ Giai đoạn 0

Backprop = áp dụng **chain rule** để lan truyền gradient của loss ngược về từng weight. Giai đoạn 0 bạn chỉ cần hiểu khái niệm; bây giờ cần hiểu **vòng lặp huấn luyện** đầy đủ:

```
1. Forward pass   : input qua model -> output -> tính loss
2. Zero grad      : xoá gradient cũ (tránh cộng dồn)
3. Backward pass  : loss.backward() -> tính gradient cho mọi tham số
4. Optimizer step : W = W - lr * grad (gradient descent, có biến thể)
```

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(16 * 112 * 112, 10),
)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

x = torch.randn(4, 3, 224, 224)        # batch 4 ảnh
y = torch.randint(0, 10, (4,))         # nhãn 4 ảnh

# Vòng lặp 1 batch:
optimizer.zero_grad()                   # 2. xoá grad cũ
logits = model(x)                       # 1. forward
loss = loss_fn(logits, y)               # 1. tính loss
loss.backward()                         # 3. backward — tự động tính gradient
optimizer.step()                        # 4. cập nhật weight
```

**Điều mấu chốt:** bạn **không cần** tự code backprop tay — PyTorch dùng **Autograd** (giữ computational graph, tự động lan truyền gradient khi gọi `loss.backward()`). Việc của bạn là hiểu *tại sao* 4 bước trên lặp lại, và *tại sao* phải gọi `zero_grad()` trước mỗi batch (nếu không, gradient cộng dồn giữa các batch → sai).

⚠️ **Bẫy rất phổ biến:** quên `optimizer.zero_grad()` → gradient cộng dồn → loss "nhảy loạn". Đây là lỗi #1 khi người mới copy code nhầm thứ tự.

### A2b. Loss function cho classification

**Cross-entropy** (đã học Giai đoạn 0 về phần xác suất) là loss chuẩn cho multi-class:

```python
loss_fn = nn.CrossEntropyLoss()
# Chú ý: trong PyTorch, CrossEntropyLoss tự combine LogSoftmax + NLLLoss
# -> input là LOGITS (chưa qua softmax), KHÔNG phải xác suất đã softmax
loss = loss_fn(logits, y)
```

**Nếu bạn tự softmax trước rồi đưa vào CrossEntropyLoss → kết quả sai trầm trọng** (softmax 2 lần). Luôn đưa **raw logits** vào `nn.CrossEntropyLoss()`.

| Loại bài toán | Loss nên dùng | Ghi chú |
|---|---|---|
| Multi-class (1 ảnh → 1 trong N lớp) | `CrossEntropyLoss` | Lớp chuẩn cho classification |
| Multi-label (1 ảnh → nhiều lớp) | `BCEWithLogitsLoss` | Nhãn dạng `[1,0,1,0...]`, dùng sigmoid |
| Lớp không cân bằng (ít ảnh lỗi) | CrossEntropy có `weight` | `weight=[w0,w1,...]` lớn cho lớp ít mẫu → phạt nặng hơn khi sai lớp hiếm |

---

## A3. Data Augmentation — nhân bản dữ liệu ảo để chống overfit

> Trong QC thực tế, bạn gần như **không bao giờ có đủ ảnh lỗi** (lỗi hiếm). Augmentation là "chiêu cốt lõi" để tận dụng tối đa vài trăm ảnh lỗi hiếm hoi có được.

### A3a. Các phép augmentation cơ bản (torchvision.transforms)

| Phép | Tham số | Dùng khi nào trong QC |
|---|---|---|
| `RandomHorizontalFlip(p=0.5)` | xác suất lật | Sản phẩm đối xứng (2D phẳng) — tăng gấp đôi số mẫu |
| `RandomRotation(degrees=(0, 10))` | góc xoay nhỏ | Bù độ lệch góc đặt sản phẩm trên băng chuyền |
| `ColorJitter(brightness, contrast, saturation, hue)` | cường độ mỗi kênh | Mô phỏng ánh sáng nhà máy thay đổi giữa các ca |
| `RandomAffine(translate=..., scale=...)` | dịch, co giãn | Vị trí/kích thước sản phẩm không cố định |
| `RandomResizedCrop(...)` | cắt và scale ngẫu nhiên | Tăng robustness với nhiều tỷ lệ gần/ xa |

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),          # lật ngang
    transforms.RandomRotation(degrees=10),           # xoay nhỏ
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# QUAN TRỌNG: transform cho tập VALIDATION/TEST KHÔNG được augmentation ngẫu nhiên —
# chỉ resize + normalize, để đánh giá đúng trên dữ liệu thật
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### A3b. Mixup — kỹ thuật nâng cao

**Mixup:** trộn 2 ảnh + 2 nhãn lại theo hệ số `λ` (sample từ Beta distribution). Buộc model ra "xác suất trung gian", cải thiện khả năng tổng quát hoá và làm mượt ranh giới quyết định.

```python
import numpy as np
import torch

def mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    idx = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[idx]
    mixed_y = (lam * y + (1 - lam) * y[idx])   # nhãn cũng trộn
    return mixed_x, mixed_y
```

⚠️ **Mixup cần nhãn không one-hot cứng để trộn** — với `CrossEntropyLoss` có thể dùng nhãn mềm/quy đổi. Nhiều pipeline dùng mixup kèm **label smoothing** thay vì nhãn one-hot cứng.

### A3c. Minh hoạ trực quan augmentation

<div align="center">
   <img src="augmentation_grid.jpg" alt="Augmentation grid" style="max-width:680px; width:100%; height:auto;" />
   <div style="font-size:0.95rem; color:#666; margin-top:6px;">Hình minh hoạ (tự sinh khi chạy bài tập): 1 ảnh gốc → nhiều biến thể qua flip/rotate/color jitter/crop. Nguyên tắc: augmentation phải **giữ nguyên ý nghĩa nhãn** — đừng lật ảnh sản phẩm có chữ/logo định hướng, đừng xoay ảnh chữ in vì sẽ đọc ngược.</div>
</div>

**Nguyên tắc vàng cho QC:** augmentation phải mô phỏng **đúng những biến thể thật** của dây chuyền (ánh sáng nhấp nháy, độ lệch vị trí, nhiễu cảm biến), **không** tạo ra biến thể phi thực tế (lật ảnh có chữ, xoay 90° ảnh sản phẩm có phương hướng). Augmentation sai chiều = dạy model học đặc trưng nhiễu.

---

## A4. Overfitting & Phương pháp chống overfit

### A4a. Nhận diện overfitting qua curve loss/accuracy

| Triệu chứng | Ý nghĩa |
|---|---|
| Train loss giảm, **val loss tăng** (điểm rẽ quạ) | **Overfit** — model học thuộc lòng train, không tổng quát. Cần chống (bên dưới) |
| Cả train lẫn val loss đều cao | **Underfit** — model quá yếu/quá ít tham số/lr sai. Cần model lớn hơn hoặc train lâu hơn |
| Train = val, cả hai cùng cao | Dữ liệu khó, model yếu, hoặc học sai feature |

**Cách debug:** in `train_loss`, `val_loss`, `train_acc`, `val_acc` sau mỗi epoch — quy tắc: nếu `val_loss` tăng 2 epochs liên tiếp trong khi `train_loss` giảm → dừng và chống overfit ngay.

### A4b. Bộ công cụ chống overfit (dùng theo thứ tự ưu tiên trong QC)

| Phương pháp | Cơ chế | Ghi chú QC |
|---|---|---|
| **Data augmentation** | Thêm biến thể ảo | Số 1 với QC vì ít dữ liệu |
| **Transfer learning / fine-tune** | Tận dụng feature đã học từ ảnh khổng lồ | Số 1 với QC vì dữ liệu nhỏ & khai thác pretrained |
| **Dropout** | Ngẫu nhiên tắt neuron khi train | `nn.Dropout(p=0.5)` — chỉ tắt khi train, tự tắt khi eval |
| **Weight decay (L2 reg)** | Cộng `λ*‖W‖²` vào loss, kéo weight về 0 | `optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)` |
| **Early stopping** | Dừng khi val loss không cải thiện | Đơn giản, hiệu quả, không đổi model |
| **Batch normalization** | Chuẩn hoá activation mỗi layer | Ổn định + có tác dụng điều chuẩn nhẹ |

> [!NOTE]
> **Quy tắc đặt vị trí Dropout:**
> - **Fully Connected (Linear Layer):** Vị trí hiệu quả nhất do chứa lượng tham số khổng lồ (dễ overfitting). Thường dùng tỉ lệ cao ($p = 0.4 \rightarrow 0.5$).
> - **Feature Map (Conv Layer):** Không dùng Dropout thường vì các pixel lân cận có tính liên kết cao. Cần dùng **Spatial Dropout** (`nn.Dropout2d`) để tắt *nguyên một kênh màu (feature map)*, ép mạng không ỷ lại vào một bộ lọc duy nhất. Tỉ lệ thường rất nhỏ ($p = 0.1 \rightarrow 0.2$).
> - **Pooling Layer:** **Tuyệt đối không dùng**. Pooling không có tham số để học nên không có khái niệm overfitting.

⚠️ **Bẫy quan trọng với Dropout & BatchNorm:** `model.eval()` và `model.train()` **PHẢI** được gọi đúng lúc. Quên `model.eval()` khi test → dropout vẫn tắt neuron → kết quả không ổn định/không tái lập. Quên `model.train()` khi train → batch norm dùng stats sai.

```python
model = ResNet18()   # ví dụ
model.train()        # TRƯỚC khi train — bật dropout/batch norm tính theo batch
# ... training loop ...
model.eval()         # TRƯỚC khi eval/test — tắt dropout, dùng running stats
with torch.no_grad():   # KHÔNG tính gradient khi eval → nhanh + đỡ RAM
    logits = model(x)
```

---

## A5. Kiến trúc phổ biến & Transfer Learning

### A5a. Ba kiến trúc bạn cần hiểu

**1. ResNet (2015) — kiến trúc nền tảng nhất**

Ý tưởng cốt lõi: **residual connection (skip connection)** — cho đầu ra 1 block `F(x)` được cộng thẳng với đầu vào `x`: `output = F(x) + x`.

- Giải quyết **vanishing gradient**: gradient đi đường 'tắt' không bị triệt tiêu khi mạng sâu.
- Đã cho phép train mạng 100+ layer (trước đó hầu như không thể).
- Các biến thể: `ResNet18/34/50/101/152` — số chữ số phía sau là số layer; càng sâu càng mạnh nhưng càng nặng.

```python
from torchvision.models import resnet18, ResNet18_Weights

# Cách dùng chuẩn: lấy pretrained + đổi layer FC cho số lớp của bạn
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features          # 512 cho resnet18
model.fc = nn.Linear(num_features, NUM_CLASSES)   # thay 1000 lớp ImageNet -> N lớp lỗi của bạn
```

> [!NOTE]
> **Làm sao mô hình biết mình cần phát hiện lỗi gì? (Bản chất Transfer Learning)**
> Tại thời điểm bạn vừa chạy xong đoạn code trên, mô hình **hoàn toàn KHÔNG BIẾT** lỗi của bạn là gì.
> Lớp `nn.Linear` bạn vừa gắp bỏ vào là một tờ giấy trắng (chứa các trọng số khởi tạo ngẫu nhiên).
> Tuy nhiên, các lớp Convolution ở phần thân (nhờ được train sẵn trên hàng triệu ảnh ImageNet) đã là những "chuyên gia" chiết xuất đặc trưng hình ảnh: góc cạnh, vệt xước, kết cấu vật liệu, sự thay đổi màu sắc...
> Do đó, việc bạn cần làm tiếp theo là **Huấn luyện (Fine-tuning)**. Bạn đưa tập ảnh lỗi thực tế vào. Mạng sẽ tận dụng năng lực phân tích hình ảnh siêu việt có sẵn ở phần thân, và chỉ mất một thời gian rất ngắn để cập nhật (học) tờ giấy trắng `nn.Linear` ở cuối để mapping: *"À, tập hợp các góc cạnh kiểu A này chính là Lỗi Trầy Xước"*.

**2. EfficientNet (2019) — tối ưu hoá hiệu suất/tài nguyên**

- **Compound scaling**: scale **đồng thời** cả depth (số layer), width (số channel), resolution (kích thước ảnh) theo một hệ số — thay vì chỉ tăng 1 chiều.
- **"Accuracy cao / Chi phí thấp"** không có nghĩa là nó vừa nhanh vừa chính xác hơn tất cả mọi thứ. Ý nghĩa đúng là: *Nếu mình chấp nhận tiêu một lượng tài nguyên (FLOPs, RAM, thời gian xử lý) nhất định, thì EfficientNet sẽ cho ra accuracy tốt nhất so với các mạng khác cùng mức ngân sách đó.* Ví dụ: với 1 giây xử lý cho mỗi ảnh, EfficientNet-B0 đạt 77% accuracy, trong khi ResNet50 (tốn tài nguyên tương đương) chỉ đạt 76%.

```python
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
```

> [!NOTE]
> **`classifier[1]` là gì?**
> Khác với ResNet (chỉ có 1 lớp `model.fc` duy nhất ở cuối), phần đầu phân loại của EfficientNet là một **danh sách (Sequential)** gồm nhiều lớp xếp chồng:
> ```
> model.classifier = Sequential(
>     [0] Dropout(p=0.2)           ← lớp thứ 0: chống Overfitting
>     [1] Linear(1280 → 1000)      ← lớp thứ 1: lớp FC phân loại thật sự
> )
> ```
> Do đó `classifier[1]` là cách truy cập vào đúng **lớp FC ở vị trí số 1** trong danh sách đó (tương tự indexing list Python). Ta thay thế lớp này để đổi từ 1000 lớp ImageNet về `NUM_CLASSES` lớp lỗi của mình. Lớp `Dropout` ở vị trí `[0]` giữ nguyên.

**3. Vision Transformer (ViT, 2020) — kỷ nguyên attention**

- Khác CNN (convolution): chia ảnh thành các **patch** (VD 16x16 px), xếp chồng thành chuỗi, dùng **self-attention** (như trong LLM/Giai đoạn 3-4 của roadmap LLM).
- Điểm mạnh: nhìn **toàn cục** ngay từ layer đầu (CNN nhìn cục bộ rồi mở rộng dần).
- Điểm yếu: cần **rất nhiều dữ liệu** để train từ đầu — thực tế luôn fine-tune pretrained.

> **Với QC ít dữ liệu, ưu tiên thực tế là:** ResNet/EfficientNet (pretrained) trước; thử ViT pretrained nếu dữ liệu tương đối đủ. CNN vẫn là lựa chọn an toàn nhất cho bài toán ảnh công nghiệp với ít mẫu.

### A5b. Transfer Learning & Fine-tuning — kỹ thuật mang tính sống còn với QC

Vì bạn có rất ít ảnh, việc tận dụng model đã train trên ImageNet (14 triệu ảnh, 1000 lớp) là không-thể-thiếu.

| Khái niệm | Ý nghĩa | Ví dụ cụ thể |
|---|---|---|
| **Transfer learning** | Tận dụng feature đã học từ bài toán lớn, mang sang bài toán nhỏ của bạn | Lấy ResNet đã phân loại 1000 lớp → dùng lại các conv layer (nhận diện cạnh, texture) |
| **Feature extraction** | **Đóng băng** toàn bộ backbone (không train), **chỉ train** layer classifier mới | `for p in model.parameters(): p.requires_grad = False`, chỉ mở FC mới |
| **Fine-tuning** | Mở khoá một phần/all backbone để train **tiếp** với lr nhỏ | Mở vài layer cuối + classifier, lr nhỏ (1e-4/1e-5) |

```python
# ==== FEATURE EXTRACTION: đóng băng backbone ====
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False          # đóng băng toàn bộ
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)  # chỉ layer mới train

# ==== FINE-TUNING: mở khoá vài layer cuối ====
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)   # lr cao cho classifier mới
# Sau 1-2 epoch ổn định, mở dần:
for name, param in model.named_parameters():
    if 'layer4' in name or 'fc' in name:   # mở layer cuối + classifier
        param.requires_grad = True
# fine-tuning nên dùng lr NHỎ (1e-4/1e-5) vì feature đã tốt, chỉ tinh chỉnh nhẹ
```

**Thực hành chuẩn trong QC:**
1. **Feature extraction** trước: chỉ train classifier mới → nhanh, chống overfit.
2. Nếu accuracy thấp → **fine-tune**: mở dần từng khối cuối lên, lr nhỏ dần.
3. Theo dõi val_loss mỗi bước, dừng khi không cải thiện (early stopping).

⚠️ **Bẫy:** đừng fine-tune toàn bộ bằng lr mặc định lớn — sẽ "phá" feature ImageNet đã học tốt. Quy tắc: **classifier mới dùng lr cao, backbone fine-tune dùng lr thấp hơn 10-100 lần**.

---

## A6. Learning Rate Scheduling & Vòng lặp huấn luyện hoàn chỉnh

### A6a. Vì sao cần scheduling

Learning rate cố định quá lớn → loss dao động quanh điểm tối ưu; quá nhỏ → hội tụ chậm. Scheduling: **bắt đầu lr vừa phải, giảm dần** để hội tụ chính xác.

| Scheduler | Cơ chế | Dùng khi nào |
|---|---|---|
| `StepLR` | Giảm lr 1 hệ số sau mỗi N epoch | Đơn giản, dễ hiểu |
| `CosineAnnealingLR` | Giảm lr theo đường cong cosin về ~0 | Phổ biến cho fine-tune hiện đại |
| `ReduceLROnPlateau` | Tự giảm khi val_loss không cải thiện | Tiện khi không muốn đặt cứng lịch |
| `OneCycleLR` | Tăng lên rồi giảm về 0 trong 1 chu kỳ | Tốc độ hội tụ nhanh |

```python
import torch.optim.lr_scheduler as lr_scheduler

scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)   # mỗi 7 epoch giảm lr 10 lần

for epoch in range(num_epochs):
    train_one_epoch(...)
    val_loss, val_acc = evaluate(...)
    scheduler.step()                      # phải GỌI SAU mỗi epoch
    print(f"Epoch {epoch}: lr={optimizer.param_groups[0]['lr']:.2e}, val_acc={val_acc:.3f}")
```

### A6b. Template vòng lặp huấn luyện đầy đủ (dùng làm bộ khung cho toàn giai đoạn)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.fc.parameters(), 'lr': lr},              # classifier: lr cao
        {'params': [p for n, p in model.named_parameters()
                    if 'fc' not in n and p.requires_grad], 'lr': lr / 10},  # backbone: lr thấp
    ])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, preds = logits.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        val_loss, val_acc = evaluate(model, val_loader, device, loss_fn)
        scheduler.step()
        print(f"Epoch {epoch}: train_loss={total_loss/len(train_loader):.4f} "
              f"train_acc={correct/total:.3f} | val_acc={val_acc:.3f} val_loss={val_loss:.4f}")
    return model

def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += loss_fn(logits, labels).item()
            _, preds = logits.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total
```

### A6c. Dataset & DataLoader — gắn kết với kỹ năng OOP Giai đoạn 0

Nhớ lại `__len__`/`__getitem__` ở Giai đoạn 0 — giờ bạn áp dụng đầy đủ với transforms:

```python
import os
from PIL import Image
from torch.utils.data import Dataset

class DefectDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')  # PIL để dùng torchvision
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

train_loader = DataLoader(
    DefectDataset(train_paths, train_labels, train_transform),
    batch_size=32, shuffle=True, num_workers=2,
)
val_loader = DataLoader(
    DefectDataset(val_paths, val_labels, val_transform),
    batch_size=32, shuffle=False, num_workers=2,
)
```

---

## A7. Checklist lỗi thường gặp (Phần A)

- [ ] `loss.backward()` báo lỗi → chắc chắn `x.requires_grad` đúng hoặc model là `nn.Module` (không phải hàm gốc)
- [ ] Loss không giảm / bằng nhau mỗi epoch → quên `optimizer.zero_grad()` hoặc lr quá nhỏ
- [ ] Quên `model.train()`/`model.eval()` → dropout/batchnorm sai → kết quả eval không tái lập
- [ ] Đưa xác suất đã softmax vào `CrossEntropyLoss` → sai trầm trọng (loss này tự log-softmax, chỉ nhận raw logits)
- [ ] Val loss tăng trong khi train loss giảm → **overfit** → tăng augmentation / dropout / weight decay / dừng (early stopping)
- [ ] Kích thước ảnh input sai → kiểm tra `transforms.Normalize` có đúng mean/std ImageNet, và `ToTensor` chia cho 255
- [ ] Fine-tune với lr quá lớn → "phá" feature ImageNet → bắt đầu với lr rất nhỏ cho backbone
- [ ] Quên `model.to(device)` hoặc để tensor trên CPU khi model trên GPU → `RuntimeError: Expected all tensors to be on the same device`

---

## A8. Bài tập thực hành đề xuất (Phần A — gắn domain QC)

**Bài 1 — CNN từ đầu (tự xây, không pretrained):**
1. Xây `nn.Sequential`: `Conv(3→16,3,pad1) → ReLU → MaxPool(2) → Conv(16→32,3,pad1) → ReLU → MaxPool(2) → Flatten → Linear(32*56*56 → N)`.
2. Train trên dataset nhị phân "lỗi / không lỗi" (~100 ảnh mỗi lớp), in curve.
3. Thêm augmentation + dropout → so sánh overfit trước/sau.

> [!NOTE]
> **Dataset công khai để thực hành (thay thế ảnh nhà máy thực):**
> Khi chưa đi làm, hình ảnh nhà máy thật không có — đây là điều hoàn toàn bình thường. Dùng dataset benchmark công khai cho mục đích học là đúng hướng. Khi đi làm thật, bạn sẽ áp dụng lại đúng quy trình này lên data thật.
>
> | Dataset | Loại lỗi | Link | Ghi chú |
> |---|---|---|---|
> | **NEU Steel** *(khuyên dùng cho Bài 1)* | Thép tấm: vết nứt, xước, rỗ bề mặt | [Kaggle NEU](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) | Nhỏ (1800 ảnh, 6 lớp), train nhanh ngay trên CPU. Có thể gộp 6 lớp thành 2 để làm binary. |
> | **MVTec AD** | 15 loại vật liệu công nghiệp (da, ốc vít, vải, kim loại...) | [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad) | Chuẩn nhất thế giới cho bài toán QC, dùng tốt cho Bài 2-3. |
> | **KolektorSDD2** | Vết nứt linh kiện điện tử | [vicos.si](https://www.vicos.si/resources/kolektorsdd2/) | Gần với bài toán inspection thực tế. |
> | **Severstal Steel (Kaggle)** | 4 loại lỗi thép tấm, có cả mask | [kaggle.com/c/severstal-steel-defect-detection](https://www.kaggle.com/c/severstal-steel-defect-detection) | Dùng khi học Segmentation ở Phase sau. |

**Bài 2 — Transfer learning + fine-tune:**
1. Tải `resnet18` pretrained, feature extraction, train.
2. Fine-tune mở `layer4` + FC, lr nhỏ.
3. So sánh accuracy/độ ổn định giữa 2 cách — rút ra khi nào nên dùng cách nào.

**Bài 3 — Đánh giá & hiểu lỗi:**
1. In confusion matrix (không chỉ accuracy) — xem lỗi nào bị nhầm nhiều.
2. Hiển thị vài ảnh bị phân loại sai → tự hỏi: đây là lỗi đặc trưng (giống nhau), lỗi ánh sáng, hay augmentation chưa đủ?

---

## A9. Câu hỏi tự kiểm tra (chuẩn bị sớm cho phỏng vấn)

- Vì sao CNN cần 3 ý tưởng local connectivity, weight sharing, hierarchical features?
+ **Local Connectivity**: Convolution chỉ nhìn một vùng nhỏ (kernel 3×3) xung quanh mỗi điểm ảnh, thay vì nhìn toàn bộ ảnh như FC. Hợp lý vì đặc trưng ảnh mang tính cục bộ (viền mắt nằm gần mắt, không nằm gần chân).
+ **Weight Sharing**: Dùng 1 kernel trượt khắp ảnh — tiết kiệm tham số, quan trọng hơn là tạo **tính bất biến vị trí (Translation Invariance)**: kernel nhận ra vết xước ở góc trên trái cũng sẽ nhận ra nó ở góc dưới phải.
+ **Hierarchical Features**: Tầng đầu học đặc trưng đơn giản (cạnh, góc, màu sắc); tầng sau tổng hợp thành đặc trưng phức tạp hơn (vân bề mặt, hình dạng vật thể).

- `Conv2d(3, 16, 3, padding=1)` với input `(1,3,224,224)` → output shape bao nhiêu? Có bao nhiêu tham số? (gợi ý: `16*3*3*3 + 16`)
+ **Output shape**: `(1, 16, 224, 224)` — `padding=1` bảo toàn H và W, số kênh đầu ra là 16.
+ **Số tham số**: `16 × 3 × 3 × 3 + 16 = 448` (16 bộ lọc × 3×3×3 trọng số + 16 bias).

- Max pooling giúp gì về receptive field và số tham số?
+ **Receptive Field tăng**: Sau MaxPool stride=2, mỗi pixel ở feature map tiếp theo "
hìn" được vùng rộng gấp đôi của ảnh gốc → các tầng sau nhận diện được đặc trưng to hơn (cả vùng mặt thay vì chỉ mép viền nhỏ).
+ **Số tham số giảm**: Feature map thu nhỏ 2 lần → tầng Conv tiếp theo xử lý ít điểm hơn. Bản thân MaxPool không có tham số học.

- Vòng lặp huấn luyện gồm mấy bước? Vì sao phải `zero_grad()` mỗi batch?
+ **Các bước** (lặp theo từng batch): `zero_grad()` → Forward (tính output) → Tính Loss → `backward()` (tính gradient) → `optimizer.step()` (cập nhật trọng số). Khởi tạo trọng số xảy ra 1 lần trước khi train, không thuộc vòng lặp.
+ **`zero_grad()`**: PyTorch mặc định **cộng dồn gradient** qua các lần gọi backward. Nếu không xóa, gradient batch sau sẽ cộng vào batch trước → gradient phình to, Loss nhảy lung tung, không hội tụ về cực tiểu.

- Overfitting nhận diện thế nào qua curve? Liệt kê 4 cách chống.
+ **Nhận diện**: Overfitting xảy ra khi **Train Acc** tiếp tục tăng nhưng **Val Acc** chứng lại rồi giảm (2 đường giao nhau và chạy xa nhau). Phải so sánh Train và Val — không phải "test và val" (Test set chỉ dùng 1 lần duy nhất ở cuối).
+ **4 nhóm chống Overfitting**: (1) Dữ liệu: Data Augmentation, thu thập thêm. (2) Mô hình: Dropout, giảm độ phức tạp. (3) Regularization: Weight Decay / L2. (4) Dừng sớm: Early Stopping theo `val_loss`.

- Transfer learning vs fine-tuning khác nhau cốt lõi ở điểm nào? Khi nào dùng cái nào?
+ **Transfer Learning** là khái niệm tổng quát (umbrella term): tận dụng tri thức từ bài toán nguồn (ImageNet) sang bài toán đích. Fine-Tuning là **một dạng của Transfer Learning**, không phải khái niệm ngang hàng.
+ **Feature Extraction** (đóng băng Backbone): Dùng khi dữ liệu ít + ảnh giống ImageNet (chó, mèo, xe cộ).
+ **Fine-Tuning** (mở khóa Backbone với LR nhỏ): Dùng khi ảnh đặc thù/ngách (thép, X-quang y tế) hoặc có đủ dữ liệu để train sâu hơn.

- Vì sao lr cho backbone fine-tune phải nhỏ hơn lr cho classifier nhiều lần?
+ Backbone đã được train sẵn, các bộ lọc rất tốt để nhận ra cạnh, góc, vân bề mặt. Nếu LR lớn, gradient từ tầng FC mới (chưa biết gì) truyền ngược về sẽ **phá hủy toàn bộ tri thức đã học** — hiện tượng gọi là **Catastrophic Forgetting**.
+ Classifier (FC) mới hoàn toàn chưa biết phân biệt 6 lớp lỗi thép → cần LR lớn hơn để học nhanh hơn.

- BatchNorm/Dropout ảnh hưởng thế nào nếu bỏ qua `model.train()`/`model.eval()`?
+ **`model.train()`** bật: Dropout tắt ngẫu nhiên nơ-rôn theo xác suất `p`; BatchNorm chuẩn hóa theo mean/std của batch hiện tại và cập nhật running statistics.
+ **Hậu quả nếu quên `model.eval()` khi đánh giá**: BatchNorm tính theo batch nhỏ 1-2 ảnh → mean/std bị nhiễu, output biến động mạnh. Dropout tiếp tục tắt nơ-rôn ngẫu nhiên → cùng 1 ảnh đưa vào 2 lần cho 2 kết quả khác nhau → kết quả đánh giá không tin cậy. Nên kết hợp `model.eval()` với `@torch.no_grad()` để tắt luôn việc tính gradient vào VRAM.

---

# PHẦN B — OBJECT DETECTION (Tuần 7-9)

> Mục tiêu sau 3 tuần: hiểu bài toán detection (bounding box + lớp), nắm được metric mAP, biết cách gán nhãn (annotation) dữ liệu, và **vận hành model YOLO** (train + evaluate) — đây là model được dùng rộng rãi nhất thực tế cho QC/nhà máy.

---

## B1. Bài toán Object Detection — định nghĩa

**So với classification:** classification trả lời *"trong ảnh có gì?"* (1 nhãn/ảnh). Detection trả lời *"có những gì, ở đâu?"* → output là **danh sách bounding box**, mỗi box gồm `(x, y, w, h)` + `class` + `confidence`.

```
Input: 1 ảnh
Output (detection): [
    {box: (x1,y1,x2,y2), class: "scratch", conf: 0.93},
    {box: (12,30,88,75), class: "dent",    conf: 0.88},
    ...
]
```

**Use case QC trực tiếp:** thay vì chỉ nói "có lỗi", bạn biết được **từng lỗi ở toạ độ nào** — cần để: khoanh vùng trên màn hình cho operator, tính số lượng lỗi, đo kích thước từng lỗi, gửi toạ độ cho robot loại bỏ.

### B1a. Các thuật ngữ cốt lõi

| Thuật ngữ | Ý nghĩa | Ghi chú |
|---|---|---|
| **Bounding box** | Hình chữ nhật bao quanh đối tượng | Thường là `(x, y, w, h)` hoặc `(x1, y1, x2, y2)`, có thể chuẩn hoá (0-1) |
| **Anchor box** | Các "khuôn hình" kích thước/tỷ lệ định sẵn | Model dự đoán offset ($t_x, t_y, t_w, t_h$) so với anchor gần nhất — xem chi tiết B1b |
| **IoU** | Tỷ số giữa diện tích giao / hợp 2 box | Đo độ khớp box dự đoán với box đúng — xem chi tiết B1c |
| **NMS** | Loại bỏ box trùng lặp quanh cùng 1 đối tượng | Cần vì model dự đoán nhiều box cho 1 vật — xem chi tiết B1d |
| **Confidence threshold** | Ngưỡng xác suất giữ/loại box | Dưới ngưỡng → bỏ box (giảm false positive) |
| **One-stage vs Two-stage** | Detection 1 bước vs 2 bước (tìm vùng nghi ngờ → phân loại) | Xem B3 |

### B1b. Anchor Box — bản chất, cơ chế Offset & ví dụ số cụ thể

#### 1. Tại sao phải cần Anchor Box? (Vấn đề cốt lõi)
Nếu bắt mạng nơ-ron từ một grid cell dự đoán trực tiếp toạ độ tuyệt đối $(x, y, w, h)$ từ con số 0:
1. **Không gian tìm kiếm vô tận:** Box có thể nhỏ 5px hoặc to 400px. Giá trị mục tiêu dao động quá lớn khiến gradient không ổn định, loss nhảy loạn xạ và cực kỳ khó hội tụ.
2. **Xung đột hình dạng (Shape ambiguity):** Tại cùng 1 grid cell, có thể xuất hiện nhiều loại vật thể khác nhau — ví dụ: một **vết xước dài thẳng đứng** ($h \gg w$) và một **vết rỗ tròn** ($w \approx h$). Nếu model chỉ có 1 đầu ra cho mỗi cell, nó sẽ bị "lưỡng lự" và sinh ra dự đoán méo mó (trung bình cộng của cả hai).

> **Giải pháp (Tư tưởng Anchor Box):** Đặt sẵn tại mỗi grid cell một tập hợp các "khuôn mẫu" (priors) với kích thước và tỷ lệ khác nhau (ví dụ: khuôn vuông, khuôn đứng cao, khuôn dẹt ngang). Model **không cần đoán toạ độ tuyệt đối**, mà chỉ cần học cách **tinh chỉnh (offset)**: *"Khuôn mẫu này cần dịch tâm bao nhiêu pixel và co/dãn chiều rộng/chiều cao bao nhiêu % để khít với vật thể?"*

---

#### 2. Cơ chế toán học: Model thực sự dự đoán cái gì? (Theo chuẩn YOLO)

Tại một grid cell có toạ độ góc trên-trái là $(c_x, c_y)$, xét một anchor box có kích thước định sẵn là $(p_w, p_h)$:

Model ConvNet sẽ xuất ra **4 số thực offset:** $(t_x, t_y, t_w, t_h)$ cho mỗi anchor. Toạ độ thực sự của Bounding Box $(b_x, b_y, b_w, b_h)$ được giải mã như sau:

$$\begin{aligned}
b_x &= \sigma(t_x) + c_x \\
b_y &= \sigma(t_y) + c_y \\
b_w &= p_w \cdot e^{t_w} \\
b_h &= p_h \cdot e^{t_h}
\end{aligned}$$

**Ý nghĩa của các hàm biến đổi:**
- **Hàm Sigmoid $\sigma(t) \in (0, 1)$:** Ép tâm box $(b_x, b_y)$ **chỉ được phép nằm bên trong grid cell hiện tại**, không bị trôi tự do ra các ô khác.
- **Hàm mũ $e^t > 0$:** 
  - Đảm bảo chiều rộng $b_w$ và chiều cao $b_h$ luôn luôn dương.
  - Khi $t_w = 0, t_h = 0 \implies e^0 = 1 \implies b_w = p_w, b_h = p_h$ (giữ nguyên kích thước anchor).
  - Khi $t_w > 0$, box nở to hơn anchor ($e^{t_w} > 1$). Khi $t_w < 0$, box co nhỏ hơn anchor ($e^{t_w} < 1$).

---

#### 3. Ví dụ số thực tế từng bước (Step-by-step Calculation)

Giả sử bạn đang huấn luyện model Object Detection (ví dụ YOLOv3/v4) kiểm tra lỗi bề mặt thép:
- **Ảnh đầu vào:** $416 \times 416$ pixel.
- **Feature map (Grid):** $13 \times 13$ cells (mỗi cell ứng với một vùng $32 \times 32$ pixel trên ảnh gốc).
- **Vị trí đang xét:** Grid cell tại hàng 5, cột 7 $\implies c_x = 7, c_y = 5$.
- **Anchor được gán sẵn tại cell này:** Một anchor hình chữ nhật dọc chuyên bắt vết xước dài (*scratches*):
  $$p_w = 40\text{ px}, \quad p_h = 160\text{ px}$$
- **Model suy luận và xuất ra 4 giá trị offsets:**
  $$t_x = 0.405, \quad t_y = -0.300, \quad t_w = 0.180, \quad t_h = -0.050$$

**Bước 1: Tính toạ độ tâm box $(b_x, b_y)$ theo toạ độ Grid & Đổi sang Pixel ảnh gốc**
- $\sigma(t_x) = \sigma(0.405) = \frac{1}{1 + e^{-0.405}} \approx 0.60$
  $$\implies b_x = 7 + 0.60 = 7.60\text{ (grid unit)} \implies X_{\text{center}} = 7.60 \times 32 = \mathbf{243.2\text{ px}}$$
- $\sigma(t_y) = \sigma(-0.300) = \frac{1}{1 + e^{0.300}} \approx 0.425$
  $$\implies b_y = 5 + 0.425 = 5.425\text{ (grid unit)} \implies Y_{\text{center}} = 5.425 \times 32 = \mathbf{173.6\text{ px}}$$

**Bước 2: Tính kích thước $(b_w, b_h)$ theo Pixel**
- $b_w = p_w \times e^{t_w} = 40 \times e^{0.180} = 40 \times 1.1972 \approx \mathbf{47.89\text{ px}}$ *(box nở rộng hơn anchor ~20%)*
- $b_h = p_h \times e^{t_h} = 160 \times e^{-0.050} = 160 \times 0.9512 \approx \mathbf{152.2\text{ px}}$ *(box co ngắn hơn anchor ~5%)*

**Bước 3: Quy đổi sang toạ độ hiển thị $(x_1, y_1, x_2, y_2)$ để vẽ lên ảnh**
- $x_1 = X_{\text{center}} - \frac{b_w}{2} = 243.2 - 23.95 = \mathbf{219.25\text{ px}}$
- $y_1 = Y_{\text{center}} - \frac{b_h}{2} = 173.6 - 76.10 = \mathbf{97.50\text{ px}}$
- $x_2 = X_{\text{center}} + \frac{b_w}{2} = 243.2 + 23.95 = \mathbf{267.15\text{ px}}$
- $y_2 = Y_{\text{center}} + \frac{b_h}{2} = 173.6 + 76.10 = \mathbf{249.70\text{ px}}$

> **Nhận xét trực quan:** Thay vì phải dự đoán các số toạ độ tuyệt đối khó khăn $[219.25, 97.50, 267.15, 249.70]$, mạng nơ-ron chỉ cần dự đoán các giá trị offset rất nhỏ quanh mức 0 $[-0.300, 0.405, 0.180, -0.050]$. Việc học này giúp gradient cực kỳ mượt và ổn định!

---

#### 4. Kích thước Anchor Box từ đâu mà có?

1. **Faster R-CNN (Truyền thống - Thủ công):** 
   - Tự quy định 3 scales ($128^2, 256^2, 512^2$) $\times$ 3 aspect ratios ($1:1, 1:2, 2:1$) = 9 anchors.
   - Nhược điểm: Dễ bị lệch nếu tập dữ liệu đặc thù (ví dụ khuyết tật bề mặt có tỷ lệ dải hẹp $1:8$ hoặc siêu nhỏ $10\times 10$).
2. **YOLOv2 đến YOLOv7 (Tự động hóa bằng K-Means Clustering & Genetic Algorithm):**
   - **Vấn đề cốt lõi của Anchor thủ công:** Việc người lập trình tự đoán kích thước anchor (như Faster R-CNN) rất dễ bị lệch pha với thực tế. Nếu tập dữ liệu là khuyết tật vết nứt kim loại (*crazing, scratches*) có dạng dải hẹp $1:8$ hoặc siêu nhỏ $12 \times 12\text{ px}$, các anchor hình vuông to $128 \times 128$ sẽ khiến mô hình khởi đầu rất xa Ground Truth $\rightarrow$ Gradient cập nhật rất lớn, khó hội tụ.
   - **Thuật toán K-Means Clustering trên Bounding Box:**
     - Lấy toàn bộ Bounding Box Ground Truth $(w_i, h_i)$ trong toàn bộ tập Train (chỉ lấy chiều rộng $w$ và chiều cao $h$, bỏ qua vị trí toạ độ tâm $(x, y)$).
     - **Tại sao KHÔNG dùng khoảng cách Euclid chuẩn $d = \sqrt{(w_1 - w_2)^2 + (h_1 - h_2)^2}$?**
       - Khoảng cách Euclid phụ thuộc vào kích thước tuyệt đối: Một box lớn $400 \times 400$ lệch 40 px tạo ra lỗi $\sqrt{40^2 + 40^2} \approx 56.57$. Trong khi một box nhỏ $20 \times 20$ lệch 15 px (lệch gần hết cả vật thể!) chỉ tạo ra lỗi $\sqrt{15^2 + 15^2} \approx 21.21$.
       - Hệ quả: K-Means với khoảng cách Euclid sẽ bị **thiên vị gom cụm theo các box lớn**, bỏ rơi các vật thể nhỏ!
     - **Metric khoảng cách bất biến theo tỷ lệ (Scale-Invariant Metric):**
       $$d(\text{box}, \text{centroid}) = 1 - \text{IoU}(\text{box}, \text{centroid})$$
       - Khi tính IoU giữa 2 box, ta tịnh tiến tâm của 2 box về trùng nhau tại gốc tọa độ $(0, 0)$.
       - Vì $0 \le \text{IoU} \le 1 \implies 0 \le d \le 1$.
       - Bất kể box lớn hay nhỏ, nếu hình dạng và tỷ lệ tương đồng thì IoU sẽ cao $\rightarrow$ Khoảng cách $d$ nhỏ.
     - **Số lượng cụm $k$ (The Elbow Method):**
       - YOLOv2 chọn $k=5$ anchors (đạt điểm cân bằng giữa độ phức tạp mô hình và Recall cao).
       - Từ YOLOv3 đến YOLOv7, kiến trúc dùng Feature Pyramid Network (FPN / PANet) chia ra 3 tỷ lệ feature map (Stride 8, 16, 32), nên thuật toán chọn **$k=9$ cụm**, phân bổ đều **3 anchors cho mỗi scale**:
         - *Scale nhỏ (Stride 8 - $52 \times 52$):* 3 anchors nhỏ (bắt vật thể nhỏ).
         - *Scale vừa (Stride 16 - $26 \times 26$):* 3 anchors trung bình (bắt vật thể vừa).
         - *Scale lớn (Stride 32 - $13 \times 13$):* 3 anchors to (bắt vật thể lớn/bao trùm).
     - **Cải tiến AutoAnchor bằng Genetic Algorithm (YOLOv5/v7):**
       - Trước khi train, YOLOv5 chạy K-Means tạo 9 anchor khởi đầu.
       - Tiếp tục dùng **Thuật toán Di truyền (Genetic Algorithm)**: Cho các kích thước anchor đột biến nhẹ qua 1000 thế hệ nhằm tối đa hóa chỉ số **Best Possible Recall (BPR)** (đảm bảo >98% ground truth boxes có ít nhất một anchor khớp IoU > 0.29).

3. **YOLOv8, YOLOv11, FCOS (Kỷ nguyên hiện đại: Bỏ hoàn toàn Anchor — Anchor-Free):**
   - **Tại sao các mô hình hiện đại lại "khai tử" Anchor Box?**
     - ❌ **Bùng nổ siêu tham số (Hyperparameter Overhead):** Người dùng phải chọn số lượng $k$, scale, aspect ratio, ngưỡng IoU matching (IoU > 0.5 là positive, < 0.4 là negative). Khi đổi sang bài toán mới (ảnh y tế, viễn thám drone, lỗi bề mặt thép...), ta phải chạy lại K-Means thiết kế lại dàn anchor từ đầu.
     - ❌ **Mất cân bằng mẫu trầm trọng (Extreme Class Imbalance):** Một ảnh $640 \times 640$ sinh ra tới ~8400 vị trí grid $\times 3\text{ anchors} = \mathbf{25,200\text{ anchor boxes}}$. Trong khi cả bức ảnh thường chỉ có 3 - 5 vật thể thực sự. Hơn $99.9\%$ anchors là nền (background/negative), gây khó khăn cho hàm loss.
     - ❌ **Chi phí tính toán ma trận IoU khổng lồ:** Trong mỗi batch huấn luyện, việc tính toán IoU giữa 25,200 anchors với hàng loạt ground-truth boxes tốn rất nhiều tài nguyên VRAM và CPU/GPU.
     - ❌ **Kém linh hoạt với vật thể có tỷ lệ bất thường:** Các khuyết tật như vết nứt uốn lượn, sợi chỉ mảnh, góc nghiêng... không khớp vừa vặn với các anchor hình chữ nhật định sẵn.
   - **Nguyên lý hoạt động của kiến trúc Anchor-Free (FCOS, YOLOv8, YOLOv11):**

     #### 💡 Bản chất cốt lõi: Ẩn dụ "Đứng tại chỗ nhìn ra 4 mép tường"
     Để dễ hiểu nhất, hãy so sánh 2 cách tư duy phát hiện vật thể:
     - **Anchor-Based (YOLOv2 - YOLOv7):** Tại mỗi vị trí ô lưới, bạn cầm sẵn **3 cái khung mẫu cứng** (nhỏ, vừa, lớn) ướm thử vào ảnh. Mạng nơ-ron phải học cách: *"Dịch chuyển tâm bao nhiêu, co giãn bề ngang và bề dọc bao nhiêu lần ($t_x, t_y, t_w, t_h$) từ cái khung mẫu này để vừa khít vật thể?"* $\rightarrow$ **Bắt buộc phải mượn khung mẫu làm bàn đạp.**
     - **Anchor-Free (FCOS, YOLOv8, YOLOv11):** **Vứt bỏ hoàn toàn khung mẫu!** Không cần chuẩn bị bất kỳ hình chữ nhật nào trước. Thay vào đó, mô hình đứng tại một điểm tọa độ bất kỳ $(x, y)$ trên ảnh và trả lời đúng 2 câu hỏi:
       1. *Câu hỏi 1 (Phân loại):* "Điểm $(x, y)$ này có đang nằm bên trong vật thể nào không (ví dụ: vết xước, vết rỉ, ô tô)?"
       2. *Câu hỏi 2 (Kích thước Bounding Box):* "Từ điểm $(x, y)$ này nhìn ra 4 phía xung quanh, cách các mép của vật thể bao nhiêu pixel?"
          - Sang mép bên **TRÁI** bao xa? $\rightarrow l$ (left)
          - Lên mép bên **TRÊN** bao xa? $\rightarrow t$ (top)
          - Sang mép bên **PHẢI** bao xa? $\rightarrow r$ (right)
          - Xuống mép bên **DƯỚI** bao xa? $\rightarrow b$ (bottom)

     #### 📐 Sơ đồ hình học trực quan

     ```text
                        Y (Mép trên: y1 = y - t)
                        ─────────────────────────▲
                        │                        │
                        │                        │ t
                        │                        │
     (Mép trái: x1 = x - l)◄─────── (x, y) ───────► (Mép phải: x2 = x + r)
                        │      l      ▲    r     │
                        │             │          │
                        │             │ b        │
                        │             ▼          │
                        ─────────────────────────▼
                        (Mép dưới: y2 = y + b)
     ```

     **Công thức khôi phục toạ độ Bounding Box (cực kỳ tự nhiên, không cần hàm số mũ $e^t$ hay anchor):**
     $$\begin{cases} 
     x_1 = x - l & \text{(Toạ độ mép trái)} \\
     y_1 = y - t & \text{(Toạ độ mép trên)} \\
     x_2 = x + r & \text{(Toạ độ mép phải)} \\
     y_2 = y + b & \text{(Toạ độ mép dưới)}
     \end{cases}$$
     $$\text{Chiều rộng } W = l + r, \qquad \text{Chiều cao } H = t + b$$

     #### 🔢 Ví dụ số thực tế từng bước (Step-by-step Numerical Example)

     Giả sử ta có một bức ảnh kích thước $640 \times 640$:
     - **Vật thể thực tế (Ground Truth):** Một vết gỉ bề mặt thép (*patches*) có toạ độ hộp: 
       $$[x_1 = 100, \, y_1 = 150, \, x_2 = 300, \, y_2 = 270]$$
       *(Chiều rộng $W = 300 - 100 = 200\text{ px}$, Chiều cao $H = 270 - 150 = 120\text{ px}$)*.
     
     - **Xét một điểm neo $(x, y)$:** Điểm tại toạ độ $(x = 180, y = 200)$ trên ảnh.
       - Kiểm tra: Vì $100 < 180 < 300$ và $150 < 200 < 270 \implies$ **Điểm này nằm trọn bên trong vết gỉ!**
     
     - **Khoảng cách mục tiêu (Ground Truth Targets) mà mạng cần học dự đoán:**
       - $l_{\text{target}} = x - x_1 = 180 - 100 = \mathbf{80\text{ px}}$ *(từ điểm $(180, 200)$ sang mép trái là 80 px)*
       - $t_{\text{target}} = y - y_1 = 200 - 150 = \mathbf{50\text{ px}}$ *(từ điểm $(180, 200)$ lên mép trên là 50 px)*
       - $r_{\text{target}} = x_2 - x = 300 - 180 = \mathbf{120\text{ px}}$ *(từ điểm $(180, 200)$ sang mép phải là 120 px)*
       - $b_{\text{target}} = y_2 - y = 270 - 200 = \mathbf{70\text{ px}}$ *(từ điểm $(180, 200)$ xuống mép dưới là 70 px)*

     - **Khi chạy suy luận (Inference):** 
       Mạng nhận vào ảnh, tại vị trí $(180, 200)$ nó bắn ra dự đoán: $\hat{l} = 79.5, \, \hat{t} = 50.2, \, \hat{r} = 120.8, \, \hat{b} = 69.1$.
       Toạ độ Bounding Box được dựng lại ngay lập tức:
       - $x_1 = 180 - 79.5 = \mathbf{100.5\text{ px}}$
       - $y_1 = 200 - 50.2 = \mathbf{149.8\text{ px}}$
       - $x_2 = 180 + 120.8 = \mathbf{300.8\text{ px}}$
       - $y_2 = 200 + 69.1 = \mathbf{269.1\text{ px}}$
       > **Nhận xét:** Kết quả box $[100.5, 149.8, 300.8, 269.1]$ khớp gần như tuyệt đối với Ground Truth $[100, 150, 300, 270]$ mà không cần tính toán bất kỳ Anchor Box mẫu nào!

     #### ❓ 3 Bài toán cốt lõi của Anchor-Free & Lời giải từ FCOS đến YOLOv8/v11

     ##### 1. Bài toán 1: "Mập mờ chồng lấn" (Ambiguity) — Khi một điểm rơi vào hai vật thể cùng lúc
     - **Hiện tượng:** Trong một bức ảnh, người đứng trước xe buýt hoặc con mèo nằm trên ghế sofa. Một điểm $(x, y)$ rơi vào vùng giao nhau của cả 2 bounding box. Điểm này phải học dự đoán nhãn nào? Học $(l, t, r, b)$ của vật nhỏ hay vật to? Nếu không có cơ chế phân tách, tín hiệu gradient sẽ xung đột dữ dội khiến mạng không thể hội tụ.
     - **Giải pháp của FCOS (FPN đa tầng phân vùng kích thước):**
       FCOS chia 5 tầng feature map ($P_3 \to P_7$) với các stride khác nhau ($8, 16, 32, 64, 128$). Với mỗi điểm dương, tính kích thước tối đa:
       $$m = \max(l, t, r, b)$$
       Sau đó phân bổ nhiệm vụ nghiêm ngặt cho từng tầng:
       - **Tầng $P_3$ (Stride 8):** Chuyên trách $m \in [0, 64\text{ px}]$ $\implies$ Bắt người, vật thể nhỏ.
       - **Tầng $P_4$ (Stride 16):** Chuyên trách $m \in [64, 128\text{ px}]$ $\implies$ Bắt vật thể vừa-nhỏ.
       - **Tầng $P_5$ (Stride 32):** Chuyên trách $m \in [128, 256\text{ px}]$ $\implies$ Bắt vật thể trung bình.
       - **Tầng $P_6$ (Stride 64):** Chuyên trách $m \in [256, 512\text{ px}]$ $\implies$ Bắt xe hơi, xe buýt.
       - **Tầng $P_7$ (Stride 128):** Chuyên trách $m \in [512, \infty\text{ px}]$ $\implies$ Bắt vật thể bao trùm cực đại.
       > Nhờ đó, người ($m \approx 50$) tự động rơi vào tầng $P_3$, xe buýt ($m \approx 300$) tự động rơi vào tầng $P_6$. Cùng 1 vị trí toạ độ ảnh nhưng ở 2 tầng feature map độc lập $\implies$ **Triệt tiêu hoàn toàn sự mập mờ!**
       > *(Nếu 2 vật thể cùng kích thước, cùng tầng mà vẫn đè lên nhau: FCOS ưu tiên gán cho vật thể có **diện tích nhỏ hơn** để bảo vệ tín hiệu của vật nhỏ không bị vật lớn nuốt chửng)*.

     ##### 2. Bài toán 2: "Điểm ở rìa mép đoán tệ" & Bước tiến từ FCOS Centerness sang YOLOv8 TAL
     - **Hiện tượng:** Một vật thể có thể chứa hàng chục đến hàng trăm điểm $(x, y)$ bên trong:
       - Điểm nằm **chính giữa tâm**: Khoảng cách 4 phía cân đối ($l \approx r, t \approx b$), trường nhìn đối xứng $\implies$ Dự đoán kích thước Bounding Box cực kỳ chuẩn xác.
       - Điểm nằm **sát rìa mép** (ví dụ gót chân, mép áo): Một khoảng cách gần bằng $0$ ($l \approx 2\text{ px}$), khoảng cách đối diện lại quá dài ($r \approx 198\text{ px}$) $\implies$ Trường nhìn lệch hẳn một bên, sinh ra vô số box méo mó, chất lượng kém.

     - **Cách giải quyết cấp 1 (FCOS — Nhánh Center-ness):**
       FCOS bổ sung thêm một nhánh Convolution song song chỉ để dự đoán độ "gần tâm" của điểm:
       $$\text{centerness}^* = \sqrt{\frac{\min(l, r)}{\max(l, r)} \times \frac{\min(t, b)}{\max(t, b)}}$$
       - Điểm ngay tâm ($l=r, t=b$): $\text{centerness}^* = \sqrt{1 \times 1} = \mathbf{1.0}$.
       - Điểm sát mép ($l \to 0$): $\text{centerness}^* \to \mathbf{0.0}$.
       - Nhánh này được huấn luyện bằng hàm **BCE Loss**.
       - **Khi chạy suy luận (Inference):**
         $$\text{Điểm số cuối cùng} = \text{Classification Score} \times \text{Centerness Score}$$
         Các box từ điểm gần rìa dù có điểm phân loại cao nhưng nhân với centerness $\approx 0$ sẽ bị kéo tụt điểm xuống $\implies$ Thuật toán NMS tự động loại bỏ các box rác này, chỉ giữ lại box từ điểm gần tâm.

     - **Cách giải quyết cấp 2 hiện đại (YOLOv8/v11 — Task-Aligned Assigner: TAL):**
       - *Tại sao Centerness vẫn chưa tối ưu?* FCOS dùng **gán nhãn tĩnh theo hình học (Static Assignment)**: cứ điểm nào nằm trong box là coi như Positive, bất kể mạng có học tốt đặc trưng ở điểm đó hay không. Điều này khiến nhánh Phân loại (Classification) và Định vị (Regression) bị **lệch pha (misaligned)**: có điểm đoán đúng class nhưng box bị lệch, có điểm vẽ box rất đẹp nhưng class score lại thấp.
       - *Bước nhảy vọt của TAL:* Chuyển sang **Gán nhãn động (Dynamic Assignment)**. Mô hình cho mạng chạy forward trước, sau đó tính chỉ số căn chỉnh chất lượng (Alignment Metric $t$):
         $$t = s^\alpha \times \text{IoU}^\beta$$
         *(trong đó $s$ là xác suất phân loại đúng nhãn, $\text{IoU}$ là độ trùng khớp của box vừa dự đoán với Ground Truth, $\alpha=0.5, \beta=6.0$)*.
       - Giá trị $\beta=6.0$ rất lớn thể hiện sự ưu tiên cực cao cho độ chính xác hình học $\text{IoU}$.
       - TAL chọn top $K$ điểm có chỉ số $t$ cao nhất làm **Positive Samples**, phần còn lại gán làm **Negative**. Nhờ đó, mạng chỉ tập trung tối ưu cho những vị trí **vừa nhận diện chuẩn nhãn, vừa vẽ box khít tuyệt đối**!

     ##### 3. Bài toán 3: "Ranh giới bất định" & Bản chất toán học của DFL (Distribution Focal Loss)
     - **Hiện tượng:** Trong thực tế, ranh giới của vật thể không phải lúc nào cũng là 1 đường viền sắc nét (ví dụ: vết rỉ sét loang lổ, khói bụi, đường nứt vi mô, vật thể bị che khuất một phần bởi bóng râm). Ngay cả con người khi gắn nhãn (annotator) cũng có sự sai lệch vài pixel ở mép.
     - **Nhược điểm của cách cũ (Dirac Delta):** Bắt mạng nơ-ron phải hồi quy chính xác một con số thực duy nhất $l = 80.4\text{ px}$. Việc ép mạng đoán một số tuyệt đối khi dữ liệu có độ nhòe tự nhiên làm gradient bị dao động mạnh và mô hình học rất cứng nhắc.

     - **Bản chất toán học của DFL trong YOLOv8/v11:**
       Thay vì ép đoán 1 con số thực, ta coi khoảng cách $y$ là một **biến ngẫu nhiên có phân phối xác suất liên tục**.
       1. **Rời rạc hóa:** Chia dải khoảng cách thành $16$ mốc nguyên (từ $0$ đến $15$ theo đơn vị stride):
          $$\mathcal{Y} = \{0, 1, 2, \dots, 15\}$$
       2. **Dự đoán phân phối (Softmax):** Mạng nơ-ron xuất ra $16$ giá trị logit cho mỗi cạnh, sau đó đi qua hàm **Softmax** để tạo thành một phân phối xác suất $\{P_0, P_1, \dots, P_{15}\}$ với $\sum P_i = 1$.
       3. **Tính toạ độ bằng Kỳ vọng toán học (Expectation):**
          $$\hat{y} = \sum_{i=0}^{15} P_i \times i = P_0 \times 0 + P_1 \times 1 + \dots + P_{15} \times 15$$
       4. **Hàm mất mát DFL (DFL Loss):**
          Giả sử giá trị nhãn thực tế $y$ rơi vào khoảng giữa 2 số nguyên $y_i \le y \le y_{i+1}$ (ví dụ $y = 7.4$ nằm giữa mốc $7$ và $8$). Hàm DFL ép phân phối xác suất dồn sự tự tin tối đa về 2 mốc lân cận này:
          $$\text{DFL}(P_i, P_{i+1}) = - \Big( (y_{i+1} - y) \log(P_i) + (y - y_i) \log(P_{i+1}) \Big)$$
          *(Với $y = 7.4 \implies$ mốc $7$ chịu trọng số $8 - 7.4 = 0.6$, mốc $8$ chịu trọng số $7.4 - 7 = 0.4$)*.

     ```text
     Xác suất P_i
       1.0 ┼               ▲ (Đỉnh phân phối nhọn: Mép vật thể rõ ràng, độ tự tin cao)
           │              │ │
       0.6 ┼             ┌┴─┴┐
           │             │   │
       0.2 ┼       ┌─┐   │   │   ┌─┐  (Phân phối bẹt: Mép vật thể bị mờ/che khuất)
       0.0 ┼───┬───┴─┴───┴───┴───┴─┴───┬───► Mốc giá trị rời rạc (0 -> 15)
           0   2    4    6   7 8   10  12  15
     ```

     > 🚀 **Ưu thế vượt bậc của DFL:** 
     > - Thể hiện trọn vẹn **sự bất định (uncertainty)**: Khi mép rõ, phân phối tạo đỉnh nhọn; khi mép mờ/che khuất, phân phối dàn phẳng tự nhiên.
     > - Tăng từ $1 - 2\%$ mAP trên benchmark COCO mà **hoàn toàn không làm chậm tốc độ suy luận (Inference FPS)**, vì lúc chạy thực tế chỉ cần một phép nhân ma trận trọng số $\sum P_i \times i$ siêu nhẹ!

| Tiêu chí so sánh | Anchor-Based (YOLOv2 - YOLOv7) | Anchor-Free (YOLOv8, YOLOv11, FCOS) |
| :--- | :--- | :--- |
| **Bản chất dự đoán** | Dự đoán độ lệch offset $(\Delta x, \Delta y, \Delta w, \Delta h)$ so với box mẫu có sẵn | Dự đoán trực tiếp khoảng cách $(l, t, r, b)$ từ điểm pixel đến 4 cạnh |
| **Số lượng siêu tham số** | Nhiều (số cụm $k$, tỷ lệ khung hình, ngưỡng IoU matching) | Rất ít (không cần chọn kích thước/tỷ lệ box trước) |
| **Tự động thích nghi dữ liệu** | Phải chạy lại K-Means khi sang tập dữ liệu mới | Tự động thích nghi với mọi bài toán/kích cỡ vật thể |
| **Số lượng box ứng viên** | Rất lớn (khoảng 25,000 - 80,000 anchors/ảnh) | Nhỏ gọn hơn (chỉ bằng số lượng điểm grid feature map) |
| **Mất cân bằng mẫu** | Cực kỳ nặng nề ($>99.9\%$ anchor là nền âm tính) | Giảm thiểu đáng kể nhờ cơ chế Center/TAL sampling |
| **Xử lý vật thể dị dạng** | Kém (khó khớp với vật thể siêu dài, siêu hẹp, hình que) | Xuất sắc (linh hoạt thích ứng mọi hình thù bất đối xứng) |

### B1c. IoU (Intersection over Union) — Bản chất, công thức & code

**IoU** (Giao trên Hợp) là thước đo định lượng mức độ trùng khớp giữa **2 hình chữ nhật** (thường là Bounding Box do Model dự đoán vs Ground-Truth thực tế).

$$\text{IoU} = \frac{\text{Diện tích phần Giao (Intersection)}}{\text{Diện tích phần Hợp (Union)}} = \frac{|A \cap B|}{|A \cup B|}$$

```text
    Box A (Model)          Box B (Ground Truth)
   ┌───────────┐
   │           │
   │      ┌────┼──────┐
   │      │████│      │   <--- Phần tô đen ████ là GIAO (Intersection = A ∩ B)
   └──────┼────┘      │
          │           │
          └───────────┘
   Toàn bộ diện tích bao phủ bởi cả 2 box là HỢP (Union = Area(A) + Area(B) - Intersection)
```

- $\text{IoU} = 0$: Hai box hoàn toàn không chạm nhau.
- $0 < \text{IoU} < 1$: Hai box giao nhau một phần.
- $\text{IoU} = 1$: Hai box trùng khít hoàn hảo $100\%$.

---

#### 1. Cách xác định tọa độ hình chữ nhật phần Giao (Intersection)

Quy ước mỗi box lưu 4 số: `(x1, y1, x2, y2)` tương ứng góc **Trên-Trái** $(x_1, y_1)$ và **Dưới-Phải** $(x_2, y_2)$ trong hệ tọa độ ảnh ($x$ tăng dần sang phải, $y$ tăng dần đi xuống).

Khi 2 box $A$ và $B$ đè lên nhau, phần giao nhau cũng là một hình chữ nhật:
1. **Góc trên-trái của phần giao:** phải bị ép vào trong, tức là lấy giá trị lớn hơn:
   $$x_{\text{inter1}} = \max(x_1^A, x_1^B), \quad y_{\text{inter1}} = \max(y_1^A, y_1^B)$$
2. **Góc dưới-phải của phần giao:** phải bị chặn lại ở mép gần hơn, tức là lấy giá trị nhỏ hơn:
   $$x_{\text{inter2}} = \min(x_2^A, x_2^B), \quad y_{\text{inter2}} = \min(y_2^A, y_2^B)$$
3. **Kích thước phần giao:**
   - Chiều rộng $= x_{\text{inter2}} - x_{\text{inter1}}$
   - Chiều cao $= y_{\text{inter2}} - y_{\text{inter1}}$
   - **Tại sao cần `max(0, ...)`?** Nếu 2 box nằm tách rời nhau (không giao nhau), mép phải sẽ nhỏ hơn mép trái ($x_{\text{inter2}} < x_{\text{inter1}}$), phép trừ ra số âm. Lấy $\max(0, \dots)$ để diện tích giao tự động bằng $0$ thay vì âm $\times$ âm thành dương!

---

#### 2. Cách tính diện tích phần Hợp (Union)

Theo nguyên lý bù trừ tập hợp:
$$\text{Union} = \text{Area}_A + \text{Area}_B - \text{Intersection}$$
*(Phải trừ đi `Intersection` 1 lần vì khi cộng Area A và Area B thì phần giao nhau đã bị tính lặp 2 lần).*

---

#### 3. Ví dụ tính tay từng bước

Cho 2 box:
- $\text{Box}_1 = (0, 0, 10, 10) \rightarrow \text{Rộng} = 10, \text{Cao} = 10 \rightarrow \text{Area}_1 = 10 \times 10 = 100$
- $\text{Box}_2 = (5, 5, 15, 15) \rightarrow \text{Rộng} = 10, \text{Cao} = 10 \rightarrow \text{Area}_2 = 10 \times 10 = 100$

**Bước 1: Tìm phần giao**
- $x_{\text{inter1}} = \max(0, 5) = 5$
- $y_{\text{inter1}} = \max(0, 5) = 5$
- $x_{\text{inter2}} = \min(10, 15) = 10$
- $y_{\text{inter2}} = \min(10, 15) = 10$
- $\text{Rộng giao} = \max(0, 10 - 5) = 5$
- $\text{Cao giao} = \max(0, 10 - 5) = 5$
- $\text{Intersection} = 5 \times 5 = 25$

**Bước 2: Tìm phần hợp**
- $\text{Union} = \text{Area}_1 + \text{Area}_2 - \text{Intersection} = 100 + 100 - 25 = 175$

**Bước 3: Tính IoU**
- $\text{IoU} = \frac{25}{175} = \frac{1}{7} \approx 0.1429$

---

#### 4. Code chuẩn Python

```python
def compute_iou(box1, box2):
    """
    box = (x1, y1, x2, y2)
    x1, y1: tọa độ góc trên-trái (top-left)
    x2, y2: tọa độ góc dưới-phải (bottom-right)
    """
    # 1. Tọa độ góc trên-trái & dưới-phải của hình chữ nhật giao nhau
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 2. Diện tích phần giao (max với 0 để tránh số âm khi 2 box không chạm nhau)
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    # 3. Diện tích từng box
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 4. Diện tích phần hợp
    union_area = area1 + area2 - inter_area

    # 5. IoU = Giao / Hợp (tránh chia cho 0)
    return inter_area / union_area if union_area > 0 else 0.0

# Kiểm tra thử với ví dụ trên
print(compute_iou((0, 0, 10, 10), (5, 5, 15, 15)))  # Output: ~0.142857 (tức 1/7)
```

**Quy ước trong Object Detection:**
- Box dự đoán được coi là **True Positive (TP)** khi có $\text{IoU} \ge \text{threshold}$ với ground-truth (thường lấy ngưỡng $\text{IoU} \ge 0.5$ cho chuẩn PASCAL VOC, hoặc tính trung bình trên dải $0.5:0.95$ cho chuẩn COCO).

### B1d. Non-Max Suppression (NMS) — xử lý box trùng lặp

Một vật thể thực tế thường bị model dự đoán bởi **nhiều bounding box chồng chéo nhau** (chỉ xê dịch vài pixel). Mục tiêu của NMS là: **Chỉ giữ lại 1 box tốt nhất cho mỗi vật thể và triệt tiêu (suppress) các box còn lại bị trùng**.

#### Thuật toán NMS diễn giải chi tiết từng bước:

```text
Giả sử ban đầu ta có danh sách các box kèm điểm số confidence (sau khi đã lọc bỏ bớt các box có score < conf_threshold):

1. Sắp xếp danh sách box theo thứ tự Confidence giảm dần.
2. Lấy ra box có điểm cao nhất hiện tại (gọi là box A) -> chắc chắn giữ lại: đưa box A vào danh sách kết quả `keep`.
3. So sánh box A với từng box còn lại trong danh sách:
   - Tính IoU(box A, box còn lại).
   - Nếu IoU >= iou_threshold: chứng tỏ box này đè lên box A quá nhiều (cùng chỉ một vật) nhưng điểm thấp hơn -> LOẠI BỎ (triệt tiêu).
   - Nếu IoU < iou_threshold: chứng tỏ box này nằm ở vị trí khác (vật thể khác) -> TIẾP TỤC GIỮ TRONG DANH SÁCH XÉT.
4. Lặp lại bước 2 & 3 với box có điểm cao nhất tiếp theo trong số các box còn lại, cho tới khi danh sách rỗng.
```

> **Ví dụ trực quan:**
> Ảnh có 2 chú chó (Chó 1 bên trái, Chó 2 bên phải):
> - Model đưa ra: Box 1 (conf 0.92, bao Chó 1), Box 2 (conf 0.75, bao Chó 1), Box 3 (conf 0.88, bao Chó 2).
> - **Vòng 1:** Chọn Box 1 (cao nhất: 0.92) vào `keep`.
>   - So Box 1 với Box 2: `IoU = 0.78 >= 0.5` $\rightarrow$ **Loại Box 2** (trùng Chó 1).
>   - So Box 1 với Box 3: `IoU = 0.02 < 0.5` $\rightarrow$ **Giữ Box 3** (ở vị trí khác).
> - **Vòng 2:** Danh sách còn lại chỉ có Box 3. Lấy Box 3 vào `keep`. Hết danh sách.
> - **Kết quả:** `keep = [Box 1, Box 3]`, chuẩn xác 2 con chó, loại được box thừa Box 2.

```python
def nms(boxes, scores, iou_threshold=0.5):
    """boxes: list of (x1,y1,x2,y2), scores: list of confidence"""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if compute_iou(boxes[i], boxes[j]) < iou_threshold]
    return keep
```

**Tham số quan trọng:** `NMS_threshold` (thường 0.4-0.5) và **`confidence_threshold`** (thường 0.25-0.5). Trong QC, nếu bỏ sót lỗi đắt tiền hơn báo nhầm → hạ confidence threshold để ưu tiên recall (xem mAP & precision/recall dưới).

---

## B2. mAP — Giải mã Thước đo Đánh giá Object Detection (Từ Trực giác đến Bản chất)

### 💡 1. Tại sao không thể dùng Accuracy? Và tại sao lại có tới 2 chữ "Trung bình" (Mean & Average)?

Trong bài toán phân loại ảnh (Classification): 1 ảnh $\to$ 1 nhãn $\to$ Tính $\text{Accuracy} = \frac{\text{Số ảnh đoán đúng}}{\text{Tổng số ảnh}}$. Rất đơn giản!

Nhưng trong **Object Detection**, mọi chuyện phức tạp hơn gấp 10 lần:
- Một ảnh có thể có 0, 1, hoặc hàng chục vật thể.
- Mô hình vừa phải **đoán đúng loại vật thể (Class)**, vừa phải **vẽ đúng vị trí (Bounding Box)**, vừa có **độ tự tin (Confidence Score)**.
- Nếu mô hình vẽ một hộp lệch 3 pixel thì có tính là đúng không? Nếu mô hình vẽ 100 cái hộp đè lên cùng 1 vật thể thì tính điểm thế nào?

Để giải quyết, người ta tạo ra thước đo **mAP (Mean Average Precision)**. Tên gọi này có 2 tầng "trung bình" lồng vào nhau:
1. **Average Precision (AP):** Điểm số trung bình hiệu năng của mô hình trên **MỘT LỚP** cụ thể (khi quét qua mọi ngưỡng Confidence).
2. **Mean Average Precision (mAP):** Lấy trung bình cộng điểm AP của **TẤT CẢ CÁC LỚP** trong tập dữ liệu.

---

### 🎛️ 2. Ẩn dụ "Cái núm xoay Confidence" — Nguồn gốc của đường cong Precision-Recall (P-R Curve)

Rất nhiều bạn học thuộc công thức tính AP nhưng không hiểu: *"Tại sao lại vẽ được đường cong Precision-Recall?"*

Hãy tưởng tượng mô hình của bạn có một **chiếc núm xoay Confidence Threshold** (vặn từ $1.0$ xuống $0.0$):
- **Khi vặn núm ở mức cực cao (ví dụ $0.95$):** 
  Mô hình chỉ giữ lại những box nó chắc chắn $100\%$. 
  $\implies$ Hầu như không có box nào báo nhầm $\implies$ **Precision chạm đỉnh ($100\%$)**. 
  $\implies$ Nhưng nó bỏ sót gần hết vật thể khó $\implies$ **Recall lẹt đẹt ($15\%$)**.
- **Khi vặn núm xuống mức trung bình (ví dụ $0.50$):** 
  Mô hình chấp nhận thêm các box kém tự tin hơn. 
  $\implies$ Bắt được thêm nhiều vật thể $\implies$ **Recall tăng lên ($60\%$)**. 
  $\implies$ Nhưng bắt đầu có một vài box báo nhầm $\implies$ **Precision giảm nhẹ ($85\%$)**.
- **Khi vặn núm xuống mức cực thấp (ví dụ $0.05$):** 
  Mô hình giữ lại tất cả mọi box, kể cả nghi ngờ mơ hồ. 
  $\implies$ Bắt trọn vẹn gần như mọi vật thể $\implies$ **Recall chạm đỉnh ($95\%$)**. 
  $\implies$ Nhưng báo nhầm và rác tràn ngập ảnh $\implies$ **Precision rơi tự do ($25\%$)**.

> 📌 **Kết luận cốt lõi:** Mỗi vị trí vặn núm cho ra một cặp số $(\text{Recall}, \text{Precision})$. Khi ta vặn núm trượt liên tục từ $1.0$ về $0.0$, tập hợp các điểm đó tạo thành **Đường cong Precision-Recall (P-R Curve)**!
> 
> **AP (Average Precision)** chính là **toàn bộ diện tích nằm dưới đường cong P-R đó (Area Under Curve - AUC)**. Nếu đường cong này càng phồng to về góc trên-phải (Precision cao mà Recall cũng cao), AP càng tiệm cận $1.0$ ($100\%$).

---

### ⚖️ 3. Trọng tài IoU & Luật tính điểm TP, FP, FN

Để biết một box dự đoán là "Đúng" (TP) hay "Sai" (FP), ta cần một thước đo làm trọng tài: **Ngưỡng $\text{IoU}_{\text{threshold}}$** (thường chọn $0.5$):

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. TRUE POSITIVE (TP)       2. FALSE POSITIVE (FP - Lệch)  3. FALSE POSITIVE (FP - Trùng)│
│    ┌──────────────┐            ┌──────────────┐               ┌──────────────┐      │
│    │  GT (Thật)   │            │  GT (Thật)   │               │  GT (Thật)   │      │
│    │  ┌───────────┼──┐         └──────────────┘               │ ┌──────────┐ │      │
│    │  │ Dự đoán   │  │             ┌──────────────┐           │ │Pred 1(TP)│ │      │
│    └──┼───────────┘  │             │Dự đoán (FP)  │           │ └──────────┘ │      │
│       └──────────────┘             │(IoU < 0.5)   │           │ ┌──────────┐ │      │
│     (IoU ≥ 0.5 & Đúng class)       └──────────────┘           │ │Pred 2(FP)│ │      │
│                                                               └─┴──────────┴─┘      │
│                                                               (Trùng GT -> Phạt FP) │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. FALSE NEGATIVE (FN - Bỏ sót lỗi)                                         │
│    ┌──────────────┐                                                         │
│    │  GT (Thật)   │  <--- Bỏ sót hoàn toàn! Không có box nào phủ qua.        │
│    └──────────────┘       (Trong kiểm tra lỗi công nghiệp: Đây là lỗi nặng nhất!)│
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Luật "1 chọi 1" (One-to-One Matching) chống gian lận:
Một vật thể thật (Ground Truth - GT) **chỉ được phép ghép cặp với duy nhất 1 box dự đoán có Confidence cao nhất** để tính là TP. Nếu mô hình vẽ thêm 5 box khác cùng bao quanh vật thể đó, cả 5 box thừa đều bị phạt là **False Positive (FP)**!

| Chỉ số | Công thức | Ý nghĩa thực tế trong nhà máy / QC |
| :--- | :--- | :--- |
| **Precision** (Độ chuẩn xác) | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | Trong 100 lần model kêu "Có lỗi!", có bao nhiêu lần là lỗi thật? (Tránh báo động giả gây tốn công kiểm tra lại). |
| **Recall** (Độ bao phủ) | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | Trong 100 vết lỗi thực tế trên dây chuyền, model bắt được bao nhiêu vết? (Tránh để lọt phế phẩm ra thị trường). |

---

### 🔢 4. Ví dụ tính toán AP từng bước bằng bảng số cụ thể

Giả sử trên tập test, lớp "Vết nứt" (*crazing*) có tổng cộng **$\text{Total GT} = 5$ vật thể thật**. Mô hình dự đoán ra 6 box:

1. **Bước 1:** Xếp các box giảm dần theo Confidence.
2. **Bước 2:** So khớp IoU với vật thể thật để gán TP hoặc FP.
3. **Bước 3:** Tính cộng dồn $\text{Acc\_TP}$ và $\text{Acc\_FP}$, từ đó tính Precision và Recall tại từng dòng:

| Thứ hạng | Box ID | Confidence | Kết quả (IoU ≥ 0.5) | $\text{Acc\_TP}$ (Cộng dồn) | $\text{Acc\_FP}$ (Cộng dồn) | $\text{Precision} = \frac{\text{Acc\_TP}}{\text{Acc\_TP} + \text{Acc\_FP}}$ | $\text{Recall} = \frac{\text{Acc\_TP}}{\text{Total GT = 5}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Box A | **$0.95$** | **TP** | 1 | 0 | $1 / 1 = \mathbf{1.00}$ ($100\%$) | $1 / 5 = \mathbf{0.20}$ ($20\%$) |
| **2** | Box B | **$0.88$** | **TP** | 2 | 0 | $2 / 2 = \mathbf{1.00}$ ($100\%$) | $2 / 5 = \mathbf{0.40}$ ($40\%$) |
| **3** | Box C | **$0.72$** | **FP** (lệch vị trí) | 2 | 1 | $2 / 3 = \mathbf{0.67}$ ($67\%$) | $2 / 5 = \mathbf{0.40}$ ($40\%$) |
| **4** | Box D | **$0.65$** | **TP** | 3 | 1 | $3 / 4 = \mathbf{0.75}$ ($75\%$) | $3 / 5 = \mathbf{0.60}$ ($60\%$) |
| **5** | Box E | **$0.50$** | **FP** (trùng lặp) | 3 | 2 | $3 / 5 = \mathbf{0.60}$ ($60\%$) | $3 / 5 = \mathbf{0.60}$ ($60\%$) |
| **6** | Box F | **$0.30$** | **TP** | 4 | 2 | $4 / 6 = \mathbf{0.67}$ ($67\%$) | $4 / 5 = \mathbf{0.80}$ ($80\%$) |

*(GT thứ 5 không được box nào bắt trúng $\implies \text{FN} = 1$, Recall tối đa đạt $80\%$)*.

#### Kỹ thuật All-Point Interpolation (Làm mượt đường cong theo chuẩn COCO):
Đường P-R thực tế đi giật cục hình răng cưa. Chuẩn COCO làm mượt bằng cách: Tại mỗi mức Recall $r$, thay Precision bằng giá trị Precision lớn nhất ở bên phải nó ($p_{\text{interp}}(r) = \max_{\tilde{r} \ge r} p(\tilde{r})$). 

```text
Precision
  1.0 ┼──────┐ p_interp
      │      │          (Đường cong răng cưa gốc)
  0.8 ┼      └───┐      ▲
      │          └───┐  │
  0.6 ┼              └──┴───────┐
      │                         │
  0.4 ┼                         │
      │                         │
  0.0 ┼──────┬──────┬──────┬────┴─┬──────► Recall
     0.0    0.2    0.4    0.6    0.8    1.0
```

Diện tích phần dưới đường cong làm mượt này chính là **$\text{AP}$ (Average Precision) của lớp đó**:
$$\text{AP} = \int_{0}^{1} p_{\text{interp}}(r) \, dr$$

---

### 🏷️ 5. Phân biệt các biến thể mAP khi đi phỏng vấn

**mAP** là trung bình cộng giá trị **AP** của toàn bộ $C$ lớp đối tượng:
$$\mathbf{mAP} = \frac{1}{C} \sum_{c=1}^C \text{AP}_c$$

| Ký hiệu | Tên gọi chuẩn | Cách tính toán | Ý nghĩa thực tế |
| :--- | :--- | :--- | :--- |
| **mAP@0.5** *(mAP50)* | Pascal VOC Metric | Cố định ngưỡng $\text{IoU} \ge 0.5$. Chỉ cần box bao quanh trúng $\ge 50\%$ diện tích là tính điểm. | Dễ đạt điểm cao, dùng để so sánh nhanh; phù hợp bài toán chỉ cần biết "vật thể ở quanh đây". |
| **mAP@0.75** *(mAP75)* | Strict Metric | Cố định ngưỡng $\text{IoU} \ge 0.75$. Đòi hỏi hộp bọc phải rất khít với vật thể. | Quan trọng cho cánh tay robot gắp vật thể (cần toạ độ chính xác cao). |
| **mAP@0.5:0.95** *(COCO mAP)* | **Tiêu chuẩn vàng COCO** | Tính mAP tại 10 mốc IoU khác nhau: $[0.50, 0.55, 0.60, \dots, 0.95]$ rồi lấy **trung bình cộng của cả 10 mốc**. | **Khắt khe & toàn diện nhất:** Đánh giá cả khả năng nhận diện lẫn độ sắc nét từng pixel của box. |
| **$\text{mAP}_S, \text{mAP}_M, \text{mAP}_L$** | Metric theo kích thước | Chia theo diện tích: Small ($< 32^2\text{ px}$), Medium ($32^2 - 96^2\text{ px}$), Large ($> 96^2\text{ px}$). | Cực kỳ quan trọng trong nhà máy để kiểm tra model có bắt được **lỗi siêu nhỏ** ($\text{mAP}_S$) hay không. |

---

### 💻 6. Code tính toán nhanh bằng `torchmetrics`

```python
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# 1. Khởi tạo metric theo chuẩn COCO
metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

# 2. Dữ liệu dự đoán từ Model (Predictions)
preds = [
    {
        "boxes": torch.tensor([[100.0, 150.0, 300.0, 270.0], [50.0, 60.0, 120.0, 180.0]]),
        "scores": torch.tensor([0.92, 0.65]),
        "labels": torch.tensor([0, 1]), # 0: crazing, 1: scratches
    }
]

# 3. Nhãn thực tế (Ground Truth)
targets = [
    {
        "boxes": torch.tensor([[100.0, 150.0, 300.0, 270.0], [45.0, 55.0, 125.0, 175.0]]),
        "labels": torch.tensor([0, 1]),
    }
]

# 4. Tính toán kết quả
metric.update(preds, targets)
results = metric.compute()

print(f"COCO mAP (0.5:0.95) : {results['map'].item():.4f}")
print(f"VOC mAP@50          : {results['map_50'].item():.4f}")
print(f"mAP Small (lỗi nhỏ) : {results['map_small'].item():.4f}")
```

---

### 🏭 7. Chiến lược thực chiến trong Kiểm định Chất lượng (QC)

> ⚠️ **Đừng để điểm mAP "đánh lừa" trên dây chuyền sản xuất!**
> 1. **Chi phí Bỏ sót lỗi (FN) $\gg$ Chi phí Báo nhầm (FP):**
>    - Để lọt 1 thanh thép nứt ra thị trường $\implies$ Nguy cơ sập công trình, kiện tụng tiền tỷ, mất uy tín.
>    - Báo nhầm 1 thanh thép lành là có lỗi $\implies$ Công nhân kiểm tra lại mất 3 giây.
> 2. **Hành động kỹ thuật:**
>    - Khi deploy lên camera nhà máy, hãy **chủ động hạ `confidence_threshold` từ $0.5$ xuống $0.20 - 0.25$**.
>    - Chấp nhận Precision giảm nhẹ (chấp nhận thêm một ít báo động giả) để **đẩy Recall lên tối đa $\ge 98\%$**, đảm bảo không một phế phẩm nào lọt qua dây chuyền!

---

## B3. One-stage vs Two-stage detectors

| Tiêu chí | Two-stage (VD: Faster R-CNN) | One-stage (VD: YOLO) |
|---|---|---|
| Cách hoạt động | Bước 1: RPN (Region Proposal Network) tìm ~2000 vùng nghi ngờ; Bước 2: phân loại + refine box từng vùng | Duyệt **một lần**: chia ảnh thành lưới, mỗi ô dự đoán box + class trực tiếp |
| Độ chính xác (vị trí) | Cao hơn (có bước refine) | Thấp hơn chút nhưng đủ cho phần lớn ứng dụng |
| **Tốc độ** | Chậm (khó realtime) | **Rất nhanh** (hàng chục-hàng trăm FPS) |
| Độ phức tạp triển khai | Cao | **Thấp — API đơn giản** |
| Dùng khi nào | Cần độ chính xác vị trí tối đa, không chạy realtime | **Realtime trên line nhà máy, QC** ✅ |

**Kết luận thực dụng cho QC:** **YOLO (one-stage)** gần như luôn được chọn vì vừa nhanh vừa đủ chính xác cho phát hiện lỗi trên băng chuyền. Faster R-CNN dùng khi yêu cầu box/vị trí cực kỳ chính xác và không cần realtime.

- **R-CNN / Fast R-CNN / Faster R-CNN:** dòng két-stage; Faster R-CNN là phiên bản gộp toàn pipeline (RPN + classifier) train end-to-end — ít nhất bạn cần hiểu ý tưởng để trả lời phỏng vấn, nhưng không cần tự code.

---

## B4. YOLO — vận hành model detection phổ biến nhất

### B4a. Tư duy "chia lưới" của YOLO

YOLO chia ảnh thành lưới SxS ô. Mỗi ô chịu trách nhiệm dự đoán các box có tâm nằm trong ô đó. Mỗi box gồm: `(Δx, Δy, Δw, Δh, objectness, class_probs)`.

- **Objectness**: xác suất ô đó có đối tượng ở đây không.
- Các phiên bản liên tục phát triển: **YOLOv8** (bản ổn định phổ biến nhất hiện nay, dùng `ultralytics`), **YOLOv11** (bản mới nhất, model `yolo11`).

### B4b. Cài đặt & workflow thực tế với Ultralytics

```bash
pip install ultralytics
```

**Dataset format:** mỗi ảnh đi kèm 1 file `.txt` cùng tên; mỗi dòng: `class_id x_center y_center width height` (tất cả chuẩn hoá 0-1).

```
# images/train/001.jpg  +  labels/train/001.txt
# 001.txt:
# 0 0.512 0.330 0.120 0.245   -> class 0 (scratch), box ở giữa, w=12%, h=24.5%
```

Cấu trúc thư mục chuẩn YOLO (cần biết để train):

```
dataset/
├── images/
│   ├── train/  *.jpg
│   └── val/    *.jpg
├── labels/
│   ├── train/  *.txt   (1 file trên 1 ảnh)
│   └── val/    *.txt
└── data.yaml   (config: đường dẫn + danh sách class)
```

```yaml
# data.yaml
train: dataset/images/train
val:   dataset/images/val
nc: 2
names: ['scratch', 'dent']
```

**Train:**

```python
from ultralytics import YOLO

# Load pretrained YOLOv8n (n = nano, nhẹ nhất, phù hợp dữ liệu nhỏ + realtime)
model = YOLO('yolov8n.pt')
results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,          # 'cpu' nếu không có GPU
    patience=20,       # early stopping nếu val không cải thiện
)
```

**Evaluate & inference:**

```python
metrics = model.val(data='data.yaml')
print(metrics.box.map)        # mAP@0.5:0.95
print(metrics.box.map50)      # mAP@0.5
print(metrics.box.maps)       # AP từng lớp

results = model.predict('test_image.jpg', conf=0.25, save=True)
# conf=0.25: bỏ box dưới 25% confidence -> điều chỉnh để balance precision/recall
```

**Export để triển khai nhà máy (rất quan trọng):**

```python
# Export sang ONNX/TensorRT để chạy realtime nhanh trên line
model.export(format='onnx')          # portable, chạy mọi nơi
# model.export(format='engine')      # TensorRT - tối ưu tốc độ trên GPU
```

### B4c. Data annotation — gán nhãn dữ liệu

**COCO format** — định dạng annotation chuẩn (JSON):

```json
{
  "images": [{"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480}],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1,
     "bbox": [100, 80, 50, 40], "area": 2000}   // [x,y,w,h]
  ],
  "categories": [{"id": 1, "name": "scratch"}]
}
```

**Công cụ gán nhãn — ưu tiên cho QC:**

| Công cụ | Đặc điểm | Chọn khi nào |
|---|---|---|
| **CVAT** | Open-source, deploy nội bộ (không lộ dữ liệu ra ngoài), hỗ trợ team, export YOLO/COCO | **Chọn cho QC** — dữ liệu nhà máy cần ở nội bộ |
| Roboflow | Đám mây, tiện, có sẵn augmentation + dataset versioning | Dữ liệu không nhạy cảm, cần tiện nhanh |

**Thực hành tốt khi gán nhãn QC:**
1. Vẽ box **sát** vật thể, không thừa background nhiều — ảnh hưởng IoU/mAP.
2. Thống nhất 1 người/1 tiêu chuẩn độ chính xác của box (nhiều người hay lệch nhau).
3. Luôn giữ **training set riêng / validation set riêng** (không trùng ảnh).
4. Ghi chú các trường hợp khó (lỗi mờ, lỗi gần biên) — vì model thường tái hiện đúng độ khó đó.

### B4d. Chú ý về transfer learning trong detection

- YOLO pretrained trên COCO (80 lớp) **không có** lớp "vết lỗi nhà máy" của bạn → bạn **bắt buộc phải fine-tune** trên bộ dữ liệu lỗi của chính mình.
- Khởi đầu từ weights pretrained (`yolov8n.pt`) học **nhanh và chính xác hơn nhiều** so với train từ đầu — tận dụng backbone đã biết nhận diện cạnh/đặc trưng chung.
- Dữ liệu QC thường ít → dùng model nhỏ (`n`/`s`) trước, tăng lên (`m`/`l`) chỉ khi cần chính xác hơn.

---

## B5. Checklist lỗi thường gặp (Phần B)

- [ ] YOLO train không nhận format dataset → kiểm tra lại cấu trúc `images/` + `labels/`, `data.yaml` đúng đường dẫn & số lớp `nc`
- [ ] Tất cả dự đoán conf rất thấp / không ra box → `imgsz` quá nhỏ so với vật thể nhỏ, hoặc thiếu fine-tune (model chưa biết lớp lỗi)
- [ ] Quá nhiều box trùng cho 1 vật → NMS threshold quá cao; giảm `NMS_threshold`
- [ ] Bỏ sót lỗi (FN cao) → tăng model size, tăng dữ liệu lỗi, hoặc hạ confidence threshold để ưu tiên recall
- [ ] Dataset có quá ít ảnh lỗi so với ảnh ok → lỗi lớp mất cân bằng; cần cân bằng kỹ trước khi train
- [ ] Box không sát vật thể → lỗi annotation; xem lại chất lượng gán nhãn, không phải lỗi model

---

## B6. Bài tập thực hành đề xuất (Phần B — gắn domain QC)

1. **Tự code IoU + NMS** (mục B1b/B1c) và unit test nhanh trên vài box giả.
2. **Gán nhãn 1 bộ mini**: tải/vẽ 50-100 ảnh lỗi giả lập, gán nhãn bằng CVAT (deploy local) hoặc Roboflow, export sang format YOLO.
3. **Train YOLOv8n** trên bộ mini đó (dùng data.yaml chuẩn), chạy 30-50 epochs.
4. **Đánh giá**: đọc `metrics.box.map50`, in precision/recall, hiển thị vài ảnh predict có box dự đoán + confidence.
5. **Tune threshold**: thay đổi `conf`, quan sát trade-off precision/recall, chọn giá trị phù hợp chi phí FN của bạn.

---

## B7. Câu hỏi tự kiểm tra (chuẩn bị sớm cho phỏng vấn)

- IoU dùng để làm gì? Vẽ hình minh hoạ 2 box IoU=0 và IoU=1.
- NMS xử lý vấn đề gì? Giải thích các bước thuật toán.
- One-stage (YOLO) và two-stage (Faster R-CNN) khác nhau thế nào? Khi nào chọn cái nào cho QC?
- mAP tính như thế nào? mAP@0.5 và mAP@0.5:0.95 khác nhau ý nghĩa ra sao?
- Precision cao nhưng Recall thấp nghĩa là gì trong bối cảnh detection lỗi nhà máy? Nên ưu tiên cái nào?
- YOLO pretrained trên COCO, vì sao vẫn phải fine-tune trên dữ liệu lỗi của mình?
- COCO format là gì? Gồm những trường (`images`, `annotations`, `categories`) nào?

---

# PHẦN C — IMAGE SEGMENTATION (Tuần 10-11)

> Mục tiêu sau 2 tuần: hiểu 3 nhánh segmentation, code được U-Net, dùng được Detectron2 và Segment Anything (SAM), và biết metric IoU/Dice cho segmentation. Segmentation là "đỉnh cao" của loạt bài toán vì cho biết **từng pixel** thuộc về lỗi nào.

---

## C1. Ba loại segmentation — khác nhau ở mức chi tiết

| Loại | Trả lời gì | Output | Ví dụ QC |
|---|---|---|---|
| **Semantic segmentation** | *Pixel này thuộc lớp nào?* | Map pixel→class (mỗi pixel 1 nhãn, không phân biệt instance) | Phân biệt vùng "lỗi" vs "nền" nhưng 2 vết lỗi đều là 1 màu |
| **Instance segmentation** | *Pixel này thuộc instance nào?* | Semantic + phân biệt từng vật thể riêng | Phân biệt **từng** vết lỗi riêng lẻ, đếm chính xác số lỗi |
| **Panoptic segmentation** | Kết hợp cả hai | "Stuff" (nền) semantic + "thing" (đối tượng) instance | Ảnh có nền + nhiều lỗi, cần cả 2 |

**Vì sao cần Instance (không chỉ Semantic) trong QC:** semantic chỉ nói "ở đây có lỗi" cho cả vùng; instance tách **từng lỗi** để đếm số lượng, đo kích thước từng lỗi, theo dõi từng lỗi qua khung hình.

### C1a. Metric cho segmentation: IoU & Dice

**Pixel IoU** (khác box IoU): tỷ lệ pixel dự đoán đúng giao / hợp.

```python
def iou_per_class(pred_mask, true_mask, num_classes):
    """pred_mask/true_mask: (H, W) indent label 0..num_classes-1"""
    batch_iou = []
    for cls in range(num_classes):
        p = (pred_mask == cls); t = (true_mask == cls)
        inter = (p & t).sum(); union = (p | t).sum()
        batch_iou.append(inter / union if union > 0 else 1.0)
    return batch_iou   # mIoU = mean của danh sách này

def dice_loss(pred, target, smooth=1.0):
    # pred: xác suất softmax (prob), target: one-hot
    inter = (pred * target).sum()
    return 1 - (2 * inter + smooth) / (pred.sum() + target.sum() + smooth)
```

| Metric | Ý nghĩa | Khi nào dùng |
|---|---|---|
| **Pixel accuracy** | % pixel đúng | Dễ sai lệch khi nền chiếm đa số (người mới hay mắc) |
| **mIoU** | Trung bình IoU theo lớp | Chuẩn đánh giá segmentation (COCO) |
| **Dice coefficient** | `2·TP / (2·TP + FP + FN)` — giống F1 | **Tốt với lớp không cân bằng** (lỗi nhỏ trên nền lớn) — rất hợp QC |

⚠️ **Bẫy pixel accuracy:** nếu ảnh có 95% nền + 5% lỗi, model dự đoán toàn "nền" đạt 95% accuracy nhưng hoàn toàn vô dụng → **luôn dùng mIoU/Dice, không dùng accuracy** cho segmentation (nhất là với lỗi nhỏ).

---

## C2. U-Net — kiến trúc segmentation cổ điển quan trọng nhất

### C2a. Tư duy Encoder-Decoder

U-Net gồm 2 nhánh:

```
Encoder (co lại - contract)              Decoder (giãn ra - expand)
- Conv+Pool nhiều lần                     - UpSample + Conv nhiều lần
- Học feature ngày càng trừu tượng        - Khôi phục lại độ phân giải pixel
- Giảm HxW, tăng số channel               - Tăng HxW, giảm số channel
        |                                       ↑
        +--------- Skip connections (nối) ------+
                  Chuyển feature chi tiết từ encoder sang decoder
```

**Skip connections = điểm mấu chốt:** khi khôi phục độ phân giải ở decoder, feature bị "mờ" mất chi tiết nhỏ. Skip connection **nối trực tiếp feature encoder** (giữ chi tiết vị trí) vào decoder tương ứng → giữ lại được **ranh giới sắc nét** — cực quan trọng cho lỗi nhỏ/mảnh. (Liên hệ: cũng là ý tưởng "residual" như ResNet — cung cấp đường thông tin song song.)

### C2b. Code U-Net tối giản trong PyTorch

```python
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UNetMini(nn.Module):
    def __init__(self, in_channels=3, n_classes=2):
        super().__init__()
        # Encoder
        self.e1 = DoubleConv(in_channels, 64)
        self.e2 = DoubleConv(64, 128)
        self.e3 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d3 = DoubleConv(256, 128)     # 128 (up) + 128 (skip) = 256
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2 = DoubleConv(128, 64)      # 64 (up) + 64 (skip) = 128
        self.out = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        s1 = self.e1(x)                    # giữ lại cho skip
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        x = self.up3(s3); x = self.d3(torch.cat([x, s2], dim=1))  # nối skip
        x = self.up2(x); x = self.d2(torch.cat([x, s1], dim=1))
        return self.out(x)                 # (B, n_classes, H, W) logits / pixel

model = UNetMini(in_channels=3, n_classes=2)
print(model(torch.randn(1, 3, 128, 128)).shape)   # (1, 2, 128, 128)
```

**Nối skip (concatenate) dọc theo channel:** `torch.cat([decoder_up, encoder_skip], dim=1)` → channel tăng gấp đôi → `DoubleConv` xử lý. Đây là "thương hiệu" của U-Net.

**Output & Loss:** output là **`n_classes` bản đồ logits** (mỗi pixel dự đoán 1 trong N lớp). Loss: `CrossEntropyLoss` (giữ chiều không gian) hoặc kết hợp `DiceLoss` (mạnh với lỗi nhỏ):

```python
# CrossEntropy cho segmentation: flatten spatial, nhãn là int mask
loss = nn.CrossEntropyLoss()(logits, target_long)   # logits (B,C,H,W), target (B,H,W) long

# Hoặc kết hợp Dice để handle lỗi nhỏ:
# total_loss = 0.5 * CE + 0.5 * DiceLoss
```

---

## C3. Mask R-CNN — instance segmentation

**Ý tưởng:** mở rộng Faster R-CNN (detection two-stage) thêm một nhánh **mặt nạ** (mask).

```
Faster R-CNN output: bounding box + class
Mask R-CNN  output: thêm 1 binary mask (H,W) cho mỗi box
                    = bbox + class + mask (chính xác hình dạng pixel của lỗi)
```

- **RoIAlign**: bước trích feature cho từng box — cải tiến của RoIPool, giữ sự liên tục toạ độ (không làm tròn) → mask chính xác hơn.
- Dùng **Detectron2** (thư viện của Meta AI) để vận hành nhanh.

```python
# Detectron2: instance segmentation out of the box
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

outputs = predictor(image)   # -> instances: boxes, scores, classes, pred_masks
```

---

## C4. Segment Anything Model (SAM) — nền tảng segmentation không cần gán nhãn riêng

### C4a. SAM là gì & hoạt động thế nào

**SAM (Meta AI, 2023):** model **promptable** segmentation — với 1 **prompt** (điểm click, bounding box, hoặc mask thô), SAM trả về mask của đối tượng. Được train trên **Segment Anything 1B (SA-1B)** — ~11 triệu ảnh, 1 tỷ mask → khả năng "cắt" bất kỳ đối tượng nào mà không cần fine-tune cho lớp cụ thể.

Kiến trúc gồm 3 phần chính:
- **Image encoder** (ViT — liên hệ Phần A): trích feature ảnh.
- **Prompt encoder**: mã hoá prompt (điểm/box/mask).
- **Mask decoder**: kết hợp 2 feature trên → sinh mask.

### C4b. Trình diễn & ứng dụng QC

```python
import torch
from segment_anything import sam_model_registry, SamPredictor

checkpoint = "sam_vit_b_01ec64.pth"
sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
sam.to("cuda" if torch.cuda.is_available() else "cpu")
predictor = SamPredictor(sam)

predictor.set_image(image)                       # encode ảnh 1 lần
# Prompt bằng bounding box của lỗi (VD: lấy từ YOLO ở Phần B!)
mask, score, _ = predictor.predict(
    box=np.array([x1, y1, x2, y2]), multimask_output=False,
)
# -> mask: mask 2D của đối tượng trong box
```

**Điểm mạnh trong QC:**
- Kết hợp tuyệt vời với YOLO: **YOLO tìm box lỗi → SAM sinh mask chính xác** — có được boundary sắc nét mà không cần gán nhãn mask.
- Phân đoạn khu vực quan tâm (mask camera, mask sản phẩm) để chỉ tập trung model vào vùng cần kiểm tra.
- Có thể dùng để **tự động tạo nhãn**: SAM cắt đối tượng → người duyệt → đưa vào training set cho Mask R-CNN/U-Net.

**Lưu ý:** SAM **không trả về lớp** (chỉ trả mask đối tượng, không biết lỗi là loại gì) → thường trao đổi với model classification hoặc detection để ghép nhãn.

---

## C5. So sánh nhanh các công cụ segmentation cho QC

| Công cụ | Loại | Ưu điểm | Nhược điểm | Use case QC |
|---|---|---|---|---|
| **U-Net** | Semantic | Nhẹ, tự train được, nhạy lỗi nhỏ | Cần gán nhãn mask | Phân đoạn lỗi/nền, sản phẩm |
| **Mask R-CNN** | Instance | Phân biệt từng lỗi + box + mask | Nặng, chậm hơn | Cần đếm/đo từng lỗi riêng |
| **SAM** | Promptable | Không cần train, mask sắc nét | Không trả lớp, có thể chậm | Assist annotation, tạo nhãn tự động |
| **SAM + YOLO** | Pipeline | Box (YOLO) + mask nét (SAM) | 2 bước giải đoạn | Đường ống QC hiệu quả không cần nhãn mask |

---

## C6. Checklist lỗi thường gặp (Phần C)

- [ ] Đánh giá segmentation bằng `accuracy` toàn ảnh → sai lệch do nền chiếm đa số → dùng **mIoU/Dice**
- [ ] Nhãn mask không khớp số lớp trong model → kiểm tra `n_classes` & cách mã hoá mask (int label vs one-hot)
- [ ] Quên `torch.cat` đúng channel khi nối skip trong U-Net → channel decoder tính sai → lỗi shape
- [ ] Loss là NaN trong segmentation nhỏ → dùng Dice kèm smooth, hoặc kiểm tra mask có `-1` (ignore index) gây `NaN`
- [ ] Mask R-CNN/SAM ra mask lệch biên → ảnh input khác resolution hoặc cần resize chuẩn về img_size của model
- [ ] Mất chi tiết lỗi nhỏ sau upsampling → tăng resolution đầu vào hoặc dùng skip connection (U-Net); không chỉ cắm thẳng decode

---

## C7. Bài tập thực hành đề xuất (Phần C — gắn domain QC)

1. **Code & train U-Net mini** trên dataset segmentation nhỏ (có thể là ảnh nhị phân "lỗi/nền" tự sinh): train 30-50 epochs, đánh giá bằng `mIoU` + `Dice`, quan sát skip connection giữ ranh giới.
2. **So sánh CE loss vs Dice loss** trên lớp lỗi nhỏ — quan sát vì sao Dice tốt hơn với mất cân bằng.
3. **Chạy Detectron2** Mask R-CNN pretrained COCO trên ảnh có vật thể quen thuộc (người, xe...) để hiểu output instances gồm box/mask/class.
4. **Chạy SAM** với prompt box (lấy từ YOLO hoặc tự vẽ) trên ảnh lỗi → lưu mask → so sánh với mask thủ công/ground-truth.
5. **Dựng pipeline hoàn chỉnh mini:** `YOLO detect → SAM segment` trên vài ảnh lỗi, xuất mask cho từng lỗi.

---

## C8. Câu hỏi tự kiểm tra (chuẩn bị sớm cho phỏng vấn)

- Semantic vs instance vs panoptic segmentation khác nhau thế nào? Cho ví dụ QC cho từng loại.
- Vì sao không dùng pixel accuracy cho segmentation lỗi nhỏ? mIoU/Dice giải quyết gì?
- U-Net gồm mấy nhánh? Vai trò skip connection là gì, vì sao nó bảo toàn ranh giới sắc nét?
- Encoder (contract) và Decoder (expand) trong U-Net tương ứng với nhiệm vụ gì?
- Mask R-CNN thêm gì so với Faster R-CNN? RoIAlign giải quyết vấn đề gì?
- SAM khác U-Net/Mask R-CNN ở điểm nào (về nhu cầu training & loại output)?
- Vì sao kết hợp YOLO + SAM là đường ống hiệu quả cho QC không cần nhãn mask?

---

# TỔNG KẾT GIAI ĐOẠN 2 (Tuần 4-11) — BỨC TRANH TOÀN CẢNH

## T9. Ba bài toán — một lộ trình tăng dần

```
Classification (A)              Detection (B)               Segmentation (C)
"Có lỗi hay không?"             "Lỗi ở đâu (box)?"          "Lỗi chiếm pixel nào?"
│                               │                           │
ResNet/EfficientNet/ViT         YOLO (one-stage)            U-Net (semantic)
Transfer learning               Faster R-CNN (two-stage)    Mask R-CNN (instance)
CrossEntropy                    mAP / IoU / NMS             mIoU / Dice
Data augmentation               Data annotation (COCO)      SAM (promptable)
Overfitting control             Confidence threshold         Pixel-wise classification
```

**Mối liên hệ xuyên suốt để Capstone:**
1. **Backbone chung:** ResNet/EfficientNet/vit làm backbone cho cả classifier, YOLO, Mask R-CNN — feature đã học ở Phần A được tái sử dụng.
2. **Transfer learning ở mọi nơi:** luôn khởi đầu từ pretrained vì dữ liệu QC nhỏ.
3. **Augmentation & chống overfit** (Phần A) áp dụng nguyên vẹn cho cả detection/segmentation.
4. **Pipeline Capstone tiềm năng:** `Preprocess (Phase 1) → Detect box lỗi (YOLO) → Segment mask (SAM) → Classify loại lỗi (ResNet) → Track & đếm (Phase 11)`.

## T10. Các thư viện/công cụ bạn sẽ dùng được sau giai đoạn này

| Công cụ | Mục đích | Cài đặt |
|---|---|---|
| `torch`, `torchvision` | Train CNN, transfer learning | `pip install torch torchvision` |
| `ultralytics` | YOLOv8/yolo11 train + inference | `pip install ultralytics` |
| `detectron2` | Mask R-CNN, instance segmentation | Xem docs (cần build) |
| `segment-anything` | SAM promptable segmentation | `pip install segment-anything` |
| `torchmetrics` | mAP, IoU, Dice metric chuẩn | `pip install torchmetrics` |
| CVAT | Gán nhãn detect/segment nội bộ | Deploy Docker (liên hệ Phase LLM Docker) |

## T11. Checklist tổng hợp giai đoạn (tự review trước khi lên Phase 3)

- [ ] Tự xây và train được CNN nhỏ từ đầu (forward/backward/optimizer hoạt động)
- [ ] Fine-tune được ResNet/EfficientNet pretrained → đạt val_acc tốt, không overfit
- [ ] Hiểu và áp dụng được augmentation, dropout, weight decay, early stopping, lr scheduling
- [ ] Code được IoU + NMS, hiểu mAP@0.5 vs mAP@0.5:0.95
- [ ] Gán nhãn được 1 bộ dataset (CVAT/Roboflow), export sang YOLO/COCO format
- [ ] Train + evaluate + export được YOLOv8n, tune được confidence threshold
- [ ] Code & train được U-Net, đánh giá bằng mIoU/Dice (không phải accuracy)
- [ ] Chạy được Detectron2 Mask R-CNN và SAM (prompt box)
- [ ] Dựng được pipeline mini `YOLO detect → SAM segment`

## T12. Tiêu chí "đủ chuẩn để sang Phase 3" (ngưỡng rõ ràng)

Mỗi bài toán bạn cần có **1 bản demo chạy được + 1 trang kết quả** (không chỉ lý thuyết):

1. **Classification:** model fine-tuned đạt **≥ 90% val accuracy** trên bộ QC nhị phân lỗi/ok, có confusion matrix.
2. **Detection:** YOLO **mAP@0.5 ≥ 0.7** trên dataset lỗi tự gán nhãn, có ảnh predict để xem.
3. **Segmentation:** U-Net **mIoU ≥ 0.7** (hoặc Dice ≥ 0.8) trên dataset lỗi/nền, hoặc pipeline YOLO+SAM sinh mask tốt trên thực tế.

> Nếu chưa đạt ngưỡng: tăng dữ liệu/quality annotation trước, không vội tăng model to. Trong QC, **dữ liệu và annotation thường quyết định hơn model** — đây là bài học cuối giai đoạn đáng nhớ nhất.
