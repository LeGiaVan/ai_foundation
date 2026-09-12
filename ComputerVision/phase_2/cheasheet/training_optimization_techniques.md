# CHEATSHEET: CÁC PHƯƠNG PHÁP TỐI ƯU HÓA HUẤN LUYỆN MÔ HÌNH (DEEP LEARNING TRAINING TRICKS)

> Bản hướng dẫn toàn diện từ lý thuyết, bản chất toán học, trực giác hình học đến code mẫu chuẩn PyTorch giúp mô hình hội tụ nhanh hơn, chống Overfitting và đạt độ chính xác (Accuracy) cao nhất.

---

## 🗺️ BẢN ĐỒ TỔNG QUAN: BẮT BỆNH VÀ KÊ ĐƠN

Trong quá trình huấn luyện Deep Learning, mọi vấn đề thường rơi vào 3 tình huống chính:

```
                      TÌNH TRẠNG MÔ HÌNH
                              │
     ┌────────────────────────┼────────────────────────┐
     ▼                        ▼                        ▼
[Underfitting]           [Overfitting]          [Instability / Slow]
(Học dốt cả Train & Val) (Train tốt, Val tụt dốc) (Loss nhảy loạn xạ, tràn VRAM)
     │                        │                        │
  • Tăng Model Capacity    • Data Augmentation      • Learning Rate Scheduler
  • Tăng Learning Rate     • Early Stopping         • Gradient Clipping
  • Thêm Warmup            • Weight Decay (AdamW)   • Mixed Precision (AMP)
  • Bỏ bớt Regularization  • Dropout / DropPath     • Batch Normalization
```

---

## PHẦN 1: CHIẾN LƯỢC ĐIỀU KHIỂN TỐC ĐỘ HỌC (LEARNING RATE SCHEDULING)

Learning Rate (LR) là siêu tham số (**hyperparameter**) quan trọng số 1 trong Deep Learning.

* **Nếu LR quá lớn**: Bước nhảy quá dài, mô hình nhảy vọt qua cực tiểu (Overshoot), Loss dao động mạnh hoặc phân kỳ (bùng nổ NaN).
* **Nếu LR quá nhỏ**: Bước đi quá chậm, mô hình mất hàng chục epoch không giảm loss, hoặc bị kẹt mãi ở cực tiểu địa phương cạn (Poor Local Minima).
* 💡 **Giải pháp**: Bắt đầu bằng bước chân vừa phải, sau đó **nhỏ dần theo thời gian** khi tiến gần về đáy vực!

---

### 1.1. Warmup (Khởi động làm nóng)
* **Ý tưởng**: Trong $N$ epoch đầu tiên, bắt đầu bằng LR cực nhỏ (ví dụ $10^{-6}$), rồi tăng dần tuyến tính lên LR mục tiêu ($10^{-3}$).
* **Tại sao cần?**: Khi mới bắt đầu, trọng số mô hình còn hỗn loạn, gradient sinh ra rất lớn và giật cục. Nếu để LR lớn ngay từ đầu, gradient lớn sẽ **phá hủy hoàn toàn các đặc trưng tiền huấn luyện (Pretrained Weights)** của ResNet/ViT. Warmup giúp mô hình "bình tĩnh" thích nghi trước khi tăng tốc.

---

### 1.2. ReduceLROnPlateau (Giảm khi tắc đường)
* **Cơ chế**: Lắng nghe chỉ số `val_loss`. Nếu sau $K$ epoch liên tiếp (**patience**) mà `val_loss` không chịu giảm thêm, tự động chia LR cho một hệ số (ví dụ giảm đi 2 lần hoặc 10 lần).
* **Ưu điểm**: Thích ứng linh hoạt theo độ học thực tế của mô hình, không cứng nhắc.
* **Code PyTorch**:
```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

for epoch in range(epochs):
    train_loss = train_one_epoch(...)
    val_loss, val_acc = evaluate(...)
    
    # LƯU Ý: ReduceLROnPlateau cần truyền val_loss vào hàm step()
    scheduler.step(val_loss)
```

---

### 1.3. CosineAnnealingLR (Hạ nhiệt theo đồ thị Cosine)
* **Cơ chế**: Giảm dần LR theo đường cong hàm Cosine từ đỉnh cao nhất về tiệm cận 0 (`eta_min`).
* **Tại sao rất được ưa chuộng?**: Ở giai đoạn giữa và cuối, LR giảm rất mượt mà không bị giật cục đột ngột như StepLR, giúp mô hình trượt êm ái vào các **vùng lòng chảo phẳng (Flat Minima)** — nơi có khả năng tổng quát hóa tốt nhất.
* **Code PyTorch**:
```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-6)

for epoch in range(epochs):
    train(...)
    val(...)
    scheduler.step()  # Không cần truyền tham số
```

---

### 1.4. CosineAnnealingWarmRestarts (Nhảy ra khỏi hố bẫy)
* **Cơ chế**: Sau khi LR hạ xuống đáy, nó bất ngờ "bật tăng" trở lại mức cao ban đầu để bắt đầu một chu kỳ mới.
* **Tác dụng**: Cú hích LR cao đột ngột giúp mô hình đủ năng lượng thoát khỏi các **cực tiểu địa phương xấu (Sharp Minima)** để đi tìm vùng trũng tốt hơn.

---

## PHẦN 2: DỪNG SỚM (EARLY STOPPING) & CHECKPOINTING

### 2.1. Bản chất của Early Stopping
* Trong thực tế, tập Train luôn giảm loss càng ngày càng sâu, nhưng tập Val sẽ đạt đáy tại một **Điểm Vàng (Sweet Spot)** rồi sau đó tăng ngược trở lại (Overfitting).
* **Early Stopping**: Giống như vị trọng tài. Khi thấy `val_loss` đã qua điểm cực tiểu mà không tiến bộ thêm sau $P$ epoch (**patience**), trọng tài sẽ **thổi còi dừng cuộc chơi ngay lập tức** để tránh lãng phí điện năng và thời gian.

```
Loss
 │      Train Loss (cứ giảm mãi)
 │╲     Val Loss (đáy ở Epoch 7 rồi ngóc lên)
 │ ╲       ╭─────────────╮
 │  ╲  ___╱              │  <-- Dừng tại đây! (Patience = 3)
 └───█───────────────────┴──────── Epoch
   Epoch 7 (Sweet Spot)
```

---

### 2.2. Tại sao nên theo dõi `val_loss` thay vì `val_acc`?
* `val_acc` là đại lượng **bậc thang gián đoạn** (rời rạc). Ví dụ: 100 ảnh, đoán đúng 85 ảnh hay 86 ảnh.
* `val_loss` là đại lượng **số thực liên tục**, phản ánh cả **độ tự tin** của mô hình (ví dụ: đoán đúng nhưng xác suất 51% hay 99%). Do đó, `val_loss` phát hiện dấu hiệu Overfitting nhạy bén và sớm hơn `val_acc` rất nhiều.

---

### 2.3. Code hoàn chỉnh Class EarlyStopping chuẩn công nghiệp

```python
import numpy as np
import torch

class EarlyStopping:
    """Dừng huấn luyện sớm nếu val_loss không cải thiện sau một số epoch định trước."""
    def __init__(self, patience=5, delta=0.0, save_path="best_model.pth"):
        self.patience = patience          # Số epoch chịu đựng
        self.delta = delta                # Mức cải thiện tối thiểu coi là có tiến bộ
        self.save_path = save_path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss, model):
        # Nếu loss cải thiện rõ rệt
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
            print(f"  --> [BEST] Val loss giam xuong {val_loss:.4f}. Da luu model!")
        else:
            self.counter += 1
            print(f"  --> [INFO] Khong cai thien ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
                print("  --> [STOP] Kich hoat Early Stopping!")

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_path)
```

---

## PHẦN 3: KỸ THUẬT CHỐNG QUÁ KHỚP (REGULARIZATION)

### 3.1. Weight Decay (L2 Regularization) & Vì sao dùng AdamW thay vì Adam?
* **Mục tiêu**: Phạt các trọng số có giá trị quá lớn. Trọng số càng nhỏ thì hàm quyết định của mô hình càng mượt mà, ít bị nhạy cảm với nhiễu.
* **Bí mật AdamW**:
  * Trong thuật toán `Adam` cổ điển, Weight Decay bị tính gộp chung vào gradient bậc một và bậc hai $\rightarrow$ làm sai lệch ý nghĩa toán học của L2 Regularization.
  * `AdamW` (Loshchilov & Hutter, 2017) tách riêng bước trừ Weight Decay trực tiếp ra khỏi gradient updates.
  * 💡 **Quy tắc vàng**: **Luôn luôn ưu tiên dùng `torch.optim.AdamW` thay vì `torch.optim.Adam`!**

---

### 3.2. Dropout vs Spatial Dropout (Dropout2d)
* **`nn.Dropout(p=0.5)`**: Ngẫu nhiên ngắt kết nối $50\%$ số nơ-ron trong lượt forward. Bắt buộc các nơ-ron còn lại phải tự học đặc trưng mà không được "dựa dẫm" vào nơ-ron hàng xóm (Co-adaptation).
* **`nn.Dropout2d(p=0.2)`**: Trong ảnh, các pixel cạnh nhau có tính tương quan cực cao. Nếu tắt 1 pixel mà các pixel bao quanh vẫn còn thì mô hình vẫn đoán được. `Dropout2d` ngắt **nguyên một kênh màu (toàn bộ 1 feature map)**, buộc mô hình phải học đặc trưng từ nhiều kênh độc lập. Thường áp dụng sau các tầng Conv.

---

### 3.3. Label Smoothing (Làm mịn nhãn)
* **Vấn đề**: One-hot vector gán nhãn cứng: $[0, 0, 1, 0, 0, 0]$ (tức khẳng định 100% là lớp Pa). Điều này ép mô hình sinh ra logits cực lớn tiệm cận vô cùng $\rightarrow$ mô hình quá tự tin và dễ overfit.
* **Label Smoothing**: Co nhãn lại thành $[0.02, 0.02, 0.90, 0.02, 0.02, 0.02]$.
* **Code PyTorch**:
```python
# Tích hợp sẵn trong CrossEntropyLoss từ PyTorch 1.10+
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

## PHẦN 4: TĂNG CƯỜNG DỮ LIỆU HIỆN ĐẠI (ADVANCED DATA AUGMENTATION)

Bên cạnh các phép xoay, lật cơ bản (`RandomHorizontalFlip`, `RandomRotation`), 2 kỹ thuật hiện đại sau đây đem lại bước nhảy vọt về độ chính xác:

### 4.1. Mixup (Zhang et al., 2017)
* Lấy 2 bức ảnh ngẫu nhiên $A$ và $B$, trộn mờ chúng lại với nhau theo tỷ lệ $\lambda \in [0, 1]$:
  $$\text{Ảnh mới} = \lambda \cdot A + (1 - \lambda) \cdot B$$
  $$\text{Nhãn mới} = \lambda \cdot \text{Nhãn } A + (1 - \lambda) \cdot \text{Nhãn } B$$

### 4.2. CutMix (Yun et al., 2019)
* Cắt một góc hình chữ nhật trên ảnh $B$ và đắp đè lên ảnh $A$. Tỷ lệ nhãn được tính theo diện tích phần đắp đè.
* Giúp mô hình không chỉ nhìn vào 1 đặc điểm nổi bật nhất (ví dụ cái mỏ con vịt) mà phải học nhận dạng vật thể ngay cả khi vật thể bị che khuất một phần.

---

## PHẦN 5: BÍ KÍP TĂNG TỐC VÀ ỔN ĐỊNH HUẤN LUYỆN (ENGINEERING TRICKS)

### 5.1. Automatic Mixed Precision (AMP - FP16)
* **Vấn đề**: Mặc định PyTorch tính toán ở độ chính xác đơn 32-bit (`float32`).
* **AMP**: Tự động chuyển các phép nhân ma trận sang nửa độ chính xác 16-bit (`float16`), chỉ giữ lại các phép tính nhạy cảm ở 32-bit.
* **Lợi ích thực tế trên GPU (như RTX 2050)**:
  * ⚡ **Tốc độ nhanh gấp 1.5 - 2 lần**.
  * 📉 **Bộ nhớ VRAM giảm gần một nửa** $\rightarrow$ có thể tăng gấp đôi `batch_size` mà không lo báo lỗi tràn RAM!
* **Code PyTorch**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, labels in loader:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    
    # 1. Forward trong ngữ cảnh autocast
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)
        
    # 2. Backward có bảo vệ tránh underflow
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

### 5.2. Gradient Clipping (Cắt tỉa Gradient)
* **Vấn đề**: Đôi khi gặp 1 batch dữ liệu dị biệt hoặc kiến trúc quá sâu, gradient bị nhân dồn vọt lên hàng nghìn $\rightarrow$ bùng nổ gradient (**Exploding Gradients**), làm vỡ nát toàn bộ trọng số mạng (Loss hóa `NaN`).
* **Giải pháp**: Nếu độ dài vector gradient vượt quá ngưỡng `max_norm`, co ngắn nó lại.
* **Code PyTorch**:
```python
loss.backward()
# Đặt ngay trước optimizer.step()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

### 5.3. Exponential Moving Average (EMA) của Trọng số
* Thay vì lấy trọng số ở epoch cuối cùng, ta duy trì một bản sao trọng số được tính trung bình động theo cấp số nhân qua các epoch:
  $$W_{\text{EMA}} = \alpha \cdot W_{\text{EMA}} + (1 - \alpha) \cdot W_{\text{hiện tại}}$$
* Giúp triệt tiêu các rung lắc ngẫu nhiên, cho ra mô hình có độ tin cậy và mượt mà cao hơn khi đưa vào sản xuất.

---

## 🎯 TỔNG HỢP PIPELINE CHUẨN ĐỂ HUẤN LUYỆN 1 MÔ HÌNH XUẤT SẮC

Khi bắt đầu một bài toán Vision mới, đây là "công thức chiến thắng" (Winning Recipe) tiêu chuẩn:

1. **Optimizer**: Dùng `AdamW` với initial learning rate `1e-3` (hoặc `3e-4` nếu fine-tuning Transformer/ViT).
2. **LR Scheduler**: `CosineAnnealingLR` (kèm 1-2 epoch Warmup nếu fine-tuning pretrained).
3. **Data Augmentation**: Lật ngang + Lật dọc + Xoay nhẹ $\pm 15^\circ$ + ColorJitter.
4. **Regularization**: `weight_decay = 1e-2`, `label_smoothing = 0.1`.
5. **Đánh giá & Dừng**: `EarlyStopping(patience=5)` theo dõi `val_loss`, luôn lưu `best_model.pth`.
6. **Tăng tốc phần cứng**: Bật `torch.cuda.amp` (FP16) để train nhanh gấp đôi và mát máy!
