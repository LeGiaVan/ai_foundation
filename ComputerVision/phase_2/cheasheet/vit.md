# Vision Transformer — ViT (Google Brain, 2020)

ViT không phát minh lại Transformer — nó lấy **nguyên xi kiến trúc Transformer từ NLP** và thử áp dụng lên ảnh. Câu hỏi trung tâm là: *"Nếu chia ảnh thành các mảnh nhỏ và coi mỗi mảnh như một token, liệu Transformer có học được không?"* — Kết quả: hoàn toàn được, và còn vượt CNN nếu có đủ dữ liệu.

---

## 1. Bối cảnh: Tại sao cần ViT khi CNN đã rất tốt?

CNN (ResNet, EfficientNet...) học đặc trưng **cục bộ** — một neuron trong layer đầu chỉ "nhìn" vùng 3×3 pixel xung quanh nó. Để hiểu mối quan hệ xa (ví dụ: đầu mèo ở góc trên bên trái liên quan đến đuôi mèo ở góc dưới bên phải), CNN phải xếp chồng nhiều layer để thông tin lan dần từ cục bộ đến toàn cục — **tốn nhiều layer, nhìn toàn cục chậm**.

Transformer trong NLP (BERT, GPT) thì khác: mỗi token **nhìn thấy mọi token khác ngay từ layer đầu tiên** — đây là sức mạnh của Self-Attention. ViT đem tư duy này sang ảnh.

---

## 2. Ý tưởng cốt lõi: Ảnh → Chuỗi các "Patch Token"

Thay vì xử lý ảnh pixel-by-pixel như CNN, ViT chia ảnh thành các **patch vuông**, sau đó coi mỗi patch là một "từ" (token) trong câu.

**Ví dụ cụ thể:**

```
Ảnh đầu vào: 224 × 224 × 3  (ảnh màu tiêu chuẩn)
Kích thước patch: 16 × 16 pixel

→ Số patch = (224 / 16) × (224 / 16) = 14 × 14 = 196 patch

→ Mỗi patch có kích thước: 16 × 16 × 3 = 768 số
```

Vậy một bức ảnh 224×224 trở thành **chuỗi 196 token**, mỗi token là vector 768 chiều.
Giống hệt câu văn 196 từ trong NLP — Transformer xử lý y chang.

---

## 3. Toàn bộ pipeline ViT — từng bước cực kỳ chi tiết

### Tổng sơ đồ:

```
Ảnh [224×224×3]
      │
      ▼
  Chia patch  →  196 patch, mỗi patch [16×16×3]
      │
      ▼
  Flatten + Linear Projection  →  196 × [768]   ← "Patch Embedding"
      │
      ▼
  Thêm [CLS] token             →  197 × [768]   ← token 0 dành riêng cho phân loại
      │
      ▼
  Cộng Positional Encoding     →  197 × [768]   ← cho mạng biết vị trí của từng patch
      │
      ▼
  ┌─────────────────────────────────────┐
  │  Transformer Encoder × L lần        │
  │                                     │
  │  ┌──────────────────────────────┐   │
  │  │  Layer Norm                  │   │
  │  │       ↓                      │   │
  │  │  Multi-Head Self-Attention   │   │
  │  │       ↓                      │   │
  │  │  (+) Skip Connection         │   │
  │  │       ↓                      │   │
  │  │  Layer Norm                  │   │
  │  │       ↓                      │   │
  │  │  MLP (Feed Forward)          │   │
  │  │       ↓                      │   │
  │  │  (+) Skip Connection         │   │
  │  └──────────────────────────────┘   │
  └─────────────────────────────────────┘
      │
      ▼
  Lấy output của [CLS] token   →  [768]
      │
      ▼
  MLP Head (Classification)    →  [số lớp]   ← VD: 1000 lớp ImageNet
      │
      ▼
  Softmax → Xác suất từng lớp
```

---

### Bước 1 — Patch Embedding: Ảnh → Ma trận token

**Mục tiêu:** Cắt ảnh thành các mảnh nhỏ (patch) và biến mỗi mảnh thành một vector đặc trưng (token) để đưa vào Transformer. Quá trình này mô phỏng cách NLP biến các từ thành Word Embedding.

**Cụ thể quá trình diễn ra qua 3 thao tác:**

1. **Băm ảnh (Grid Splitting):** 
   Ảnh gốc (ví dụ $224 \times 224 \times 3$) được chia lưới thành các hình vuông không chồng lấp, kích thước $16 \times 16$.
   - Số lượng mảnh thu được: $(224 / 16) \times (224 / 16) = 14 \times 14 = 196$ patch.
   
2. **Duỗi thẳng (Flatten):** 
   Mỗi mảnh $16 \times 16 \times 3$ chứa 768 điểm ảnh. Ta xếp 768 con số này thành một hàng ngang (vector 1D có kích thước 768). Tuy nhiên, đây mới chỉ là giá trị pixel thô, chưa mang nhiều đặc trưng cấp cao.

3. **Linear Projection (Nhúng - Embedding):** 
   Đem vector pixel thô $X$ kích thước `[1 × 768]` nhân với một **Ma trận trọng số học được $W$** có kích thước `[768 × 768]`. Phép toán này tạo ra một vector mới $Z = X \cdot W$ cũng có kích thước `[1 × 768]`.
   
   **Bản chất của phép nhân này là gì?**
   - Ma trận $W$ chứa 768 cột, mỗi cột đóng vai trò như một **"bộ lọc đặc trưng" (filter/template)**.
   - Khi nhân vector mảnh ảnh $X$ với cột thứ $j$ của $W$ (chính là phép Tích vô hướng), ta đang đo lường xem mảnh ảnh này **khớp đến mức nào** với bộ lọc thứ $j$.
   - Ví dụ: Trong quá trình huấn luyện, cột $W_1$ tự học cách phát hiện "cạnh chéo", cột $W_2$ học cách phát hiện "mảng màu đỏ", cột $W_3$ phát hiện "góc nhọn"...
   - Kết quả đầu ra là vector $Z$ (chính là Token Embedding). Lúc này, $Z$ không còn chứa độ sáng RGB thô kệch nữa, mà nó chứa **768 điểm số (activations)**. Điểm số tại vị trí $j$ càng cao chứng tỏ mảnh ảnh đó càng mang đậm đặc trưng mà bộ lọc thứ $j$ đang tìm kiếm. Ta đã nâng cấp thành công từ "pixel thô" lên thành "ngữ nghĩa thị giác".

```text
Patch thô (16x16x3) → Flatten [768] → Nhân ma trận W [768×768] → Token Embedding [768]
```

> [!TIP]
> **Mẹo lập trình thực tế (Implementation Hack):**
> Trong Pytorch, thay vì phải viết code cắt ảnh, flatten rồi nhân ma trận một cách rườm rà, toàn bộ **Bước 1** này được thực hiện cực kỳ thanh lịch chỉ bằng đúng **1 lớp Convolution**:
> ```python
> nn.Conv2d(in_channels=3, out_channels=768, kernel_size=16, stride=16)
> ```
> - `kernel_size=16` và `stride=16`: Đảm bảo cuộn từng ô vuông $16 \times 16$ vừa khít, không trượt đè lên nhau (Non-overlapping).
> - `out_channels=768`: Biến 3 kênh màu của mỗi ô thành một vector 768 chiều.
> 
> Output nhận được sẽ là một Tensor `[Batch, 768, 14, 14]`. Sau đó chỉ việc `reshape()` và đổi chỗ các trục để thu được kích thước `[Batch, 196, 768]` là coi như đã hoàn thành Patch Embedding!

**Sau bước này, toàn bộ ảnh biến thành:**
```text
[196 × 768]   ← ma trận 2D (196 token, mỗi token 768 chiều)
```

---

### Bước 2 — [CLS] Token: "Người đại diện" tổng hợp thông tin

ViT vay mượn một ý tưởng cực kỳ thông minh từ mô hình ngôn ngữ BERT: thêm một **token đặc biệt [CLS]** (viết tắt của Classification) vào đầu chuỗi dữ liệu.

```text
Trước: [patch_1, patch_2, ..., patch_196]      → 196 token
Sau:   [CLS, patch_1, patch_2, ..., patch_196] → 197 token
```

**Bản chất của `[CLS]` Token là gì?**
- Nó hoàn toàn không chứa bất kỳ điểm ảnh (pixel) thật nào. Ban đầu, nó chỉ là một vector 768 chiều chứa các con số ngẫu nhiên.
- Tuy nhiên, nó là một tham số **có thể học (learnable parameter)**, nghĩa là thuật toán Gradient Descent sẽ tự do nhào nặn các con số trong vector này qua từng vòng huấn luyện.

**Cơ chế "Hút" thông tin và lý do vứt bỏ 196 mảnh:**
- **Quá trình tổng hợp:** Khi cụm 197 token này chui vào các lớp Transformer Encoder, chúng liên tục thực hiện **Self-Attention** (tất cả nhìn vào tất cả). Token `[CLS]` (đóng vai trò Query) sẽ đi đánh giá mức độ quan trọng với toàn bộ 196 patch còn lại (đóng vai trò Key). Vì `[CLS]` không bị trói buộc vào bất kỳ vị trí vật lý cụ thể nào trên ảnh nên nó rất trung lập. Dưới áp lực của hàm Loss phân loại, mạng Neural tự động ép `[CLS]` học cách: *"Hãy đi lùng sục và thu thập các đặc trưng quan trọng nhất (như tai, đuôi, lông...) từ 196 mảnh kia và gom hết vào bản thân mình"*.
- **Bản tóm tắt hoàn hảo:** Sau khi đi qua nhiều lớp Transformer, 196 token patch kia dù có thông minh lên thì về cơ bản vẫn mang tính "cục bộ" (ví dụ patch số 1 vẫn đại diện mạnh nhất cho góc trên bên trái). Ngược lại, vector `[CLS]` lúc này đã "hấp thụ" tinh hoa của toàn bộ bức ảnh và trở thành một **đại diện toàn cục (Global Representation)**.
- **Phân loại cuối cùng:** Vì vector `[CLS]` đã chứa đủ mọi thông tin cần thiết để kết luận bức ảnh là gì, ta chỉ cần trích xuất duy nhất vector đầu ra của `[CLS]` để đưa vào mạng Classifier (MLP) là xong. Việc vứt bỏ 196 đầu ra kia ở bước cuối giúp bài toán tập trung và tiết kiệm tài nguyên. *(Ghi chú: Thay vì dùng `[CLS]`, một số kiến trúc khác chọn cách tính trung bình cộng của 196 patch - gọi là Global Average Pooling - nhưng tác giả ViT nguyên bản chọn `[CLS]` để giữ mọi thứ y hệt như mô hình NLP ban đầu).*

---

### Bước 3 — Positional Encoding: Cho mạng biết patch ở đâu trong ảnh

**Vấn đề nghiêm trọng:** Transformer thuần túy **không có khái niệm vị trí** — nó chỉ xử lý một "tập hợp" token, không quan tâm token nào đứng trước token nào. Nếu hoán đổi patch góc trên trái với patch giữa ảnh, Transformer sẽ cho output y chang — điều này rõ ràng sai.

**Giải pháp — Positional Encoding:**

```
Token sau bước 2:      [197 × 768]

Positional Encoding:   [197 × 768]   ← Ma trận tham số học được (Learnable Parameter)

Cộng trực tiếp (Element-wise Addition):
[197 × 768] (Token)  +  [197 × 768] (Position)  =  [197 × 768]  ← Kích thước giữ nguyên
```

Mỗi vị trí từ $0$ đến $196$ (0=CLS, 1=patch góc trên trái, ..., 196=patch góc dưới phải) sẽ được cộng thêm một vector vị trí riêng biệt để đánh dấu "tọa độ" của nó trước khi đi vào khối xử lý.

> [!TIP]
> **ViT có dùng hàm Sin/Cos như Transformer gốc không?**
> Câu trả lời là **KHÔNG**. Transformer gốc trong NLP dùng hàm Sin/Cos tính toán sẵn bằng tay vì độ dài câu văn có thể biến đổi dài ngắn vô tận. Ngược lại, ViT xử lý ảnh đã được đưa về kích thước cố định (ví dụ: luôn là $224 \times 224$, tạo ra cố định 196 patch). 
> 
> Vì số lượng vị trí là cố định, các tác giả ViT chọn sử dụng **Learnable Positional Encoding 1D**. Tức là họ khởi tạo ngẫu nhiên một ma trận tham số `[197 × 768]` và ném nó vào cho mô hình tự học (`requires_grad=True`). Qua các epoch, quá trình Backpropagation sẽ tự động điều chỉnh các vector này sao cho phản ánh đúng khoảng cách không gian nhất (tự học được rằng patch 1 nằm gần patch 2, và nằm cực xa patch 196).

**Tại sao cộng (không nối/concatenate)?** Cộng giữ nguyên chiều vector — đơn giản và đủ hiệu quả. Nối sẽ tăng chiều lên 1536, tốn gấp đôi tài nguyên mà thực nghiệm không tốt hơn.

---

### Bước 4 — Transformer Encoder: Trái tim của ViT

Đây là phần được lặp lại L lần (ViT-Base: L=12, ViT-Large: L=24).

Mỗi lần lặp gồm 2 sub-block, đều có **skip connection** và **Layer Norm**:

#### Sub-block 1: Multi-Head Self-Attention (MHSA)

**Mục tiêu:** Mỗi token "hỏi thăm" tất cả token khác — patch A quan tâm đến patch B bao nhiêu?

Input: `X` có shape `[197 × 768]`

**Step 4.1 — Tạo Q, K, V (Query, Key, Value):**

```
X [197 × 768]  nhân với 3 ma trận học được:

  W_Q [768 × 768]  →  Q [197 × 768]  ← "Tôi đang tìm kiếm gì?"
  W_K [768 × 768]  →  K [197 × 768]  ← "Tôi có thể cung cấp gì?"
  W_V [768 × 768]  →  V [197 × 768]  ← "Thông tin thực sự của tôi"
```

Tương tự hệ thống tìm kiếm:
- **Q (Query)**: câu hỏi — "tôi, patch số 5 (vùng tai mèo), đang tìm kiếm patch nào liên quan?"
- **K (Key)**: nhãn — "tôi, patch số 120 (vùng râu mèo), có đặc trưng này"
- **V (Value)**: nội dung thực — "đây là thông tin đầy đủ của patch số 120"

**Step 4.2 — Tính Attention Score:**

```
Score = Q × K^T  →  [197 × 197]   ← ma trận tương đồng mọi cặp token

Mỗi ô [i, j] = mức độ token i "chú ý" đến token j

Ví dụ hàng i=5 (patch tai mèo):
  [5,1]=0.02  [5,2]=0.01  ...  [5,120]=0.89  ...  [5,197]=0.03
  → patch tai mèo chú ý nhiều nhất đến patch số 120 (râu mèo)
```

**Step 4.3 — Scale và Softmax:**

```
Score_scaled = Score / √768   ← chia căn(d_k) để tránh softmax bão hòa

Score_softmax = Softmax(Score_scaled)  →  [197 × 197]

→ Mỗi hàng i là phân phối xác suất: token i "phân bổ" 100% sự chú ý cho 197 token
```

Tại sao chia căn(d_k)? Tích vô hướng Q·K tăng theo chiều vector — với d_k=768, các giá trị rất lớn làm softmax "đổ" gần như toàn bộ weight vào 1 vị trí (gần như one-hot), gradient về 0, học rất khó.

**Step 4.4 — Weighted Sum với V:**

```
Output_attention = Score_softmax × V  →  [197 × 768]

→ Token i "thu thập" thông tin từ tất cả token khác,
  được trọng số hóa bởi mức độ chú ý đã tính.
```

**Step 4.5 — Multi-Head (nhiều "đầu" attention):**

ViT-Base dùng **h=12 head**. Thay vì 1 attention trên 768 chiều, chia ra 12 attention song song trên 64 chiều (768/12=64):

```
Head 1: Q1,K1,V1 (768→64) → output [197 × 64]   ← học quan hệ "loại A"
Head 2: Q2,K2,V2 (768→64) → output [197 × 64]   ← học quan hệ "loại B"
...
Head 12: ...               → output [197 × 64]   ← học quan hệ "loại L"

Concatenate: [197 × 64×12] = [197 × 768]

Linear Projection W_O [768 × 768]  →  [197 × 768]
```

Mỗi head học một "góc nhìn" khác nhau về quan hệ giữa các patch:
- Head 1 có thể học quan hệ cạnh — patch có cạnh ngang chú ý đến nhau.
- Head 2 có thể học quan hệ màu sắc.
- Head 5 có thể học quan hệ xa (đầu-đuôi con vật).

**Skip connection và Layer Norm:**

```
X_after_attn = LayerNorm(X + MHSA(X))
```

Cộng X vào output attention (giống ResNet) để tránh vanishing gradient khi xếp chồng nhiều layer.

---

#### Sub-block 2: MLP (Feed Forward Network)

Mỗi token, **độc lập**, đi qua một mạng 2 lớp FC:

```
Input: [197 × 768]

  ↓  FC1: [768 → 3072]  + GELU activation   ← mở rộng 4×
  ↓  FC2: [3072 → 768]                       ← nén về
  ↓  Skip connection + LayerNorm

Output: [197 × 768]
```

**Tại sao cần MLP sau Attention?**
- MHSA giỏi học **quan hệ giữa các token** (token nào chú ý đến token nào).
- MLP giỏi học **biến đổi phi tuyến bên trong từng token** (xử lý thông tin đã được tổng hợp từ attention).
- Hai việc này bổ sung cho nhau — bỏ MLP đi thì mạng mất khả năng biến đổi phi tuyến.

**Tại sao GELU thay vì ReLU?**
GELU (Gaussian Error Linear Unit) smooth hơn ReLU — không "chết cứng" như ReLU ở vùng âm, giúp gradient chảy tốt hơn qua nhiều layer Transformer.

---

### Bước 5 — Classification Head: Output cuối

Sau L lần qua Transformer Encoder:

```
Output: [197 × 768]  ← 197 token, mỗi token 768 chiều

→ Chỉ lấy token 0 — đó là [CLS] token: [768]

→ MLP Head:
   FC: [768 → num_classes]   (ví dụ 1000 lớp ImageNet)
   Softmax → xác suất từng lớp

→ Dự đoán: lớp có xác suất cao nhất
```

---

## 4. Tại sao Positional Encoding quan trọng — ví dụ trực quan

Tưởng tượng bạn bị đưa cho 196 mảnh ghép hình đã được cắt ra, nhưng **không biết mảnh nào ở đâu**. Bạn nhìn thấy: "mảnh này có lông vàng", "mảnh kia có màu trắng", "mảnh nọ có đường viền đen" — nhưng không biết chúng ở góc nào của bức tranh. Việc nhận ra là bức tranh chó hay mèo vẫn có thể làm được phần nào (dựa vào đặc trưng từng mảnh), nhưng sẽ rất khó hiểu hình dạng tổng thể.

Positional Encoding = ghi số thứ tự lên từng mảnh ghép hình trước khi đưa cho Transformer — nó biết mảnh 1 là góc trên trái, mảnh 14 là cuối hàng đầu, mảnh 15 là đầu hàng hai...

---

## 5. So sánh CNN vs ViT

| Tiêu chí | CNN (ResNet/EfficientNet) | ViT |
|---|---|---|
| **Đơn vị xử lý** | Pixel (convolution cục bộ) | Patch 16×16 (token toàn cục) |
| **Nhìn toàn cục** | Cần nhiều layer để lan dần | Ngay từ layer 1 (self-attention) |
| **Yêu cầu data** | Ít data → vẫn học được | Cần **rất nhiều data** (JFT-300M, ImageNet-21k) |
| **Inductive bias** | Cao (locality, translation equivariance) | Thấp — học từ data |
| **Tham số tiêu biểu** | 25M (ResNet-50) | 86M (ViT-Base) |
| **Tốc độ inference** | Nhanh trên GPU | Chậm hơn (attention quadratic với số patch) |
| **Transfer learning** | Dễ fine-tune ít data | Cần pretrain lớn, fine-tune mới ổn |

**Inductive bias là gì?**
CNN có sẵn giả định: "pixel gần nhau thường liên quan nhau" (locality) và "pattern như nhau ở mọi vị trí trong ảnh" (translation equivariance). Đây là tri thức được *hard-code* vào kiến trúc — giúp CNN học tốt với ít data hơn. ViT không có các giả định này — nó phải học tất cả từ data, nên cần data nhiều hơn rất nhiều.

---

## 6. Các biến thể ViT chính

| Model | Layers (L) | Hidden size (D) | Heads (h) | Tham số |
|---|---|---|---|---|
| **ViT-Base/16** | 12 | 768 | 12 | ~86M |
| **ViT-Large/16** | 24 | 1024 | 16 | ~307M |
| **ViT-Huge/14** | 32 | 1280 | 16 | ~632M |
| **DeiT-Base** | 12 | 768 | 12 | ~86M |
| **Swin-Base** | 4 stages | 128→1024 | 4→32 | ~88M |

`/16` = patch size 16×16, `/14` = patch size 14×14 (nhỏ hơn → nhiều patch hơn → chi tiết hơn → tốn hơn).

**DeiT (2021):** ViT train được trên ImageNet-1k (1.2M ảnh) nhờ augmentation mạnh và knowledge distillation từ CNN teacher — giải quyết điểm yếu "cần quá nhiều data" của ViT gốc.

**Swin Transformer (2021):** Thêm lại inductive bias kiểu CNN vào ViT — chia ảnh thành cửa sổ nhỏ (window), attention chỉ trong cửa sổ (local), sau đó shift window để kết nối vùng lân cận. Nhanh hơn ViT gốc, dễ áp dụng vào bài toán detection/segmentation hơn.

---

## 7. Toàn bộ luồng ViT-Base (tóm tắt số liệu)

```
Input ảnh:          [224 × 224 × 3]
                          │
Chia patch:         196 patch, mỗi patch [16×16×3=768]
                          │
Linear Projection:  [196 × 768]   (W: 768×768, ~590K params)
                          │
Thêm [CLS]:         [197 × 768]
                          │
+ Positional Enc:   [197 × 768]   (197 × 768 = ~151K params, học được)
                          │
               ╔═════════════════════╗
               ║  × 12 Encoder Blocks ║
               ║                     ║
               ║  MHSA:  ~2.4M/block ║
               ║  MLP:   ~4.7M/block ║
               ║  Total: ~7.1M/block ║
               ╚═════════════════════╝
                          │
Lấy [CLS] output:   [768]
                          │
MLP Head:           [768 → 1000]   (ImageNet)
                          │
Softmax:            [1000]  xác suất

Tổng tham số ViT-Base: ~86M
```

---

## 8. Use case

- **Khi có dữ liệu rất lớn (> 1M ảnh)** hoặc khi dùng pretrained ViT (DeiT/Swin đã pretrain sẵn) → ViT cạnh tranh hoặc vượt CNN.
- **Khi cần hiểu quan hệ toàn cục ngay từ đầu** — ví dụ nhận dạng đối tượng có mối quan hệ xa nhau trong ảnh (đếm người trong đám đông, phân tích bố cục phức tạp).
- **Bài toán QC công nghiệp với ít ảnh:** ưu tiên ResNet/EfficientNet pretrained — ViT cần fine-tune cẩn thận hơn và thường không chiếm ưu thế rõ ràng với < 10K mẫu.
- **Dùng Swin Transformer** nếu cần kết hợp giữa hiệu năng của ViT và khả năng áp dụng vào detection/segmentation (vì Swin tạo feature map đa tỉ lệ như CNN, tương thích với FPN).
