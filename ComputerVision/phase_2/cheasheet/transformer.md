# Transformer — "Attention Is All You Need" (Google, 2017)

Transformer không dùng RNN, không dùng CNN — nó chỉ dựa hoàn toàn vào **cơ chế Attention** để học mối quan hệ giữa các phần tử trong chuỗi. Đây là kiến trúc nền tảng của toàn bộ kỷ nguyên AI hiện đại: BERT, GPT, T5, và cả ViT.

---

## 1. Bối cảnh: Vấn đề của RNN trước Transformer

Trước Transformer, bài toán xử lý chuỗi (dịch máy, tóm tắt văn bản...) dùng **RNN (LSTM/GRU)**. RNN xử lý tuần tự — token 1 → token 2 → token 3 — và truyền thông tin qua một vector ẩn (hidden state).

**Hai vấn đề lớn của RNN:**

1. **Quên thông tin xa (vanishing gradient):** Muốn hiểu từ đầu câu ảnh hưởng đến từ cuối câu, gradient phải lan ngược qua hàng trăm bước — suy yếu dần, mạng "quên" ngữ cảnh xa.

2. **Không song song hóa được:** Token 3 phải chờ token 1 và 2 xử lý xong → không tận dụng được GPU để train nhanh trên tập dữ liệu lớn.

👉 Transformer giải quyết cả hai: mọi token **nhìn thấy nhau trực tiếp**, và toàn bộ chuỗi được xử lý **song song**.

---

## 2. Bức tranh tổng thể: Transformer gốc là Encoder-Decoder

Paper gốc "Attention Is All You Need" thiết kế Transformer cho bài toán **dịch máy** (sequence-to-sequence):

```
Câu tiếng Anh (input)      Câu tiếng Việt (output đang tạo)
        │                              │
   ┌────▼────┐                   ┌────▼────┐
   │ ENCODER │  ──── context ──► │ DECODER │ ──► token tiếp theo
   └─────────┘                   └─────────┘
```

- **Encoder:** Đọc toàn bộ câu input, nén thành biểu diễn ngữ nghĩa (context).
- **Decoder:** Nhận context từ Encoder + các token đã sinh ra trước đó → sinh token tiếp theo.

> **Lưu ý:** ViT chỉ dùng **Encoder** (không cần Decoder vì phân loại ảnh không cần sinh chuỗi). BERT cũng chỉ dùng Encoder. GPT chỉ dùng Decoder. Hiểu Encoder là đủ để hiểu ViT.

---

## 3. Pipeline Transformer Encoder — từng bước cực kỳ chi tiết

### Tổng sơ đồ Encoder:

```
Input tokens: ["Con", "mèo", "ngủ", "trên", "ghế"]
      │
      ▼
  Token Embedding       →  [5 × 512]  ← mỗi từ thành vector 512 chiều
      │
      ▼
  + Positional Encoding →  [5 × 512]  ← cho mạng biết thứ tự từng từ
      │
      ▼
  ┌──────────────────────────────────────────┐
  │  Encoder Block × N lần (N=6 trong paper) │
  │                                          │
  │  ┌────────────────────────────────────┐  │
  │  │  Layer Norm                        │  │
  │  │       ↓                            │  │
  │  │  Multi-Head Self-Attention (MHSA)  │  │
  │  │       ↓                            │  │
  │  │  (+) Skip Connection               │  │
  │  │       ↓                            │  │
  │  │  Layer Norm                        │  │
  │  │       ↓                            │  │
  │  │  Feed Forward Network (FFN/MLP)    │  │
  │  │       ↓                            │  │
  │  │  (+) Skip Connection               │  │
  │  └────────────────────────────────────┘  │
  └──────────────────────────────────────────┘
      │
      ▼
  Output: [5 × 512]  ← 5 token, mỗi token "đã hiểu ngữ cảnh toàn câu"
```

---

### Bước 1 — Token Embedding: Từ → Vector số

**Vấn đề:** Mạng neural không hiểu chữ — cần chuyển từ thành số.

```
Vocabulary (từ điển): 50,000 từ

"Con"  → index 1234  → vector [512 chiều]  (lookup bảng embedding)
"mèo"  → index 5678  → vector [512 chiều]
"ngủ"  → index 910   → vector [512 chiều]
...
```

Ma trận embedding `[50000 × 512]` được **học trong quá trình training** — các từ có nghĩa gần nhau sẽ có vector gần nhau trong không gian 512 chiều. Ví dụ: vector("vua") - vector("đàn ông") ≈ vector("nữ hoàng") - vector("phụ nữ").

**Sau bước này:**

```
Input: [5 × 512]   (5 từ, mỗi từ là vector 512 chiều)
```

---

### Bước 2 — Positional Encoding: Cho mạng biết thứ tự từ

**Vấn đề nghiêm trọng:** Self-Attention xử lý song song toàn bộ câu — nó không biết từ nào đứng trước, từ nào đứng sau. Câu "mèo ăn cá" và "cá ăn mèo" sẽ cho kết quả giống hệt nhau nếu không có positional encoding — rõ ràng sai.

**Giải pháp trong paper gốc — dùng hàm sin/cos (không học được):**

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Trong đó:
- `pos` = vị trí của token (0, 1, 2, ...)
- `i` = chiều thứ i của vector (0, 1, 2, ..., 255)
- `d_model` = 512

**Ví dụ trực quan:**

```
Vị trí 0 ("Con"):  [sin(0), cos(0), sin(0/100), cos(0/100), ...]  = [0, 1, 0, 1, ...]
Vị trí 1 ("mèo"):  [sin(1), cos(1), sin(1/100), cos(1/100), ...]  = [0.84, 0.54, 0.01, ...]
Vị trí 2 ("ngủ"):  [sin(2), cos(2), sin(2/100), cos(2/100), ...]  = [0.91, -0.42, 0.02, ...]
```

Mỗi vị trí có một "dấu tay" (fingerprint) duy nhất gồm 512 số — mạng học cách đọc fingerprint này để biết thứ tự.

**Tại sao dùng sin/cos thay vì số nguyên (0, 1, 2...)?**
Dùng số nguyên thô làm giá trị positional encoding sẽ tạo ra độ lệch lớn giữa các vị trí xa nhau và làm mất cân bằng với token embedding. Sin/cos luôn nằm trong [-1, 1], cùng tầm với embedding, và có tính tuần hoàn giúp mạng tổng quát hóa sang câu dài hơn khi inference.

**Cộng vào embedding:**

```
Token Embedding [5 × 512]
+
Positional Encoding [5 × 512]
=
[5 × 512]   ← shape không đổi, nhưng mỗi token giờ "mang theo" thông tin vị trí
```

---

### Bước 3 — Self-Attention: Trái tim của Transformer

**Mục tiêu:** Khi xử lý từ "ngủ", mạng cần biết **"Con"** và **"mèo"** là chủ thể đang ngủ — tức là từ "ngủ" cần "chú ý" nhiều đến "mèo" và "Con". Self-Attention học mối quan hệ này.

**Ví dụ ngữ nghĩa trước khi đi vào toán:**

```
Câu: "Con mèo ngủ trên ghế"

Khi xử lý từ "ngủ":
  → "ngủ" chú ý đến "mèo" rất nhiều  (0.72)  ← ai đang ngủ?
  → "ngủ" chú ý đến "Con"  vừa phải  (0.18)  ← bổ ngữ
  → "ngủ" chú ý đến "ghế"  ít         (0.07)  ← nơi chốn
  → "ngủ" chú ý đến "trên" rất ít    (0.03)
  → "ngủ" chú ý đến chính nó          (0.00)
                                      ------
                                       1.00   ← tổng = 1 (softmax)
```

#### Toán học của Self-Attention:

**Step 3.1 — Tạo Q, K, V từ mỗi token:**

```
Input X: [5 × 512]

Nhân với 3 ma trận trọng số học được:

  W_Q [512 × 64]  →  Q [5 × 64]   ← Query: "tôi đang tìm kiếm thông tin gì?"
  W_K [512 × 64]  →  K [5 × 64]   ← Key:   "tôi có thể cung cấp thông tin gì?"
  W_V [512 × 64]  →  V [5 × 64]   ← Value: "thông tin thực sự của tôi là gì?"
```

*(Lưu ý: Trong Multi-Head Attention với 8 heads, bước này thực chất dùng **8 BỘ MA TRẬN KHÁC NHAU**: $(W_{Q1}, W_{K1}, W_{V1})$, ..., $(W_{Q8}, W_{K8}, W_{V8})$. Do đó nó sinh ra 8 bộ $(Q, K, V)$ riêng biệt phản ánh 8 góc nhìn khác nhau về dữ liệu.)*

**Tại sao 64 chiều thay vì 512? — Ý tưởng Multi-Head**

Về lý thuyết có thể chạy 1 attention duy nhất trên 512 chiều — nhưng khi đó toàn bộ attention chỉ học được **1 loại quan hệ** giữa các token. Trong khi 1 câu có nhiều loại quan hệ cùng lúc:

```
Câu: "Con mèo trắng ngủ trên ghế gỗ"

Quan hệ chủ-vị:       "mèo"   → "ngủ"
Quan hệ tính từ-DT:   "trắng" → "mèo",  "gỗ" → "ghế"
Quan hệ vị trí:       "ngủ"   → "trên"  → "ghế"
```

**Giải pháp: chia 512 chiều thành 8 phần, mỗi phần học 1 loại quan hệ**

```
512 chiều tổng
──────────────────────────────────────────────────────
│ 64 chiều │ 64 chiều │ 64 chiều │ ... │ 64 chiều │
│ Head 1   │ Head 2   │ Head 3   │     │ Head 8   │
│ chủ-vị   │ tính từ  │ vị trí   │     │ ...      │
──────────────────────────────────────────────────────
    512 / 8 = 64 chiều mỗi head
```

> [!NOTE] 
> **Bản chất việc "học quan hệ":**
> Cách nói "Head 1 học chủ-vị" là **cách diễn giải trực quan (intuition)** của con người khi visualize mô hình đã train xong. Thực tế máy không biết ngữ pháp:
> - 8 Head được khởi tạo ma trận trọng số ngẫu nhiên khác nhau.
> - Qua huấn luyện (Backpropagation), để giảm hàm Loss hiệu quả, 8 Head tự động "chuyên môn hóa" vào các khía cạnh khác nhau (đặc tính tự nảy sinh - emergent property).
> - Việc chạy 8 Head này diễn ra **song song cùng lúc** bằng phép nhân ma trận tối ưu trên GPU, không phải qua vòng lặp `for`.

Mỗi head chạy attention **độc lập** trên 64 chiều của riêng nó, rồi ghép lại:

```
Head 1 → output [5 × 64]
Head 2 → output [5 × 64]
...
Head 8 → output [5 × 64]
         ↓  concatenate
      [5 × 512]    ← khôi phục về 512 chiều ban đầu
```

Con số 64 **không có gì đặc biệt** — chỉ là kết quả phép chia:

```
d_k = d_model / h = 512 / 8 = 64
```

Nếu chọn h=4 → d_k=128; h=16 → d_k=32. Paper chọn h=8 vì thực nghiệm cho kết quả tốt nhất.

**Analogy hệ thống tìm kiếm YouTube:**
- Bạn gõ "học lập trình Python" → đây là **Query** của bạn.
- Mỗi video có **Key** = tag/mô tả (để hệ thống so khớp với query).
- Khi tìm thấy video phù hợp, nội dung thực của video = **Value** (cái bạn thực sự nhận được).

**Step 3.2 — Tính Attention Score (độ tương đồng Q và K):**

```
Score = Q × K^T  →  [5 × 5]

Ví dụ hàng của từ "ngủ" (index 2):
         Con   mèo   ngủ  trên  ghế
Score = [3.2,  8.7,  1.1, 0.8,  1.4]
         ↑
         Giá trị cao = Query "ngủ" khớp tốt với Key "mèo"
```

> [!TIP]
> **Tại sao lại dùng phép nhân $Q \times K^T$?**
> Bản chất của phép toán này là tính **Tích vô hướng (Dot Product)** giữa vector Query của một từ và vector Key của tất cả các từ khác. 
> 
> Trong hình học, Tích vô hướng là một phép đo **độ tương đồng (similarity)** giữa 2 vector:
> - Nếu 2 vector hướng về cùng một phía (đặc trưng khớp nhau) → Tích vô hướng **lớn**.
> - Nếu 2 vector vuông góc (không liên quan gì nhau) → Tích vô hướng bằng **0**.
>
> Cụ thể với ví dụ từ "ngủ":
> 1. Từ "ngủ" mang theo vector $Q_{ngủ}$ (mang thông điệp ẩn: "Tôi là động từ, tôi cần tìm chủ thể thực hiện").
> 2. Nó lấy vector $Q_{ngủ}$ nhân vô hướng lần lượt với vector $K$ của tất cả các từ trong câu:
>    - $Q_{ngủ} \cdot K_{Con} = 3.2$
>    - $Q_{ngủ} \cdot K_{mèo} = 8.7$ (Cực kỳ khớp! $K_{mèo}$ báo hiệu nó là danh từ động vật phù hợp).
>    - $Q_{ngủ} \cdot K_{trên} = 0.8$ (Hoàn toàn lệch pha).
> 3. Kết quả thu được là các **điểm số thô (raw scores)**. Điểm càng cao, chứng tỏ hai từ càng "hợp cạ" ở khía cạnh mà Head này đang quan tâm.

**Step 3.3 — Scale (chia căn d_k):**

```
Score_scaled = Score / √64 = Score / 8

Tại sao? Khi chiều d_k lớn, tích vô hướng Q·K có phương sai lớn
→ các giá trị Score chênh lệch cực lớn
→ Softmax "cứng" gần như one-hot
→ gradient ≈ 0, mạng không học được

Chia √d_k để chuẩn hóa phương sai về ~1.
```

> [!NOTE] 
> **Chứng minh Toán học sơ bộ:**
> Giả sử các thành phần của vector Query ($q$) và Key ($k$) là các biến ngẫu nhiên độc lập, có kỳ vọng ($E$) bằng $0$ và phương sai ($Var$) bằng $1$.
> *(Hệ quả: $Var(X) = E[X^2] - (E[X])^2 \Rightarrow 1 = E[X^2] - 0 \Rightarrow E[X^2] = 1$)*
> 
> Tích vô hướng của chúng là: $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$
> 
> **1. Phương sai của 1 cặp số nhân nhau $Var(q_i k_i)$:**
> - Áp dụng công thức: $Var(q_i k_i) = E[(q_i k_i)^2] - (E[q_i k_i])^2$
> - Vì độc lập $\Rightarrow E[q_i k_i] = E[q_i] \times E[k_i] = 0 \times 0 = 0$.
> - Và $E[(q_i k_i)^2] = E[q_i^2 k_i^2] = E[q_i^2] \times E[k_i^2] = 1 \times 1 = 1$.
> $\Rightarrow Var(q_i k_i) = 1 - 0 = 1$.
> 
> **2. Phương sai của tổng (Tích vô hướng):**
> - Tích vô hướng là tổng của $d_k$ cặp như vậy. Theo tính chất phương sai của tổng các biến độc lập: 
>   $Var(q \cdot k) = \sum_{i=1}^{d_k} Var(q_i k_i) = 1 + 1 + ... + 1 = d_k$.
> 
> Vậy, điểm số thô ban đầu có phương sai là **$d_k$** (tức là dao động rất mạnh khi $d_k$ lớn).
> Theo định lý phương sai: $Var(X / c) = \frac{1}{c^2} Var(X)$.
> Để ép phương sai này về lại 1, ta cần chia biến $X$ (điểm tích vô hướng) cho một hằng số $c$:
> $Var(\frac{q \cdot k}{c}) = \frac{1}{c^2} Var(q \cdot k) = \frac{d_k}{c^2}$
> Để kết quả bằng 1 $\rightarrow c^2 = d_k \rightarrow c = \sqrt{d_k}$.
> 
> Đó là lý do toán học gốc rễ giải thích vì sao phép chia cho $\sqrt{d_k}$ lại kéo phương sai về chuẩn $\approx 1$, giúp hàm Softmax mượt mà hơn và tránh được lỗi triệt tiêu đạo hàm (Vanishing Gradient).

**Step 3.4 — Softmax → Attention Weight:**

```
Attention_weight = Softmax(Score_scaled)  →  [5 × 5]

Hàng của "ngủ" sau softmax:
         Con   mèo   ngủ  trên  ghế
weight = [0.18, 0.72, 0.03, 0.02, 0.05]
                ↑
         Tổng = 1.0  (phân phối xác suất)
```

**Step 3.5 — Nhân với V (Weighted Sum):**

```
Output = Attention_weight × V  →  [5 × 64]

Output của "ngủ" = 0.18 × V("Con") + 0.72 × V("mèo") + 0.03 × V("ngủ") + ...

→ Vector output của "ngủ" chủ yếu chứa thông tin của "mèo" (72%)
   kèm một chút thông tin của "Con" (18%) và các từ khác
→ Từ "ngủ" đã "thu thập" ngữ cảnh từ cả câu!
```

**Công thức gọn:**

```
Attention(Q, K, V) = Softmax(QK^T / √d_k) × V
```

---

### Bước 4 — Multi-Head Attention: Nhiều "góc nhìn" song song

**Vấn đề với 1 head:** Chỉ học được 1 loại quan hệ — ví dụ chỉ học quan hệ ngữ pháp chủ-vị. Nhưng trong câu có nhiều loại quan hệ cùng lúc.

**Giải pháp:** Chạy 8 attention **song song**, mỗi cái học một loại quan hệ khác nhau.

```
Input X: [5 × 512]

Head 1: Q1,K1,V1 (512→64) → output [5 × 64]  ← học quan hệ chủ-vị
Head 2: Q2,K2,V2 (512→64) → output [5 × 64]  ← học quan hệ tính từ-danh từ
Head 3: Q3,K3,V3 (512→64) → output [5 × 64]  ← học quan hệ động từ-trạng từ
...
Head 8: Q8,K8,V8 (512→64) → output [5 × 64]  ← học quan hệ tham chiếu (coreference)

Concatenate: [5 × 64×8] = [5 × 512]

Linear Projection W_O [512 × 512]  →  [5 × 512]   ← trộn thông tin từ 8 head
```

> [!TIP]
> **Tại sao Concat xong lại phải đi qua thêm một lớp Linear ($W^O$)?**
> Khi nối (concat) 8 Head lại, ta chỉ đang thực hiện thao tác cơ học: xếp 8 vector 64 chiều đứng cạnh nhau. Lúc này thông tin của 8 Head vẫn hoàn toàn rời rạc (cách ly với nhau). Lớp ma trận trọng số $W^O$ giải quyết 3 nhiệm vụ cực kỳ quan trọng:
> 1. **Trộn lẫn thông tin (Information Mixing):** Nhờ phép nhân ma trận, thông tin từ Head 1 (ngữ pháp) có thể giao thoa và kết hợp chéo với Head 2 (ngữ nghĩa)... biến 8 mảnh ghép rời rạc thành một vector thống nhất có ý nghĩa tổng hợp.
> 2. **Cân nhắc độ ưu tiên (Weighting):** $W^O$ chứa các tham số học được. Nó đóng vai trò như màng lọc cuối cùng: tự quyết định xem với từ đang xét thì nên ưu tiên/nhấn mạnh kết quả của Head nào, và triệt tiêu tín hiệu rác của Head nào.
> 3. **Đưa về không gian chuẩn:** Transformer gồm nhiều khối Layer xếp chồng lên nhau. $W^O$ đảm bảo output của khối Attention được đưa về lại đúng không gian 512 chiều gốc (đồng bộ về distribution), giúp tín hiệu luân chuyển mượt mà vào lớp Feed Forward phía sau mà không bị xô lệch đặc trưng.

**Skip Connection + Layer Norm:**

```
X_out = LayerNorm(X + MultiHead(X))   →  [5 × 512]
```

- **Skip connection (residual):** Cộng X gốc vào output → gradient chảy trực tiếp, tránh vanishing gradient khi xếp chồng 6-12 block.
- **Layer Norm (không phải Batch Norm):** Chữ "Layer" ở đây không có nghĩa là "5 lớp". Với ma trận đầu vào `X` kích thước `[5 × 512]` (tức là 5 token, mỗi token có 512 đặc trưng), Layer Norm sẽ hoạt động độc lập trên **từng token một**. Nó sẽ lấy 512 con số của token thứ nhất, tính trung bình và phương sai rồi chuẩn hóa dãy số đó. Tức là nó thực hiện thao tác chuẩn hóa 5 lần, mỗi lần quét ngang 1 hàng.

> [!TIP]
> **Tại sao Transformer dùng Layer Norm mà không phải Batch Norm?**
> - **Batch Norm (chuẩn hóa theo cột dọc):** Tính trung bình của đặc trưng thứ $j$ dọc theo tất cả các token trong toàn Batch. Nhưng trong NLP, các câu dài ngắn khác nhau, ta phải độn thêm các token rác (`<PAD>`) cho bằng nhau. Batch Norm sẽ cộng gộp cả các giá trị rác vô nghĩa này vào tính trung bình, làm hỏng hoàn toàn độ chính xác của mạng.
> - **Layer Norm (chuẩn hóa theo hàng ngang):** Thao tác chuẩn hóa được gói gọn trong nội bộ 512 đặc trưng của bản thân token đó. Nó không dòm ngó sang token bên cạnh, cũng không liên quan đến câu khác trong batch. Do đó, dù câu văn có dài ngắn ra sao hay bị chèn bao nhiêu padding đi chăng nữa, giá trị đặc trưng của token đang xét vẫn không bị ảnh hưởng!

---

### Bước 5 — Feed Forward Network (FFN): Xử lý phi tuyến

Sau khi Self-Attention "thu thập ngữ cảnh" cho mỗi token, FFN **biến đổi phi tuyến** thông tin đó.

```
Input: [5 × 512]

Mỗi token xử lý ĐỘC LẬP qua 2 lớp FC:

  FC1: [512 → 2048]  + ReLU   ← mở rộng 4× (paper gốc dùng ReLU, ViT dùng GELU)
  FC2: [2048 → 512]            ← nén về

Output: [5 × 512]
```

**Tại sao cần FFN sau Attention?**

| Attention | FFN |
|---|---|
| Học **quan hệ giữa các token** | Học **biến đổi phi tuyến trong từng token** |
| Token "thu thập" thông tin từ cả câu | Token "tiêu hóa" thông tin vừa thu thập |
| Tuyến tính (Q,K,V đều là phép nhân ma trận) | Phi tuyến (nhờ activation ReLU/GELU) |

Attention thuần túy là phép biến đổi tuyến tính — không có phi tuyến tính thì xếp chồng bao nhiêu layer cũng tương đương với 1 layer. FFN thêm phi tuyến tính thiết yếu.

**Skip Connection + Layer Norm:**

```
X_final = LayerNorm(X_attn_out + FFN(X_attn_out))  →  [5 × 512]
```

---

### Bước 6 — Xếp chồng N block

Paper gốc dùng N=6 block. Output của block này là input của block tiếp theo.

```
Block 1: [5 × 512] → [5 × 512]   (học quan hệ đơn giản, cục bộ)
Block 2: [5 × 512] → [5 × 512]   (học quan hệ phức tạp hơn)
...
Block 6: [5 × 512] → [5 × 512]   (học quan hệ ngữ nghĩa trừu tượng)

Mỗi block, từng token lại "hỏi thăm" tất cả token khác với "trình độ hiểu biết" cao hơn.
```

---

## 4. Decoder — Kẻ sinh chuỗi (Dành cho Dịch máy, Chatbot)

> *Ghi chú: Nếu bạn chỉ học ViT (Vision Transformer) thì có thể bỏ qua phần này vì ViT chỉ xài Encoder. Nhưng nếu bạn muốn hiểu ChatGPT hoạt động thế nào thì đây là phần cốt lõi!*

Khác với Encoder (đọc hết cả câu một lúc để lấy ngữ cảnh), Decoder hoạt động theo cơ chế **Autoregressive (tự hồi quy)** — tức là sinh ra từng từ một, lấy từ vừa sinh ra làm đầu vào cho vòng lặp tiếp theo. 

Để làm được điều đó, 1 Block của Decoder có tới **3 bước xử lý (thay vì 2 như Encoder)**:

**Bước 1: Masked Self-Attention (Tự chú ý... nhưng bị che tương lai)**
Giả sử Decoder đang dịch câu "I love you" sang "Tôi yêu em". Hiện tại nó đã sinh ra chữ "Tôi", và đang xử lý để đẻ ra chữ "yêu".
- Ở bước này, chữ "yêu" chỉ được phép tính Attention (nhìn lại) chữ "Tôi" ở quá khứ. Nó **BỊ CẤM** nhìn thấy chữ "em" ở tương lai (vì chữ "em" thực tế chưa được sinh ra).
- *Thủ thuật toán học:* Người ta đắp một cái mặt nạ (Mask). Các vị trí từ ở tương lai bị đánh điểm Score là $-\infty$. Khi đi qua hàm Softmax, $-\infty$ sẽ bị ép về $0\%$. Nhờ vậy, từ hiện tại không thể "ăn gian" nhìn trộm tương lai.

**Bước 2: Cross-Attention (Bắc cầu sang hỏi Encoder)**
Đây là bước giao tiếp giữa hai bán cầu não.
- Decoder mang vector của từ hiện tại (đóng vai trò **Query** - câu hỏi).
- Nó đem câu hỏi này chạy sang kho dữ liệu của Encoder (nơi chứa bản mã hóa trọn vẹn của câu gốc "I love you" — đóng vai trò là **Key** và **Value**).
- Phép tính Attention diễn ra: Query của Decoder rà soát xem nó khớp với Key nào bên Encoder nhất, từ đó "hút" Value (ngữ nghĩa) tương ứng mang về. Đây là cách Decoder biết bám sát vào nội dung gốc để dịch không bị lạc đề.

**Bước 3: Feed Forward Network (FFN)**
Giống hệt bên Encoder. Sau khi nhào nặn thông tin quá khứ (từ Bước 1) và thông tin gốc (từ Bước 2), dữ liệu được đưa qua FFN để "tiêu hóa" và biến đổi phi tuyến (tìm ra quy luật sâu xa hơn).

**Bước 4: Sinh từ tiếp theo (Vòng lặp ngoài cùng)**
Sau khi chui qua $N$ block Decoder như trên, vector đầu ra cuối cùng sẽ đi qua một lớp Mạng nơ-ron tuyến tính (Linear) để phóng to kích thước bằng đúng số lượng từ trong từ điển (ví dụ 50,000 từ). Cuối cùng, hàm Softmax sẽ biến 50,000 con số này thành xác suất phần trăm (tổng = 1). Từ nào có % cao nhất sẽ được chọn làm từ tiếp theo!

>Cứ thế, từ vừa được chọn lại chui ngược lại vào Decoder để sinh ra từ kế tiếp, cho đến khi máy tính tự đẻ ra thẻ `<EOS>` (End Of Sentence - Kết thúc câu).
---

## 5. Sơ đồ đầy đủ toàn bộ Transformer

```
INPUT (tiếng Anh):  "The cat sleeps"
                          │
                   Token Embedding
                   + Pos. Encoding
                          │  [3 × 512]
                ┌─────────▼─────────┐
                │   ENCODER STACK   │
                │  ┌─────────────┐  │
                │  │  Block 1    │  │
                │  │  MHSA+FFN   │  │
                │  └──────┬──────┘  │
                │         │  × 6    │
                │  ┌──────▼──────┐  │
                │  │  Block 6    │  │
                │  │  MHSA+FFN   │  │
                │  └─────────────┘  │
                └────────┬──────────┘
                         │ Context [3 × 512]
                         │
OUTPUT (tiếng Việt đang sinh): "Con mèo ___"
                          │
                   Token Embedding
                   + Pos. Encoding
                          │  [2 × 512]
                ┌─────────▼─────────┐
                │   DECODER STACK   │
                │  ┌─────────────┐  │
                │  │  Block 1    │  │
                │  │  Masked MHSA│  │
                │  │  Cross-Attn ◄──── Context từ Encoder
                │  │  FFN        │  │
                │  └──────┬──────┘  │
                │         │  × 6    │
                │  ┌──────▼──────┐  │
                │  │  Block 6    │  │
                │  └─────────────┘  │
                └────────┬──────────┘
                         │  [2 × 512]
                    Linear [512 → vocab_size]
                    Softmax
                         │
                  "ngủ"  ← token tiếp theo được dự đoán
```

---

## 6. Tóm tắt các thành phần và vai trò

| Thành phần | Vai trò | Đặc điểm quan trọng |
|---|---|---|
| **Token Embedding** | Từ → vector số | Ma trận 50K×512, học được |
| **Positional Encoding** | Mã hóa vị trí từ | Sin/cos (gốc) hoặc học được (ViT) |
| **Q, K, V** | 3 "phiên bản" của token để tính attention | Mỗi cái là phép chiếu tuyến tính |
| **Attention Score** | Độ tương đồng Q·K | Scale bằng √d_k để ổn định |
| **Softmax** | Chuyển score thành phân phối xác suất | Tổng = 1, tập trung vào token quan trọng |
| **Multi-Head** | Nhiều loại attention song song | 8 head × 64 chiều = 512 chiều |
| **FFN** | Biến đổi phi tuyến từng token | Mở rộng 4× rồi nén về |
| **Skip Connection** | Tránh vanishing gradient | Cộng thẳng input vào output |
| **Layer Norm** | Ổn định training | Chuẩn hóa trên từng token |
| **Masked Attention** | Chỉ nhìn quá khứ (Decoder) | Che tương lai bằng -∞ |
| **Cross-Attention** | Kết nối Decoder với Encoder | Q từ Decoder, K/V từ Encoder |

---

## 7. Độ phức tạp tính toán — điểm yếu của Attention

Self-Attention tính Score matrix `Q × K^T` có kích thước **[n × n]** với n = độ dài chuỗi.

```
Chi phí tính toán = O(n²  × d)
Chi phí bộ nhớ   = O(n²)
```

Với câu 512 token → ma trận 512×512 = 262,144 phần tử → OK.
Với câu 10,000 token → ma trận 10K×10K = 100,000,000 phần tử → RAM explode.

**Đây là lý do ViT phải chia ảnh thành patch 16×16 thay vì xử lý từng pixel:**
- Ảnh 224×224 = 50,176 pixel → attention matrix 50K×50K → không thể tính được.
- Chia thành patch → 196 token → attention matrix 196×196 → hoàn toàn ổn.

---

## 8. Tóm tắt số liệu Transformer gốc (paper "Attention Is All You Need")

```
d_model    = 512       ← chiều vector embedding
d_ff       = 2048      ← chiều mở rộng trong FFN (4 × d_model)
h          = 8         ← số attention head
d_k = d_v  = 64       ← chiều mỗi head (d_model / h)
N          = 6         ← số block Encoder và Decoder

Tham số Encoder: ~31M
Tham số Decoder: ~31M
Tổng: ~62M tham số (bản Base)
```

---

## 9. Từ Transformer đến ViT — 1 bước nhỏ về kiến trúc

| | Transformer Encoder gốc (NLP) | ViT |
|---|---|---|
| **Input** | Chuỗi token từ | Chuỗi patch từ ảnh |
| **Embedding** | Word Embedding (50K từ → 512) | Linear Projection (768 pixel → 768) |
| **Pos. Encoding** | Sin/cos cố định | Học được (trainable) |
| **Token đặc biệt** | Không có (hoặc [SEP] trong BERT) | [CLS] token ở đầu |
| **Số block** | 6 (paper gốc) | 12 (ViT-Base) |
| **d_model** | 512 | 768 |
| **Heads** | 8 | 12 |
| **FFN activation** | ReLU | GELU |
| **Output** | Sequence embedding | [CLS] token → phân loại |

👉 ViT về bản chất là **Transformer Encoder gốc với input khác** — hiểu Transformer rồi thì ViT chỉ là thay "từ" bằng "patch ảnh".
