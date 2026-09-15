# Cheatsheet Toàn Diện: ROC-AUC vs. PR-AUC (Precision-Recall AUC)

> **Mục tiêu cốt lõi**: Nắm vững bản chất toán học, cách dựng đường cong từ bảng số liệu thực tế, hiểu sâu **"Ảo giác ROC" (The ROC Illusion)** trong dữ liệu mất cân bằng nghiêm trọng, và lý do vì sao **Object Detection bắt buộc phải dùng PR-AUC (AP/mAP)** thay vì ROC-AUC.

---

## 1. Nền tảng bắt buộc: Confusion Matrix & Bản chất 4 góc nhìn

Để đánh giá một mô hình phân loại nhị phân (Binary Classification) hoặc phát hiện vật thể xuất điểm xác suất (Score từ $0.0 \to 1.0$), ta cần đặt ra một **ngưỡng quyết định (Classification Threshold $\tau$)**:
- Nếu $\text{Score} \ge \tau \implies$ Dự đoán là **Dương (Positive)**.
- Nếu $\text{Score} < \tau \implies$ Dự đoán là **Âm (Negative)**.

Tại bất kỳ ngưỡng $\tau$ cố định nào, ta thu được bảng **Confusion Matrix (Ma trận nhầm lẫn)**:

```
                      THỰC TẾ (Ground Truth)
                   Dương (P)          Âm (N)
              ┌──────────────────┬──────────────────┐
    Dương (P) │  True Positive   │  False Positive  │ ◄── Tổng đoán Dương
DỰ            │       (TP)       │       (FP)       │     = TP + FP
ĐOÁN          ├──────────────────┼──────────────────┤
     Âm (N)   │  False Negative  │  True Negative   │ ◄── Tổng đoán Âm
              │       (FN)       │       (TN)       │     = FN + TN
              └──────────────────┴──────────────────┘
                 ▲                  ▲
                 │                  │
            Tổng mẫu Dương     Tổng mẫu Âm
            thực tế: P         thực tế: N
            = TP + FN          = FP + TN
```

### 3 Chỉ số then chốt dẫn xuất:

1. **TPR (True Positive Rate) / Recall / Sensitivity (Độ nhạy)**:
   $$\text{TPR} = \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{\text{TP}}{P}$$
   - *Ý nghĩa đời thực*: Trong tổng số mẫu thực sự Dương (như ca có bệnh, đồ vật thật, gian lận thật), mô hình **bắt trúng được bao nhiêu %**?

2. **FPR (False Positive Rate) / Fall-out (Tỷ lệ báo động giả)**:
   $$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} = \frac{\text{FP}}{N}$$
   - *Ý nghĩa đời thực*: Trong tổng số người thực sự Khỏe mạnh (Âm thật), mô hình **báo động giả nhầm bao nhiêu %**?

3. **Precision / Positive Predictive Value (Độ chuẩn xác)**:
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
   - *Ý nghĩa đời thực*: Trong tất cả những ca mô hình vung tay hô "DƯƠNG TÍNH", thì **có bao nhiêu % là Dương thật**?

> [!IMPORTANT]
> **Điểm khác biệt chí mạng giữa FPR và Precision:**
> - $\text{FPR}$ chia cho $N = \text{FP} + \text{TN}$ (Tổng số mẫu **ÂM THẬT**).
> - $\text{Precision}$ chia cho $\text{TP} + \text{FP}$ (Tổng số mẫu **MÔ HÌNH BÁO DƯƠNG**), hoàn toàn **KHÔNG quan tâm tới $TN$**!

---

## 2. Dữ liệu thực nghiệm mẫu: Quét ngưỡng từng bước

Xét bài toán có **10 mẫu dữ liệu**, thực tế có **3 Dương ($P=3$)** và **7 Âm ($N=7$)** (tỷ lệ dương $30\%$, mất cân bằng nhẹ):

| Thứ hạng | Score dự đoán | Nhãn thực tế (Ground Truth) |
| :---: | :---: | :---: |
| **#1** | **0.95** | **Dương (+)** |
| **#2** | **0.90** | **Dương (+)** |
| **#3** | 0.85 | Âm (-) |
| **#4** | **0.80** | **Dương (+)** |
| **#5** | 0.70 | Âm (-) |
| **#6** | 0.60 | Âm (-) |
| **#7** | 0.50 | Âm (-) |
| **#8** | 0.40 | Âm (-) |
| **#9** | 0.30 | Âm (-) |
| **#10** | 0.10 | Âm (-) |

### Bảng tính chi tiết khi hạ ngưỡng $\tau$ từ cao xuống thấp:

*Nguyên tắc: Khi hạ ngưỡng, những mẫu có score $\ge \tau$ sẽ được gán là Dương.*

| Ngưỡng $\tau$ | Mẫu mới nhận vào | $\text{TP}$ | $\text{FP}$ | $\text{FN}$ | $\text{TN}$ | $\text{TPR = Recall}$ ($\frac{\text{TP}}{3}$) | $\text{FPR}$ ($\frac{\text{FP}}{7}$) | $\text{Precision}$ ($\frac{\text{TP}}{\text{TP}+\text{FP}}$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ban đầu ($\tau > 1$)** | *(Chưa nhận mẫu nào)* | 0 | 0 | 3 | 7 | **0.00** | **0.00** | **1.00** *(quy ước)* |
| **$\ge 0.95$** | #1 (Dương) | 1 | 0 | 2 | 7 | **0.33** | **0.00** | **1.00** ($1/1$) |
| **$\ge 0.90$** | #2 (Dương) | 2 | 0 | 1 | 7 | **0.67** | **0.00** | **1.00** ($2/2$) |
| **$\ge 0.85$** | #3 (Âm) | 2 | 1 | 1 | 6 | **0.67** | **0.14** | **0.67** ($2/3$) |
| **$\ge 0.80$** | #4 (Dương) | 3 | 1 | 0 | 6 | **1.00** | **0.14** | **0.75** ($3/4$) |
| **$\ge 0.70$** | #5 (Âm) | 3 | 2 | 0 | 5 | **1.00** | **0.29** | **0.60** ($3/5$) |
| **$\ge 0.60$** | #6 (Âm) | 3 | 3 | 0 | 4 | **1.00** | **0.43** | **0.50** ($3/6$) |
| **$\ge 0.50$** | #7 (Âm) | 3 | 4 | 0 | 3 | **1.00** | **0.57** | **0.43** ($3/7$) |
| **$\ge 0.40$** | #8 (Âm) | 3 | 5 | 0 | 2 | **1.00** | **0.71** | **0.38** ($3/8$) |
| **$\ge 0.30$** | #9 (Âm) | 3 | 6 | 0 | 1 | **1.00** | **0.86** | **0.33** ($3/9$) |
| **$\ge 0.10$** | #10 (Âm) | 3 | 7 | 0 | 0 | **1.00** | **1.00** | **0.30** ($3/10$) |

---

## 3. ROC Curve (Receiver Operating Characteristic) & ROC-AUC

- **Trục tung ($Y$)**: $\text{TPR}$ (True Positive Rate / Recall) — Bắt được bao nhiêu % mẫu dương thật?
- **Trục hoành ($X$)**: $\text{FPR}$ (False Positive Rate) — Báo động nhầm bao nhiêu % mẫu âm thật?

### Sơ đồ trực quan ROC Curve:

```
TPR (Recall)
 1.0 ┼───────────────────●──●──●──●──●──●──● (1.0, 1.0)
     │                 ／┆
 0.8 │               ／  ┆  ◄── Điểm (0.14, 1.00) tại ngưỡng 0.80
     │             ●     ┆
 0.6 │             │     ┆      ╭─────────────────────────────────╮
     │             │     ┆      │ ★ Điểm lý tưởng: (FPR=0, TPR=1) │
 0.4 │             │    ／      │   (Góc trên cùng bên trái)      │
     │             ●   ／       │                                 │
 0.2 │             │ ／         │ --- Đường chéo: Đoán mò ngẫu     │
     │             │／          │     nhiên (Random Guess, AUC=0.5│
 0.0 ┼─────────────●────────────┴─────────────────────────────────┘
    0.0           0.2   0.4   0.6   0.8   1.0   FPR (False Alarm Rate)
```

### Đặc điểm cốt lõi của ROC:
1. **Điểm xuất phát & kết thúc**: Luôn bắt đầu từ $(0, 0)$ (ngưỡng cực cao $\tau > 1$, không đoán ai là dương) và kết thúc tại $(1, 1)$ (ngưỡng cực thấp $\tau \le 0.1$, đoán tất cả đều là dương).
2. **Đường cơ sở ngẫu nhiên (Random Baseline)**: Là **đường chéo nối từ $(0, 0) \to (1, 1)$** với diện tích $\text{AUC} = 0.5$. Một mô hình đoán mò ngẫu nhiên sẽ có xác suất đoán đúng bằng xác suất đoán sai ($\text{TPR} = \text{FPR}$) ở mọi ngưỡng.
3. **Ý nghĩa xác suất của ROC-AUC**:
   $$\text{ROC-AUC} = P\big(\text{Score}(P_{\text{ngẫu nhiên}}) > \text{Score}(N_{\text{ngẫu nhiên}})\big)$$
   *Nói một cách dân dã*: Nếu bốc ngẫu nhiên 1 người bệnh và 1 người khỏe, $\text{ROC-AUC}$ chính là xác suất mô hình chấm điểm người bệnh cao hơn người khỏe.
4. **Tính chất bất biến (Class Prevalence Invariant)**: ROC **không thay đổi** nếu tỷ lệ nhãn Dương/Âm trong tập dữ liệu thay đổi, bởi vì:
   - $\text{TPR}$ chỉ tính dựa trên cột Dương ($P$).
   - $\text{FPR}$ chỉ tính dựa trên cột Âm ($N$).
   - Hai cột này hoàn toàn độc lập với nhau.

---

## 4. PR Curve (Precision-Recall) & PR-AUC (Average Precision - AP)

- **Trục tung ($Y$)**: $\text{Precision}$ — trong các ca báo dương, bao nhiêu % đúng?
- **Trục hoành ($X$)**: $\text{Recall}$ ($\text{TPR}$) — bắt được bao nhiêu % ca dương trong thực tế?

### Sơ đồ trực quan PR Curve:

```
Precision
 1.0 ┼──●──────────●
     │             │\
 0.8 │             │ \   ● (Recall=1.0, Precision=0.75)
     │             │  \  │\
 0.6 │             │   ● │ \  ● (Recall=1.0, Precision=0.60)
     │             │     │  \ │\
 0.4 │             │     │   ●● \●
 0.3 ┼- - - - - - - - - - - - - - ● ◄── BASELINE ĐOÁN MÒ (P / Total = 3/10 = 0.3)
 0.2 │
 0.0 ┼────────────────────────────────────
    0.0           0.33  0.67      1.0   Recall (TPR)
```

### Đặc điểm của PR Curve:
1. **Khu vực lý tưởng**: Góc **trên cùng bên phải** $(Recall=1.0, Precision=1.0)$ — vừa bắt hết không sót ai, vừa không bắt nhầm ai.
2. **Đường cơ sở ngẫu nhiên (Random Baseline)**: 
   > [!CAUTION]
   > **Rất nhiều người nhầm lẫn!** Đường cơ sở của PR **KHÔNG PHẢI là đường chéo $0.5$**!
   > Đường cơ sở của PR là **một đường nằm ngang** có giá trị bằng chính **tỷ lệ lớp Dương trong dữ liệu**:
   $$\text{Baseline}_{\text{PR}} = \frac{P}{P + N}$$
   - Trong ví dụ này: $3 / 10 = 0.3$.
   - Nếu bài toán ung thư tỷ lệ bệnh là $0.1\%$, baseline của PR-AUC sẽ là **$0.001$**!
3. **Hình dạng zíc-zắc**: Đường PR không nhất thiết phải đơn điệu giảm. Khi hạ ngưỡng để bắt thêm mẫu, nếu gặp mẫu đúng thì Recall tăng và Precision có thể tăng nhẹ lại; nếu gặp mẫu sai thì Precision tụt dốc.
4. **PR-AUC chính là AP (Average Precision)**: Trong Object Detection (VOC / COCO), diện tích dưới đường cong PR được làm mịn (Interpolation) chính là chỉ số chuẩn **mAP** (mean Average Precision).

---

## 5. Cốt lõi: "Ảo Giác ROC" (The ROC Illusion) khi Mất Cân Bằng Dữ Liệu

Tại sao không dùng ROC-AUC cho tất cả mọi bài toán mà phải sinh ra PR-AUC? Hãy so sánh qua một kịch bản dữ liệu mất cân bằng cực độ.

### Kịch bản thực tế: Hệ thống phát hiện gian lận thẻ tín dụng
- Tổng giao dịch: $100,100$ giao dịch.
- **Gian lận thật ($P$)**: $100$ giao dịch.
- **Bình thường ($N$)**: $100,000$ giao dịch ($N$ gấp $1000$ lần $P$).

Mô hình thiết lập ngưỡng và dự đoán ra kết quả:
- Bắt được $80$ vụ gian lận $\implies \text{TP} = 80, \text{FN} = 20 \implies \text{Recall} = 80\%$.
- Báo động nhầm $2,000$ giao dịch lành tính thành gian lận $\implies \text{FP} = 2,000, \text{TN} = 98,000$.

### Bây giờ hãy tính ROC vs PR:

| Chỉ số | Công thức tính | Kết quả | Cảm giác mang lại |
|---|---|:---:|---|
| **$\text{FPR}$ (Trục X của ROC)** | $\frac{\text{FP}}{N} = \frac{2,000}{100,000}$ | **$2\%$ (0.02)** | **Cực kỳ xuất sắc!** Báo nhầm có $2\%$ số người vô tội. Đường ROC bám sát vách trục tung, $\text{ROC-AUC} \approx 0.98$. Sếp khen mô hình đỉnh cao! |
| **$\text{Precision}$ (Trục Y của PR)** | $\frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{80}{80 + 2,000}$ | **$3.85\%$ (0.0385)** | **Thảm họa thực tế!** Cứ 100 lần hệ thống bấm chuông báo động khóa thẻ, thì có tới hơn 96 lần là khóa nhầm khách vip vô tội. Đội hỗ trợ khách hàng vỡ trận! |

```
                       CÙNG MỘT MÔ HÌNH:
        Góc nhìn ROC                     Góc nhìn PR
┌───────────────────────────┐    ┌───────────────────────────┐
│ FPR = 0.02 (Rất nhỏ!)     │    │ Precision = 0.038 (Cực tệ)│
│ ROC-AUC ~ 0.98            │    │ PR-AUC ~ 0.05             │
│                           │    │                           │
│ "Mô hình gần như hoàn hảo"│    │ "Hệ thống toàn báo động   │
│ (Ảo giác do TN quá lớn!)  │    │  giả, không dùng được!"   │
└───────────────────────────┘    └───────────────────────────┘
```

> [!WARNING]
> **Kết luận về Ảo giác ROC**:
> Khi tập âm tính $N$ quá khổng lồ, mẫu số của $\text{FPR} = \frac{\text{FP}}{N}$ bị kéo lên cực lớn. Cho dù $\text{FP}$ có tăng vọt từ $100 \to 2,000$ (tăng 20 lần báo động giả), $\text{FPR}$ vẫn chỉ nhích từ $0.1\% \to 2\%$. **ROC-AUC hoàn toàn bị "mù" trước sự bùng nổ của False Positives.**
> Trong khi đó, $\text{Precision}$ phản ứng cực kỳ nhạy bén vì mẫu số của nó là $(\text{TP} + \text{FP})$ — nó trừng phạt thẳng tay sự gia tăng của $\text{FP}$!

---

## 6. Vì sao Object Detection (YOLO, FCOS) BẮT BUỘC dùng PR-AUC (mAP)?

Trong thị giác máy tính, Object Detection là bài toán **mất cân bằng nền/vật thể (Foreground/Background Imbalance) ở mức độ vô cực**:

1. **Khái niệm $TN$ không tồn tại hoặc bằng vô tận**:
   - Một ảnh kích thước $800 \times 800$ có $640,000$ pixel. Một mô hình như FCOS hay RetinaNet đánh giá hàng chục nghìn điểm/anchor.
   - Một con mèo trong ảnh chỉ chiếm chừng vài trăm điểm (Positive).
   - Còn lại hàng vạn vùng không chứa mèo là điểm Âm. Nếu coi mỗi pixel trống hay mỗi bounding box vô định là 1 $TN$, thì $TN \to \infty$!
2. **Hậu quả nếu dùng ROC-AUC trong Object Detection**:
   $$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} \approx \frac{\text{FP}}{\infty} \approx 0$$
   - Ở mọi ngưỡng confidence, $\text{FPR}$ sẽ luôn xấp xỉ bằng $0$.
   - Đường ROC của mọi model (kể cả model rất dở phát hiện box lung tung) sẽ luôn nằm ép chặt vào trục $Y$ và $\text{ROC-AUC}$ luôn đạt $\approx 0.999$.
   - **Chỉ số ROC hoàn toàn mất tính phân loại để đánh giá detector nào tốt hơn!**
3. **PR-AUC (Average Precision - AP) là vị cứu tinh**:
   - $AP$ chỉ quan tâm:
     - Model tìm ra bao nhiêu vật thể thật? ($\text{Recall}$)
     - Trong số các bounding box model vẽ ra, bao nhiêu box vẽ trúng (IoU $\ge 0.5$)? ($\text{Precision}$)
   - Hàng triệu pixel nền trống ($TN$) không bao giờ xuất hiện trong công thức của Precision và Recall. Do đó, AP phản ánh trung thực $100\%$ chất lượng phát hiện của detector.

---

## 7. Bảng so sánh & Quyết định: Khi nào dùng cái nào?

| Tiêu chí | ROC-AUC | PR-AUC (Average Precision) |
|---|---|---|
| **Trục tọa độ** | $Y: \text{TPR}$, $X: \text{FPR}$ | $Y: \text{Precision}$, $X: \text{Recall}$ |
| **Random Baseline** | Luôn cố định là **đường chéo $= 0.5$** | **Đường ngang $= \frac{P}{P+N}$** (thay đổi theo dữ liệu) |
| **Sự phụ thuộc vào $TN$** | **Có** (mẫu số của FPR có $TN$) | **Hoàn toàn KHÔNG** quan tâm đến $TN$ |
| **Dữ liệu cân bằng (50/50)** | Rất tốt, chuẩn mực | Rất tốt, tương đương ROC |
| **Dữ liệu mất cân bằng nặng (1:100, 1:1000)** | **Bị ảo giác** (quá lạc quan, che giấu FP) | **Trung thực, phản ánh chính xác** chất lượng |
| **Độ nhạy khi đổi tỷ lệ test set** | Không đổi (Invariant) | Đổi theo tỷ lệ dương của test set |
| **Ứng dụng thực tế điển hình** | Chẩn đoán y tế tổng quát (chi phí FP và FN ngang nhau), phân loại cân bằng | **Object Detection (COCO mAP)**, Phát hiện gian lận tài chính, Tìm kiếm thông tin (Search Engine), Lọc Spam |

---

## 8. Thực chiến Python: Tự vẽ & tính cả 2 đường cong

Đoạn mã Python sử dụng thư viện `scikit-learn` và `matplotlib` để tái hiện chính xác ví dụ 10 mẫu trên:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

# 1. Dữ liệu từ ví dụ 10 mẫu
# 1: Dương, 0: Âm
y_true = np.array([1,    1,    0,    1,    0,    0,    0,    0,    0,    0])
scores = np.array([0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.10])

# 2. Tính toán ROC và AUC
fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
roc_auc = roc_auc_score(y_true, scores)

# 3. Tính toán PR và AP (PR-AUC)
precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)
ap_score = average_precision_score(y_true, scores)
baseline_pr = np.sum(y_true) / len(y_true)  # 3/10 = 0.3

# 4. Trực quan hóa
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot ROC Curve ---
axes[0].plot(fpr, tpr, color='#1f77b4', lw=2.5, marker='o', label=f'ROC curve (AUC = {roc_auc:.3f})')
axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random Baseline (AUC = 0.50)')
axes[0].set_xlim([-0.02, 1.02])
axes[0].set_ylim([-0.02, 1.05])
axes[0].set_xlabel('False Positive Rate (FPR)', fontsize=12)
axes[0].set_ylabel('True Positive Rate (TPR / Recall)', fontsize=12)
axes[0].set_title('ROC Curve (TPR vs FPR)', fontsize=14, fontweight='bold')
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend(loc="lower right", fontsize=11)

# --- Plot PR Curve ---
axes[1].step(recall, precision, color='#d62728', lw=2.5, where='post', marker='o', label=f'PR curve (AP = {ap_score:.3f})')
axes[1].axhline(y=baseline_pr, color='gray', linestyle='--', lw=1.5, label=f'Random Baseline (P/Total = {baseline_pr:.2f})')
axes[1].set_xlim([-0.02, 1.02])
axes[1].set_ylim([-0.02, 1.05])
axes[1].set_xlabel('Recall', fontsize=12)
axes[1].set_ylabel('Precision', fontsize=12)
axes[1].set_title('Precision-Recall Curve (PR-AUC / AP)', fontsize=14, fontweight='bold')
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend(loc="lower left", fontsize=11)

plt.tight_layout()
plt.show()

print(f"ROC-AUC Score : {roc_auc:.4f}")
print(f"PR-AUC (AP)   : {ap_score:.4f}")
```

---

## 9. Câu thần chú phỏng vấn cần nhớ nằm lòng

> 1. **"ROC dùng khi âm dương cân đối, hoặc quan tâm đến năng lực xếp hạng tổng thể."**
> 2. **"PR dùng khi lớp dương hiếm (imbalance), hoặc khi chi phí của False Alarm (báo động giả) là rất đắt."**
> 3. **"Object Detection không có TN (True Negative vô hạn), nên ROC-AUC luôn bằng 1 và vô dụng $\implies$ Bắt buộc phải dùng mAP (PR-AUC)!"**
