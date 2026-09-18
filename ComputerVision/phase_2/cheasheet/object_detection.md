
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

     #### ❓ 2 Vấn đề hóc búa của Anchor-Free & Cách giải quyết

     ##### 1. Bài toán 1: "Một vật thể có cả trăm điểm $(x, y)$ nằm bên trong, điểm nào cũng đòi đoán thì tin ai?"
     - **Hiện tượng:** Điểm nằm ngay giữa tâm vật thể nhìn ra 4 phía rất cân đối nên đoán kích thước cực chuẩn. Nhưng điểm nằm ở sát rìa mép ngoài (ví dụ gót chân) nhìn sang mép đối diện ở quá xa nên đoán rất dễ lệch.
     - **FCOS giải quyết bằng Centerness:** Thêm một nhánh phụ tính điểm "độ gần tâm". Điểm càng xa tâm thì độ tin cậy càng bị kéo tụt xuống gần 0.
     - **YOLOv8/v11 giải quyết bằng TAL (Task-Aligned Assigner):** 
       Không dùng quy tắc cứng nhắc, YOLOv8 chấm điểm từng vị trí dự đoán theo công thức:
       $$t = s^\alpha \times \text{IoU}^\beta$$
       *(trong đó $s$ là điểm số phân loại xem có đúng loại vật thể không, $\text{IoU}$ là độ khớp của box vừa vẽ)*.
       Chỉ những vị trí nào **vừa đoán đúng loại vật thể ($s$ cao), vừa vẽ box chuẩn khít ($\text{IoU}$ cao)** thì mới được chọn làm mẫu chuẩn (Positive) để cập nhật trọng số.

     ##### 2. Bài toán 2: "Ranh giới vật thể bị mờ/nhòe, làm sao ép mạng đoán chính xác 1 con số pixel?"
     - **Hiện tượng:** Mép của một đám rỉ sét hay đường nứt kim loại trong thực tế thường bị nhòe (mờ dần pixel). Bắt mạng phải đoán cứng nhắc một con số thực đơn lẻ $l = 80.0\text{ px}$ (Dirac Delta) là phi thực tế và làm mạng học rất cứng nhắc.
     - **YOLOv8/v11 giải quyết bằng DFL (Distribution Focal Loss):** 
       Thay vì đoán 1 con số chết, mạng đưa ra một **phân phối xác suất** quanh mép vật thể:
       - *"Tôi đoán 70% khả năng mép trái cách 80 px, 20% khả năng là 81 px, 10% khả năng là 79 px"*.
       - Giá trị khoảng cách cuối cùng là kỳ vọng toán học của phân phối đó.
       - Cách này giúp mô hình thể hiện được **sự bất định (uncertainty)** tại đường viền vật thể, giúp Bounding Box dự đoán mượt mà, chính xác và bám sát thực tế hơn rất nhiều.

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

## B2. mAP — metric chuẩn để đánh giá detection

### B2a. Precision / Recall trong ngữ cảnh detection

| Thuật ngữ | Công thức | Ý nghĩa |
|---|---|---|
| **TP (True Positive)** | IoU ≥ threshold với ground-truth & đúng class | Box đúng, trúng vật |
| **FP (False Positive)** | IoU < threshold hoặc sai class | Báo có lỗi nhưng không phải / sai chỗ |
| **FN (False Negative)** | Ground-truth không được box nào phủ | **Bỏ sót lỗi — nguy hiểm nhất trong QC** |
| **Precision** | TP / (TP + FP) | Trong số box model báo, bao nhiêu % đúng? |
| **Recall** | TP / (TP + FN) | Trong số lỗi thật, model bắt được bao nhiêu %? |

### B2b. AP và mAP

- **AP (Average Precision):** diện tích dưới đường Precision-Recall khi thay đổi confidence threshold từ cao xuống thấp.
- **mAP (Mean AP):** trung bình AP trên **tất cả các lớp**.

```python
# Thư viện tính mAP: pip install torchmetrics
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

metric = MeanAveragePrecision()
# preds = [{'boxes': tensor(...), 'scores': tensor(...), 'labels': tensor(...)}]
# target = [{'boxes': tensor(...), 'labels': tensor(...)}]
metric.update(preds, target)
result = metric.compute()
print(result['map'])            # mAP (IoU 0.5:0.95)
print(result['map_50'])         # mAP@IoU=0.5 (hay dùng cho QC, dễ thấy rõ)
```

**Ghi chú QC quan trọng về mAP:**
- **mAP@0.5** dễ đạt cao & trực quan — dùng để so sánh nhanh các model.
- **mAP@0.5:0.95** (COCO) khắt khe hơn, nhạy với độ chính xác *vị trí* box — quan trọng nếu bạn cần toạ độ chính xác để robot/đo kích thước.
- Trong QC đừng chỉ ngắm mAP: hãy xét **chi phí FN > chi phí FP** (bỏ sót lỗi đắt hơn) và tune `confidence_threshold` theo precision/recall trade-off phù hợp.

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
