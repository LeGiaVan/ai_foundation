# Anchor-Free trong Object Detection — Giải thích chi tiết

## 1. Anchor-based là gì? (để hiểu tại sao cần Anchor-Free)

Các mô hình detection cổ điển (Faster R-CNN, SSD, YOLOv2/v3, RetinaNet...) hoạt động theo nguyên lý:

- Trước khi train, người ta **đặt sẵn** hàng nghìn "hộp neo" (anchor boxes) có kích thước và tỷ lệ khung hình (aspect ratio) cố định lên khắp bức ảnh (ví dụ: 3 tỷ lệ × 3 kích thước tại mỗi vị trí trên feature map).
- Mạng chỉ cần học cách: (1) phân loại anchor đó là vật thể hay nền, (2) "tinh chỉnh" (regress) độ lệch giữa anchor và bounding box thật.

Vấn đề của cách này:
- Phải **thiết kế thủ công** số lượng, tỷ lệ, kích thước anchor — đây là hyperparameter rất nhạy cảm, sai là model tệ hẳn.
- Sinh ra **quá nhiều anchor** (có thể hàng chục nghìn trên 1 ảnh) → mất cân bằng dữ liệu dương/âm nghiêm trọng, tốn bộ nhớ và tính toán IoU giữa toàn bộ anchor với ground-truth.
- Anchor cố định khó khớp với vật thể có hình dạng bất thường (rất dài, rất dẹt).

**Anchor-free** ra đời để loại bỏ hoàn toàn bước "đặt hộp neo trước" này.

## 2. Anchor-Free là gì?

Ý tưởng cốt lõi: thay vì so khớp với các hộp định sẵn, mô hình **dự đoán trực tiếp vị trí vật thể** dựa trên các điểm (points) hoặc pixel trên feature map — giống như bài toán segmentation theo từng pixel.

Có 2 trường phái chính:

### A. Keypoint-based (dựa trên điểm đặc trưng)
Phát hiện vật thể bằng cách tìm các **điểm đặc trưng** (góc, tâm) rồi ghép chúng lại thành bounding box.

- **CornerNet** (2018): phát hiện 2 góc (top-left, bottom-right) của bounding box bằng heatmap, sau đó dùng "embedding" để ghép cặp góc thuộc cùng 1 vật thể.
- **CenterNet** (2019): coi vật thể là **1 điểm duy nhất — tâm của bounding box**. Mạng dự đoán heatmap tâm + offset + kích thước (width, height) tại điểm đó. Đơn giản hơn CornerNet vì không cần ghép cặp.
- **ExtremeNet**: phát hiện 4 điểm cực trị (trên, dưới, trái, phải) + tâm.

### B. Dense prediction / Center-based (dự đoán dày đặc trên từng pixel)
Đây là hướng phổ biến và ảnh hưởng nhất — tiêu biểu là **FCOS**.

## 3. Cơ chế hoạt động chi tiết — FCOS

FCOS (Fully Convolutional One-Stage Object Detector) hoạt động giống hệt semantic segmentation: dự đoán một vector 4 chiều (l, t, r, b) mã hóa vị trí bounding box tại mỗi pixel thuộc vùng vật thể.

### Bước 1: Ảnh biến thành lưới điểm như thế nào?

Giả sử ảnh đầu vào có kích thước **800×800 pixel**. Ảnh đi qua backbone (ResNet...) và bị giảm kích thước dần (downsample). Ở một tầng feature map có **stride = 8** (mỗi pixel trên feature map đại diện cho vùng 8×8 pixel trên ảnh gốc), feature map sẽ có kích thước **100×100** (800/8 = 100).

→ Tổng cộng có **10.000 điểm** trên feature map này, mỗi điểm là một "ứng viên" để dự đoán.

Điểm tại vị trí (i, j) trên feature map tương ứng với tọa độ trên ảnh gốc:
```
x = i × stride + stride/2
y = j × stride + stride/2
```

### Bước 2: Điểm nào là "dương" (positive), điểm nào là "nền"?

Khác với anchor-based (phải tính IoU), FCOS chỉ cần kiểm tra hình học đơn giản:

- Nếu điểm (x, y) sau khi ánh xạ về ảnh gốc **rơi vào bên trong** một ground-truth box → **dương**, phải học dự đoán 4 số (l, t, r, b) của box đó.
- Nếu điểm nằm ngoài mọi box → **âm** (background).

Ví dụ: box có góc trái-trên (200,110), góc phải-dưới (480,310). Điểm tại (340, 210) là điểm dương, với:
```
l = 340 - 200 = 140
t = 210 - 110 = 100
r = 480 - 340 = 140
b = 310 - 210 = 100
```

Có thể có hàng chục điểm dương cho cùng 1 vật thể (khác anchor-based, mỗi anchor chỉ khớp 1 box theo IoU cao nhất).

### Bước 3: Nhiều điểm — làm sao biết vật to hay nhỏ để không rối? (FPN đa tầng)

**Vấn đề gốc rễ:** nếu chỉ dùng 1 feature map, một điểm nằm trong vùng chồng lấn của 2 vật thể (ví dụ người nhỏ đứng trước xe buýt lớn) sẽ rơi vào cả 2 ground-truth box cùng lúc → mạng không biết nên học theo vật nào. Đây gọi là **ambiguity**.

**Giải pháp — FPN (Feature Pyramid Network):** tạo ra 5 tầng feature map với stride khác nhau (P3→P7), mỗi tầng chỉ chịu trách nhiệm cho vật thể có kích thước nằm trong 1 khoảng nhất định:

| Tầng | Stride | Khoảng m = max(l,t,r,b) cho phép |
|---|---|---|
| P3 | 8 | 0 – 64 |
| P4 | 16 | 64 – 128 |
| P5 | 32 | 128 – 256 |
| P6 | 64 | 256 – 512 |
| P7 | 128 | 512 – ∞ |

Với mỗi điểm dương, tính m = max(l,t,r,b) (nửa kích thước gần đúng của vật thể), rồi chỉ tầng có khoảng phù hợp mới chấp nhận điểm đó.

Ví dụ: người có m≈60 → chỉ P3 chấp nhận. Xe buýt có m≈300 → chỉ P6 chấp nhận. Cùng 1 tọa độ ảnh gốc nhưng ở 2 tầng khác nhau là 2 điểm độc lập, không chia sẻ trọng số → hết xung đột.

**Trường hợp 2 vật thể cùng kích thước, cùng tầng, vẫn chồng lấn:**

Ví dụ: box "sofa" (260×220 = 57.200 px²) và box "mèo" (180×150 = 27.000 px²) chồng lấn một phần.

Luật: **chỉ những điểm nằm trong vùng giao nhau** mới bị mập mờ và cần chọn — FCOS chọn gán cho box có **diện tích nhỏ hơn** (ở đây là mèo). Lý do: vật nhỏ vốn đã có ít điểm dương hơn vật lớn (ít pixel bên trong box hơn), nếu không ưu tiên thì vật nhỏ dễ bị "nuốt chửng" tín hiệu huấn luyện bởi vật lớn đứng cạnh/đè lên.

Các điểm **ngoài** vùng giao nhau không bị ảnh hưởng gì — vẫn giữ nguyên nhãn của box chứa nó (điểm chỉ trong sofa → vẫn là sofa; điểm chỉ trong mèo → vẫn là mèo).

**Điểm quan trọng dễ hiểu nhầm:** việc một điểm bị "giành" bởi box nhỏ hơn **không có nghĩa là box lớn bị cắt mất một phần** trong kết quả cuối cùng. Mỗi điểm dương — dù nằm ở vị trí nào trong box — luôn học dự đoán đủ **4 khoảng cách đến 4 cạnh của toàn bộ box gốc**, không phải chỉ cạnh gần nó. Ví dụ điểm nằm ở góc trái sofa vẫn phải dự đoán ra khoảng cách `r` tới tận cạnh phải sofa, dù đoạn đó bị con mèo che/chồng lên phía trước — vì (l,t,r,b) là khoảng cách hình học tới box thật, không phụ thuộc việc vùng đó có bị vật khác che hay không.

→ Kết quả khi suy luận: các điểm còn lại của sofa (dù ít hơn do mất một số điểm ở vùng giao) vẫn mỗi điểm tự dựng ra **1 box sofa hoàn chỉnh**; các điểm của mèo cũng tự dựng ra **1 box mèo hoàn chỉnh**. NMS gộp các box trùng lặp của cùng 1 vật lại. Kết quả cuối là **2 box đầy đủ chồng lấn lên nhau**, giống hệt 2 ground-truth ban đầu — không có chuyện "nửa trái là sofa, nửa phải là mèo".

### Bước 4: Center-ness — xử lý vấn đề "điểm ở rìa dự đoán tệ"

Điểm ở gần rìa box vẫn hợp lệ là điểm dương, nhưng dự đoán kém chính xác (một trong 4 số l/t/r/b gần bằng 0, dễ sai số tỷ lệ lớn). FCOS thêm nhánh phụ dự đoán **center-ness**:

```
centerness = sqrt( [min(l,r)/max(l,r)] × [min(t,b)/max(t,b)] )
```

- Điểm đúng tâm hình học: l≈r, t≈b → centerness ≈ 1.
- Điểm sát rìa: một số rất nhỏ → centerness ≈ 0.

Lúc suy luận: **điểm số cuối = classification score × centerness score**. Box từ điểm gần rìa bị điểm thấp kéo xuống → bị NMS loại trước, chỉ giữ box từ điểm gần tâm.

### Bước 5: Loss function khi train

Với mỗi điểm dương, tính 3 loss cộng lại:
1. **Focal loss** cho classification.
2. **IoU loss** (hoặc GIoU) cho regression 4 số (l,t,r,b).
3. **Binary cross-entropy** cho nhánh center-ness.

Điểm âm chỉ đóng góp vào loss classification.

### Bước 6: Suy luận (inference) — dựng lại box

Với mỗi điểm có classification score cao:
```
x0 (trái) = x - l
y0 (trên) = y - t
x1 (phải) = x + r
y1 (dưới) = y + b
```

Sau đó chạy NMS trên toàn bộ box từ mọi điểm dương ở mọi tầng FPN để loại box trùng lặp.

## 4. So sánh Anchor-based vs Anchor-free

| Tiêu chí | Anchor-based | Anchor-free |
|---|---|---|
| Hộp neo định trước | Có (hàng nghìn/ảnh) | Không |
| Hyperparameter (tỷ lệ, kích thước anchor) | Cần tinh chỉnh thủ công | Không cần |
| Tính IoU giữa anchor–GT | Có, tốn tính toán | Không |
| Đơn vị dự đoán | Anchor box | Điểm/pixel |
| Ví dụ | Faster R-CNN, SSD, YOLOv2/v3, RetinaNet | FCOS, CenterNet, CornerNet, YOLOX, YOLOv8 |

## 5. Ưu điểm & nhược điểm

**Ưu điểm:**
- Kiến trúc đơn giản hơn, ít hyperparameter → dễ tái sử dụng cho bài toán khác (segmentation, tracking, pose estimation).
- Không bị giới hạn bởi tỷ lệ/kích thước anchor cố định → linh hoạt hơn với vật thể có hình dạng bất thường.
- Giảm số lượng candidate cần xử lý, giảm mất cân bằng dương/âm.
- Tốc độ inference thường nhanh hơn.

**Nhược điểm:**
- Vấn đề "ambiguity" khi nhiều vật thể chồng lấn — cần thêm cơ chế như FPN, center-ness để xử lý.
- Một số kỹ thuật keypoint-based (CornerNet) cần bước ghép cặp phức tạp, tăng độ trễ.
- Label assignment (gán điểm nào là dương) trở thành bài toán quan trọng mới — các nghiên cứu sau này (ATSS, OTA, SimOTA trong YOLOX) tập trung cải tiến chính bước này.

## 6. Xu hướng hiện tại

Từ 2020 trở đi, anchor-free gần như trở thành mainstream: YOLOX, YOLOv6/v7/v8, và cả các mô hình dựa trên Transformer (DETR và họ hàng) đều bỏ hẳn anchor, thay bằng dự đoán trực tiếp qua điểm hoặc qua "object query".

## 7. Tài liệu tham khảo

- **FCOS**: Tian, Z. et al., *"FCOS: A simple and strong anchor-free object detector"* — https://arxiv.org/abs/2006.09214
- **CornerNet**: Law, H., Deng, J., *"CornerNet: Detecting Objects as Paired Keypoints"* — arXiv:1808.01244
- **CenterNet (Objects as Points)**: Zhou, X. et al. — arXiv:1904.07850
- **ExtremeNet**: Zhou, X. et al., *"Bottom-up Object Detection by Grouping Extreme and Center Points"* — arXiv:1901.08043
- **FoveaBox**: Kong, T. et al. — arXiv:1904.03797
- **ATSS**: Zhang, S. et al., *"Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection"* — arXiv:1912.02424
- Code chính thức FCOS: git.io/AdelaiDet