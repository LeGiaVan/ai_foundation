# Cheatsheet: Pipeline AI Detection thực tế trên dây chuyền sản xuất

> Mục đích: hiểu **toàn bộ vòng đời vật lý** của một hệ thống CV kiểm tra chất lượng trong nhà máy — từ lúc sản phẩm đi qua camera đến lúc nó bị loại bỏ hoặc thông qua — mà không cần đào sâu từng tuần trong roadmap học.

---

## 0. Sơ đồ tổng quan (6 bước)

```
[1] Cảm biến trigger
        │
        ▼
[2] Camera chụp ảnh
        │
        ▼
[3] Edge AI inference
        │
        ▼
[4] Quyết định pass/fail
        │
        ▼
[5] PLC điều khiển actuator
        │
        ▼
[6] Logging & dashboard ──┐
        │                 │
        ▼                 │
  (retrain định kỳ) ◄─────┘
```

Ghi nhớ nguyên tắc cốt lõi: **đây là một vòng lặp thời gian thực có ràng buộc vật lý cứng (hard real-time constraint)** — không giống train model trên Colab, mọi bước phải xảy ra trong một cửa sổ thời gian rất ngắn, đồng bộ với tốc độ băng chuyền.

---

## 1. Cảm biến trigger

| Mục | Nội dung |
|---|---|
| Vai trò | Báo cho camera biết "sản phẩm đã đến đúng vị trí, chụp ngay" |
| Thiết bị phổ biến | Cảm biến quang điện (photoelectric sensor), cảm biến tiệm cận (proximity sensor), encoder trên băng chuyền |
| Vì sao không để camera tự chụp liên tục? | Chụp liên tục (free-run) → tốn tài nguyên, ảnh dễ bị mờ do sai thời điểm, khó đồng bộ vị trí vật thể trong khung hình |
| Tín hiệu ra | Digital signal (0V/24V) hoặc pulse, đi thẳng vào camera hoặc qua PLC trung gian |
| Câu hỏi hay gặp | "Camera của bạn free-run hay hardware-triggered?" → luôn trả lời **hardware-triggered** trong môi trường production thật |

---

## 2. Camera chụp ảnh

| Mục | Nội dung |
|---|---|
| Loại camera | Industrial camera (GigE Vision, USB3 Vision) — khác hẳn webcam thường vì cần global shutter (chống nhòe khi vật thể di chuyển nhanh) |
| Ánh sáng | **Quan trọng hơn cả model** — ánh sáng không ổn định là nguyên nhân số 1 gây model fail trong production. Dùng đèn chuyên dụng: backlight (soi ngược, tốt cho đo biên dạng), ring light (chiếu đều, tốt cho lỗi bề mặt), dome light (khử phản chiếu trên vật thể bóng) |
| Lens | Tiêu cự và khoảng cách làm việc (working distance) phải cố định — nếu sản phẩm không nằm đúng khoảng lấy nét, ảnh mờ dù model tốt đến đâu |
| Thời gian phơi sáng (exposure time) | Phải đủ ngắn để "đóng băng" vật thể đang chuyển động — công thức gần đúng: `exposure_time < pixel_size / tốc_độ_băng_chuyền` |
| Format ảnh | Thường lưu ảnh gốc (raw/BMP) trước khi qua tiền xử lý, để phục vụ debug và làm dữ liệu train sau này |
| Liên hệ roadmap | Tuần 2-3 (không gian màu, filter) áp dụng ở bước tiền xử lý ngay sau khi chụp: khử nhiễu, cân bằng sáng, crop vùng ROI (Region of Interest) |

---

## 3. Edge AI inference

| Mục | Nội dung |
|---|---|
| Vì sao chạy tại chỗ (edge) thay vì gửi lên cloud? | Độ trễ mạng không chấp nhận được cho real-time; nhà máy thường hạn chế kết nối internet vì bảo mật; chi phí cloud liên tục cho hàng nghìn ảnh/ngày quá cao |
| Thiết bị phổ biến | NVIDIA Jetson (Orin/Xavier), Intel NUC + OpenVINO, Industrial PC có GPU rời |
| Bước bắt buộc trước khi deploy | Convert model → ONNX → tối ưu bằng TensorRT (NVIDIA) hoặc OpenVINO IR (Intel). Model train bằng PyTorch/TensorFlow **không chạy trực tiếp** hiệu quả trên edge |
| Kỹ thuật tăng tốc | Quantization INT8/FP16 (giảm độ chính xác số học đổi lấy tốc độ), pruning, batch size = 1 (vì xử lý từng sản phẩm một, không có batch thật) |
| Ràng buộc thời gian (latency budget) | Công thức: `thời gian cho phép = khoảng_cách(camera → điểm reject) / tốc_độ_băng_chuyền`. Ví dụ: khoảng cách 0.5m, băng chuyền 1m/s → chỉ có **500ms** cho toàn bộ bước 2→3→4→5 |
| Liên hệ roadmap | Tuần 20-21 |

---

## 4. Quyết định pass / fail

| Mục | Nội dung |
|---|---|
| Đầu vào | Confidence score từ model (detection/segmentation/anomaly score) |
| Đầu ra | Nhị phân: PASS (cho qua) / FAIL (loại bỏ) |
| Vấn đề cốt lõi | Chọn ngưỡng (threshold) không phải bài toán thuần kỹ thuật mà là **bài toán kinh doanh**: chi phí bỏ sót lỗi (false negative — sản phẩm lỗi lọt ra thị trường) so với chi phí loại nhầm hàng tốt (false positive — lãng phí sản phẩm tốt) |
| Cách tiếp cận | Vẽ đường cong Precision-Recall, chọn điểm threshold dựa trên chi phí thực tế của doanh nghiệp, không chọn theo "điểm đẹp nhất trên biểu đồ" |
| Lưu ý thực tế | Với sản phẩm an toàn (linh kiện điện tử, thực phẩm, dược phẩm) → luôn ưu tiên giảm false negative dù tăng false positive |

---

## 5. PLC điều khiển actuator

| Mục | Nội dung |
|---|---|
| Vai trò | Biến kết quả "FAIL" thành hành động vật lý thật — đẩy sản phẩm lỗi ra khỏi dây chuyền |
| Actuator phổ biến | Xi-lanh khí nén (air cylinder/air blast), tay gạt cơ khí, băng chuyền rẽ nhánh |
| Giao thức kết nối AI ↔ PLC | **Modbus TCP/RTU** (phổ biến nhất, dễ code bằng Python qua thư viện `pymodbus`), **OPC-UA** (chuẩn công nghiệp mới hơn, bảo mật tốt hơn), hoặc đơn giản nhất là **digital I/O (GPIO)** trực tiếp trên edge device nếu hệ thống nhỏ |
| Độ trễ cần tính thêm | Thời gian PLC nhận lệnh + thời gian cơ khí phản ứng (xi-lanh cần vài chục ms để đẩy) — phải cộng vào latency budget ở bước 3 |
| Đây là phần THIẾU trong hầu hết roadmap online | Vì nó thuộc lĩnh vực tự động hóa công nghiệp (Industrial Automation/OT), không phải Data Science thuần |

---

## 6. Logging & Dashboard (MLOps)

| Mục | Nội dung |
|---|---|
| Lưu gì | Ảnh gốc + ảnh có bounding box/mask + kết quả pass/fail + timestamp + confidence score |
| Dashboard hiển thị | Tỷ lệ lỗi theo giờ/ca, cảnh báo khi tỷ lệ lỗi vượt ngưỡng, trạng thái hệ thống (camera mất kết nối, edge device quá tải) |
| Model/data drift trong nhà máy | Khác ánh sáng theo ca (ngày/đêm), ống kính bị bám bụi, thay đổi lô nguyên liệu → model giảm hiệu suất dần theo thời gian dù không đổi code |
| Cách phát hiện drift | Theo dõi phân phối confidence score theo thời gian, theo dõi tỷ lệ FAIL bất thường, so sánh định kỳ với tập validation cố định |
| Vòng lặp cải thiện | Ảnh bị gắn cờ nghi ngờ (borderline confidence, hoặc bị operator phản hồi sai) → đưa vào hàng chờ gắn nhãn lại → bổ sung vào tập train → retrain định kỳ (không phải liên tục, thường theo lịch tuần/tháng) |
| Công cụ | DVC (versioning dữ liệu), MLflow/W&B (theo dõi experiment và model version), Docker (đóng gói pipeline để dễ deploy lại) |
| Liên hệ roadmap | Tuần 22-23 |

---

## Bảng thuật ngữ nhanh (Glossary)

| Thuật ngữ | Giải nghĩa ngắn |
|---|---|
| Hardware trigger | Camera chụp theo tín hiệu điện từ cảm biến, không tự chạy liên tục |
| ROI (Region of Interest) | Vùng ảnh cần xử lý, crop bớt phần không liên quan để tăng tốc |
| Global shutter | Cảm biến camera chụp toàn khung hình cùng lúc, chống nhòe vật thể di chuyển nhanh |
| Working distance | Khoảng cách cố định từ lens đến vật thể để đảm bảo lấy nét |
| ONNX | Định dạng trung gian để chuyển model giữa các framework (PyTorch → TensorRT/OpenVINO) |
| Latency budget | Tổng thời gian tối đa cho phép từ lúc chụp ảnh đến lúc actuator phản ứng |
| Modbus/OPC-UA | Giao thức giao tiếp giữa phần mềm AI và PLC/thiết bị công nghiệp |
| Data drift | Hiện tượng dữ liệu thực tế thay đổi dần khiến model cũ giảm độ chính xác |
| False negative cost | Chi phí khi bỏ sót một sản phẩm lỗi — thường là yếu tố quyết định threshold |

---

## Checklist câu hỏi tự kiểm tra hiểu bài

- [ ] Giải thích được vì sao camera nhà máy dùng hardware trigger thay vì free-run
- [ ] Tính được latency budget cho một line cụ thể (biết tốc độ băng chuyền + khoảng cách)
- [ ] Nêu được ít nhất 2 giao thức kết nối AI ↔ PLC
- [ ] Giải thích được vì sao chọn threshold là bài toán kinh doanh, không chỉ kỹ thuật
- [ ] Nêu được 2 nguyên nhân gây model drift trong nhà máy và cách phát hiện
- [ ] Vẽ lại được toàn bộ 6 bước của pipeline mà không cần nhìn tài liệu

---

## Map nhanh ra roadmap 30 tuần

| Bước pipeline | Tuần tương ứng trong roadmap |
|---|---|
| 1-2. Trigger + Camera/ánh sáng | Tuần 2-3 |
| 3. Model detect (nội dung AI) | Tuần 7-13 |
| 3. Edge inference (tối ưu hóa) | Tuần 20-21 |
| 4. Threshold/quyết định | Tuần 12-13 + phần phỏng vấn tuần 28-30 |
| 5. PLC/actuator | **Không có trong roadmap — tự tìm hiểu thêm (Modbus/OPC-UA cơ bản)** |
| 6. Logging/Dashboard/Drift | Tuần 22-23 |