# Roadmap Computer Vision Engineer Intern — TPHCM (chuyên CV cho nhà máy, 30 tuần)

Bỏ hẳn track LLM/RAG, dồn toàn bộ thời gian cho Computer Vision — đi sâu hơn nhiều so với bản kết hợp trước: có thêm 3D vision/depth (phục vụ robot tay gắp), object tracking video (phục vụ giám sát dây chuyền), và MLOps riêng cho CV (khác MLOps cho LLM). Domain xuyên suốt: hệ thống kiểm tra chất lượng bằng camera trên dây chuyền sản xuất.

---

### Giai đoạn 0 — Tuần 1: Python & Toán nền cho CV
**Từ khóa:** numpy vectorization, ma trận/tensor, đại số tuyến tính cơ bản (dot product, eigenvalue ở mức trực giác), gradient/đạo hàm, xác suất cơ bản (phân phối, kỳ vọng), OOP Python cho pipeline xử lý ảnh.
**Tài liệu:**
- 3Blue1Brown — "Essence of Linear Algebra" (playlist, trực quan hoá đại số tuyến tính, rất hợp cho người học CV): https://www.3blue1brown.com/topics/linear-algebra
- NumPy Docs — Quickstart: https://numpy.org/doc/stable/user/quickstart.html

### Giai đoạn 1 — Tuần 2-3: Image Processing nền tảng
**Từ khóa:** pixel/channel, color space (RGB/HSV/grayscale), convolution filter, Gaussian blur, edge detection (Canny, Sobel), morphological operations (erosion/dilation), thresholding, contour detection, histogram equalization.
**Tài liệu:**
- OpenCV-Python Tutorials (chính thức, đầy đủ code mẫu): https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- PyImageSearch — bài viết nền tảng về xử lý ảnh cổ điển (rất phổ biến trong cộng đồng CV): https://pyimagesearch.com/

### Giai đoạn 2 — Tuần 4-6: Deep Learning cho Vision
**Từ khóa:** CNN (convolution, pooling, stride, padding, feature map), backpropagation, kiến trúc phổ biến (ResNet, EfficientNet, Vision Transformer/ViT ở mức khái niệm), transfer learning, fine-tuning pretrained model, data augmentation (flip/rotate/color jitter/mixup), overfitting, learning rate scheduling, loss function cho classification.
**Tài liệu:**
- PyTorch Official Tutorials (Quickstart + Vision transfer learning tutorial): https://pytorch.org/tutorials/
- DeepLearning.AI — "Convolutional Neural Networks" (Andrew Ng, C4 trong Deep Learning Specialization): https://www.deeplearning.ai/courses/
- Papers with Code — mục Image Classification (tra cứu kiến trúc/benchmark khi cần): https://paperswithcode.com/task/image-classification

### Giai đoạn 3 — Tuần 7-9: Object Detection
**Từ khóa:** bounding box, anchor box, IoU (Intersection over Union), non-max suppression (NMS), one-stage vs two-stage detector, YOLO (v8/v11), Faster R-CNN, mAP (mean Average Precision), confidence threshold, data annotation (COCO format), transfer learning cho custom object.
**Tài liệu:**
- Ultralytics YOLO Docs (chính thức — train/val/predict/export chỉ vài dòng): https://docs.ultralytics.com/
- Roboflow Docs (annotation, augmentation, quản lý dataset, export nhiều format): https://docs.roboflow.com/
- CVAT (công cụ gán nhãn mã nguồn mở, deploy nội bộ được, phù hợp dữ liệu nhà máy nhạy cảm): https://opencv.github.io/cvat/docs/

### Giai đoạn 4 — Tuần 10-11: Segmentation
**Từ khóa:** semantic segmentation vs instance segmentation vs panoptic segmentation, U-Net, Mask R-CNN, Segment Anything Model (SAM), IoU cho segmentation, pixel-wise classification, dice loss.
**Tài liệu:**
- Detectron2 Docs (Mask R-CNN, instance segmentation): https://detectron2.readthedocs.io/
- Meta AI — Segment Anything (SAM, model foundation cho segmentation, zero-shot khá mạnh): https://segment-anything.com/

### Giai đoạn 5 — Tuần 12-13: Anomaly Detection & Quality Inspection
Lý do quan trọng: nhà máy hiếm khi có đủ ảnh lỗi để train supervised detection — phải học hướng unsupervised/few-shot.
**Từ khóa:** anomaly detection, one-class classification, autoencoder reconstruction error, feature-embedding-based (PatchCore, PaDiM, FastFlow), imbalanced dataset, precision/recall trade-off cho lỗi hiếm, threshold tuning theo chi phí bỏ sót lỗi (false negative cost).
**Tài liệu:**
- Anomalib (thư viện chính thức của OpenVINO/Intel, gồm PatchCore, PaDiM, FastFlow, benchmark sẵn): https://github.com/openvinotoolkit/anomalib
- MVTec AD Dataset (bộ dữ liệu chuẩn defect detection công nghiệp, dùng luyện tập trực tiếp): https://www.mvtec.com/company/research/datasets/mvtec-ad

### Giai đoạn 6 — Tuần 14-15: OCR & Barcode Reading
**Từ khóa:** text detection vs text recognition, PaddleOCR/EasyOCR pipeline, barcode/QR decoding (pyzbar), image preprocessing cho OCR (deskew, binarization, denoise).
**Tài liệu:**
- PaddleOCR (chính thức, hỗ trợ tốt text công nghiệp/nhãn sản phẩm): https://github.com/PaddlePaddle/PaddleOCR
- EasyOCR: https://github.com/JaidedAI/EasyOCR

### Giai đoạn 7 — Tuần 16-17: 3D Vision & Depth (cho robot/automation)
Mảng này cần nếu hướng tới robot tay gắp, đo kích thước vật thể, hoặc kiểm tra hình học 3D — rất hay gặp trong dây chuyền lắp ráp.
**Từ khóa:** camera calibration (intrinsic/extrinsic parameter), stereo vision, disparity map, depth camera (Intel RealSense, structured light/ToF), point cloud, point cloud registration (ICP), 3D bounding box.
**Tài liệu:**
- Intel RealSense SDK Docs (chính thức, phổ biến nhất cho depth camera công nghiệp giá hợp lý): https://dev.intelrealsense.com/docs
- Open3D Docs (thư viện xử lý point cloud, dễ dùng, có tutorial calibration/ICP): https://www.open3d.org/docs/release/

### Giai đoạn 8 — Tuần 18-19: Video & Object Tracking
**Từ khóa:** multi-object tracking (MOT), DeepSORT, ByteTrack, Kalman filter (khái niệm), track ID association, frame rate vs inference speed trade-off, video pipeline (đọc RTSP stream từ camera công nghiệp), multi-camera setup.
**Tài liệu:**
- Ultralytics Tracking Docs (tích hợp sẵn ByteTrack/BoT-SORT với YOLO, thực chiến nhanh): https://docs.ultralytics.com/modes/track/
- OpenCV — Video I/O Docs (đọc RTSP/webcam, xử lý luồng video): https://docs.opencv.org/4.x/dd/de7/group__videoio.html

### Giai đoạn 9 — Tuần 20-21: Model Optimization & Edge Deployment
Phần bắt buộc để chạy được trong nhà máy thật: inference phải real-time, tại chỗ, không phụ thuộc cloud.
**Từ khóa:** model quantization (INT8/FP16), pruning, knowledge distillation (khái niệm), ONNX export, TensorRT engine build, OpenVINO IR format, latency vs throughput, edge device (Jetson Nano/Xavier/Orin, Intel NUC), inference pipeline (batching, async inference, pinned memory).
**Tài liệu:**
- ONNX Runtime Docs: https://onnxruntime.ai/docs/
- NVIDIA TensorRT Docs: https://docs.nvidia.com/deeplearning/tensorrt/
- OpenVINO Docs: https://docs.openvino.ai/
- NVIDIA Jetson — Getting Started: https://developer.nvidia.com/embedded/learn/get-started-jetson

### Giai đoạn 10 — Tuần 22-23: MLOps riêng cho CV
**Từ khóa:** dataset versioning (DVC), experiment tracking (MLflow/Weights & Biases), model registry, data drift/concept drift cho ảnh (thay đổi ánh sáng/góc camera theo thời gian), model monitoring trong production, Docker + GPU container, CI/CD cho model (khác CI/CD cho code thường vì phải test cả accuracy).
**Tài liệu:**
- DVC Docs (data version control, chính thức): https://dvc.org/doc
- Weights & Biases — Quickstart: https://docs.wandb.ai/quickstart
- NVIDIA Container Toolkit (chạy model CV có GPU trong Docker): https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- GitHub Actions Quickstart: https://docs.github.com/en/actions/get-started/quickstart

### Giai đoạn 11 — Tuần 24-27: Capstone Project
Kiến trúc gợi ý: **Camera trên dây chuyền → tiền xử lý ảnh (Giai đoạn 1) → model phát hiện lỗi (object detection/segmentation — Giai đoạn 3-4, hoặc anomaly detection nếu lỗi hiếm — Giai đoạn 5) → chạy real-time trên edge device đã tối ưu (Giai đoạn 9) → log kết quả + tracking qua nhiều frame nếu cần (Giai đoạn 8) → dashboard hiển thị + cảnh báo, có versioning/monitoring (Giai đoạn 10).**
Không có tài liệu mới — quay lại tài liệu ở các giai đoạn trên khi cần tra cứu chi tiết lúc code.

### Giai đoạn 12 — Tuần 28-30: Portfolio, CV, phỏng vấn, apply
**Câu hỏi nên chuẩn bị:**
- "Explain the difference between object detection, segmentation, and classification"
- "How would you handle a defect detection task with only 20 defective images?" — anomaly detection, few-shot, augmentation, synthetic data
- "One-stage vs two-stage detector — trade-offs?"
- "Why deploy on edge instead of cloud for a factory camera system?" — latency, network reliability, data privacy, cost per inference
- "How do you validate a model's mAP is good enough for production, and what threshold would you choose given the cost of a missed defect?"
- "How would you handle camera drift/lighting changes over time in production?" — data drift monitoring, periodic retraining
- "Explain how you'd calibrate a camera and why it matters for measurement tasks"

---

## Bảng tổng hợp theo tuần

| Tuần | Nội dung trọng tâm |
|---|---|
| 1 | numpy, đại số tuyến tính cơ bản, xác suất |
| 2-3 | OpenCV, convolution filter, edge detection, morphology |
| 4-6 | CNN, ResNet/EfficientNet/ViT, transfer learning, augmentation |
| 7-9 | YOLO, IoU, NMS, mAP, annotation, custom object detection |
| 10-11 | U-Net, Mask R-CNN, SAM, segmentation |
| 12-13 | anomaly detection, PatchCore/PaDiM, MVTec AD |
| 14-15 | OCR, PaddleOCR, barcode/QR |
| 16-17 | camera calibration, stereo vision, depth camera, point cloud |
| 18-19 | multi-object tracking, DeepSORT/ByteTrack, video pipeline |
| 20-21 | quantization, ONNX, TensorRT, OpenVINO, Jetson |
| 22-23 | DVC, W&B, model monitoring, CI/CD cho model |
| 24-27 | Capstone: hệ thống kiểm tra chất lượng end-to-end |
| 28-30 | Portfolio, phỏng vấn CV |

---

Ghi chú: Giai đoạn 7 (3D Vision) chỉ thực sự cần nếu bạn nhắm tới vị trí liên quan robot/automation gắp-đặt vật thể. Nếu công ty mục tiêu chỉ làm kiểm tra chất lượng 2D thuần (phần lớn vị trí entry-level QC vision), có thể bỏ giai đoạn này để rút gọn còn ~28 tuần mà vẫn giữ đủ độ sâu ở phần detection/anomaly/edge deployment — đây là 3 mảng nhà tuyển dụng quan tâm nhất.