"""
Script huấn luyện YOLOv8n trên tập dữ liệu SunFlower Detect (100 ảnh).
Thực hiện các bước 3 & 4 trong Cheatsheet Object Detection (B6).
"""

import os
import sys
from pathlib import Path


def train():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[LỖI] Chưa cài đặt thư viện ultralytics.")
        print("Vui lòng chạy lệnh: pip install ultralytics")
        sys.exit(1)

    # 1. Đường dẫn dataset & file cấu hình
    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir / "SunFlower Detect.v1i.yolov8"
    yaml_path = dataset_dir / "data.yaml"
    project_dir = base_dir / "runs"

    if not yaml_path.exists():
        print(f"[LỖI] Không tìm thấy file cấu hình: {yaml_path}")
        sys.exit(1)

    print("=" * 60)
    print("🌻 BẮT ĐẦU HUẤN LUYỆN YOLOV8N TRÊN TẬP SUNFLOWER DETECT")
    print(f"📁 Dataset YAML: {yaml_path}")
    print(f"📁 Thư mục lưu runs: {project_dir}")
    print("=" * 60)

    # 2. Khởi tạo model pre-trained (COCO weights để transfer learning)
    model = YOLO("yolov8n.pt")

    # 3. Huấn luyện model
    # epochs=30: đủ để model hội tụ trên 73 ảnh train
    # imgsz=640: độ phân giải chuẩn của YOLOv8
    # batch=8: phù hợp cho cả CPU và GPU nhỏ
    results = model.train(
        data=str(yaml_path),
        epochs=30,
        imgsz=640,
        batch=8,
        workers=2,
        project=str(project_dir),
        name="sunflower_yolov8n",
        exist_ok=True
    )

    print("\n" + "=" * 60)
    print("✅ HUẤN LUYỆN HOÀN TẤT!")
    print("=" * 60)

    # 4. Đánh giá (Evaluate) mô hình trên tập validation/test
    print("\n🔍 ĐANG ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP VALIDATION/TEST...")
    metrics = model.val()

    print("\n📊 KẾT QUẢ ĐÁNH GIÁ METRICS (Bước 4):")
    print(f"  - mAP@50      : {metrics.box.map50:.4f}")
    print(f"  - mAP@50-95   : {metrics.box.map:.4f}")
    print(f"  - Precision   : {metrics.box.mp:.4f}")
    print(f"  - Recall      : {metrics.box.mr:.4f}")

    # 5. Thử nghiệm dự đoán (Predict) trên 3 ảnh test đầu tiên
    test_images_dir = dataset_dir / "test" / "images"
    test_images = list(test_images_dir.glob("*.jpg"))[:3]

    if test_images:
        print(f"\n🎯 DỰ ĐOÁN THỬ NGHIỆM TRÊN {len(test_images)} ẢNH TEST:")
        preds = model.predict(
            source=[str(p) for p in test_images],
            conf=0.25,
            save=True,
            project=str(project_dir),
            name="predictions"
        )
        print(f"👉 Ảnh kết quả dự đoán đã được lưu tại: {project_dir / 'predictions'}")


if __name__ == "__main__":
    train()
