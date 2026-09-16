"""
Module chuyên biệt: Tinh chỉnh Ngưỡng Tự Tin (Tune Confidence Threshold).
Khảo sát sự đánh đổi Precision / Recall và tối ưu hoá chi phí kiểm tra lỗi (Domain QC).
Phục vụ mục B6 (Bước 5) trong Cheatsheet Object Detection.
"""

import sys
import io
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Đảm bảo in tiếng Việt chuẩn trên Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def tune_confidence_threshold():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "runs" / "sunflower_yolov8n" / "weights" / "best.pt"
    data_yaml = base_dir / "SunFlower Detect.v1i.yolov8" / "data.yaml"
    output_dir = base_dir / "runs" / "tune_threshold"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"[LỖI] Không tìm thấy trọng số mô hình tại: {model_path}")
        print("Hãy chạy huấn luyện trước bằng file train_yolov8.py!")
        return

    print("=" * 80)
    print("🎯 KHẢO SÁT SỰ ĐÁNH ĐỔI (TRADE-OFF) PRECISION - RECALL & TỐI ƯU CHI PHÍ QC")
    print(f"📦 Model: {model_path}")
    print(f"📊 Dataset: {data_yaml}")
    print("=" * 80)

    model = YOLO(str(model_path))

    # Dải ngưỡng confidence cần quét từ thấp (vét lỗi) đến cao (chắc chắn)
    conf_thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    # Giả lập bài toán kinh tế trong nhà máy sản xuất (Domain QC):
    # - Chi phí bỏ sót 1 lỗi (FN): $100 (lọt phế phẩm ra khách hàng, đền bù hợp đồng)
    # - Chi phí báo động giả 1 lần (FP): $10 (công nhân dừng máy kiểm tra lại)
    COST_FN = 100.0
    COST_FP = 10.0
    TOTAL_GT = 99  # Tổng số vật thể thật trong tập test

    results_table = []

    print("\n⏳ Đang quét qua các ngưỡng confidence...")
    for conf in conf_thresholds:
        # Chạy validation ở ngưỡng conf tương ứng
        val_res = model.val(
            data=str(data_yaml),
            conf=conf,
            verbose=False,
            plots=False
        )

        p = float(val_res.box.mp)  # Precision trung bình
        r = float(val_res.box.mr)  # Recall trung bình

        # F1-Score: Trung bình điều hoà giữa Precision và Recall
        if p + r > 0:
            f1 = 2 * (p * r) / (p + r)
        else:
            f1 = 0.0

        # Ước lượng số lượng TP, FN, FP
        tp = round(r * TOTAL_GT)
        fn = TOTAL_GT - tp
        fp = round(tp / p - tp) if p > 0 else 0

        # Tổng chi phí rủi ro QC
        qc_cost = (fn * COST_FN) + (fp * COST_FP)

        results_table.append({
            "conf": conf,
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "qc_cost": qc_cost
        })

    # =========================================================================
    # 1. IN BẢNG ĐÁNH ĐỔI PRECISION vs RECALL
    # =========================================================================
    print("\n" + "=" * 85)
    print(f"{'Ngưỡng Conf':^12} | {'Precision':^10} | {'Recall':^10} | {'F1-Score':^10} | {'Ước tính (TP/FN/FP)':^20} | {'Chi phí QC ($)':^14}")
    print("=" * 85)

    for row in results_table:
        counts = f"{row['tp']:>2} TP / {row['fn']:>2} FN / {row['fp']:>2} FP"
        print(f"{row['conf']:^12.2f} | {row['precision']:^10.3f} | {row['recall']:^10.3f} | {row['f1']:^10.3f} | {counts:^20} | {row['qc_cost']:^14.0f}")
    print("=" * 85)

    # =========================================================================
    # 2. PHÂN TÍCH VÀ KHUYẾN NGHỊ NGƯỠNG TỐI ƯU
    # =========================================================================
    # Điểm tối ưu F1 cao nhất (Cân bằng học thuật)
    best_f1_row = max(results_table, key=lambda x: x["f1"])
    # Điểm tối ưu chi phí QC thấp nhất (Thực chiến nhà máy)
    best_cost_row = min(results_table, key=lambda x: x["qc_cost"])

    print("\n💡 PHÂN TÍCH VÀ QUYẾT ĐỊNH NGƯỠNG:")
    print(f"  1. [Mục tiêu Cân Bằng F1-Score]:")
    print(f"     -> Chọn Confidence = {best_f1_row['conf']:.2f}")
    print(f"     -> Precision = {best_f1_row['precision']:.3f}, Recall = {best_f1_row['recall']:.3f}, F1 = {best_f1_row['f1']:.3f}")
    print(f"     -> Phù hợp: Khi chi phí của báo động giả (FP) và bỏ sót lỗi (FN) tương đương nhau.")

    print(f"\n  2. [Mục tiêu Nhà Máy QC Thực Tế (Chi phí FN = ${COST_FN}, FP = ${COST_FP})]:")
    print(f"     -> Chọn Confidence = {best_cost_row['conf']:.2f}")
    print(f"     -> Chi phí tối thiểu: ${best_cost_row['qc_cost']:.0f} (Chỉ bỏ sót {best_cost_row['fn']} lỗi)")
    print(f"     -> Phù hợp: Dây chuyền sản xuất khắt khe, thà kiểm tra nhầm còn hơn lọt hàng lỗi ra ngoài!")

    # =========================================================================
    # 3. TRỰC QUAN HÓA BẰNG ĐỒ THỊ VÀ LƯU FILE ẢNH
    # =========================================================================
    confs = [r["conf"] for r in results_table]
    precisions = [r["precision"] for r in results_table]
    recalls = [r["recall"] for r in results_table]
    f1s = [r["f1"] for r in results_table]
    costs = [r["qc_cost"] for r in results_table]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Đồ thị 1: Precision / Recall / F1 theo Threshold
    ax1.plot(confs, precisions, 'b-o', label='Precision (Độ chuẩn xác)', lw=2)
    ax1.plot(confs, recalls, 'g-s', label='Recall (Độ bao phủ)', lw=2)
    ax1.plot(confs, f1s, 'r--^', label='F1-Score', lw=2)
    ax1.axvline(best_f1_row['conf'], color='purple', linestyle=':', label=f"Max F1 ({best_f1_row['conf']:.2f})")
    ax1.set_title("Sự Đánh Đổi Precision vs Recall theo Ngưỡng Confidence", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Confidence Threshold", fontsize=11)
    ax1.set_ylabel("Điểm Metric (0.0 - 1.0)", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='best')

    # Đồ thị 2: Chi phí QC theo Threshold
    ax2.plot(confs, costs, 'm-d', label=f'Tổng Chi Phí QC (FN=${COST_FN}, FP=${COST_FP})', lw=2.5)
    ax2.axvline(best_cost_row['conf'], color='red', linestyle='--', label=f"Chi Phí Thấp Nhất ({best_cost_row['conf']:.2f})")
    ax2.set_title("Chi Phí Rủi Ro QC Nhà Máy theo Ngưỡng Confidence", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Confidence Threshold", fontsize=11)
    ax2.set_ylabel("Tổng Chi Phí Ước Tính ($)", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='best')

    plt.tight_layout()
    plot_file = output_dir / "confidence_tradeoff_qc.png"
    plt.savefig(str(plot_file), dpi=300)
    plt.close()

    print(f"\n📈 Đã vẽ và lưu đồ thị đánh đổi tại:")
    print(f"👉 {plot_file}")


if __name__ == "__main__":
    tune_confidence_threshold()
