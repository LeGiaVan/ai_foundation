"""
Module tự triển khai IoU (Intersection over Union) và NMS (Non-Maximum Suppression) từ số 0.
Phục vụ bài tập B6 trong Cheatsheet Object Detection (Domain QC / Vision Inspection).
"""

import sys
import io
import numpy as np
from typing import List, Tuple, Union

# Đảm bảo in tiếng Việt trên Windows console không bị lỗi mã hóa
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def compute_iou(box1: Union[List[float], np.ndarray], 
                box2: Union[List[float], np.ndarray]) -> float:
    """
    Tính IoU giữa 2 bounding box dạng [x1, y1, x2, y2].
    (x1, y1): toạ độ góc trên bên trái (top-left)
    (x2, y2): toạ độ góc dưới bên phải (bottom-right)
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2

    # 1. Tìm toạ độ phần giao nhau (Intersection Box)
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    # 2. Chiều rộng và chiều cao phần giao nhau
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # 3. Diện tích từng box
    area1 = max(0.0, b1_x2 - b1_x1) * max(0.0, b1_y2 - b1_y1)
    area2 = max(0.0, b2_x2 - b2_x1) * max(0.0, b2_y2 - b2_y1)

    # 4. Diện tích hợp nhất (Union Area) = Area1 + Area2 - Intersection
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0

    return float(inter_area / union_area)


def compute_iou_vectorized(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Tính IoU giữa 1 box [4] với một mảng N boxes [N, 4] bằng NumPy vectorization.
    Hữu ích để tăng tốc thuật toán NMS.
    """
    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_box = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])

    union_area = area_box + area_boxes - inter_area
    union_area = np.maximum(union_area, 1e-8)  # Tránh chia cho 0

    return inter_area / union_area


def non_maximum_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25
) -> List[int]:
    """
    Thuật toán Khử Cực Đại Không Tối Đa (NMS - Non-Maximum Suppression).
    
    Tham số:
    ---------
    boxes: np.ndarray shape (N, 4) dạng [x1, y1, x2, y2]
    scores: np.ndarray shape (N,) điểm tự tin (confidence score)
    iou_threshold: float, ngưỡng IoU để coi 2 box là trùng lặp cùng 1 vật thể
    score_threshold: float, loại bỏ ngay các box dưới ngưỡng confidence này

    Trả về:
    ---------
    keep: List[int], danh sách các chỉ số (indices) của box được giữ lại
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    # Bước 1: Lọc bỏ các box có điểm confidence dưới score_threshold
    valid_mask = scores >= score_threshold
    indices = np.where(valid_mask)[0]

    if len(indices) == 0:
        return []

    valid_boxes = boxes[indices]
    valid_scores = scores[indices]

    # Bước 2: Sắp xếp các box theo thứ tự confidence giảm dần
    order = np.argsort(valid_scores)[::-1]

    keep = []

    # Bước 3: Lặp qua từng box có điểm cao nhất
    while len(order) > 0:
        # Lấy box có score cao nhất hiện tại
        top_idx = order[0]
        original_idx = indices[top_idx]
        keep.append(int(original_idx))

        if len(order) == 1:
            break

        # Box hiện tại đang xét
        current_box = valid_boxes[top_idx]
        # Các box còn lại có score thấp hơn
        remaining_boxes = valid_boxes[order[1:]]

        # Bước 4: Tính IoU giữa current_box với các box còn lại
        ious = compute_iou_vectorized(current_box, remaining_boxes)

        # Bước 5: Chỉ giữ lại các box có IoU < iou_threshold (không bị chồng lấn quá mức)
        # Các box có IoU >= iou_threshold bị loại (suppressed)
        non_overlap_indices = np.where(ious < iou_threshold)[0]

        # Cập nhật lại danh sách order cho vòng lặp tiếp theo (+1 vì so sánh với order[1:])
        order = order[non_overlap_indices + 1]

    return keep


# =====================================================================
# BỘ TEST CASES TOÀN DIỆN (UNIT TESTS)
# =====================================================================

def run_tests():
    print("=" * 65)
    print("🚀 BẮT ĐẦU CHẠY KIỂM THỬ: IoU & NMS (Domain QC / Vision Inspection)")
    print("=" * 65)

    # -------------------------------------------------------------
    # NHÓM TEST 1: Kiểm thử độ chính xác của hàm compute_iou
    # -------------------------------------------------------------
    print("\n[PHẦN 1] Kiểm tra hàm compute_iou với các hình thái hình học:")

    # Case 1.1: Hai box hoàn toàn trùng khít
    b1 = [10, 10, 50, 50]
    b2 = [10, 10, 50, 50]
    iou = compute_iou(b1, b2)
    assert np.isclose(iou, 1.0), f"Thất bại: Mong đợi 1.0 nhưng nhận {iou}"
    print(f"  ✅ Test 1.1 - Hai box trùng khít hoàn toàn: IoU = {iou:.4f} (Kỳ vọng: 1.0000)")

    # Case 1.2: Hai box hoàn toàn tách biệt
    b1 = [0, 0, 10, 10]
    b2 = [20, 20, 30, 30]
    iou = compute_iou(b1, b2)
    assert np.isclose(iou, 0.0), f"Thất bại: Mong đợi 0.0 nhưng nhận {iou}"
    print(f"  ✅ Test 1.2 - Hai box rời rạc không chạm: IoU = {iou:.4f} (Kỳ vọng: 0.0000)")

    # Case 1.3: Hai box chỉ chạm mép ngoài nhau (tiếp xúc biên)
    b1 = [0, 0, 10, 10]
    b2 = [10, 0, 20, 10]
    iou = compute_iou(b1, b2)
    assert np.isclose(iou, 0.0), f"Thất bại: Chạm mép diện tích giao phải là 0, nhận {iou}"
    print(f"  ✅ Test 1.3 - Hai box tiếp xúc mép ngoài: IoU = {iou:.4f} (Kỳ vọng: 0.0000)")

    # Case 1.4: Giao nhau một phần (Tính toán giải tích kiểm chứng)
    # Box A: [0, 0, 2, 2] -> diện tích = 4
    # Box B: [1, 1, 3, 3] -> diện tích = 4
    # Giao nhau: [1, 1, 2, 2] -> chiều rộng = 1, chiều cao = 1 -> diện tích giao = 1
    # Hợp: 4 + 4 - 1 = 7 -> IoU = 1 / 7 ≈ 0.142857
    b1 = [0, 0, 2, 2]
    b2 = [1, 1, 3, 3]
    iou = compute_iou(b1, b2)
    expected_iou = 1.0 / 7.0
    assert np.isclose(iou, expected_iou, atol=1e-5), f"Thất bại: Mong đợi {expected_iou} nhưng nhận {iou}"
    print(f"  ✅ Test 1.4 - Giao nhau 1 phần tính tay: IoU = {iou:.6f} (Kỳ vọng: {expected_iou:.6f})")

    # Case 1.5: Box nhỏ lọt thỏm trong box to
    # Box A: [0, 0, 4, 4] -> S = 16
    # Box B: [1, 1, 3, 3] -> S = 4
    # Giao: [1, 1, 3, 3] -> S = 4; Hợp: 16 -> IoU = 4/16 = 0.25
    b1 = [0, 0, 4, 4]
    b2 = [1, 1, 3, 3]
    iou = compute_iou(b1, b2)
    assert np.isclose(iou, 0.25), f"Thất bại: Mong đợi 0.25 nhưng nhận {iou}"
    print(f"  ✅ Test 1.5 - Box con nằm trong box to: IoU = {iou:.4f} (Kỳ vọng: 0.2500)")

    # -------------------------------------------------------------
    # NHÓM TEST 2: Kịch bản QC phát hiện lỗi ngoại quan linh kiện
    # -------------------------------------------------------------
    print("\n[PHẦN 2] Kịch bản thực chiến QC Nhà Máy (Inspection):")
    print("Giả lập ảnh camera công nghiệp chụp bo mạch PCB có 2 lỗi thật:")
    print("  - Vị trí 1 (Góc trái): Vết xước (Scratch) -> Model sinh ra 3 box đè lên nhau.")
    print("  - Vị trí 2 (Góc phải): Lỗ thủng (Hole/Dent) -> Model sinh ra 2 box đè lên nhau.")
    print("  - Vị trí 3: Báo động giả ngẫu nhiên có score thấp.")

    boxes = np.array([
        # Cụm vết xước 1
        [10.0, 10.0, 50.0, 50.0],  # Box 0: score 0.92 (chuẩn nhất)
        [12.0,  8.0, 48.0, 52.0],  # Box 1: score 0.85 (chồng lấn nặng Box 0)
        [ 8.0, 12.0, 53.0, 49.0],  # Box 2: score 0.65 (chồng lấn nặng Box 0)
        
        # Cụm lỗ thủng 2
        [200.0, 150.0, 250.0, 200.0],  # Box 3: score 0.88 (chuẩn nhất cụm 2)
        [198.0, 152.0, 252.0, 198.0],  # Box 4: score 0.73 (chồng lấn nặng Box 3)

        # Báo động giả độ tự tin thấp
        [100.0, 100.0, 140.0, 140.0]   # Box 5: score 0.15 (nhiễu)
    ])

    scores = np.array([0.92, 0.85, 0.65, 0.88, 0.73, 0.15])

    # Thực hiện NMS với iou_threshold=0.5, score_threshold=0.3
    keep_indices = non_maximum_suppression(
        boxes=boxes,
        scores=scores,
        iou_threshold=0.5,
        score_threshold=0.3
    )

    print(f"\n  Kết quả NMS giữ lại các box chỉ số: {keep_indices}")
    for idx in keep_indices:
        print(f"    -> Box #{idx}: Toạ độ={boxes[idx].tolist()}, Score={scores[idx]}")

    # Kỳ vọng: Giữ lại đúng Box 0 và Box 3!
    assert keep_indices == [0, 3], f"Thất bại: Kỳ vọng [0, 3], nhưng nhận được {keep_indices}"
    print("  ✅ Test 2.1 - NMS lọc đúng 2 box tối ưu đại diện cho 2 lỗi, triệt tiêu box thừa và nhiễu!")

    # -------------------------------------------------------------
    # NHÓM TEST 3: Edge Cases
    # -------------------------------------------------------------
    print("\n[PHẦN 3] Kiểm tra Edge Cases:")

    # Case 3.1: Không có box nào vượt qua score_threshold
    empty_keep = non_maximum_suppression(boxes, scores, score_threshold=0.95)
    assert empty_keep == [], f"Thất bại: Kỳ vọng rỗng nhưng nhận {empty_keep}"
    print(f"  ✅ Test 3.1 - Score threshold quá cao (0.95): Trả về rỗng [] chính xác.")

    # Case 3.2: Danh sách box rỗng đầu vào
    empty_input = non_maximum_suppression(np.empty((0, 4)), np.empty(0))
    assert empty_input == [], f"Thất bại: Kỳ vọng rỗng nhưng nhận {empty_input}"
    print(f"  ✅ Test 3.2 - Input rỗng ban đầu: Xử lý an toàn không crash.")

    # Case 3.3: Tất cả các box độc lập không hề chạm nhau
    sep_boxes = np.array([
        [0, 0, 10, 10],
        [20, 20, 30, 30],
        [40, 40, 50, 50]
    ])
    sep_scores = np.array([0.9, 0.8, 0.7])
    all_keep = non_maximum_suppression(sep_boxes, sep_scores, iou_threshold=0.5, score_threshold=0.5)
    assert all_keep == [0, 1, 2], f"Thất bại: Kỳ vọng giữ cả 3 [0, 1, 2], nhận {all_keep}"
    print(f"  ✅ Test 3.3 - Các box hoàn toàn không giao nhau: Giữ lại toàn bộ.")

    print("\n" + "=" * 65)
    print("🎉 TẤT CẢ 8 TEST CASES ĐÃ VƯỢT QUA THÀNH CÔNG (100% PASSED)!")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
