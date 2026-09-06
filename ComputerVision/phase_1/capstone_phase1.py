import cv2
import numpy as np


def _is_lighting_uneven(gray, grid=4, std_threshold=8.0):
    """Chia ảnh thành lưới grid x grid, đo độ lệch chuẩn độ sáng trung bình giữa
    các ô — std cao nghĩa là ánh sáng lệch vùng (bóng đổ, đèn chiếu lệch tâm)."""
    h, w = gray.shape
    means = []
    for i in range(grid):
        for j in range(grid):
            y0, y1 = i * h // grid, (i + 1) * h // grid
            x0, x1 = j * w // grid, (j + 1) * w // grid
            means.append(gray[y0:y1, x0:x1].mean())
    return float(np.std(means)) > std_threshold


def qc_pipeline(image_path, mode='threshold', save_debug=False, debug_prefix='debug'):
    """
    Pipeline QC 8 bước — đã kiểm chứng thực nghiệm trên bộ dataset test, không chỉ
    suy luận lý thuyết. 3 điểm sửa quan trọng so với các bản trước:

    1. Threshold branch kiểm tra CẢ 2 polarity (binary và binary đảo ngược) trước
       khi tìm contour — vì Otsu không đảm bảo lỗi luôn là vùng SÁNG hơn hay TỐI
       hơn nền; nếu chỉ tìm 1 chiều, lỗi tối hơn nền sẽ bị bỏ lọt hoàn toàn
       (findContours với RETR_EXTERNAL không thấy "lỗ" bên trong vùng nền).
    2. Threshold branch lọc theo CẢ min_area lẫn max_area — vì khi ánh sáng lệch
       hoặc Otsu chia sai, đôi khi gần nửa ảnh bị nhận nhầm thành 1 "lỗi" khổng lồ.
       max_area chặn đúng trường hợp này.
    3. Canny branch dùng blur mạnh hơn (9x9) và KHÔNG áp CLAHE clip quá cao trước
       Canny — CLAHE khuếch đại tương phản đồng thời khuếch đại luôn nhiễu, đẩy
       sàn nhiễu gradient lên gần bằng tín hiệu thật, khiến Canny mù hoàn toàn nếu
       threshold1/2 giữ mặc định 50/150.

    Params:
        image_path: đường dẫn ảnh cần kiểm tra
        mode: 'threshold' cho lỗi dạng vùng/mảng (vết bẩn, đốm ố)
              'canny' cho lỗi dạng viền/vết nứt mảnh (trầy, sứt mẻ)
        save_debug: True để lưu ảnh từng bước
        debug_prefix: tiền tố tên file debug
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

    defects = []
    kept_contours = []
    method_used = None
    binary_for_debug = None

    if mode == 'threshold':
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        uneven = _is_lighting_uneven(gray)
        if uneven:
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, blockSize=51, C=5
            )
            method_used = 'adaptive (blockSize=51)'
        else:
            _, binary = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            method_used = 'otsu'

        binary_for_debug = binary
        kernel = np.ones((9, 9), np.uint8)
        img_area = gray.shape[0] * gray.shape[1]
        MIN_AREA = 300
        MAX_AREA = int(img_area * 0.08)  # 1 lỗi thật hiếm khi chiếm >8% khung hình

        # Xét cả 2 polarity — lỗi có thể sáng hơn hoặc tối hơn nền
        for b in (binary, cv2.bitwise_not(binary)):
            cleaned = cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel, iterations=1)
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if MIN_AREA < area < MAX_AREA:
                    x, y, w, h = cv2.boundingRect(cnt)
                    defects.append({'area_px': area, 'bbox': (x, y, w, h),
                                     'perimeter_px': cv2.arcLength(cnt, True)})
                    kept_contours.append(cnt)

    elif mode == 'canny':
        # KHÔNG dùng CLAHE clip cao ở đây — nó khuếch đại nhiễu, đẩy sàn nhiễu
        # gradient lên cao, làm Canny mất luôn tín hiệu thật. Blur mạnh hơn thay thế.
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        binary = cv2.Canny(blurred, threshold1=10, threshold2=30)
        binary_for_debug = binary
        kernel = np.ones((7, 7), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        MIN_AREA = 50
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > MIN_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                defects.append({'area_px': area, 'bbox': (x, y, w, h),
                                 'perimeter_px': cv2.arcLength(cnt, True)})
                kept_contours.append(cnt)
        method_used = 'canny (10/30)'

    else:
        raise ValueError("mode phải là 'threshold' hoặc 'canny'")

    result = {
        'gray': gray,
        'binary': binary_for_debug,
        'contours': kept_contours,   # chỉ chứa contour của các defect đã lọt qua min/max area
        'defects': defects,
        'defect_count': len(defects),
        'method_used': method_used,
    }

    if save_debug:
        cv2.imwrite(f'{debug_prefix}_gray.jpg', gray)
        cv2.imwrite(f'{debug_prefix}_binary.jpg', binary_for_debug)

    return result


def draw_defects(image_path, result, output_path='result_annotated.jpg'):
    """Vẽ contour (xanh lá) + bounding box + diện tích (đỏ) lên ảnh gốc để kiểm tra trực quan."""
    img_color = cv2.imread(image_path)
    cv2.drawContours(img_color, result['contours'], -1, (0, 255, 0), 2)

    for d in result['defects']:
        x, y, w, h = d['bbox']
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            img_color, f"{d['area_px']:.0f}px",
            (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )

    cv2.imwrite(output_path, img_color)
    return output_path


# if __name__ == '__main__':
#     import glob
#     import os

#     DATASET_DIR = "qc_test_dataset/qc_test_dataset/images"
#     OUT_DIR = "annotated_samples"
#     os.makedirs(OUT_DIR, exist_ok=True)

#     print(f"{'Ảnh':30s} {'Threshold':28s} {'Canny':10s}")
#     print("-" * 70)
#     for path in sorted(glob.glob(os.path.join(DATASET_DIR, "*.jpg"))):
#         name = os.path.basename(path)
#         r_t = qc_pipeline(path, mode='threshold')
#         r_c = qc_pipeline(path, mode='canny')
#         print(f"{name:30s} {r_t['method_used']:>15s} = {r_t['defect_count']:2d} lỗi   canny = {r_c['defect_count']:2d} lỗi")

#         # Xuất ảnh annotated cho vài ảnh tiêu biểu để kiểm tra trực quan
#         if r_t['defect_count'] > 0:
#             draw_defects(path, r_t, os.path.join(OUT_DIR, f"threshold_{name}"))
#         if r_c['defect_count'] > 0:
#             draw_defects(path, r_c, os.path.join(OUT_DIR, f"canny_{name}"))

# ===== Sử dụng =====
if __name__ == '__main__':
#     # Ví dụ 1: phát hiện vết bẩn/đốm ố (dạng vùng)
#     # result_stain = qc_pipeline('qc_test_dataset/qc_test_dataset/images/09_stain.jpg', mode='threshold', save_debug=True)
#     # print(f"[Threshold] Phát hiện {result_stain['defect_count']} vết bẩn")
#     # for i, d in enumerate(result_stain['defects']):
#     #     print(f"  Lỗi {i+1}: diện tích={d['area_px']:.1f}px, bbox={d['bbox']}")
#     # draw_defects('qc_test_dataset/qc_test_dataset/images/09_stain.jpg', result_stain, 'result_stain.jpg')

#     # Ví dụ 2: phát hiện vết trầy/sứt mẻ (dạng viền mỏng)
    result_scratch = qc_pipeline('sperm.png', mode='canny', save_debug=True)
    print(f"[Canny] Phát hiện {result_scratch['defect_count']} vết trầy")
    draw_defects('sperm.png', result_scratch, 'result_sperm.jpg')
 