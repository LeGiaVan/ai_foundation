"""
Tạo dataset ảnh giả lập để test pipeline QC:
- clean: ảnh sạch, không lỗi
- stain: có vết bẩn/đốm ố (vùng rộng, đồng nhất) -> test nhánh threshold
- scratch: có vết trầy/nứt (đường mỏng, độ sáng thất thường) -> test nhánh canny
- mixed: có cả 2 loại lỗi cùng lúc
- uneven_light: ảnh sạch nhưng ánh sáng chiếu lệch 1 góc (test CLAHE/adaptive threshold)

Mỗi ảnh có nhiễu hạt (grain noise) và texture nhẹ để giống ảnh camera công nghiệp thật.
Ground truth lưu trong labels.csv để bạn tự đối chiếu kết quả pipeline.
"""

import cv2
import numpy as np
import os
import csv
import random

random.seed(42)
np.random.seed(42)

OUT_DIR = "/mnt/user-data/outputs/qc_test_dataset"
IMG_DIR = os.path.join(OUT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

W, H = 640, 480
BASE_GRAY = 180  # nền sáng kiểu bề mặt kim loại/nhựa dưới đèn LED


def make_base_surface(uneven_light=False, texture_strength=3):
    """Tạo nền bề mặt sản phẩm: màu xám đều + texture nhẹ + (tuỳ chọn) ánh sáng lệch góc.

    Lưu ý: mức nhiễu ở đây được canh chỉnh thấp hơn bản đầu (texture_strength 6->3,
    sensor noise 3->1.5) vì bản đầu tạo nhiễu nền đủ lớn để tự nó gây ra false positive
    ngay cả trên ảnh "sạch" khi Otsu ép chia đôi phân bố unimodal — một lỗi thiết kế
    dataset, không phải lỗi của pipeline.
    """
    surface = np.full((H, W), BASE_GRAY, dtype=np.float32)

    texture = np.random.normal(0, texture_strength, (H, W)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (5, 5), 0)
    surface += texture

    if uneven_light:
        yy, xx = np.mgrid[0:H, 0:W]
        gradient = (xx / W) * 40 - 18
        surface += gradient

    sensor_noise = np.random.normal(0, 1.5, (H, W)).astype(np.float32)
    surface += sensor_noise

    return np.clip(surface, 0, 255).astype(np.uint8)


def add_stain(img, cx, cy, radius, darkness=45):
    """Vết bẩn/đốm ố: vùng blob tối hơn nền, đồng nhất, biên hơi mờ tự nhiên."""
    mask = np.zeros((H, W), dtype=np.float32)
    cv2.circle(mask, (cx, cy), radius, 1.0, -1)
    # méo hình blob cho tự nhiên hơn hình tròn hoàn hảo
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    img_f = img.astype(np.float32)
    img_f -= mask * darkness
    return np.clip(img_f, 0, 255).astype(np.uint8), (cx - radius, cy - radius, 2 * radius, 2 * radius)


def add_scratch(img, x1, y1, x2, y2, max_dip=32, width=2):
    """Vết trầy: đường mỏng, độ sáng thay đổi thất thường dọc theo đường (không đồng nhất).

    max_dip tăng từ 18 lên 32 so với bản đầu — bản đầu tạo tín hiệu quá yếu, gần
    như chìm dưới sàn nhiễu sau khi CLAHE khuếch đại, khiến Canny không bắt được
    dù pipeline đúng logic. Đây là lỗi thiết kế dataset, đã sửa lại cho thực tế hơn.
    """
    line_mask = np.zeros((H, W), dtype=np.float32)
    cv2.line(line_mask, (x1, y1), (x2, y2), 1.0, width)
    line_mask = cv2.GaussianBlur(line_mask, (3, 3), 0)

    n_pts = 50
    variation = np.random.uniform(-max_dip, max_dip * 0.6, n_pts)
    variation_map = cv2.resize(variation.reshape(1, -1), (W, H), interpolation=cv2.INTER_LINEAR)

    img_f = img.astype(np.float32)
    img_f += line_mask * variation_map
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    bbox = (max(x_min - 3, 0), max(y_min - 3, 0), (x_max - x_min) + 6, (y_max - y_min) + 6)
    return np.clip(img_f, 0, 255).astype(np.uint8), bbox


def add_grain(img, amount=4):
    noise = np.random.normal(0, amount, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


records = []
idx = 0

def save(img, name, defect_type, bboxes):
    global idx
    fname = f"{idx:02d}_{name}.jpg"
    cv2.imwrite(os.path.join(IMG_DIR, fname), img)
    records.append({
        "filename": fname,
        "defect_type": defect_type,
        "bboxes_xywh": ";".join([f"{b[0]},{b[1]},{b[2]},{b[3]}" for b in bboxes])
    })
    idx += 1


# ===== 1. Ảnh sạch, ánh sáng đều =====
for i in range(3):
    img = make_base_surface(uneven_light=False)
    img = add_grain(img)
    save(img, "clean", "none", [])

# ===== 2. Ảnh sạch, ánh sáng lệch góc (test CLAHE / adaptive threshold) =====
for i in range(3):
    img = make_base_surface(uneven_light=True)
    img = add_grain(img)
    save(img, "clean_uneven_light", "none", [])

# ===== 3. Vết bẩn/đốm ố (test nhánh threshold) =====
for i in range(4):
    img = make_base_surface(uneven_light=random.choice([True, False]))
    cx, cy = random.randint(150, W - 150), random.randint(120, H - 120)
    radius = random.randint(20, 45)
    img, bbox = add_stain(img, cx, cy, radius, darkness=random.randint(35, 60))
    img = add_grain(img)
    save(img, "stain", "stain", [bbox])

# ===== 4. Vết trầy/nứt (test nhánh Canny) =====
for i in range(4):
    img = make_base_surface(uneven_light=random.choice([True, False]))
    x1, y1 = random.randint(60, 200), random.randint(60, H - 60)
    x2, y2 = x1 + random.randint(150, 300), y1 + random.randint(-80, 80)
    img, bbox = add_scratch(img, x1, y1, x2, y2, max_dip=random.randint(28, 45))
    img = add_grain(img)
    save(img, "scratch", "scratch", [bbox])

# ===== 5. Cả 2 loại lỗi cùng lúc (test toàn bộ pipeline, cả 2 nhánh) =====
for i in range(3):
    img = make_base_surface(uneven_light=random.choice([True, False]))
    cx, cy = random.randint(120, 250), random.randint(120, H - 120)
    radius = random.randint(18, 35)
    img, bbox1 = add_stain(img, cx, cy, radius, darkness=random.randint(35, 55))
    x1, y1 = random.randint(350, 450), random.randint(60, H - 60)
    x2, y2 = x1 + random.randint(100, 180), y1 + random.randint(-60, 60)
    img, bbox2 = add_scratch(img, x1, y1, x2, y2, max_dip=random.randint(28, 40))
    img = add_grain(img)
    save(img, "mixed", "stain+scratch", [bbox1, bbox2])

# ===== 6. Trường hợp khó: vết trầy rất mờ (test giới hạn của threshold cố định) =====
for i in range(2):
    img = make_base_surface(uneven_light=False)
    x1, y1 = random.randint(100, 300), random.randint(100, 350)
    x2, y2 = x1 + random.randint(120, 220), y1 + random.randint(-40, 40)
    img, bbox = add_scratch(img, x1, y1, x2, y2, max_dip=16)  # dip nhẹ hơn scratch thường, vẫn đủ trên sàn nhiễu
    img = add_grain(img)
    save(img, "scratch_faint", "scratch_faint", [bbox])

# ===== Ghi labels.csv =====
csv_path = os.path.join(OUT_DIR, "labels.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "defect_type", "bboxes_xywh"])
    writer.writeheader()
    writer.writerows(records)

print(f"Đã tạo {idx} ảnh trong {IMG_DIR}")
print(f"Ground truth: {csv_path}")
