import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Đọc ảnh grayscale
gray = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# 2. Blur nhẹ trước (dùng chung cho cả 2 để so sánh công bằng)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# ===== KẾT QUẢ 1: SOBEL (raw, chưa qua non-max suppression / threshold kép) =====
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
magnitude = cv2.magnitude(sobel_x, sobel_y)
sobel_result = cv2.convertScaleAbs(magnitude)   # đưa về 0-255 để hiển thị

# ===== KẾT QUẢ 2: CANNY (đầy đủ pipeline 5 bước) =====
canny_result = cv2.Canny(blurred, threshold1=50, threshold2=150)

# 3. Hiển thị song song để so sánh
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(gray, cmap='gray')
axes[0].set_title('Ảnh gốc (grayscale)')
axes[0].axis('off')

axes[1].imshow(sobel_result, cmap='gray')
axes[1].set_title('Sobel (magnitude thô)')
axes[1].axis('off')

axes[2].imshow(canny_result, cmap='gray')
axes[2].set_title('Canny (5 bước hoàn chỉnh)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('sobel_vs_canny.png', dpi=150)
plt.show()