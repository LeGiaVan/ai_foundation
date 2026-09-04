import cv2

# # 1. Luôn dùng ảnh grayscale cho Canny
# gray = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# # 2. QUAN TRỌNG: Blur ảnh để khử nhiễu trước khi tìm cạnh
# # Nếu không blur, Canny rất nhạy cảm với các đốm nhiễu lấm tấm
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# # 3. Chạy Canny Edge Detection
# # gradient > 150 -> chắc chắn là cạnh
# # gradient < 50 -> chắc chắn vứt bỏ
# edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
# cv2.imwrite("edges.jpg", edges)


gray = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)   # đạo hàm theo x
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)   # đạo hàm theo y
magnitude = cv2.magnitude(sobel_x, sobel_y)  

# Chuyển đổi về kiểu uint8 (0-255) trước khi lưu ảnh để tránh cảnh báo
magnitude_8u = cv2.convertScaleAbs(magnitude)
cv2.imwrite("sobel_magnitude.jpg", magnitude_8u)