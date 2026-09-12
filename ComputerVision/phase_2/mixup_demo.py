import torch
from torchvision import transforms
from PIL import Image

def mixup_demo():
    # 1. Tải 2 ảnh (Ảnh 1 là quả cà chua, Ảnh 2 là bảng mạch từ phase 1)
    img1 = Image.open('tomato.png').convert('RGB')
    img2_path = 'apple.png'
    img2 = Image.open(img2_path).convert('RGB')
    
    # 2. Resize 2 ảnh về cùng một kích thước và chuyển thành Tensor (0.0 -> 1.0)
    # Bắt buộc phải cùng kích thước thì mới cộng pixel với nhau được
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor() 
    ])
    
    tensor1 = preprocess(img1)
    tensor2 = preprocess(img2)
    
    # 3. Kỹ thuật Mixup: Trộn theo hệ số lambda
    # Giả sử ta lấy 60% đặc trưng của quả cà chua và 40% của bảng mạch
    lam = 0.6
    mixed_tensor = lam * tensor1 + (1 - lam) * tensor2
    
    # Ở bước huấn luyện (Train), ta cũng trộn nhãn y hệt: 
    # mixed_label = 0.6 * label_tomato + 0.4 * label_board
    
    # 4. Chuyển ngược lại thành ảnh thường để xem bằng mắt và lưu
    mixed_img = transforms.ToPILImage()(mixed_tensor)
    mixed_img.save('mixup_result.jpg')
    
    # In ra câu tiếng Anh để tránh lỗi Unicode trên console
    print(f"Mixup applied with lambda={lam}.")
    print("Saved the mixed image to 'mixup_result.jpg'.")

if __name__ == '__main__':
    mixup_demo()
