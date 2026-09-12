from torchvision import transforms
from PIL import Image

# 1. Tách riêng phần augmentation (chưa chuyển sang Tensor) để dễ lưu/xem bằng mắt thường
aug_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),          # lật ngang
    transforms.RandomRotation(degrees=10),           # xoay nhỏ
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
])

# 2. Các bước chuyển sang Tensor và Chuẩn hoá (để đưa vào model)
tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Kết hợp cả 2 cho tập train thực tế
train_transform = transforms.Compose([
    aug_transform,
    tensor_transform
])

img_path = 'tomato.png'
img = Image.open(img_path)

# Áp dụng augmentation và lưu ảnh
aug_img = aug_transform(img)
aug_img.save('tomato_aug.png')
print("Saved augmented image to 'tomato_aug.png'")

# Xem thử kích thước của ảnh nếu đưa qua toàn bộ pipeline
tensor_img = train_transform(img)
print("Tensor shape (after ToTensor and Normalize):", tensor_img.shape)