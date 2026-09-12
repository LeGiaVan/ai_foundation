"""
Bài tập A8: Huấn luyện mô hình Phân loại Lỗi Bề mặt (QC Defect Classification)
Sử dụng Dataset NEU Surface Defect Database (6 lớp lỗi bề mặt thép)

Cấu trúc khung bài tập gồm 5 bước chuẩn PyTorch:
1. Cấu hình Hyperparameters & Device
2. Chuẩn bị Dataset & DataLoader (dùng ImageFolder + Transforms)
3. Định nghĩa Mô hình CNN (Bài 1 - CNN từ đầu & Bài 2 - ResNet18 Transfer Learning)
4. Huấn luyện (Training Loop) & Đánh giá (Validation Loop)
5. Tính Confusion Matrix & Đánh giá kết quả (Bài 3)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG & HYPERPARAMETERS
# ==========================================
CONFIG = {
    "data_dir": None,  # Đường dẫn tới thư mục dataset (VD: từ kagglehub hoặc thư mục giải nén)
    "img_size": (224, 224),
    "batch_size": 32,
    "lr": 0.001,
    "epochs": 10,
    "num_classes": 6,   # NEU dataset có 6 lớp lỗi
    "model_type": "custom_cnn",  # Chọn: "custom_cnn" (Bài 1) hoặc "resnet18" (Bài 2)
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ==========================================
# 2. MÔ HÌNH CNN TỪ ĐẦU (Bài 1)
# ==========================================
class CustomCNN(nn.Module):
    """
    Theo đề bài A8 - Bài 1:
    Conv(3->16, 3, pad=1) -> ReLU -> MaxPool(2)
    -> Conv(16->32, 3, pad=1) -> ReLU -> MaxPool(2)
    -> Flatten -> Linear(32 * 56 * 56 -> N)
    """
    def __init__(self, num_classes=6):
        super(CustomCNN, self).__init__()
        
        # Ký hiệu bên dưới theo quy ước trực quan: H×W×C (chiều cao × chiều rộng × kênh màu)
        # (Lưu ý: PyTorch lưu tensor theo [Batch, C, H, W] — thứ tự đảo ngược với cách viết dưới đây)
        #
        # Input:   224×224×3
        # Conv1:   224×224×16  (padding=1 giữ nguyên H×W)
        # MaxPool1: 112×112×16  (stride=2 chia đôi H và W)
        # Conv2:   112×112×32  (padding=1 giữ nguyên H×W)
        # MaxPool2:  56×56×32  (stride=2 chia đôi H và W)
        # Flatten:  56*56*32 = 100,352 → đây là đầu vào của Linear
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def build_model(model_type, num_classes):
    if model_type == "custom_cnn":
        print("[INFO] Khởi tạo Custom CNN (Bài 1)...")
        model = CustomCNN(num_classes=num_classes)
    elif model_type == "resnet18":
        print("[INFO] Khởi tạo ResNet18 Pretrained (Bài 2)...")
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Thay thế layer FC cuối cùng bằng số lớp bài toán QC
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model

# ==========================================
# 3. NẠP DỮ LIỆU & TRANSFORMS
# ==========================================
def get_dataloaders(data_dir, img_size, batch_size):
    # Transform căn bản cho Validation/Test
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transform bổ sung Data Augmentation cho Train (Bài 1 bước 3)
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # =====================================================================
    # NOTE: Tạo 2 Dataset độc lập để tránh bug tham chiếu bộ nhớ.
    # Nếu chỉ dùng 1 ImageFolder rồi gán val_dataset.dataset.transform = val_transform,
    # cả train_dataset cũng sẽ bị đổi sang val_transform và mất Data Augmentation!
    # =====================================================================
    train_dataset_base = datasets.ImageFolder(root=data_dir, transform=train_transform)
    val_dataset_base = datasets.ImageFolder(root=data_dir, transform=val_transform)
    
    # Chia Train / Validation (80% train, 20% val) cố định seed 42
    train_size = int(0.8 * len(train_dataset_base))
    val_size = len(train_dataset_base) - train_size
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(len(train_dataset_base)), [train_size, val_size], generator=generator
    )
    
    # Tạo 2 Subset độc lập tương ứng với 2 bộ transform
    train_dataset = Subset(train_dataset_base, train_indices)
    val_dataset = Subset(val_dataset_base, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"[DATA] Tổng số ảnh: {len(train_dataset_base)} | Train: {train_size} | Val: {val_size}")
    print(f"[DATA] Các lớp (classes): {train_dataset_base.classes}")
    return train_loader, val_loader, train_dataset_base.classes

# ==========================================
# 4. VÒNG LẶP HUẤN LUYỆN & ĐÁNH GIÁ
# ==========================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc, all_preds, all_labels

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    device = torch.device(CONFIG["device"])
    print(f"[SYSTEM] Đang chạy trên thiết bị: {device}")

    # 1. Kiểm tra thư mục dữ liệu
    data_dir = CONFIG["data_dir"]
    if data_dir is None or not os.path.exists(data_dir):
        print("\n[HƯỚNG DẪN] Bạn cần chạy file `download_NEU.py` trước để tải dataset!")
        print("Sau đó gán đường dẫn tải về vào `CONFIG['data_dir']` trong file này.\n")
        return

    # 2. Khởi tạo DataLoaders
    train_loader, val_loader, class_names = get_dataloaders(
        data_dir, CONFIG["img_size"], CONFIG["batch_size"]
    )

    # 3. Khởi tạo Model, Loss, Optimizer
    model = build_model(CONFIG["model_type"], len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    # 4. Training Loop
    print("\n--- BẮT ĐẦU HUẤN LUYỆN ---")
    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch:02d}/{CONFIG['epochs']:02d}] | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}%")

    # 5. Đánh giá cuối cùng (Bài 3: Confusion Matrix)
    _, _, final_preds, final_labels = evaluate(model, val_loader, criterion, device)
    
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print("\n--- PHÂN TÍCH KẾT QUẢ (BÀI 3) ---")
        print("Classification Report:")
        print(classification_report(final_labels, final_preds, target_names=class_names))
        print("Confusion Matrix:")
        print(confusion_matrix(final_labels, final_preds))
    except ImportError:
        print("\n[Gợi ý] Cài thêm `scikit-learn` (`pip install scikit-learn`) để in Confusion Matrix bài 3 đẹp mắt!")

if __name__ == "__main__":
    main()
