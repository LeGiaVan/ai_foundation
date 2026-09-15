"""
Bài tập A8 - Phần 2: Transfer Learning với ResNet-18 (NEU Surface Defect)
Bao gồm 2 Chiến thuật Transfer Learning:
  1. Chiến thuật 1: Feature Extraction (Frozen Backbone, chỉ train FC)
  2. Chiến thuật 2: Fine-Tuning toàn diện (Unfrozen Backbone với Differential LR)
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG & HYPERPARAMETERS
# ==========================================
CONFIG = {
    "data_dir": r"d:\ai_foundation\ComputerVision\phase_2\A8\dataset\NEU-DET\train\images",
    "img_size": (224, 224),
    "batch_size": 32,
    "epochs": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
device = torch.device(CONFIG["device"])
print(f"[INFO] PyTorch: {torch.__version__} | Device: {CONFIG['device']}")

# ==========================================
# 2. CHUẨN BỊ DỮ LIỆU
# ==========================================
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

val_tf = transforms.Compose([
    transforms.Resize(CONFIG["img_size"]),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_tf = transforms.Compose([
    transforms.Resize(CONFIG["img_size"]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_ds = ImageFolder(root=CONFIG["data_dir"], transform=train_tf)
val_ds   = ImageFolder(root=CONFIG["data_dir"], transform=val_tf)
class_names = train_ds.classes
num_classes = len(class_names)

tr_n = int(0.8 * len(train_ds))
vl_n = len(train_ds) - tr_n
gen = torch.Generator().manual_seed(42)
tr_idx, vl_idx = random_split(range(len(train_ds)), [tr_n, vl_n], generator=gen)

tr_set = Subset(train_ds, tr_idx)
vl_set = Subset(val_ds, vl_idx)

train_loader = DataLoader(tr_set, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=0)
val_loader   = DataLoader(vl_set, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

print(f"[DATA] Tổng: {len(train_ds)} | Train: {tr_n} | Val: {vl_n} | Lớp: {num_classes}")

# ==========================================
# 3. HÀM TRAIN & EVALUATE
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, dev):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, dev):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        out = model(x)
        loss = criterion(out, y)
        running_loss += loss.item() * x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)

# ==========================================
# 4. CHIẾN THUẬT 1: FEATURE EXTRACTION (FROZEN)
# ==========================================
def run_strategy_1():
    print("\n" + "="*60)
    print("CHIẾN THUẬT 1: ĐÓNG BĂNG TOÀN BỘ BACKBONE (CHỈ TRAIN FC)")
    print("="*60)
    
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for p in m.parameters():
        p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    m = m.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(m.fc.parameters(), lr=1e-3)
    
    for ep in range(1, CONFIG["epochs"] + 1):
        tl, ta = train_one_epoch(m, train_loader, criterion, optimizer, device)
        vl, va, _, _ = evaluate(m, val_loader, criterion, device)
        print(f"[S1 - Ep {ep:02d}] Train: {tl:.4f} / {ta*100:.1f}% || Val: {vl:.4f} / {va*100:.1f}%")
    return m

# ==========================================
# 5. CHIẾN THUẬT 2: FINE-TUNING VỚI DIFFERENTIAL LR
# ==========================================
def run_strategy_2():
    print("\n" + "="*60)
    print("CHIẾN THUẬT 2: MỞ KHÓA BACKBONE VỚI DIFFERENTIAL LEARNING RATES")
    print("="*60)
    
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for p in m.parameters():
        p.requires_grad = True
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    m = m.to(device)
    
    backbone_params = [p for name, p in m.named_parameters() if not name.startswith("fc")]
    fc_params       = [p for name, p in m.named_parameters() if name.startswith("fc")]
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": 1e-5},  # LR siêu nhỏ cho Backbone
        {"params": fc_params,       "lr": 1e-4},  # LR vừa phải cho FC
    ], weight_decay=1e-2)
    
    for ep in range(1, CONFIG["epochs"] + 1):
        tl, ta = train_one_epoch(m, train_loader, criterion, optimizer, device)
        vl, va, _, _ = evaluate(m, val_loader, criterion, device)
        print(f"[S2 - Ep {ep:02d}] Train: {tl:.4f} / {ta*100:.1f}% || Val: {vl:.4f} / {va*100:.1f}%")
    
    vl, va, preds, labels_true = evaluate(m, val_loader, criterion, device)
    print("\n=== BÁO CÁO PHÂN LOẠI CHI TIẾT (CHIẾN THUẬT 2) ===")
    print(classification_report(labels_true, preds, target_names=class_names, digits=4))
    return m

if __name__ == "__main__":
    m1 = run_strategy_1()
    m2 = run_strategy_2()
