import cv2
import numpy as np


class ImagePipeline:
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __call__(self, img):
        """
        Bọc toàn bộ pipeline trong __call__ để có thể gọi pipeline(img) trực tiếp
        """
        self.load_image(img)
        img = self.resize(img)
        img = self.normalize(img)
        img = self.to_chw(img)
        return img

    def load_image(self, path):
        """
        load(): đọc ảnh bằng OpenCV → in shape, dtype, min, max để luyện thói quen debug
        """
        img = cv2.imread(path)
        print(f"Shape: {img.shape}")
        print(f"Dtype: {img.dtype}")
        print(f"Min: {img.min()}")
        print(f"Max: {img.max()}")
    
    def resize(self, img, target_size=(224, 224)):
        """
        resize(): resize về target size
        """
        img = cv2.resize(img, target_size)
        return img
    
    def normalize(self, img):
        """
        normalize(): chuẩn hóa ảnh về 0-1 sau đó trừ mean/chia std (dùng broadcasting)
        """
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
        return img
    
    def to_chw(self, img):
        """
        to_chw(): chuyển ảnh từ HWC sang CHW
        """
        img = img.transpose(2, 0, 1)
        return img
    
class SimpleDataset:
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.image_paths[idx], self.labels[idx]