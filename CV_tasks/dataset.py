import os
import torchvision.transforms as transforms
from numpy.array_api import astype
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.datasets import MNIST,SVHN,USPS
from scipy.io import loadmat
import os
import numpy as np

class CVDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        label = self.y[idx]
        return img, label

class OfficeCaltech10Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []  # 存储图像路径
        self.labels = []  # 存储标签

        # 类名到数字标签的映射
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(os.listdir(root_dir))}

        # 遍历文件夹，加载图像路径和数字标签
        for class_name, idx in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(idx)  # 存储数字标签

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

def load_data(task="digit"):
    if task == "digit":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = USPS(root='./dataset', train=True, download=True, transform=transform)
        test_dataset = MNIST(root='./dataset', train=False, download=True, transform=transform)
    elif task == "office":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = OfficeCaltech10Dataset(root_dir='dataset/office_caltech_10/amazon', transform=transform)
        test_dataset = OfficeCaltech10Dataset(root_dir='dataset/office_caltech_10/caltech', transform=transform)
    else:
        print("Wrong Task!")
        exit(-1)
    return train_dataset, test_dataset


