import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import random

class ECGImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        # 收集所有图片路径和对应的标签
        for dir in os.listdir(root_dir):
            full_dir = os.path.join(root_dir, dir)
            for file_name in os.listdir(full_dir):
                if file_name.endswith('.png'):
                    parts = file_name.split('_')
                    if len(parts) >= 2:
                        label = parts[1].split('.')[0]  # 去除扩展名
                        self.image_files.append(os.path.join(full_dir, file_name))
                        self.labels.append(label)

        # 创建类别到索引的映射
        self.classes = sorted(set(self.labels))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('L')  # 转为灰度图
        label_str = self.labels[idx]
        label = self.class_to_idx[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),  # 从 16384 -> 128
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


transform = transforms.Compose([
    transforms.ToTensor(),
])

print(torch.__version__)
print("Collecting image files...")

dataset = ECGImageDataset(root_dir=r'.\plots\labeled_data', transform=transform)
print("Total images:", len(dataset))

# 划分训练集和测试集 (80% 训练集, 20% 测试集)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建训练集和测试集的 DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Dataset loaded.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = SimpleCNN(num_classes=len(dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 4
print("Starting training...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader):.4f}")

    # 在每个 epoch 结束时评估模型性能
        # 获取 'N' 对应的类别索引
    n_class_idx = dataset.class_to_idx.get('N', None)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)

            # 排除标签为 'N' 的样本
            if n_class_idx is not None:
                mask = (labels != n_class_idx)
                images = images[mask]
                labels = labels[mask]

            if len(labels) == 0:
                continue  # 跳过没有有效标签的 batch

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total > 0:
        print(f"Test Accuracy (excluding 'N') after Epoch {epoch+1}: {100 * correct / total:.2f}%")
    else:
        print(f"No valid samples to evaluate after Epoch {epoch+1}.")


print("Label mapping:")
for i, cls in enumerate(dataset.classes):
    print(f"{i}: {cls}")

model_save_path = 'ecg_cnn_model2.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
