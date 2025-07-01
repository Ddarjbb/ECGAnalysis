import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import re
# 定义自定义数据集
class ECGImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')  # 转换为灰度图
        # 提取第一个 [xxx] 中的内容作为标签字符串
        match = re.search(r"\[(.*?)\]", self.image_files[idx])
        if match:
            label_str = match.group(1)  # 获取 [] 中的内容
        else:
            label_str = ' '  # 默认为空或设为 'Unknown'
        # 将标签字符串转换为类别ID
        # 将标签字符串转换为类别ID
        label_set = {
            'N': 0,  # 假设 'N' 仍然是一个类别
            'L': 1,
            'R': 2,
            'A': 3,
            'a': 4,
            'J': 5,
            'S': 6,
            'V': 7,
            'F': 8,
            '[': 9,
            '!': 10,
            ']': 11,
            'e': 12,
            'j': 13,
            'E': 14,
            'p': 15,
            'f': 16,
            'x': 17,
            'Q': 18,
            'i': 19,
            'q': 20
        }
        label = label_set.get(label_str, 0)  # 取第一个字符作为类别，未知标签默认为0

        if self.transform:
            image = self.transform(image)
        return image, label


# 定义轻量级CNN模型
class LightCNN(nn.Module):
    def __init__(self, num_classes=21):
        super(LightCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # 输入通道1，输出通道8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # 输出通道16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 输出通道32
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # 替代Flatten，输出固定维度
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)  # 直接映射到类别数
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平
        x = self.classifier(x)
        return x


def main():
    # 数据预处理和加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化
    ])

    dataset = ECGImageDataset(image_dir=r'.\plots', transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    print("Dataset loaded.")
    # 初始化模型、损失函数和优化器
    model = LightCNN(num_classes=21)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)
    num_epochs = 10
    print("Start training...")
    for epoch in range(num_epochs):
        print("Epoch:", epoch+1)
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Validation Accuracy: {100 * correct / total:.2f}%')


if __name__ == '__main__':
    main()
