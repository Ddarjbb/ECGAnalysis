import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os


# 定义 CNN 模型结构（必须和训练时一致）
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 加载数据集以获取类别映射（仅用于读取 label mapping）
class ECGImageDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = []
        self.labels = []

        for dir in os.listdir(root_dir):
            full_dir = os.path.join(root_dir, dir)
            for file_name in os.listdir(full_dir):
                parts = file_name.split('_')
                if len(parts) >= 2:
                    label = parts[1].split('.')[0]
                    self.image_files.append(file_name)
                    self.labels.append(label)

        # 创建类别映射
        self.classes = sorted(set(self.labels))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}


def predict_image(model, image_path, transform, class_names, device):
    """ 对单张图像进行预测 """
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]

    return predicted_class


def main():

    model_path = 'ecg_cnn_model.pth'
    dataset_path = r".\plots\labeled_data"
    image_path = r"D:\ECGAnalysis\plots\labeled_data\233\263_+.png"

    dataset = ECGImageDataset(dataset_path)
    class_names = dataset.classes


    num_classes = len(class_names)
    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 进行预测
    predicted_class = predict_image(model, image_path, transform, class_names, device)
    print(f"result：{predicted_class}")


if __name__ == '__main__':
    main()
