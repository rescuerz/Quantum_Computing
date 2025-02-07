import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

class MNISTDataset(Dataset):
    def __init__(self, trainingSize: float = 0.6, testSize: float = 0.2, validationSize: float = 0.2, num_classes: int = 10):
        # 加载数据集
        self.digits = load_digits()
        self.num_classes = num_classes

        # 选择前num_classes个类别的数据
        mask = self.digits.target < num_classes
        self.X = self.digits.images[mask][:, np.newaxis, :, :]
        self.y = self.digits.target[mask]

        # 转换标签为one-hot编码
        self.y_ = torch.unsqueeze(torch.tensor(self.y, dtype=int), 1)
        self.y_hot = torch.scatter(torch.zeros((self.y.size, self.num_classes)), 1, self.y_, 1).numpy()

        # 验证数据划分比例
        if not np.isclose(trainingSize + testSize + validationSize, 1.0):
            raise ValueError("The sum of sizes must be 1.")

        # 数据集划分
        self.X_train, self.X_test, self.y_train_hot, self.y_test_hot = train_test_split(
            self.X, 
            self.y_hot, 
            test_size=testSize + validationSize, 
            shuffle=True,
            stratify=self.y  # 分层采样确保每个类别比例一致
        )

        self.X_test, self.X_val, self.y_test_hot, self.y_val_hot = train_test_split(
            self.X_test, 
            self.y_test_hot, 
            test_size=testSize/(testSize + validationSize), 
            shuffle=True,
            stratify=np.argmax(self.y_test_hot, axis=1)  # 分层采样确保每个类别比例一致
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # 返回单个样本和其one-hot标签
        return self.X[idx], self.y_hot[idx]

    def getTrainData(self):
        """获取训练数据"""
        return self.X_train, self.y_train_hot

    def getTestData(self):
        """获取测试数据"""
        return self.X_test, self.y_test_hot

    def getValidationData(self):
        """获取验证数据"""
        return self.X_val, self.y_val_hot

    def show(self):
        """展示数据集描述和样本图像"""

        # 根据分类数设置展示数量
        if self.num_classes == 2:
            total_samples = 8
        elif self.num_classes == 4:
            total_samples = 8
        elif self.num_classes == 10:
            total_samples = 10
        else:
            raise ValueError("The number of classes should be 2, 4, or 10.")

        # 每个类别的样本数
        samples_per_class = total_samples // self.num_classes

        # 收集样本
        mask = self.digits.target < self.num_classes
        images = self.digits.images[mask]
        labels = self.digits.target[mask]

        selected_images = []
        selected_labels = []
        for cls in range(self.num_classes):
            # 获取当前类别的所有样本索引
            cls_indices = np.where(labels == cls)[0]
            # 确保每个类别都有均匀的样本
            selected_indices = cls_indices[:samples_per_class]
            selected_images.extend(images[selected_indices])
            selected_labels.extend(labels[selected_indices])

        # 将多余的样本补齐（如果total_samples无法被num_classes整除）
        remaining_samples = total_samples - len(selected_images)
        if remaining_samples > 0:
            extra_indices = np.random.choice(len(images), remaining_samples, replace=False)
            selected_images.extend(images[extra_indices])
            selected_labels.extend(labels[extra_indices])

        # 创建图像展示
        fig, axes = plt.subplots(2, total_samples//2, figsize=(total_samples, 4))
        if total_samples == 1:
            axes = [axes]  # 如果只有一个样本，确保axes是列表
        else:
            axes = axes.ravel()

        for i in range(total_samples):
            axes[i].set_axis_off()
            axes[i].imshow(selected_images[i], cmap=plt.cm.gray_r, interpolation='nearest')
            axes[i].set_title(f'Digit: {selected_labels[i]}')

        plt.tight_layout()
        plt.show()


# 测试代码
if __name__ == '__main__':
    num_classes = [2, 4, 10]
    for num_class in num_classes:
        print(f"{num_class}分类测试：")
        mnist = MNISTDataset(num_classes=num_class)
        mnist.show()
    
        # 获取并打印各数据集大小
        train_data, train_labels = mnist.getTrainData()
        test_data, test_labels = mnist.getTestData()
        val_data, val_labels = mnist.getValidationData()

        print(f"训练集大小: {len(train_data)}")
        print(f"测试集大小: {len(test_data)}")
        print(f"验证集大小: {len(val_data)}")

    
    
    

    