<!-- @format -->

# 量子神经网络实验项目

本项目实现了一个基于量子计算的神经网络模型，用于探索量子机器学习在图像分类任务中的应用。

## 项目结构

```
QuantumModel/
├── data/           # 数据集相关代码
│   └── mnist_dataset.py  # MNIST数据集加载和预处理
├── models/         # 模型定义
│   ├── base_model.py     # 模型基类
│   ├── classical_model.py # 经典神经网络模型
│   ├── hybrid_model.py   # 混合量子-经典模型
│   ├── param_gen.py      # 参数生成器
│   └── quantum_model.py  # 量子模型
├── tests/          # 单元测试
│   └── test.py          # 测试用例
├── utils/          # 工具函数
│   └── utils.py   # 工具函数
├── main.py         # 主程序入口
├── requirement.txt # 项目依赖
└── result.csv      # 实验结果
```

### 目录详细说明

- **data/**
  - `mnist_dataset.py`: 实现 MNIST 数据集的加载、预处理和数据增强
- **models/**
  - `base_model.py`: 定义模型的基础接口和通用功能
  - `classical_model.py`: 实现传统神经网络模型
  - `hybrid_model.py`: 实现混合量子-经典模型架构
  - `param_gen.py`: 实现模型参数的初始化和生成
  - `quantum_model.py`: 实现量子计算模型
- **tests/**
  - `test.py`: 包含模型功能和性能的单元测试
- **utils/**
  - `utils.py`: 提供模型训练、评估等通用工具函数

## 环境要求

1. 操作系统：Windows 10/11
2. 编程语言：Python 3.10
3. 核心依赖：
   - PyTorch >= 1.10.0
   - PennyLane >= 0.28.0
   - NumPy >= 1.21.0
   - scikit-learn >= 1.0.0
   - matplotlib >= 3.5.0

## 安装步骤

1. 创建并激活虚拟环境：

```bash
conda create -n qml python=3.10
conda activate qml
```

2. 安装依赖：

```bash
pip install -r requirement.txt
```

## 运行说明

```bash
python main.py
```

## 预期输出

训练过程将显示如下信息：

```
Loading dataset...
Starting training...
Epoch [1/10] Loss: 0.693 Accuracy: 45.6%
Epoch [2/10] Loss: 0.524 Accuracy: 68.3%
...
Training completed.
Final test accuracy: 85.7%
Results saved to result.csv
```

## 注意事项

1. 混合模型(QQ 和 QC)的训练时间较长，默认已注释相关代码
2. 实验结果将自动保存在 result.csv 文件中
