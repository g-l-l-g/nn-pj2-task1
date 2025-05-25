# PyTorch CIFAR-10 动态卷积神经网络项目

本项目实现了一个灵活的卷积神经网络 (CNN) 框架，使用 PyTorch 在 CIFAR-10 数据集上进行图像分类。它支持动态构建网络架构，包括标准卷积层、ResNet块，并集成了训练、评估、多种可视化工具（如滤波器、Grad-CAM、损失函数表面图、混淆矩阵和训练曲线）以及实验管理功能。

## 项目特点

*   动态模型构建: 通过配置文件定义CNN架构，轻松实验不同深度、宽度和层类型的网络。
*   ResNet 支持: 内置 ResBlock 实现，方便构建类 ResNet 架构。
*   模块化设计: 代码结构清晰，分为配置、数据加载、模型定义、训练、评估和工具函数等模块。
*   全面的可视化:
    *   训练/验证损失和准确率曲线。
    *   卷积层滤波器可视化。
    *   Grad-CAM 可视化，理解模型决策区域。
    *   3D 损失函数表面图 (静态和交互式 Plotly HTML)。
    *   分类混淆矩阵。
*   实验管理:
    *   通过Python配置文件管理不同的实验设置。
    *   自动将训练结果（模型权重、日志、图表、配置文件副本）保存到带时间戳的运行目录中。
    *   集成 TensorBoard 进行实时训练监控。
*   可配置的训练参数: 支持多种优化器 (Adam, SGD)、学习率调度器 (MultiStepLR, CosineAnnealingLR)、损失函数 (CrossEntropyLoss, MSELoss, L1Loss) 等。

## 项目结构
```
.
├── mynn/ # 主要的神经网络库模块
│   ├── __init__.py
│   ├── config/ # 配置文件目录
│   │   ├── __init__.py
│   │   ├── project_setup_config.py
│   │   ├── training_defaults_config.py
│   │   └── experiments/ # 存放各个实验的具体配置
│   │       ├── __init__.py
│   │       ├── simple_cnn_config.py
│   │       ├── dynamic_cnn_baseline_config.py
│   │       ├── dynamic_cnn_deeper_config.py
│   │       ├── resnet_cifar_config.py
│   │       └── deep_resnet_config.py
│   ├── criterion.py # 损失函数获取
│   ├── data_loader.py # CIFAR-10 数据加载
│   ├── evaluate.py # 模型评估逻辑
│   ├── lr_scheduler.py # 学习率调度器获取
│   ├── models.py # DynamicCNN 和 ResBlock 模型定义
│   ├── optimizer.py # 优化器获取
│   ├── train.py # 模型训练逻辑
│   └── utils/ # 工具函数和可视化脚本
│       ├── __init__.py
│       ├── image_show.py
│       ├── plot_3d_loss_surface.py
│       ├── plot_confusion_matrix.py
│       ├── plot_training_results.py
│       ├── visualize_filters.py
│       └── visualize_grad_cam.py
├── confusion_matrix_visualization.py  # 可视化结果生成脚本 (独立运行)
├── filter_visualization.py            # 可视化结果生成脚本
├── grad-cam_vsualization.py           # 可视化结果生成脚本
├── loss_surface_visualization.py      # 可视化结果生成脚本
├── main.py # 主程序入口，用于运行实验
├── test.py # 用于测试已训练模型的脚本
├── README.md # 本文件
└── requirements.txt # 项目依赖
```

## 环境配置

### 创建虚拟环境 (推荐):
```
    bash
    python -m venv venv
    # 或者使用 conda
    # conda create -n mycnn_env python=3.10
```

### 激活虚拟环境:
    *   Windows: venv\Scripts\activate
    *   macOS/Linux: source venv/bin/activate
    *   Conda: conda activate mycnn_env

### 安装依赖:
```
    bash
    pip install -r requirements.txt
```
    确保您安装的 PyTorch 版本与您的 CUDA 版本兼容（如果使用 GPU）。您可以访问 PyTorch官网 (`https://pytorch.org/get-started/locally/`) 获取特定平台的安装命令。

### 下载 CIFAR-10 数据集:
    *   首次运行` mynn/data_loader.py `中的 `get_cifar10_loaders` 函数时，如果` download=True` (默认未设置，建议手动下载或首次运行时修改)，会自动下载数据集到 `mynn/config/project_setup_config.py` 中` DATASET_ROOT_DIR` 指定的路径下（默认为` ./dataset`）。
    *   或者，您可以手动下载` CIFAR-10 Python `版本数据集，并解压到 `./dataset/cifar-10-batches-py `目录下。

## 如何使用

### 配置实验

实验配置位于 `mynn/config/experiments/ `目录下。每个 `.py` 文件定义了一个或多个实验的架构和训练参数。

*   ARCHITECTURE: 一个列表，定义了网络的层及其参数。支持的层类型包括 conv, bn, relu, pool, avgpool, adaptiveavgpool, flatten, fc, dropout, resblock。
*   EXPERIMENT_CONFIG: 一个字典，包含了模型类型、架构名称、优化器、学习率、损失函数、训练轮数、批大小等超参数。

例如，`deep_resnet_config.py 定义了一个类 ResNet-34 的架构。

要添加新的实验，可以在` mynn/config/experiments/ `目录下创建一个新的 Python 文件，定义 `ARCHITECTURE` 和 `EXPERIMENT_CONFIG`，然后在` mynn/config/experiments/__init__.py `中的 `ALL_EXPERIMENT_CONFIGS` 字典中注册它。

### 运行训练实验

使用` main.py `脚本来运行训练实验。
```
bash
python main.py
```
默认情况下，`main.py `中的 `experiments_to_run`列表指定了要运行的实验。您可以修改此列表来运行特定的实验或多个实验。

```
Python
# 在 main.py 中修改
if __name__ == '__main__':
    results_summary = []
    experiments_to_run = ["deep_resnet_config"] # 修改这里来选择实验
    # experiments_to_run = ["simple_cnn", "dynamic_cnn_deeper", "deep_resnet_config"] # 运行多个
    # ...
```
训练过程中，模型权重、TensorBoard日志、配置文件副本以及训练曲线图会自动保存到 ./runs/<timestamp>/<experiment_name>/ 目录下。

### 测试已训练的模型

使用` test.py `脚本来评估已训练模型在测试集上的性能。

您需要修改` test.py`底部的 `if __name__ == '__main__':` 部分，指定要测试的实验的时间戳、实验配置名称以及模型权重文件名。
```
Python
# 在 test.py 中修改
if __name__ == '__main__':
    experiment_time = '20250525-135249' # 替换为您的实验运行时间戳
    experiment_name = 'deep_resnet_config' # 替换为您的实验配置名称
    config_file_name = "full_experiment_config.json"
    weights_file_name = f"model_best_{experiment_name}.pth" # 通常是最佳模型

    EXPERIMENT_ROOT_PATH = f"./runs/{experiment_time}/{experiment_name}"
    CONFIG_FILE_PATH = os.path.join(EXPERIMENT_ROOT_PATH, config_file_name)
    TRAINED_MODEL_WEIGHTS_PATH = os.path.join(EXPERIMENT_ROOT_PATH, weights_file_name)

    # ... (检查文件是否存在) ...
    test_model_on_test_set(config_path_=CONFIG_FILE_PATH, model_weights_path_=TRAINED_MODEL_WEIGHTS_PATH)
```
然后运行：
```
bash
python test.py
```
### 生成可视化结果

项目在` ./visualizations/` 目录下提供了一些独立的 Python 脚本，用于对已训练的模型生成各种可视化图表。这些脚本通常需要指定实验配置文件的路径和模型权重文件的路径。

在每个可视化脚本的` if __name__ == '__main__':` 部分，您需要设置正确的` experiment_time`, `experiment_name`, `CONFIG_FILE_PATH` 和 `TRAINED_MODEL_WEIGHTS_PATH`。

#### 混淆矩阵:
```
bash
python visualizations/confusion_matrix_visualization.py
```
输出保存在` visualizations/confusion_matrix/<experiment_time>/<experiment_name>/`。

#### 滤波器可视化:
```
bash
python visualizations/filter_visualization.py
```
输出保存在` visualizations/filters/<experiment_time>/<experiment_name>/`。默认可视化第一个卷积层。

#### Grad-CAM 可视化:
```
bash
python visualizations/grad-cam_vsualization.py
```
默认处理3个样本。输出保存在 `visualizations/grad_cam/<experiment_time>/<experiment_name>/`。

#### 3D 损失函数表面图:
```
bash
python visualizations/loss_surface_visualization.py
```
默认基于最佳模型权重绘制。输出保存在 `visualizations/3d_loss_surface/<experiment_time>/<experiment_name>/`。该脚本可能计算量较大。

脚本内` visualization_parameters` 可以调整绘图的细节，例如采样点数和范围。

### 使用 TensorBoard

训练过程中会生成 TensorBoard 日志。要查看它们：
确保已安装 TensorBoard (pip install tensorboard)。
在命令行中运行：
```
bash
tensorboard --logdir ./runs
```
或者指定到某个具体实验的 tensorboard_logs 目录：
```
bash
tensorboard --logdir ./runs/<timestamp>/<experiment_name>/tensorboard_logs
```
在浏览器中打开 TensorBoard 提供的链接 (通常是`http://localhost:6006`)。

## 主要模块说明

`mynn.config`: 包含项目级别的设置（如数据集路径、类别名）、训练默认参数以及所有实验的具体配置。
`mynn.data_loader`: 负责加载和预处理 CIFAR-10 数据集，支持数据增强和验证集分割。
`mynn.models`: 定义了 DynamicCNN 类，它可以根据配置文件动态创建网络。还包括 ResBlock 的实现。
`mynn.train`: 包含核心的训练循环逻辑 train_model，处理模型训练、验证、学习率调度、权重保存和 TensorBoard 日志记录。
`mynn.evaluate`: 提供了 evaluate_model 函数，用于在测试集上评估模型性能并生成混淆矩阵。
`mynn.utils`: 包含各种辅助工具，如绘图函数 (训练曲线、混淆矩阵、3D损失图)、滤波器可视化和 Grad-CAM 实现。

