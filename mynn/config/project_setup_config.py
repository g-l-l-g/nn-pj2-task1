# mynn/config/project_setup_config.py

# 数据集类别
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 模型保存路径的基础目录
RUNS_DIR_BASE = "./runs"  # Updated example path
DEFAULT_MODEL_SAVE_FILENAME = "model_final.pth"    # 最终模型基础文件名
DEFAULT_BEST_MODEL_SAVE_FILENAME = "model_best.pth"  # 最佳模型基础文件名
DEFAULT_INITIAL_MODEL_SAVE_FILENAME = "model_initial.pth"  # 新增: 初始权重文件名

# 数据集路径
DATASET_ROOT_DIR = './dataset'
CIFAR10_FOLDER_NAME = 'cifar-10-batches-py'
