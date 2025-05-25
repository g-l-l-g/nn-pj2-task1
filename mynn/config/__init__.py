# mynn/config/__init__.py

# 从新的顶级配置文件导入
from .project_setup_config import (
    CLASSES,
    RUNS_DIR_BASE, DEFAULT_MODEL_SAVE_FILENAME, DEFAULT_BEST_MODEL_SAVE_FILENAME,
    DATASET_ROOT_DIR, CIFAR10_FOLDER_NAME
)
from .training_defaults_config import (
    DEVICE,
    DEFAULT_OPTIMIZER_TYPE, DEFAULT_LEARNING_RATE, DEFAULT_MOMENTUM, DEFAULT_WEIGHT_DECAY,
    DEFAULT_BATCH_SIZE, DEFAULT_NUM_EPOCHS, DEFAULT_LOSS_FUNCTION
)

# 导入并公开所有实验配置
from .experiments import ALL_EXPERIMENT_CONFIGS

__all__ = [
    # 来自 project_setup_config.py
    "CLASSES",
    "RUNS_DIR_BASE", "DEFAULT_MODEL_SAVE_FILENAME", "DEFAULT_BEST_MODEL_SAVE_FILENAME",
    "DATASET_ROOT_DIR", "CIFAR10_FOLDER_NAME",

    # 来自 training_defaults_config.py
    "DEVICE",
    "DEFAULT_OPTIMIZER_TYPE", "DEFAULT_LEARNING_RATE", "DEFAULT_MOMENTUM", "DEFAULT_WEIGHT_DECAY",
    "DEFAULT_BATCH_SIZE", "DEFAULT_NUM_EPOCHS", "DEFAULT_LOSS_FUNCTION",

    # 来自 experiments/__init__.py
    "ALL_EXPERIMENT_CONFIGS",
]